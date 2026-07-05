"""tool_pipeline.measure.bench — Zeitmessung (CUDA-Events + Wall-Clock).

Zwei Messungen, bewusst getrennt (die Plan-Reihenfolge schiebt `verify`
dazwischen):

* `time_first_launch` — **Wall-Clock** des ersten (kalten) Launches. Der
  cuTile-JIT ist ein **host-seitiger** Kompilierschritt (mehrere hundert ms) und
  passiert lazy beim ersten `ct.launch`. CUDA-Events messen nur GPU-Zeit und
  würden ihn verpassen → hier `time.perf_counter()` + `synchronize()`. Ergibt
  `compile_ms` und füllt zugleich den Output-Tensor (den dann `verify` beurteilt).
* `benchmark` — **CUDA-Event**-getaktete **warme** Läufe → Median-`run_ms`
  (reine GPU-Kernel-Zeit).

TZ 4: Warmup + getaktete Iterationen mit **L2-Flush** zwischen den Läufen
(cold-L2 als Default, wie `triton.do_bench`) → **Verteilung** statt nur Median:
`run_ms` (Median), `min_ms`, `p90_ms`, `sigma_ms`. Der Rückgabe-dict bleibt offen;
GB/s/%-Peak liegen bewusst in `metrics.py` (aus M/N/K/dtype abgeleitet), nicht
hier. `time_first_launch` (= `compile_ms`) bleibt getrennt.
"""

from __future__ import annotations

import math
import statistics
import time
from typing import Any, Callable

import torch

# Größe des L2-Flush-Puffers: großzügig über jeder realistischen GPU-L2-Größe
# (die GB10-L2-Größe ist in den Analyse-Dateien nicht belegt → konservativ groß,
# so wie `triton.do_bench` einen ~256-MB-Clear-Buffer nutzt). Der Puffer wird
# zwischen den getakteten Iterationen genullt und verdrängt so die Kernel-Daten
# aus dem L2 → jede Messung startet „kalt“ (cold-L2, PLAN §3 Default).
_L2_FLUSH_BYTES = 256 * 1024 * 1024


def time_first_launch(launch: Callable, A: torch.Tensor, B: torch.Tensor,
                      C: torch.Tensor) -> float:
    """Wall-Clock-ms des ersten (kalten) Launches — inkl. cuTile-JIT.

    Füllt `C` mit dem Kernel-Ergebnis (für die anschließende `verify`-Prüfung).
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    launch(A, B, C)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


def _summarize_times(times_ms: list[float]) -> dict[str, float]:
    """Verteilungs-Kennzahlen aus den je Iteration gemessenen Zeiten.

    `run_ms` bleibt der **Median** (robust gegen Ausreißer, unveränderter Key).
    `p90_ms` ist das 90.-Perzentil per **nearest-rank** (deterministisch,
    interpolationsfrei — so ist der Wert im Test exakt vorhersagbar). `sigma_ms`
    ist die **Populations**-Standardabweichung über alle gemessenen Iterationen
    (nicht Stichprobe: wir messen die Grundgesamtheit der Läufe; robust für n=1).
    """
    s = sorted(times_ms)
    n = len(s)
    if n == 0:
        nan = float("nan")
        return {"run_ms": nan, "min_ms": nan, "p90_ms": nan, "sigma_ms": nan}
    p90_idx = min(n - 1, math.ceil(0.9 * n) - 1)
    return {
        "run_ms": statistics.median(s),
        "min_ms": s[0],
        "p90_ms": s[p90_idx],
        "sigma_ms": statistics.pstdev(s) if n > 1 else 0.0,
    }


def benchmark(launch: Callable, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
              warmup: int = 10, iters: int = 30, flush_l2: bool = True) -> dict[str, Any]:
    """Warme, CUDA-Event-getaktete Läufe → **Verteilung** der GPU-Kernel-Zeit.

    :param warmup:   ungetaktete Aufwärm-Läufe (stabilisieren Takt/Caches).
    :param iters:    getaktete Läufe; je Lauf ein eigenes Event-Paar.
    :param flush_l2: L2 zwischen den Iterationen leeren (cold-L2, PLAN §3
                     Default). Der Flush wird VOR dem Start-Event abgesetzt und
                     zählt daher NICHT in die gemessene Kernel-Zeit.
    :returns: ``{"run_ms"(Median), "min_ms", "p90_ms", "sigma_ms", "iters",
              "warmup"}`` — additiv zu TZ 1 (nur neue Schlüssel dazu).
    """
    for _ in range(warmup):
        launch(A, B, C)
    torch.cuda.synchronize()

    # Flush-Puffer EINMAL allozieren, dann je Iteration nullen (die Allokation
    # selbst würde sonst mitzählen). int8 → 1 Byte/Element, Größe = _L2_FLUSH_BYTES.
    # Scheitert die Allokation (GPU fast voll auf der geteilten Maschine), messen
    # wir OHNE Flush weiter — besser als einen bereits verifizierten Kernel über
    # eine OOM-Exception zu einem run_error zu degradieren.
    flush_buf = None
    if flush_l2:
        try:
            flush_buf = torch.empty(_L2_FLUSH_BYTES, dtype=torch.int8, device=C.device)
        except RuntimeError:      # torch.cuda.OutOfMemoryError ist eine RuntimeError-Unterklasse
            flush_buf = None

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if flush_buf is not None:
            # Verdrängt die Kernel-Daten aus dem L2, BEVOR das Start-Event fällt
            # (Stream-Reihenfolge) → kalt gemessen, ohne den Flush mitzutakten.
            flush_buf.zero_()
        starts[i].record()
        launch(A, B, C)
        ends[i].record()
    torch.cuda.synchronize()

    times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    d: dict[str, Any] = _summarize_times(times_ms)
    d["iters"] = iters
    d["warmup"] = warmup
    return d
