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

TZ 1 minimal: Warmup + wenige getaktete Iterationen → **Median**. Kein L2-Flush,
keine Verteilung (min/p90/σ), kein GB/s — das ist TZ 4. Der Rückgabe-dict ist
dafür offen (nur Schlüssel ergänzen).
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Callable

import torch


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


def benchmark(launch: Callable, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
              warmup: int = 10, iters: int = 30) -> dict[str, Any]:
    """Warme, CUDA-Event-getaktete Läufe → Median-`run_ms` (GPU-Kernel-Zeit).

    :param warmup: ungetaktete Aufwärm-Läufe (stabilisieren Takt/Caches).
    :param iters:  getaktete Läufe; je Lauf ein eigenes Event-Paar.
    """
    for _ in range(warmup):
        launch(A, B, C)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        launch(A, B, C)
        ends[i].record()
    torch.cuda.synchronize()

    times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return {
        "run_ms": statistics.median(times_ms),
        "iters": iters,
        "warmup": warmup,
    }
