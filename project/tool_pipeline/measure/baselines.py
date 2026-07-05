"""tool_pipeline.measure.baselines — Vergleichs-Baselines (Ober-/Untergrenze).

Die Erkenntnis „was bringt unser Tuning?“ wird erst mit Bezugspunkten greifbar:

* **cuBLAS-Obergrenze** — ``torch.matmul`` (cuBLAS-Pfad) auf denselben Operanden:
  die hochoptimierte Bibliotheks-Leistung, gegen die sich unser Kernel misst.
* **naive-cuTile-Untergrenze** — derselbe (verifizierte) Codegen mit einem winzigen
  Tile (16×16×16) ohne Swizzle = „cuTile ohne Tuning“.

Beide werden mit **derselben** Event-Timing-Schleife wie der Haupt-Kernel gemessen
(`bench.benchmark`, cold-L2) → direkt vergleichbare TFLOP/s. Jede Baseline ist
optional (`RunConfig.baselines`) und scheitert **graceful** (Eintrag mit
``available=False`` + Grund), ohne den Lauf zu kippen. Baselines gehen **nicht**
in den Config-Slug ein (der Kernel-Quelltext ändert sich durch sie nicht).
"""

from __future__ import annotations

from typing import Any

import torch

from .bench import benchmark
from .metrics import gemm_flops, tflops

# Kanonische Baseline-Namen (RunConfig.baselines: list[str]).
KNOWN_BASELINES = ("cublas", "naive")

# Tile der naiven Untergrenze: bewusst winzig = „cuTile ohne Tuning“.
_NAIVE_TILE = {"TM": 16, "TN": 16, "TK": 16}


def _naive_cutile_launch(config):
    """Baue eine `launch(A,B,C)`-Closure des naiven cuTile-Kernels (Tile 16³).

    Nutzt exakt den bewiesenen Codegen-Pfad (`emit` → `load_kernel`) mit einer
    abgeleiteten Config, die sich nur im Tile unterscheidet → eigener Slug, eigene
    (gecachte) Kernel-Datei, kein Konflikt mit dem Haupt-Kernel.
    """
    from ..codegen.compile import load_kernel
    from ..schema import RunConfig

    naive_cfg = RunConfig(
        family=config.family, expr=config.expr,
        dim_sizes=dict(config.dim_sizes),
        dtype=config.dtype, acc_dtype=config.acc_dtype,
        tile=dict(_NAIVE_TILE), swizzle=False,
    )
    return load_kernel(naive_cfg).launch


def _measure_cublas(A: torch.Tensor, B: torch.Tensor, dtype: str,
                    warmup: int, iters: int) -> dict[str, Any]:
    """cuBLAS/torch.matmul-Timing auf denselben Operanden.

    Für ``dtype='tf32'`` wird der Tensor-Core-tf32-Pfad temporär aktiviert (fairer
    Vergleich zum tf32-Kernel); sonst rechnet matmul im nativen Format der Inputs.
    """
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = (dtype == "tf32")
    try:
        out = torch.matmul(A, B)                      # Probe-Aufruf (wirft z.B. bei fp8)
        b = benchmark(lambda a, bb, c: torch.matmul(a, bb, out=out),
                      A, B, out, warmup=warmup, iters=iters)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    return b


def measure_baselines(names, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                      config, warmup: int = 5, iters: int = 20) -> dict[str, Any]:
    """Miss die angeforderten Baselines → ``{name: {available, tflops, run_ms, ...}}``.

    :param names:  Teilmenge von `KNOWN_BASELINES` (unbekannte → available=False).
    :param A, B:   dieselben Operanden wie der Haupt-Kernel (Compute-dtype).
    :param C:      Output-Tensor des Hauptlaufs (nur als dtype/Shape-Vorlage genutzt;
                   die naive Baseline schreibt in eine EIGENE Kopie, damit das
                   bereits verifizierte C nicht überschrieben wird).
    :returns:      Dict je Baseline-Name; TFLOP/s über dieselbe FLOP-Zahl wie der
                   Haupt-Kernel → direkt vergleichbar.
    """
    M, K = A.shape
    _, N = B.shape
    flops = gemm_flops(M, N, K)
    result: dict[str, Any] = {}

    for name in names:
        if name not in KNOWN_BASELINES:
            result[name] = {"available": False, "note": f"unbekannte Baseline {name!r}"}
            continue
        try:
            if name == "cublas":
                b = _measure_cublas(A, B, config.dtype, warmup, iters)
            else:  # naive
                launch = _naive_cutile_launch(config)
                Cn = torch.empty_like(C)              # eigener Output, C bleibt unberührt
                b = benchmark(launch, A, B, Cn, warmup=warmup, iters=iters)
            run_ms = b["run_ms"]
            result[name] = {
                "available": True,
                "run_ms": round(run_ms, 5),
                "tflops": round(tflops(flops, run_ms), 3),
            }
        except Exception as e:  # noqa: BLE001
            result[name] = {"available": False,
                            "note": f"{type(e).__name__}: {str(e)[:200]}"}
    return result
