"""tool_pipeline.measure.metrics — abgeleitete Kennzahlen aus der Messung.

TZ 1 minimal: nur **TFLOP/s** aus der GEMM-FLOP-Zahl und der gemessenen
`run_ms`. Erreichte GB/s, arithmetische Intensität und %-vom-Peak (mit den
GB10-Roofline-Peaks aus `hardware.py`) kommen in TZ 4 — der Rückgabe-dict ist
dafür offen (nur Schlüssel ergänzen).
"""

from __future__ import annotations

from typing import Any


def gemm_flops(M: int, N: int, K: int) -> int:
    """FLOP-Zahl eines GEMM: 2·M·N·K (je MAC eine Mult + eine Add)."""
    return 2 * M * N * K


def tflops(flops: int, ms: float) -> float:
    """TFLOP/s aus FLOP-Zahl und Laufzeit in Millisekunden."""
    if ms <= 0:
        return float("nan")
    return flops / (ms * 1e-3) / 1e12


def compute_metrics(M: int, N: int, K: int, run_ms: float) -> dict[str, Any]:
    """Kennzahlen-dict für `RunResult.metrics` (TZ 1: nur TFLOP/s)."""
    return {"tflops": tflops(gemm_flops(M, N, K), run_ms)}
