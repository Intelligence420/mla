"""tool_pipeline.measure.metrics — abgeleitete Kennzahlen aus der Messung.

Aus M/N/K + gemessener `run_ms` + dtype werden die Roofline-Zutaten berechnet:
**TFLOP/s**, **erreichte GB/s**, **arithmetische Intensität** (FLOP/Byte) und
**%-vom-Peak** (Compute & Bandbreite) — mit den GB10-Peaks aus `hardware.py`.
Bewusst torch-/cuTile-frei (headless testbar); der Rückgabe-dict bleibt offen.
"""

from __future__ import annotations

from typing import Any, Optional

from ..hardware import MEM_BANDWIDTH_GBPS, dtype_bytes, peak_tflops


def gemm_flops(M: int, N: int, K: int, B: int = 1) -> int:
    """FLOP-Zahl eines (Batched-)GEMM: 2·B·M·N·K (je MAC eine Mult + eine Add).

    `B` = Batch (Produkt der Batch-Indizes; Default 1 → Plain-GEMM unverändert).
    """
    return 2 * B * M * N * K


def gemm_bytes(M: int, N: int, K: int, dtype: str, acc_dtype: str, B: int = 1) -> int:
    """Minimaler DRAM-Traffic eines (Batched-)GEMM C=A@B in Bytes.

    Liest A=(B,M,K) und B=(B,K,N) im Compute-`dtype`, schreibt C=(B,M,N) im
    Output-/Akku-`acc_dtype`. Das ist der **algorithmische** Mindest-Traffic (ohne
    Tiling-Rereads) — die übliche Roofline-Konvention für „erreichte GB/s“ und
    arithmetische Intensität. `B` skaliert den Traffic linear (Default 1).
    """
    in_b = dtype_bytes(dtype)
    out_b = dtype_bytes(acc_dtype)
    return B * (in_b * (M * K + K * N) + out_b * (M * N))


def tflops(flops: int, ms: float) -> float:
    """TFLOP/s aus FLOP-Zahl und Laufzeit in Millisekunden."""
    if ms <= 0:
        return float("nan")
    return flops / (ms * 1e-3) / 1e12


def gbps(num_bytes: int, ms: float) -> float:
    """Erreichte GB/s aus bewegten Bytes und Laufzeit in Millisekunden."""
    if ms <= 0:
        return float("nan")
    return num_bytes / (ms * 1e-3) / 1e9


def compute_metrics(M: int, N: int, K: int, run_ms: float,
                    dtype: str, acc_dtype: str, B: int = 1) -> dict[str, Any]:
    """Kennzahlen-dict für `RunResult.metrics`.

    Keys: ``tflops`` (roh — `run.py` rundet ihn), sowie gerundet ``gbps``,
    ``arithmetic_intensity`` (FLOP/Byte, deterministisch), ``percent_peak_flops``
    und ``percent_peak_bw``. dtype-abhängige Werte sind ``None``, wenn kein Peak
    (fp32/fp64) bzw. keine Byte-Größe bekannt ist — nie ein Fehler, der die
    Mess-Stufe kippt.

    `B` (Batch, Default 1 → Plain-GEMM unverändert) skaliert **FLOPs und Bytes
    gemeinsam** ⇒ ``tflops``/``gbps`` wachsen mit B, die arithmetische Intensität
    (FLOP/Byte) bleibt batch-**unabhängig** (B kürzt sich heraus) — physikalisch
    korrekt, damit batched Punkte auf der Roofline richtig sitzen.
    """
    flops = gemm_flops(M, N, K, B)
    out: dict[str, Any] = {"tflops": tflops(flops, run_ms)}

    # GB/s + arithm. Intensität nur bei bekannter dtype-Größe (sonst graceful None).
    try:
        nbytes: Optional[int] = gemm_bytes(M, N, K, dtype, acc_dtype, B)
    except KeyError:
        nbytes = None
    if nbytes:
        achieved_gbps = gbps(nbytes, run_ms)
        out["gbps"] = round(achieved_gbps, 2)
        out["arithmetic_intensity"] = round(flops / nbytes, 2)   # FLOP/Byte
        out["percent_peak_bw"] = round(achieved_gbps / MEM_BANDWIDTH_GBPS * 100.0, 1)
    else:
        out["gbps"] = None
        out["arithmetic_intensity"] = None
        out["percent_peak_bw"] = None

    peak = peak_tflops(dtype)
    out["percent_peak_flops"] = (round(out["tflops"] / peak * 100.0, 1)
                                 if peak else None)
    return out
