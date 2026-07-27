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


# ---------------------------------------------------------------------------
# Memory-bound-Familien (TZ 7): eigene FLOP-/Byte-Zählung (KEIN 2·B·M·N·K).
# GB/s ist hier die Primärmetrik; die arithmetische Intensität ist sehr niedrig
# (das ist die Roofline-Aussage: Punkte weit links).
# ---------------------------------------------------------------------------
def elementwise_flops(num_elements: int, op: str) -> int:
    """FLOP-Zahl einer elementweisen Op: 1 FLOP/Element für `add`/`mul`,
    **0** für `copy` (reine Bandbreite — es wird nicht gerechnet)."""
    return 0 if op == "copy" else num_elements


def elementwise_bytes(num_elements: int, arity: int, dtype: str, acc_dtype: str) -> int:
    """DRAM-Traffic einer elementweisen Op: `arity` Eingaben lesen (Compute-`dtype`)
    + einen Output schreiben (`acc_dtype`), je `num_elements` Elemente."""
    in_b = dtype_bytes(dtype)
    out_b = dtype_bytes(acc_dtype)
    return num_elements * (arity * in_b + out_b)


def reduction_flops(kept_size: int, reduced_size: int) -> int:
    """FLOP-Zahl einer Summen-Reduktion ~ `kept_size·reduced_size` Additionen
    (ein Add je gelesenem Element — die übliche Zählung)."""
    return kept_size * reduced_size


def reduction_bytes(kept_size: int, reduced_size: int, dtype: str, acc_dtype: str) -> int:
    """DRAM-Traffic einer Reduktion: die ganze Eingabe lesen (`kept·reduced`,
    Compute-`dtype`) + den kleinen Output schreiben (`kept`, `acc_dtype`)."""
    in_b = dtype_bytes(dtype)
    out_b = dtype_bytes(acc_dtype)
    return kept_size * reduced_size * in_b + kept_size * out_b


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


def _finish(flops: int, nbytes: Optional[int], run_ms: float,
            dtype: str) -> dict[str, Any]:
    """Gemeinsame Roofline-Kennzahlen aus FLOP-Zahl + Byte-Traffic (family-agnostisch).

    Keys: ``tflops`` (roh — `run.py` rundet ihn), sowie gerundet ``gbps``,
    ``arithmetic_intensity`` (FLOP/Byte), ``percent_peak_flops`` und
    ``percent_peak_bw``. dtype-/byte-abhängige Werte sind ``None``, wenn kein
    Peak (fp32/fp64) bzw. keine Byte-Größe bekannt ist — nie ein Fehler, der die
    Mess-Stufe kippt. (Bei ``flops==0`` — Elementwise ``copy`` — ist die AI 0,
    also GB/s die Primärmetrik; der Punkt sitzt roofline-technisch ganz links.)
    """
    out: dict[str, Any] = {"tflops": tflops(flops, run_ms)}
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


def compute_metrics(M: int, N: int, K: int, run_ms: float,
                    dtype: str, acc_dtype: str, B: int = 1,
                    epilog: Optional[str] = None) -> dict[str, Any]:
    """Kennzahlen-dict für eine **Kontraktion** (GEMM).

    `B` (Batch, Default 1 → Plain-GEMM unverändert) skaliert **FLOPs und Bytes
    gemeinsam** ⇒ ``tflops``/``gbps`` wachsen mit B, die arithmetische Intensität
    (FLOP/Byte) bleibt batch-**unabhängig** (B kürzt sich heraus) — physikalisch
    korrekt, damit batched Punkte auf der Roofline richtig sitzen.

    ``epilog`` (TZ 9, Fusion): ``"bias"`` liest zusätzlich den Operanden D in voller
    Ausgabe-Form (B·M·N Elemente, Compute-``dtype``) ⇒ dieser Extra-Traffic geht in
    die Bytes des **fusionierten** Kernels ein. ``"relu"``/``None`` bringen keinen
    Extra-Operanden. Die Fusion spart gegenüber dem sequentiellen Pfad den DRAM-Umweg
    des Zwischentensors (2·out·M·N·B Bytes, in ``measure/fusion.py`` beziffert) — hier
    steht die (höhere) AI des fused-Punkts; der sequentielle Punkt sitzt links davon.
    """
    flops = gemm_flops(M, N, K, B)
    try:
        nbytes: Optional[int] = gemm_bytes(M, N, K, dtype, acc_dtype, B)
        if epilog == "bias":
            nbytes += dtype_bytes(dtype) * B * M * N   # D-Read (Compute-dtype)
    except KeyError:
        nbytes = None
    return _finish(flops, nbytes, run_ms, dtype)


def compute_metrics_nary(steps: list, run_ms: float,
                         dtype: str, acc_dtype: str) -> dict[str, Any]:
    """Kennzahlen-dict für eine **n-äre Kontraktion** (Kette paarweiser GEMMs, TZ 7.5-3).

    ``steps`` = Liste von ``(M, N, K, B)`` je paarweisem Schritt. Aggregiert zu
    **einem** Roofline-Punkt: ``total_flops = Σ 2·B·M·N·K``; ``total_bytes`` = Summe
    der Per-Schritt-GEMM-Bytes — das schließt den **Zwischentensor-Traffic** ein (jeder
    Schritt liest seine zwei Operanden und schreibt sein Ergebnis, das der nächste
    Schritt wieder liest). So sitzt die Kette als *ein* Punkt korrekt auf der Roofline.
    """
    total_flops = sum(gemm_flops(M, N, K, B) for (M, N, K, B) in steps)
    try:
        total_bytes: Optional[int] = sum(
            gemm_bytes(M, N, K, dtype, acc_dtype, B) for (M, N, K, B) in steps)
    except KeyError:
        total_bytes = None
    return _finish(total_flops, total_bytes, run_ms, dtype)


def compute_metrics_elementwise(num_elements: int, arity: int, op: str, run_ms: float,
                                dtype: str, acc_dtype: str) -> dict[str, Any]:
    """Kennzahlen-dict für eine **Elementwise**-Op (memory-bound). `add`/`mul` =
    1 FLOP/Element, `copy` = 0 (reine Bandbreite). GB/s ist die Primärmetrik."""
    flops = elementwise_flops(num_elements, op)
    try:
        nbytes: Optional[int] = elementwise_bytes(num_elements, arity, dtype, acc_dtype)
    except KeyError:
        nbytes = None
    return _finish(flops, nbytes, run_ms, dtype)


def compute_metrics_reduction(kept_size: int, reduced_size: int, run_ms: float,
                              dtype: str, acc_dtype: str) -> dict[str, Any]:
    """Kennzahlen-dict für eine **Reduktion** (memory-bound). ~`kept·reduced`
    Additionen; Traffic = ganze Eingabe lesen + kleinen Output schreiben."""
    flops = reduction_flops(kept_size, reduced_size)
    try:
        nbytes: Optional[int] = reduction_bytes(kept_size, reduced_size, dtype, acc_dtype)
    except KeyError:
        nbytes = None
    return _finish(flops, nbytes, run_ms, dtype)
