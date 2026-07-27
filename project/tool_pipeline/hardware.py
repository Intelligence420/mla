"""tool_pipeline.hardware — GB10-Roofline-Kennwerte (Peaks je dtype, Bandbreite).

Single Source of Truth der Hardware-Zahlen für die abgeleiteten Metriken
(%-vom-Peak in `measure/metrics.py`) und — ab TZ 5 — für das Roofline-Chart.
Bewusst **reine Daten** (torch-/cuTile-frei), damit auch die fork-sichere GUI
und headless-Tests sie ohne GPU-Kontext laden können.

Quelle der Zahlen: `project/project-development/analysis/RESULTS_gb10.md`,
empirisch auf der GB10 (Grace-Blackwell, sm_121) geklärt. TZ 4 hält es bewusst
**minimal** (nur Peak-Tabelle + Bandbreite + Byte-Größen); das Roofline-Chart,
das dieselben Zahlen weiternutzt, ist TZ 5.
"""

from __future__ import annotations

from typing import Optional

# --- Maschine (Reproduzierbarkeit) ---
GPU_NAME = "NVIDIA GB10 (Grace-Blackwell, sm_121)"

# --- Speicherbandbreite ---
# 273 GB/s theoretisch (unified LPDDR5x). Real ~70–85 % (keine publizierte
# STREAM-Zahl) → wir nehmen den theoretischen Wert als %-Peak-Nenner und
# dokumentieren die reale Spanne separat (relevant erst fürs TZ-5-Roofline).
MEM_BANDWIDTH_GBPS: float = 273.0
MEM_BANDWIDTH_REAL_FRACTION: tuple[float, float] = (0.70, 0.85)

# --- Peak-Rechenleistung je Compute-dtype (dense, gemessen mmapeak, TFLOP/s) ---
# fp16/bf16 ≈ 213, fp8 ≈ 214, tf32 ≈ 53. fp32-plain nutzt KEINE Tensor-Cores
# (kein sinnvoller Peak → None ⇒ %-Peak dort „—“ statt irreführendem Nenner);
# fp64 auf sm_12x vernachlässigbar (kein FP64-Tensor-Core) → ebenfalls None.
PEAK_TFLOPS: dict[str, Optional[float]] = {
    "fp16":    213.0,
    "bf16":    213.0,
    "tf32":    53.0,
    "fp8e4m3": 214.0,
    "fp8e5m2": 214.0,
    "fp32":    None,   # plain fp32: kein Tensor-Core-Peak
    "fp64":    None,   # kein FP64-Tensor-Core auf sm_12x
}

# --- Speichergröße je Element (Bytes) — für erreichte GB/s + arithm. Intensität.
# tf32-Operanden liegen als float32 im Speicher (4 B); der tfloat32-Cast passiert
# erst im Kernel. fp8 = 1 B, fp16/bf16 = 2 B, fp32 = 4 B, fp64 = 8 B.
DTYPE_BYTES: dict[str, int] = {
    "fp16":    2,
    "bf16":    2,
    "tf32":    4,
    "fp8e4m3": 1,
    "fp8e5m2": 1,
    "fp32":    4,
    "fp64":    8,
}


def peak_tflops(dtype: str) -> Optional[float]:
    """Peak-TFLOP/s des Compute-`dtype` (oder ``None``, wenn kein Tensor-Core-Peak)."""
    return PEAK_TFLOPS.get(dtype)


def dtype_bytes(dtype: str) -> int:
    """Speichergröße eines Elements des `dtype` in Bytes.

    :raises KeyError: für unbekannte dtype-Labels (bewusst hart — ein fehlender
                      Eintrag würde sonst still eine falsche GB/s-Zahl erzeugen;
                      der Aufrufer `compute_metrics` fängt das graceful ab).
    """
    return DTYPE_BYTES[dtype]


def ridge_point(dtype: str) -> Optional[float]:
    """Ridge-Point des `dtype` in FLOP/Byte — die arithmetische Intensität, ab der
    eine Operation vom Speicher- ins Rechen-Limit kippt (Schnittpunkt der
    Bandbreiten-Schräge mit der Peak-Decke).

    Die Bandbreiten-Schräge ist ``TFLOP/s = (MEM_BANDWIDTH_GBPS / 1000) · AI``
    (273 GB/s ⇒ Steigung 0.273); Gleichsetzen mit dem Peak ergibt
    ``ridge = peak_tflops(dtype) / (MEM_BANDWIDTH_GBPS / 1000)``. ``None``, wenn
    der `dtype` keinen Tensor-Core-Peak hat (fp32/fp64) — dann gibt es keinen
    Ridge (die Operation ist ohne Compute-Decke rein bandbreiten-beschränkt).

    Auf der GB10 sind die Ridges sehr hoch (bf16 ≈ 780, fp8 ≈ 784, tf32 ≈ 194
    FLOP/Byte); übliche GEMM-Größen (AI ≈ 128) liegen weit **links** davon ⇒
    memory-bound — die zentrale Roofline-Aussage des Tools (TZ 5).
    """
    peak = peak_tflops(dtype)
    if peak is None:
        return None
    return peak / (MEM_BANDWIDTH_GBPS / 1000.0)
