"""tool_pipeline.measure.verify — das verify-before-trust-Gate.

Jeder generierte Kernel wird gegen eine **fp32-Referenz** (`torch.einsum`)
geprüft, **bevor** seine Zahlen verwendet/angezeigt werden — das Sicherheitsnetz
gegen stille Falschergebnisse (v. a. mma-Orientierung, Risiko ①).

`verify()` ist bewusst ein **reiner Urteiler**: Es bekommt einen bereits
erzeugten Output-Tensor und vergleicht ihn mit der Referenz. Das Starten und
Timen des Kernels macht die Orchestrierung (`run.py`) bzw. die Mess-Schicht
(`bench.py`) — so passiert der teure cuTile-JIT (Kalt-Lauf) genau **einmal** und
zählt dort als `compile_ms`, statt hier verdeckt zu laufen.

TZ 3: zusätzlich `mean_abs_err` + `rel_err` (L2, relativ zur fp32-Referenznorm)
und eine nach **(dtype, acc_dtype)** gekeyte Toleranztabelle (bf16/tf32/fp8).
Der Rückgabe-dict war dafür schon offen — es kamen nur Schlüssel dazu.
"""

from __future__ import annotations

from typing import Any

import torch

from ..schema import RunConfig

# (dtype-Label, acc_dtype-Label) → (atol, rtol) für torch.allclose.
#
# Grundlage: die auf GB10 gemessenen max_abs_err (analysis/RESULTS_gb10.md,
# reproduziert) mit großzügigem Abstand nach oben — so ist ein korrekter Kernel
# über wechselnde Größen nie falsch-negativ, ein grober Fehler (v. a.
# mma-Orientierung, Risiko ①) wird aber sicher gefangen. Die Werte spiegeln die
# bewiesenen Pass-Gates aus analysis/dtype_analyse.py.
#
# Die Tabelle ist zugleich die zweite Verteidigungslinie der Acc-Regeln
# (bf16/tf32 → nur fp32; fp16/fp8 → fp16|fp32): eine unzulässige Kombi steht hier
# nicht und führt über `_tolerances` zu einem sauberen Fehler-Status.
_TOLERANCES: dict[tuple[str, str], tuple[float, float]] = {
    # fp16 (Anker): fp32-Akku unverändert aus TZ 1 (Werte aus A03/A05) …
    ("fp16", "fp32"): (2e-1, 2e-2),
    # … und fp16-Akku (fp16-Output rundet gröber; gemessen max_abs ≈ 0.22).
    ("fp16", "fp16"): (8.0, 2e-1),
    # bf16 & tf32: Akku IMMER fp32 (Pflicht); tf32 = astype-Cast im Kernel.
    ("bf16", "fp32"): (1.0, 2e-2),
    ("tf32", "fp32"): (1.0, 2e-2),
    # fp8 e4m3: fp32-Akku (max_abs ≈ 1.5e-5) oder fp16-Akku (schneller, ≈ 0.16).
    ("fp8e4m3", "fp32"): (8.0, 2e-1),
    ("fp8e4m3", "fp16"): (8.0, 2e-1),
    # fp8 e5m2 (gröberes Mantissen-Format → etwas lockerer).
    ("fp8e5m2", "fp32"): (16.0, 3e-1),
    ("fp8e5m2", "fp16"): (16.0, 3e-1),
    # Anker/Diagnose: reines fp32 (kein Tensor-Core), sehr straff.
    ("fp32", "fp32"): (1e-2, 1e-3),
}


def _tolerances(dtype: str, acc_dtype: str) -> tuple[float, float]:
    key = (dtype, acc_dtype)
    if key not in _TOLERANCES:
        raise NotImplementedError(
            f"keine Toleranzen für Kombi dtype={dtype!r}, acc_dtype={acc_dtype!r} "
            f"definiert (Acc-Regeln: bf16/tf32→fp32; fp16/fp8→fp16|fp32). "
            f"Verfügbar: {sorted(_TOLERANCES)}."
        )
    return _TOLERANCES[key]


def verify(output: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
           config: RunConfig) -> dict[str, Any]:
    """Vergleiche einen Kernel-Output mit der fp32-`torch.einsum`-Referenz.

    :param output: der vom Kernel erzeugte Tensor (beliebiger dtype; wird zum
                   Vergleich nach fp32 hochgezogen).
    :param A, B:   die Eingabe-Operanden (in ihrem Compute-dtype).
    :param config: liefert `expr` (Referenz-Kontraktion) sowie `dtype`/`acc_dtype`
                   (Toleranzen).
    :returns:      ``{"max_abs_err", "mean_abs_err", "rel_err", "passed", "atol",
                   "rtol"}`` — passend zu ``RunResult.accuracy``.
    """
    atol, rtol = _tolerances(config.dtype, config.acc_dtype)

    # fp32-Referenz: dieselbe Kontraktion in voller Präzision.
    ref = torch.einsum(config.expr, A.float(), B.float())

    out_f = output.float()
    diff = out_f - ref
    max_abs_err = diff.abs().max().item()
    mean_abs_err = diff.abs().mean().item()
    # Relativer Fehler: L2-Norm des Fehlers relativ zur fp32-Referenznorm —
    # dimensionslos und damit über Formate vergleichbar (Scatter-Y-Achse).
    ref_norm = ref.norm()
    rel_err = (diff.norm() / ref_norm).item() if ref_norm > 0 else 0.0
    passed = bool(torch.allclose(out_f, ref, atol=atol, rtol=rtol))

    return {
        "max_abs_err": max_abs_err,
        "mean_abs_err": mean_abs_err,
        "rel_err": rel_err,
        "passed": passed,
        "atol": atol,
        "rtol": rtol,
    }
