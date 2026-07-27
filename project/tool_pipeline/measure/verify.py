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

TZ 7: `verify(output, operands, config)` ist **variadisch** (1 oder 2 Operanden)
und **family-/op-abhängig** in der Referenz: Kontraktion/Reduktion über
`torch.einsum`, Elementwise `add`/`mul`/`copy` direkt aus der Op (`add`/`copy`
sind kein einsum-Ausdruck). Die Toleranztabelle bleibt family-neutral.
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
    # fp8 e4m3 & e5m2: pro Akku GETRENNT gaten (sonst prüfte der genaue fp32-Akku-
    # Pfad so lasch wie der grobe fp16-Akku-Pfad). fp32-Akku ist sehr genau
    # (max_abs ≈ 1.5e-5 @512³, da die fp8-Quantisierung aus dem Diff gegen die
    # fp8-Referenz herausfällt) → straffes Gate wie der fp16→fp32-Anker. fp16-Akku
    # ist grob (≈ 0.16) → eigenes lockeres Gate (e5m2 mit gröberer Mantisse lockerer).
    ("fp8e4m3", "fp32"): (2e-1, 2e-2),
    ("fp8e4m3", "fp16"): (8.0, 2e-1),
    ("fp8e5m2", "fp32"): (2e-1, 2e-2),
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


def _reference(config: RunConfig, operands: list) -> torch.Tensor:
    """fp32-Referenz je Operations-Familie/Op (alles in voller Präzision).

    Kontraktion **und** Reduktion (Summe) drückt `torch.einsum(expr, *ops)` aus.
    Elementwise ist gemischt: `mul` ließe sich als einsum schreiben, `add`/`copy`
    **nicht** — daher wird die Elementwise-Referenz direkt aus der Op berechnet
    (das ist genau die geforderte `A (op) B`- bzw. `A`-Referenz).
    """
    ops_f = [o.float() for o in operands]
    if config.family == "elementwise":
        op = config.op
        if op == "add":
            return ops_f[0] + ops_f[1]
        if op == "mul":
            return ops_f[0] * ops_f[1]
        if op == "copy":
            return ops_f[0]
        if op == "relu":
            return ops_f[0].clamp(min=0)
        raise NotImplementedError(
            f"Elementwise-Op {op!r} hat keine verify-Referenz."
        )
    # Kontraktion mit Epilog-Fusion (TZ 9): die Referenz ist die Kontraktion
    # GEFOLGT vom Epilog (torch.einsum(...) dann +bias / relu). Die ersten
    # len(inputs) Operanden speisen das einsum; etwaige weitere sind Epilog-Operanden
    # (bias: D in Ausgabe-Form). Nur der 2-Op-Kontraktions-Pfad setzt epilog; n-är
    # und reine Kontraktion (epilog=None) fallen unverändert auf das einsum unten.
    if config.family == "contraction" and config.epilog:
        n_in = len(config.inputs or [])
        base = torch.einsum(config.expr, *ops_f[:n_in])
        return _apply_epilog_reference(config.epilog, base, ops_f[n_in:])
    # Reduktion (Summe) + Kontraktion (ohne Epilog): einsum deckt beide ab.
    return torch.einsum(config.expr, *ops_f)


def _apply_epilog_reference(epilog: str, base: torch.Tensor,
                            extra_ops: list) -> torch.Tensor:
    """fp32-Referenz des Kontraktions-Epilogs auf dem einsum-Ergebnis ``base``.

    ``bias`` addiert den (einzigen) Extra-Operanden D (Ausgabe-Form); ``relu``
    schneidet bei 0 ab (operandenlos). Exakt die Ops, die der fusionierte Kernel
    auf dem Akku-Tile ausführt — nur hier in voller fp32-Präzision.
    """
    if epilog == "bias":
        return base + extra_ops[0]
    if epilog == "relu":
        return base.clamp(min=0)
    raise NotImplementedError(f"Epilog {epilog!r} hat keine verify-Referenz.")


def verify(output: torch.Tensor, operands: list, config: RunConfig) -> dict[str, Any]:
    """Vergleiche einen Kernel-Output mit der fp32-Referenz (family-/op-abhängig).

    :param output:   der vom Kernel erzeugte Tensor (beliebiger dtype; wird zum
                     Vergleich nach fp32 hochgezogen).
    :param operands: Liste der Eingabe-Operanden (in ihrem Compute-dtype) — 1
                     (unär/Reduktion) **oder** 2 (binär/Kontraktion). Variadisch
                     (TZ 7) statt fest `A, B`.
    :param config:   liefert `family`/`op`/`expr` (Referenz) sowie `dtype`/`acc_dtype`
                     (Toleranzen).
    :returns:        ``{"max_abs_err", "mean_abs_err", "rel_err", "passed", "atol",
                     "rtol"}`` — passend zu ``RunResult.accuracy``.
    """
    atol, rtol = _tolerances(config.dtype, config.acc_dtype)

    # fp32-Referenz: dieselbe Operation in voller Präzision.
    ref = _reference(config, operands)

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
