"""tool_pipeline.measure.verify — das verify-before-trust-Gate.

Jeder generierte Kernel wird gegen eine **fp32-Referenz** (`torch.einsum`)
geprüft, **bevor** seine Zahlen verwendet/angezeigt werden — das Sicherheitsnetz
gegen stille Falschergebnisse (v. a. mma-Orientierung, Risiko ①).

`verify()` ist bewusst ein **reiner Urteiler**: Es bekommt einen bereits
erzeugten Output-Tensor und vergleicht ihn mit der Referenz. Das Starten und
Timen des Kernels macht die Orchestrierung (`run.py`) bzw. die Mess-Schicht
(`bench.py`) — so passiert der teure cuTile-JIT (Kalt-Lauf) genau **einmal** und
zählt dort als `compile_ms`, statt hier verdeckt zu laufen.

TZ 1: nur `max_abs_err` + Pass/Fail (fp16-Toleranzen). Mean/rel-Fehler und die
dtype-abhängigen Toleranzen für bf16/tf32/fp8 kommen in TZ 3 — der Rückgabe-dict
ist dafür schon offen (nur Schlüssel ergänzen).
"""

from __future__ import annotations

from typing import Any

import torch

from ..schema import RunConfig

# dtype-Label → (atol, rtol). TZ 1: nur fp16 (Werte aus A03/A05).
# TZ 3 ergänzt hier bf16/tf32/fp8.
_TOLERANCES: dict[str, tuple[float, float]] = {
    "fp16": (2e-1, 2e-2),
}


def _tolerances(dtype: str) -> tuple[float, float]:
    if dtype not in _TOLERANCES:
        raise NotImplementedError(
            f"TZ 1: keine Toleranzen für dtype {dtype!r} definiert "
            f"(bf16/tf32/fp8 = TZ 3). Verfügbar: {sorted(_TOLERANCES)}."
        )
    return _TOLERANCES[dtype]


def verify(output: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
           config: RunConfig) -> dict[str, Any]:
    """Vergleiche einen Kernel-Output mit der fp32-`torch.einsum`-Referenz.

    :param output: der vom Kernel erzeugte Tensor (beliebiger dtype; wird zum
                   Vergleich nach fp32 hochgezogen).
    :param A, B:   die Eingabe-Operanden (in ihrem Compute-dtype).
    :param config: liefert `expr` (Referenz-Kontraktion) und `dtype` (Toleranzen).
    :returns:      ``{"max_abs_err", "passed", "atol", "rtol"}`` — passend zu
                   ``RunResult.accuracy``.
    """
    atol, rtol = _tolerances(config.dtype)

    # fp32-Referenz: dieselbe Kontraktion in voller Präzision.
    ref = torch.einsum(config.expr, A.float(), B.float())

    out_f = output.float()
    max_abs_err = (out_f - ref).abs().max().item()
    passed = bool(torch.allclose(out_f, ref, atol=atol, rtol=rtol))

    return {
        "max_abs_err": max_abs_err,
        "passed": passed,
        "atol": atol,
        "rtol": rtol,
    }
