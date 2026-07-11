"""Headless-Tests des verify-before-trust-Gates (TZ 3 / TODO 1).

`verify()` ist ein reiner Urteiler (Output-Tensor → Vergleich vs fp32-Referenz),
also ohne Dash **und ohne GPU** prüfbar: die (dtype, acc_dtype)-Toleranztabelle,
die neuen Metriken (mean_abs_err, rel_err) und die Acc-Regel-Verteidigung
(unzulässige Kombi → NotImplementedError) werden auf kleinen CPU-Tensoren geprüft.

Lauffähig standalone (`python tests/test_verify.py`, aus `project/`) **und** via
pytest. Braucht `torch` (CPU genügt), KEINE CUDA/cuTile.
"""

from __future__ import annotations

import math
import os
import sys

# project/ auf den Pfad, damit `tool_pipeline` importierbar ist (standalone-Lauf).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from tool_pipeline.measure.verify import (  # noqa: E402
    _TOLERANCES,
    _tolerances,
    verify,
)
from tool_pipeline.schema import ALLOWED_ACC, RunConfig  # noqa: E402

_ACCURACY_KEYS = {"max_abs_err", "mean_abs_err", "rel_err", "passed", "atol", "rtol"}


def _ref_output(delta=None):
    """Kleine, deterministische A/B + fp32-Referenz; optional gestörter Output."""
    torch.manual_seed(0)
    A = torch.randn(4, 3)
    B = torch.randn(3, 5)
    ref = torch.einsum("ik,kj->ij", A.float(), B.float())
    output = ref if delta is None else ref + delta
    return A, B, ref, output


def test_tolerances_cover_exactly_the_in_scope_combos():
    """Die Tabelle enthält genau die 9 in-scope (dtype, acc)-Kombis."""
    expected = {
        ("fp16", "fp32"), ("fp16", "fp16"),
        ("bf16", "fp32"),
        ("tf32", "fp32"),
        ("fp8e4m3", "fp32"), ("fp8e4m3", "fp16"),
        ("fp8e5m2", "fp32"), ("fp8e5m2", "fp16"),
        ("fp32", "fp32"),
    }
    assert set(_TOLERANCES) == expected, set(_TOLERANCES) ^ expected


def test_tolerances_agree_with_schema_acc_rules():
    """Anti-Drift: verify._TOLERANCES deckt genau die von schema.ALLOWED_ACC
    erlaubten (dtype, acc)-Kombis ab — eine Regel, zwei Verteidigungslinien."""
    from_rules = {(d, a) for d, accs in ALLOWED_ACC.items() for a in accs}
    assert set(_TOLERANCES) == from_rules, set(_TOLERANCES) ^ from_rules


def test_tolerances_fp16_gate_unchanged():
    """Regression: der fp16→fp32-Anker behält exakt die TZ-1-Toleranzen."""
    assert _tolerances("fp16", "fp32") == (2e-1, 2e-2)


def test_tolerances_reject_illegal_acc_combos():
    """Acc-Regel-Verteidigung: verbotene/unbekannte Kombis → NotImplementedError."""
    illegal = [("bf16", "fp16"), ("tf32", "fp16"), ("fp8e4m3", "bf16"), ("krass", "fp32")]
    for dtype, acc in illegal:
        try:
            _tolerances(dtype, acc)
        except NotImplementedError:
            continue
        raise AssertionError(f"({dtype}, {acc}) hätte NotImplementedError werfen müssen")


def test_verify_returns_all_accuracy_keys():
    """verify() liefert die vier alten + die zwei neuen Schlüssel (additiv)."""
    A, B, _, output = _ref_output()
    res = verify(output, [A, B], RunConfig())
    assert set(res) == _ACCURACY_KEYS, set(res) ^ _ACCURACY_KEYS


def test_verify_perfect_output_zero_error_passes():
    """Output == Referenz → alle Fehler 0, passed True."""
    A, B, _, output = _ref_output()  # output = ref
    res = verify(output, [A, B], RunConfig())
    assert res["passed"] is True
    assert res["max_abs_err"] == 0.0 and res["mean_abs_err"] == 0.0 and res["rel_err"] == 0.0


def test_verify_computes_mean_and_rel_error():
    """max/mean/rel werden korrekt aus dem Fehler-Tensor berechnet."""
    torch.manual_seed(0)
    A = torch.randn(4, 3)
    B = torch.randn(3, 5)
    ref = torch.einsum("ik,kj->ij", A.float(), B.float())
    delta = torch.full_like(ref, 0.01)
    delta[0, 0] = 0.5  # ein Ausreißer → max != mean
    res = verify(ref + delta, [A, B], RunConfig())
    assert math.isclose(res["max_abs_err"], delta.abs().max().item(), rel_tol=1e-6)
    assert math.isclose(res["mean_abs_err"], delta.abs().mean().item(), rel_tol=1e-6)
    exp_rel = (delta.norm() / ref.norm()).item()
    assert math.isclose(res["rel_err"], exp_rel, rel_tol=1e-6)


def test_verify_flags_gross_error_as_failed():
    """Ein grob falscher Output überschreitet die Toleranz → passed False."""
    A, B, ref, _ = _ref_output()
    bad = ref + 1000.0  # weit jenseits atol/rtol
    res = verify(bad, [A, B], RunConfig())
    assert res["passed"] is False
    assert res["max_abs_err"] > 100.0


def test_verify_selects_gate_by_dtype_and_acc():
    """Die zurückgegebenen atol/rtol stammen aus der (dtype, acc)-Tabelle."""
    A, B, _, output = _ref_output()
    res = verify(output, [A, B], RunConfig(dtype="fp8e5m2", acc_dtype="fp32"))
    assert (res["atol"], res["rtol"]) == _TOLERANCES[("fp8e5m2", "fp32")]


# --- Memory-bound-Referenzen (TZ 7): op-abhängig, NICHT immer einsum -----------
def _elem_cfg(op):
    expr = "ij->ij" if op == "copy" else "ij,ij->ij"
    return RunConfig(family="elementwise", op=op, expr=expr, dtype="fp32", acc_dtype="fp32")


def test_verify_elementwise_add_reference():
    """add-Referenz ist A+B (KEIN einsum-Ausdruck) — korrekter Output passt,
    ein mul-Output (was einsum liefern würde) fällt durch → beweist die op-Wahl."""
    torch.manual_seed(0)
    A = torch.randn(6, 4)
    B = torch.randn(6, 4)
    cfg = _elem_cfg("add")
    assert verify(A + B, [A, B], cfg)["passed"] is True
    # Gegenprobe: A*B (der einsum('ij,ij->ij')-Wert) ist NICHT die add-Referenz.
    assert verify(A * B, [A, B], cfg)["passed"] is False


def test_verify_elementwise_mul_reference():
    """mul-Referenz ist A*B."""
    torch.manual_seed(1)
    A = torch.randn(5, 5)
    B = torch.randn(5, 5)
    assert verify(A * B, [A, B], _elem_cfg("mul"))["passed"] is True


def test_verify_elementwise_copy_reference():
    """copy-Referenz ist A (unär, 1 Operand)."""
    torch.manual_seed(2)
    A = torch.randn(8, 3)
    assert verify(A.clone(), [A], _elem_cfg("copy"))["passed"] is True
    assert verify(A + 1.0, [A], _elem_cfg("copy"))["passed"] is False


def test_verify_reduction_reference():
    """Reduktions-Referenz ist torch.einsum('ij->i', A) = zeilenweise Summe."""
    torch.manual_seed(3)
    A = torch.randn(7, 9)
    cfg = RunConfig(family="reduction", op="sum", expr="ij->i", dtype="fp32", acc_dtype="fp32")
    assert verify(A.sum(dim=1), [A], cfg)["passed"] is True
    assert verify(A.sum(dim=0)[:7], [A], cfg)["passed"] is False  # falsche Achse


def _main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} Tests bestanden")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
