"""Korrektheitsnetz für den Codegen (TZ 1): generierter GEMM vs. fp32.

Der Codegen ist die Hauptquelle **stiller** Falschergebnisse (v. a. die
mma-Orientierung, Risiko ①). Diese Tests fahren den **echten** Codegen-Pfad
(`emit` → `compile.load_kernel` → `launch`) auf der GPU und prüfen gegen eine
fp32-`torch.einsum`-Referenz — plus einen expliziten **Orientierungs-Wächter**,
der `A@B` von seinen transponierten Doppelgängern unterscheidet.

Lauffähig standalone (`python tests/test_codegen.py`, aus `project/`) **und**
via pytest. Braucht GPU + cuTile (fährt echte Kernel).
"""

from __future__ import annotations

import os
import sys

import torch

# project/ auf den Pfad, damit `tool_pipeline` importierbar ist (standalone-Lauf).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool_pipeline.codegen.compile import clear_cache, load_kernel  # noqa: E402
from tool_pipeline.codegen.emit import emit  # noqa: E402
from tool_pipeline.codegen.templates.contraction import build_gemm_module  # noqa: E402
from tool_pipeline.measure.verify import _TOLERANCES  # noqa: E402
from tool_pipeline.schema import RunConfig, RunResult  # noqa: E402

# Output ist fp32 (= acc_dtype) → STRAFFE Toleranz: der reale fp32-Akku-Fehler
# liegt ~1e-4, eine fp16-Akku/-Output-Regression (~3e-2) MUSS damit failen.
# (Die lockeren fp16-Output-Toleranzen 2e-1/2e-2 gehören erst in TZ 3.)
ATOL, RTOL = 1e-2, 1e-3

# Compute-/Akku-dtype-Label → torch-dtype für die Test-Operanden.
_TORCH = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def _run_gemm(M: int, N: int, K: int, dtype: str = "fp16", acc: str = "fp32"):
    """Echten Codegen-Pfad fahren: emit → load → launch. Gibt (A, B, C).

    `dtype`/`acc` steuern Compute- bzw. Akku-/Output-dtype (Default fp16→fp32 =
    der TZ-1-Anker, unverändert). Die weiteren in-scope-Formate werden pro
    TZ-3-Teilschritt freigeschaltet.
    """
    cfg = RunConfig(dim_sizes={"i": M, "k": K, "j": N}, dtype=dtype, acc_dtype=acc)
    launch = load_kernel(cfg, emit(cfg)).launch
    torch.manual_seed(0)
    in_t, out_t = _TORCH[dtype], _TORCH[acc]
    A = torch.randn(M, K, dtype=in_t, device="cuda")
    B = torch.randn(K, N, dtype=in_t, device="cuda")
    C = torch.empty(M, N, dtype=out_t, device="cuda")   # Output = acc_dtype
    launch(A, B, C)
    torch.cuda.synchronize()
    return A, B, C


def _assert_matches_fp32(M: int, N: int, K: int, dtype: str, acc: str):
    """Generierter Kernel stimmt gegen die fp32-Referenz — mit der
    **Produktions-Toleranz** des Formats (Quelle: measure.verify._TOLERANCES,
    keine Duplikat-Werte)."""
    atol, rtol = _TOLERANCES[(dtype, acc)]
    A, B, C = _run_gemm(M, N, K, dtype, acc)
    ref = torch.einsum("ik,kj->ij", A.float(), B.float())
    assert C.shape == (M, N), f"{dtype}->{acc} {(M, N, K)}: Shape {tuple(C.shape)}"
    assert C.dtype == _TORCH[acc], f"{dtype}->{acc}: Output-dtype {C.dtype} != {acc}"
    err = (C.float() - ref).abs().max().item()
    assert torch.allclose(C.float(), ref, atol=atol, rtol=rtol), \
        f"{dtype}->{acc} {(M, N, K)} weicht ab: max_abs_err={err:.3e} (atol={atol}, rtol={rtol})"


def _assert_orientation(dtype: str, acc: str):
    """Orientierungs-Wächter je Format: exakt A@B, kein transponierter Doppelgänger.

    Quadratische Inputs → alle Doppelgänger sind shape-legal, nur die Zahlen
    unterscheiden sie. err_AB < 1.0 passt für alle in-scope-Formate (auch die
    fp16-/fp8-Akku-Pfade mit ~0.2 realem Fehler)."""
    M = N = K = 256
    A, B, C = _run_gemm(M, N, K, dtype, acc)
    Af, Bf, Cf = A.float(), B.float(), C.float()
    err_AB = (Cf - Af @ Bf).abs().max().item()
    imposters = {
        "A@B^T": (Cf - Af @ Bf.T).abs().max().item(),
        "B@A": (Cf - Bf @ Af).abs().max().item(),
        "(A@B)^T": (Cf - (Af @ Bf).T).abs().max().item(),
    }
    assert err_AB < 1.0, f"{dtype}->{acc}: A@B sollte passen, err={err_AB:.3e}"
    assert min(imposters.values()) > 10.0, \
        f"{dtype}->{acc}: ein Doppelgänger liegt verdächtig nah: {imposters}"


def test_gemm_correct_across_sizes():
    """Generierter GEMM stimmt gegen fp32 — glatte UND ragged (Padding-)Größen.

    Enthält bewusst nicht-tile-teilbare Größen, damit der Rand-Pfad geprüft wird
    (ZERO-Padding im K-Loop ist im MAC neutral, ct.store clippt M/N-Ränder) —
    sonst bliebe genau dieser Pfad zu 0 % abgedeckt.
    """
    sizes = [
        (512, 512, 512), (256, 384, 128), (128, 128, 64),   # glatt (Tile-Vielfache)
        (130, 100, 70), (129, 127, 65), (1, 1, 1),          # ragged: M%TM,N%TN,K%TK != 0
    ]
    for (M, N, K) in sizes:
        A, B, C = _run_gemm(M, N, K)
        ref = torch.einsum("ik,kj->ij", A.float(), B.float())
        assert C.shape == (M, N), f"falsche Output-Shape bei {(M, N, K)}: {tuple(C.shape)}"
        # Output-dtype muss fp32 bleiben (= acc_dtype); eine fp16-Akku/-Output-
        # Regression würde hier UND an der straffen Toleranz scheitern.
        assert C.dtype == torch.float32, f"Output-dtype {C.dtype} != fp32 bei {(M, N, K)}"
        err = (C - ref).abs().max().item()
        assert torch.allclose(C, ref, atol=ATOL, rtol=RTOL), \
            f"GEMM {(M, N, K)} weicht ab: max_abs_err={err:.3e}"


def test_gemm_computes_AB_not_transpose():
    """Orientierungs-Wächter: der Kernel rechnet exakt A@B, keinen Doppelgänger.

    Auf quadratischen Inputs sind A@B, A@B^T, B@A und (A@B)^T alle shape-legal —
    nur die Zahlen unterscheiden sie. Genau hier würde ein stiller
    mma-Orientierungsfehler durchschlüpfen.
    """
    M = N = K = 256
    A, B, C = _run_gemm(M, N, K)
    Af, Bf = A.float(), B.float()
    err_AB = (C - Af @ Bf).abs().max().item()
    imposters = {
        "A@B^T": (C - Af @ Bf.T).abs().max().item(),
        "B@A": (C - Bf @ Af).abs().max().item(),
        "(A@B)^T": (C - (Af @ Bf).T).abs().max().item(),
    }
    assert err_AB < 1.0, f"A@B sollte passen, err={err_AB:.3e}"
    assert min(imposters.values()) > 10.0, \
        f"ein Doppelgänger liegt verdächtig nah: {imposters}"


def test_gemm_bf16_across_sizes():
    """bf16→fp32 (Akku fp32 = Pflicht): stimmt gegen fp32 — glatt UND ragged.

    bf16 ist nativ (kein Kernel-Cast, Akku fp32) → derselbe erzeugte Kernel wie
    fp16, nur mit bf16-Operanden; prüft, dass der Codegen dtype-agnostisch bleibt.
    """
    for (M, N, K) in [(256, 256, 256), (130, 100, 70)]:
        _assert_matches_fp32(M, N, K, "bf16", "fp32")


def test_gemm_bf16_orientation():
    """bf16: Orientierungs-Wächter (rechnet A@B, keinen Doppelgänger)."""
    _assert_orientation("bf16", "fp32")


def test_emit_deterministic():
    """Gleiche Config → byte-identischer Quelltext (Cache-/Reproduzierbarkeit)."""
    assert emit(RunConfig()) == emit(RunConfig())


def test_build_gemm_module_rejects_unknown_acc():
    """Unbekannter Akkumulator-dtype → ValueError (statt still falschem Kernel)."""
    try:
        build_gemm_module({"TM": 128, "TN": 128, "TK": 64}, "fp16", "int4")
    except ValueError:
        return
    raise AssertionError("build_gemm_module hätte bei acc_dtype='int4' ValueError werfen müssen")


def test_run_end_to_end_ok():
    """Integration: run() über die ganze Pipeline → status ok, verifiziert."""
    from tool_pipeline.store import store as st
    import tool_pipeline.run as R

    orig = st.append_result
    st.append_result = lambda r, path=None: None   # Store isolieren (kein echter Append)
    try:
        res = R.run(RunConfig())
    finally:
        st.append_result = orig
    assert res.status == "ok", f"status={res.status} error={res.error}"
    assert res.accuracy["passed"] and res.metrics["tflops"] > 0


def test_run_returns_result_on_compile_error():
    """Nicht baubare Configs → RunResult mit compile_error, NIE eine Exception."""
    import tool_pipeline.run as R
    from tool_pipeline.store import store as st

    orig = st.append_result
    st.append_result = lambda r, path=None: None
    try:
        for cfg in [
            RunConfig(expr="ki,kj->ij"),                  # nicht-kanonisch → reshape lehnt ab
            RunConfig(family="elementwise"),              # falsche Familie → parse lehnt ab
            RunConfig(dtype="bf16", acc_dtype="fp16"),    # unzulässige Acc-Kombi → check_dtype_combo
        ]:
            res = R.run(cfg)
            assert isinstance(res, RunResult), "run() darf nicht werfen"
            assert res.status == "compile_error", f"{cfg.expr}/{cfg.family}/{cfg.dtype}: {res.status}"
            assert res.error
    finally:
        st.append_result = orig


def test_run_verify_failed_status():
    """Falsche Kernel-Zahlen → verify_failed (verify gemonkeypatcht); keine Metriken."""
    import tool_pipeline.run as R
    from tool_pipeline.store import store as st

    orig_store, orig_verify = st.append_result, R.verify
    st.append_result = lambda r, path=None: None
    R.verify = lambda C, A, B, cfg: {"max_abs_err": 999.0, "passed": False, "atol": 0.2, "rtol": 0.02}
    try:
        res = R.run(RunConfig())
        assert res.status == "verify_failed", f"status={res.status}"
        assert res.accuracy["passed"] is False and res.metrics == {}
    finally:
        st.append_result, R.verify = orig_store, orig_verify


def test_run_error_on_launch_failure():
    """Laufzeit-Crash in der Messung → run_error (kein Raise)."""
    import tool_pipeline.run as R
    from tool_pipeline.store import store as st

    def _boom(*a, **k):
        raise RuntimeError("simulierter Launch-Crash")

    orig_store, orig_bench = st.append_result, R.benchmark
    st.append_result = lambda r, path=None: None
    R.benchmark = _boom
    try:
        res = R.run(RunConfig())
        assert res.status == "run_error", f"status={res.status}"
        assert res.error
    finally:
        st.append_result, R.benchmark = orig_store, orig_bench


def _main() -> int:
    clear_cache()
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
