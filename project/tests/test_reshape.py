"""B1-Reshape-Korrektheit gegen torch (Risiko-④-Sicherheitsnetz, TZ 6).

Der Kern-Test: für eine Palette von Ausdrücken (Plain-GEMM, transponiert, Batched,
mehrdim. M/N, allgemeine Tensor-Kontraktion) reproduziert die Kette

    natürliche Operanden → B1-View → kanonisches Batched-Matmul → Output-Rück-View

**numerisch exakt** `torch.einsum(expr, A, B)`. Das beweist die View-/Stride-
Mathematik **ohne** den cuTile-Kernel — `torch.matmul` steht hier als
Platzhalter für die bewiesene GEMM-Struktur. Zusätzlich: die zero-copy-Vorhersage
wird gegen das reale torch-View-Verhalten gekreuzt, und der Config/Optimizer-Port
(split/fuse/permute) wird kurz auf Wirken geprüft.

Läuft auf CPU (torch, **kein** GPU/cuTile). Standalone (`python tests/test_reshape.py`,
aus `project/`) und via pytest.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from tool_pipeline.intermediate_representation.config import (  # noqa: E402
    DimType, generate_config,
)
from tool_pipeline.intermediate_representation.optimizer import Optimizer  # noqa: E402
from tool_pipeline.intermediate_representation.parse import parse  # noqa: E402
from tool_pipeline.intermediate_representation.reshape import (  # noqa: E402
    from_canonical_output, to_canonical, to_canonical_operands,
)

# Ausdruck-Palette + Größen je Index (klein, damit CPU schnell bleibt).
_CASES = {
    "ik,kj->ij":        {"i": 3, "k": 4, "j": 5},                      # Plain-GEMM
    "ki,kj->ij":        {"i": 3, "k": 4, "j": 5},                      # A transponiert
    "ij,jk->ik":        {"i": 3, "j": 4, "k": 5},                      # Standard-Namen
    "bik,bkj->bij":     {"b": 2, "i": 3, "k": 4, "j": 5},              # Batched GEMM
    "abik,abkj->abij":  {"a": 2, "b": 2, "i": 3, "k": 4, "j": 5},      # 2 Batch-Achsen
    "ijk,kl->ijl":      {"i": 2, "j": 3, "k": 4, "l": 5},              # mehrdim. M (i·j)
    "acspx,bspy->abcyx": {"a": 2, "c": 2, "s": 2, "p": 3, "x": 3,      # allg. Kontraktion
                          "b": 2, "y": 2},
}


def _natural_operands(expr, sizes):
    """Contiguous fp64-Operanden in natürlicher einsum-Shape (deterministisch)."""
    ir = parse(expr, sizes)
    torch.manual_seed(0)
    A = torch.randn(*[sizes[d] for d in ir.inputs[0]], dtype=torch.float64)
    B = torch.randn(*[sizes[d] for d in ir.inputs[1]], dtype=torch.float64)
    return ir, A, B


# --- Kern: B1-View reproduziert torch.einsum numerisch exakt ------------------
def _check_expr(expr, sizes):
    ir, A, B = _natural_operands(expr, sizes)
    canon = to_canonical(ir)

    # 1) fusionierte Größen stimmen mit der Roh-Rechnung überein.
    assert canon.a_shape == (canon.B, canon.M, canon.K), (expr, canon.a_shape)
    assert canon.b_shape == (canon.B, canon.K, canon.N), (expr, canon.b_shape)
    assert canon.c_shape == (canon.B, canon.M, canon.N), (expr, canon.c_shape)

    # 2) B1-View → kanonisches Batched-Matmul → Rück-View == torch.einsum.
    A_c, B_c = to_canonical_operands(canon, A, B)
    assert tuple(A_c.shape) == canon.a_shape and tuple(B_c.shape) == canon.b_shape
    C_canon = torch.matmul(A_c, B_c)                 # (B,M,K)x(B,K,N) -> (B,M,N)
    assert tuple(C_canon.shape) == canon.c_shape
    C_nat = from_canonical_output(canon, C_canon)
    ref = torch.einsum(expr, A, B)
    assert tuple(C_nat.shape) == tuple(ref.shape), (expr, C_nat.shape, ref.shape)
    assert torch.allclose(C_nat, ref, atol=1e-9, rtol=1e-7), \
        (expr, (C_nat - ref).abs().max().item())


def test_all_expressions_reproduce_einsum():
    """Die ganze Palette: B1-Kette == torch.einsum (numerisch exakt, fp64)."""
    for expr, sizes in _CASES.items():
        _check_expr(expr, sizes)


# --- zero-copy-Vorhersage gegen echtes torch-Verhalten kreuzen ----------------
def _actual_zero_copy(t, perm, shape):
    """True ⇔ permute+reshape liefert einen echten View (kein Kopie)."""
    v = t.permute(*perm).reshape(shape)
    return v.data_ptr() == t.data_ptr()


def test_zero_copy_prediction_matches_torch():
    """canonical.zero_copy stimmt mit dem tatsächlichen torch-View-Verhalten überein."""
    for expr, sizes in _CASES.items():
        ir, A, B = _natural_operands(expr, sizes)
        canon = to_canonical(ir)
        actual = (_actual_zero_copy(A, canon.a_perm, canon.a_shape)
                  and _actual_zero_copy(B, canon.b_perm, canon.b_shape))
        assert canon.zero_copy == actual, (expr, canon.zero_copy, actual)


def test_general_contraction_needs_copy():
    """`acspx,bspy->abcyx`: M=[a,c,x] ist in A NICHT zusammenhängend (s,p dazwischen)
    ⇒ der View braucht eine Kopie (zero_copy=False) — die Kette bleibt trotzdem korrekt."""
    ir = parse("acspx,bspy->abcyx", _CASES["acspx,bspy->abcyx"])
    assert to_canonical(ir).zero_copy is False


def test_plain_gemm_is_zero_copy():
    """Plain-GEMM (und die meisten Batched-Fälle) sind zero-copy-Views."""
    for expr in ("ik,kj->ij", "bik,bkj->bij", "ijk,kl->ijl"):
        ir = parse(expr, _CASES[expr])
        assert to_canonical(ir).zero_copy is True, expr


# --- Config/Optimizer-Port: split/fuse/permute wirken (relative Imports ok) ---
def test_config_classifies_dims():
    """generate_config klassifiziert M/N/K/C wie erwartet (Port-Sanity)."""
    cfg = generate_config("bik,bkj->bij", [(2, 3, 4), (2, 4, 5)])
    # dim_order = erstes Auftreten: b, i, k, j
    assert cfg.dim_types == [DimType.C, DimType.M, DimType.K, DimType.N]
    assert cfg.dim_sizes == [2, 3, 4, 5]


def test_optimizer_split_then_fuse_is_identity():
    """split_dim + fuse_dims == Identität (Größen + Strides unverändert)."""
    cfg = generate_config("mk,kn->mn", [(128, 64), (64, 256)])
    before = (list(cfg.dim_sizes), [list(s) for s in cfg.strides])
    opt = Optimizer(cfg)
    opt.split_dim(0, 16, 8)          # m=128 -> (16, 8)
    assert cfg.dim_sizes == [16, 8, 64, 256]
    opt.fuse_dims(0, 1)              # zurück zu m=128
    after = (list(cfg.dim_sizes), [list(s) for s in cfg.strides])
    assert before == after, (before, after)


def test_optimizer_fuse_rejects_non_adjacent():
    """fuse_dims lehnt nicht-benachbarte Dims ab (Stride-Adjazenz-Beweis)."""
    # cmk,ckn->cmn: c und k sind in Operand 0 NICHT benachbart (m dazwischen).
    cfg = generate_config("cmk,ckn->cmn", [(4, 8, 6), (4, 6, 5)])
    opt = Optimizer(cfg)
    c_id = 0  # dim_order = c,m,k,n ; c=0, k=2
    raised = False
    try:
        opt.fuse_dims(c_id, 2)
    except ValueError:
        raised = True
    assert raised, "fuse_dims hätte nicht-benachbarte Dims ablehnen müssen"


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
