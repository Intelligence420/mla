"""Headless-Tests der einsum→IR-Klassifikation (`intermediate_representation.parse`).

Prüft die allgemeine M/N/K/Batch-Klassifikation und den impliziten Output (TZ 6)
**ohne** GPU/torch — reine String-/Mengen-Logik. Deckt die Ausdruck-Palette ab,
auf der der B1-Reshape (TZ6.2) aufsetzt: Plain-GEMM, transponiert, Batched,
mehrdimensionales M sowie die allgemeine Tensor-Kontraktion; dazu die strengen
Ablehnungen (n-är, Diagonalen, freier/wiederholter Output, unbekannte Größe).

Lauffähig standalone (`python tests/test_parse.py`, aus `project/`) **und** via
pytest. Braucht nur `tool_pipeline` (torch-/cuTile-frei).
"""

from __future__ import annotations

import os
import sys

# project/ auf den Pfad, damit `tool_pipeline` importierbar ist (standalone-Lauf).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool_pipeline.intermediate_representation.parse import (  # noqa: E402
    ContractionIR,
    ElementwiseIR,
    NAryContractionIR,
    ReductionIR,
    parse,
)
from tool_pipeline.schema import RunConfig  # noqa: E402


# --- Klassifikation der Ausdruck-Palette (M/N/K/Batch, fusionierte Größen) ----
def test_plain_gemm():
    """`ik,kj->ij`: M=[i], N=[j], K=[k], kein Batch; ist kanonischer Plain-GEMM."""
    ir = parse("ik,kj->ij", {"i": 8, "k": 4, "j": 6})
    assert ir.m_dims == ["i"] and ir.n_dims == ["j"] and ir.k_dims == ["k"]
    assert ir.batch_dims == []
    assert (ir.M, ir.N, ir.K, ir.B) == (8, 6, 4, 1)
    assert ir.is_canonical_gemm() is True


def test_transposed_a():
    """`ki,kj->ij`: A transponiert (in0='ki') ⇒ KEIN kanonischer Passthrough,
    aber die Klassifikation bleibt M=[i], N=[j], K=[k]."""
    ir = parse("ki,kj->ij", {"i": 8, "k": 4, "j": 6})
    assert ir.m_dims == ["i"] and ir.n_dims == ["j"] and ir.k_dims == ["k"]
    assert ir.batch_dims == []
    assert ir.is_canonical_gemm() is False  # in0='ki' != 'ik' ⇒ echter Reshape nötig


def test_batched_gemm():
    """`bik,bkj->bij`: b ist Batch (in beiden Operanden + Output)."""
    ir = parse("bik,bkj->bij", {"b": 2, "i": 8, "k": 4, "j": 6})
    assert ir.batch_dims == ["b"]
    assert ir.m_dims == ["i"] and ir.n_dims == ["j"] and ir.k_dims == ["k"]
    assert (ir.B, ir.M, ir.N, ir.K) == (2, 8, 6, 4)
    assert ir.is_canonical_gemm() is False  # Batch ⇒ kein 2D-Passthrough


def test_multidim_m():
    """`ijk,kl->ijl`: i,j sind BEIDE M ⇒ fusioniertes M = i·j."""
    ir = parse("ijk,kl->ijl", {"i": 2, "j": 3, "k": 4, "l": 5})
    assert ir.m_dims == ["i", "j"] and ir.n_dims == ["l"] and ir.k_dims == ["k"]
    assert ir.batch_dims == []
    assert (ir.M, ir.N, ir.K) == (2 * 3, 5, 4)


def test_general_tensor_contraction():
    """`acspx,bspy->abcyx`: mehrdim. M=[a,c,x], N=[b,y], K=[s,p], kein Batch
    (die A06-Referenzform). M/N/K in Output-, K in in0-Reihenfolge."""
    ir = parse("acspx,bspy->abcyx",
               {"a": 2, "c": 3, "s": 4, "p": 5, "x": 6, "b": 7, "y": 8})
    assert ir.m_dims == ["a", "c", "x"]      # Output-Reihenfolge (abcyx)
    assert ir.n_dims == ["b", "y"]
    assert ir.k_dims == ["s", "p"]           # in0-Reihenfolge (acspx)
    assert ir.batch_dims == []
    assert ir.M == 2 * 3 * 6 and ir.N == 7 * 8 and ir.K == 4 * 5


# --- Impliziter Output (einsum-Konvention) -----------------------------------
def test_implicit_output_plain():
    """Ohne '->' = Indizes die genau einmal vorkommen, alphabetisch: `ik,kj`→`ij`."""
    ir = parse("ik,kj", {"i": 8, "k": 4, "j": 6})
    assert ir.output == "ij"
    assert ir.m_dims == ["i"] and ir.n_dims == ["j"] and ir.k_dims == ["k"]


def test_implicit_output_sorts_alphabetically():
    """Impliziter Output ist alphabetisch sortiert (nicht in Auftritts-Reihenfolge)."""
    # 'jk,ki' → einmal: j,i → sortiert 'ij' (nicht 'ji').
    ir = parse("jk,ki", {"i": 5, "j": 6, "k": 4})
    assert ir.output == "ij"


def test_implicit_output_contracts_shared_index():
    """Ein in BEIDEN Operanden stehender Index (hier b) wird implizit kontrahiert,
    NICHT als Batch behalten: `bik,bkj`→`ij` (Batched GEMM braucht expliziten Output)."""
    ir = parse("bik,bkj", {"b": 2, "i": 8, "k": 4, "j": 6})
    assert ir.output == "ij"          # b fällt raus (kommt zweimal vor)
    assert ir.batch_dims == []        # ⇒ kein Batch, sondern über b summiert
    assert "b" in ir.k_dims           # b ist jetzt eine Kontraktions-Dim


# --- Strenge Validierung (loud-fail) -----------------------------------------
def _raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"falsche Exception {type(e).__name__}: {e}") from e
    raise AssertionError(f"erwartete {exc.__name__}, aber nichts wurde geworfen")


def test_nary_chain_is_planned():
    """TZ 7.5-3: >2 Operanden (Ketten-Kontraktion) ⇒ NAryContractionIR mit einer Folge
    sauberer paarweiser 2-Op-Schritte; der letzte Schritt liefert exakt den Output."""
    sizes = {"i": 8, "j": 4, "k": 6, "l": 5}
    ir = parse("ij,jk,kl->il", sizes)
    assert isinstance(ir, NAryContractionIR)
    assert ir.inputs == ["ij", "jk", "kl"] and ir.output == "il"
    assert len(ir.steps) == 2 and ir.steps[-1]["c_expr"] == "il"
    # jeder geplante Schritt ist ein sauberes 2-Op-GEMM (parse als 2-Op → ContractionIR mit K)
    for s in ir.steps:
        sub = parse(f"{s['a_expr']},{s['b_expr']}->{s['c_expr']}", sizes)
        assert isinstance(sub, ContractionIR) and sub.k_dims


def test_nary_hadamard_rejected():
    """n-äres Hadamard (kein Index kontrahiert, z. B. abc,bca,cba->abc) ist KEINE
    Ketten-Kontraktion ⇒ Loud-Fail (kein still falsches Ergebnis)."""
    assert _raises(lambda: parse("abc,bca,cba->abc", {"a": 2, "b": 2, "c": 2}),
                   NotImplementedError)


def test_nary_implicit_output():
    """n-är mit implizitem Output (einsum-Konvention): ij,jk,kl ⇒ Output il."""
    ir = parse("ij,jk,kl", {"i": 2, "j": 2, "k": 2, "l": 2})
    assert isinstance(ir, NAryContractionIR) and ir.output == "il"


def test_two_operand_path_unchanged():
    """Regression: genau 2 Operanden bleiben ContractionIR (M/N/K unverändert)."""
    ir = parse("ik,kj->ij", {"i": 4, "k": 4, "j": 4})
    assert isinstance(ir, ContractionIR)
    assert ir.m_dims == ["i"] and ir.n_dims == ["j"] and ir.k_dims == ["k"]


def test_rejects_diagonal_in_operand():
    """Wiederholter Index je Operand (Diagonale) ⇒ NotImplementedError."""
    assert _raises(lambda: parse("ii,ij->ij", {"i": 4, "j": 4}), NotImplementedError)


def test_rejects_repeated_output_index():
    """Wiederholter Index im expliziten Output (Spur) ⇒ NotImplementedError."""
    assert _raises(lambda: parse("ik,kj->ii", {"i": 4, "k": 4, "j": 4}),
                   NotImplementedError)


def test_rejects_free_output_index():
    """Output-Index, der in keinem Operanden vorkommt ⇒ ValueError."""
    assert _raises(lambda: parse("ik,kj->ijz", {"i": 4, "k": 4, "j": 4, "z": 4}),
                   ValueError)


def test_rejects_missing_size():
    """Fehlende Größe für einen vorkommenden Index ⇒ ValueError."""
    assert _raises(lambda: parse("ik,kj->ij", {"i": 4, "k": 4}), ValueError)


def test_string_without_sizes_raises():
    """Roher Ausdruck ohne dim_sizes ⇒ ValueError."""
    assert _raises(lambda: parse("ik,kj->ij"), ValueError)


# --- RunConfig-Pfad ----------------------------------------------------------
def test_runconfig_default_classifies():
    """parse(RunConfig()) klassifiziert den Default-Ausdruck (ik,kj->ij) korrekt."""
    ir = parse(RunConfig())
    assert ir.is_canonical_gemm() is True
    assert (ir.M, ir.N, ir.K) == (512, 512, 512)  # RunConfig-Default-Größen


def test_runconfig_elementwise_family_parses():
    """family='elementwise' + gültiger Ausdruck ⇒ ElementwiseIR (TZ 7)."""
    ir = parse(RunConfig(family="elementwise", op="add", expr="ij,ij->ij",
                         dim_sizes={"i": 4, "j": 5}))
    assert isinstance(ir, ElementwiseIR)
    assert ir.arity == 2 and ir.output == "ij"


def test_runconfig_reduction_family_parses():
    """family='reduction' + gültiger Ausdruck ⇒ ReductionIR (TZ 7)."""
    ir = parse(RunConfig(family="reduction", op="sum", expr="ij->i",
                         dim_sizes={"i": 4, "j": 5}))
    assert isinstance(ir, ReductionIR)
    assert ir.kept_dims == ["i"] and ir.reduced_dims == ["j"]


def test_runconfig_unknown_family_rejected():
    """Unbekannte Operations-Familie ⇒ ValueError (loud-fail)."""
    assert _raises(lambda: parse(RunConfig(family="foo")), ValueError)


# --- Elementwise-Familie (TZ 7) ----------------------------------------------
def test_elementwise_binary():
    """`ij,ij->ij`: 2 Operanden gleicher Form; axes=['i','j'], 2D-Sicht rows·cols."""
    ir = parse("ij,ij->ij", {"i": 8, "j": 6}, family="elementwise")
    assert isinstance(ir, ElementwiseIR)
    assert ir.arity == 2 and ir.axes == ["i", "j"]
    assert ir.shape == (8, 6) and ir.num_elements == 48
    assert ir.rows == 8 and ir.cols == 6


def test_elementwise_unary_copy():
    """`ij->ij` (auch implizit `ij`): 1 Operand (copy)."""
    ir = parse("ij->ij", {"i": 8, "j": 6}, family="elementwise")
    assert ir.arity == 1 and ir.output == "ij"
    ir2 = parse("ij", {"i": 8, "j": 6}, family="elementwise")   # impliziter Output = ij
    assert ir2.arity == 1 and ir2.output == "ij"


def test_elementwise_3d():
    """`ijk,ijk->ijk`: 3D elementweise; rows=i·j, cols=k (kontiguierte Achse)."""
    ir = parse("ijk,ijk->ijk", {"i": 2, "j": 3, "k": 5}, family="elementwise")
    assert ir.rows == 6 and ir.cols == 5 and ir.num_elements == 30


def test_elementwise_rejects_transposed_operand():
    """`ij,ji->ij`: zweiter Operand transponiert ⇒ ValueError (keine reine Abbildung)."""
    assert _raises(lambda: parse("ij,ji->ij", {"i": 4, "j": 4}, family="elementwise"),
                   ValueError)


def test_elementwise_rejects_three_operands():
    """3 Operanden ⇒ NotImplementedError (n-är draußen)."""
    assert _raises(lambda: parse("ij,ij,ij->ij", {"i": 4, "j": 4}, family="elementwise"),
                   NotImplementedError)


def test_elementwise_rejects_implicit_binary():
    """Binär ohne expliziten Output (`ij,ij`) ⇒ leerer impliziter Output ⇒ ValueError."""
    assert _raises(lambda: parse("ij,ij", {"i": 4, "j": 4}, family="elementwise"),
                   ValueError)


# --- Reduktions-Familie (TZ 7) -----------------------------------------------
def test_reduction_row_sum():
    """`ij->i` (= A02 task_02): kept=[i], reduced=[j]; kept_size·reduced_size = i·j."""
    ir = parse("ij->i", {"i": 8, "j": 6}, family="reduction")
    assert isinstance(ir, ReductionIR)
    assert ir.kept_dims == ["i"] and ir.reduced_dims == ["j"]
    assert ir.kept_size == 8 and ir.reduced_size == 6
    assert ir.out_shape == (8,)


def test_reduction_col_sum():
    """`ij->j`: kept=[j], reduced=[i] (Achse 0 reduziert ⇒ host-seitiges permute)."""
    ir = parse("ij->j", {"i": 8, "j": 6}, family="reduction")
    assert ir.kept_dims == ["j"] and ir.reduced_dims == ["i"]
    assert ir.kept_size == 6 and ir.reduced_size == 8
    assert ir.out_shape == (6,)


def test_reduction_full_sum():
    """`ij->` (volle Summe): kept=[], reduced=[i,j]; kept_size=1, out_shape leer."""
    ir = parse("ij->", {"i": 8, "j": 6}, family="reduction")
    assert ir.kept_dims == [] and ir.reduced_dims == ["i", "j"]
    assert ir.kept_size == 1 and ir.reduced_size == 48
    assert ir.out_shape == ()


def test_reduction_rejects_no_reduced_axis():
    """`ij->ij` (Output = Eingabe) ist keine Reduktion ⇒ ValueError."""
    assert _raises(lambda: parse("ij->ij", {"i": 4, "j": 4}, family="reduction"),
                   ValueError)


def test_reduction_rejects_two_operands():
    """2 Operanden ⇒ NotImplementedError (Reduktion ist 1-Operand)."""
    assert _raises(lambda: parse("ij,ij->i", {"i": 4, "j": 4}, family="reduction"),
                   NotImplementedError)


def test_reduction_rejects_free_output():
    """Output-Index nicht im Operanden (`ij->ik`) ⇒ ValueError."""
    assert _raises(lambda: parse("ij->ik", {"i": 4, "j": 4, "k": 4}, family="reduction"),
                   ValueError)


def test_family_via_string_param():
    """Der family-Parameter routet auch den String-Pfad (für Controls-Helfer)."""
    assert isinstance(parse("ij->i", {"i": 4, "j": 4}, family="reduction"), ReductionIR)
    assert isinstance(parse("ij,ij->ij", {"i": 4, "j": 4}, family="elementwise"),
                      ElementwiseIR)
    # Default-Familie bleibt contraction (Controls-Validierung unverändert).
    ir = parse("ik,kj->ij", {"i": 4, "k": 4, "j": 4})
    assert ir.m_dims == ["i"]   # ContractionIR


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
