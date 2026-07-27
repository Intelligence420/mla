"""tool_pipeline.intermediate_representation.reshape — B1: Kontraktion → kanonisches Batched-GEMM.

**Ziel (TZ 6):** *jede* 2-Operanden-Kontraktion host-seitig per `permute`+`reshape`
auf die kanonische Form `(B,M,K)×(B,K,N)→(B,M,N)` bringen, damit der Codegen nur
**eine** bewiesene Struktur (A05-Orientierung, kein Swap) emittieren muss
(Risiko ④: die View-/Stride-Mathematik muss numerisch exakt sein).

`to_canonical(ir)` ist **config/optimizer-getrieben**: es baut über
`config.generate_config` die Transformations-IR (Validierung + Per-Tensor-Strides)
und prognostiziert mit dem Stride-Adjazenz-Test aus `optimizer` (dort als
`is_row_major_contiguous_run` ausgelagert), ob der View **zero-copy** ist. Die
konkrete Permutation je Operand liest es direkt aus den klassifizierten
`ContractionIR`-Dim-Listen.

**Design-Entscheidungen (TZ 6):** kanonisch **immer** `(B,M,K)` (auch B=1 → eine
bewiesene Struktur, Grid-Z=1); Reshape via `.reshape()` — freier View wo möglich,
sonst eine **Setup-Kopie** (passiert außerhalb der Zeitmessung UND der analytisch
berechneten Roofline-Metriken → verfälscht nichts; `zero_copy` protokolliert es).

Der eigentliche torch-Umbau der Operanden macht `to_canonical_operands` /
`from_canonical_output` (torch lazy importiert — das Modul bleibt für die reine
Spec-/Klassifikations-Nutzung torch-frei).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .config import _row_major_strides, generate_config
from .optimizer import is_row_major_contiguous_run
from .parse import ContractionIR


@dataclass
class Canonical:
    """Kanonische Batched-GEMM-Beschreibung **inklusive** der B1-View-Spezifikation.

    `M/N/K/B` sind die fusionierten Größen. Die View-Felder beschreiben (rein als
    Daten, ohne torch) den `permute`+`reshape`-Umbau je Operand auf `(B,M,K)` /
    `(B,K,N)` und den Rück-Umbau des kanonischen Outputs `(B,M,N)` in die
    natürliche einsum-Reihenfolge (für die `torch.einsum`-Verifikation).
    """

    M: int
    N: int
    K: int
    B: int = 1
    transform_needed: bool = False   # True ⇔ nicht der triviale 2D-Plain-GEMM
    zero_copy: bool = True           # True ⇔ beide Operanden-Views ohne Kopie

    ir: Optional[ContractionIR] = None

    # Operand 0 (A): natürliche Shape → permute → (B, M, K)
    a_natural_shape: tuple = ()
    a_perm: tuple = ()
    a_shape: tuple = ()
    # Operand 1 (B): natürliche Shape → permute → (B, K, N)
    b_natural_shape: tuple = ()
    b_perm: tuple = ()
    b_shape: tuple = ()
    # Output C: kanonisch (B, M, N) → reshape (auffalten) → permute → natürlich
    c_shape: tuple = ()
    c_unfused_shape: tuple = ()
    c_perm: tuple = ()
    out_natural_shape: tuple = ()


def _group_views_zero_copy(nat_order: str, target_order: list[str],
                           group_lengths: list[int], sizes: dict[str, int]) -> bool:
    """Ob `permute(nat→target)` + gruppenweises Fusionieren ein zero-copy-View ist.

    Für einen in `nat_order` **contiguous** liegenden Tensor: permutiere die
    Row-Major-Strides in `target_order` und prüfe je Gruppe (Länge aus
    `group_lengths`) die interne Row-major-Kontiguität (Kriterium aus
    `optimizer.is_row_major_contiguous_run` = der `fuse_dims`-Adjazenztest).
    """
    nat_strides = _row_major_strides(list(nat_order), nat_order, sizes)  # in nat_order
    perm = [nat_order.index(d) for d in target_order]
    perm_sizes = [sizes[d] for d in target_order]
    perm_strides = [nat_strides[p] for p in perm]
    idx = 0
    for glen in group_lengths:
        if not is_row_major_contiguous_run(perm_sizes[idx:idx + glen],
                                           perm_strides[idx:idx + glen]):
            return False
        idx += glen
    return True


def to_canonical(ir: ContractionIR) -> Canonical:
    """`ContractionIR` → `Canonical` mit vollständiger B1-View-Spezifikation.

    Deckt **jede** 2-Operanden-Kontraktion ab (transponiert, Batch, mehrdim. M/N/K).
    Numerisch abgesichert in `tests/test_reshape.py` gegen `torch.einsum` (Risiko ④).
    """
    sizes = ir.dim_sizes
    in0, in1, output = ir.inputs[0], ir.inputs[1], ir.output
    batch, m, k, n = ir.batch_dims, ir.m_dims, ir.k_dims, ir.n_dims
    B, M, N, K = ir.B, ir.M, ir.N, ir.K

    # Config/Optimizer-getrieben: Transformations-IR bauen (validiert Shapes/
    # Konsistenz, liefert die Per-Tensor-Strides für den zero-copy-Beweis).
    explicit_expr = f"{in0},{in1}->{output}"
    a_nat = tuple(sizes[d] for d in in0)
    b_nat = tuple(sizes[d] for d in in1)
    generate_config(explicit_expr, [a_nat, b_nat])   # loud-fail bei inkonsistenten Größen

    # --- Operand-Views: natürliche Achsen → [batch, m, k] bzw. [batch, k, n] ---
    a_order = batch + m + k
    b_order = batch + k + n
    a_perm = tuple(in0.index(d) for d in a_order)
    b_perm = tuple(in1.index(d) for d in b_order)

    # --- Output-Rück-View: (B, M, N) auffalten in (batch…, m…, n…), dann in
    #     die natürliche Output-Reihenfolge permutieren ---
    c_block = batch + m + n
    c_unfused_shape = tuple(sizes[d] for d in c_block)
    c_perm = tuple(c_block.index(d) for d in output)

    # --- zero-copy-Vorhersage (Stride-Adjazenz je Gruppe) ---
    zero_copy = (
        _group_views_zero_copy(in0, a_order, [len(batch), len(m), len(k)], sizes)
        and _group_views_zero_copy(in1, b_order, [len(batch), len(k), len(n)], sizes)
    )

    return Canonical(
        M=M, N=N, K=K, B=B,
        transform_needed=not ir.is_canonical_gemm(),
        zero_copy=zero_copy,
        ir=ir,
        a_natural_shape=a_nat, a_perm=a_perm, a_shape=(B, M, K),
        b_natural_shape=b_nat, b_perm=b_perm, b_shape=(B, K, N),
        c_shape=(B, M, N), c_unfused_shape=c_unfused_shape, c_perm=c_perm,
        out_natural_shape=tuple(sizes[d] for d in output),
    )


# ---------------------------------------------------------------------------
# torch-Anwendung der View-Spezifikation (lazy import → Modul bleibt torch-frei)
# ---------------------------------------------------------------------------
def to_canonical_operands(canonical: Canonical, A_nat, B_nat):
    """Natürliche Operanden → kanonische `(B,M,K)` / `(B,K,N)` (permute+reshape).

    `.reshape()` liefert einen View, wo die Achsen row-major zusammenhängen, sonst
    eine Kopie (Setup-Zeit, außerhalb der Messung — s. Modul-Docstring).
    """
    A = A_nat.permute(*canonical.a_perm).reshape(canonical.a_shape)
    B = B_nat.permute(*canonical.b_perm).reshape(canonical.b_shape)
    return A, B


def from_canonical_output(canonical: Canonical, C_canon):
    """Kanonischer Output `(B,M,N)` → natürliche einsum-Shape (für `verify`)."""
    return C_canon.reshape(canonical.c_unfused_shape).permute(*canonical.c_perm)
