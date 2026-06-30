"""tool_pipeline.ir.parse — einsum-Ausdruck → typisierte Kontraktions-IR.

Klassifiziert einen **2-Operanden**-einsum-Ausdruck nach dem aus A05/06 bekannten
Schema in **M / N / K / Batch**-Dimensionen:

* **Batch (C)**: Index steht in *beiden* Operanden **und** im Output.
* **K (kontrahiert)**: Index steht in *beiden* Operanden, **nicht** im Output (summiert).
* **M**: Index steht in Operand 0 **und** im Output, nicht in Operand 1.
* **N**: Index steht in Operand 1 **und** im Output, nicht in Operand 0.

Für `ik,kj->ij` ⇒ M=[i], N=[j], K=[k], Batch=[].

Bewusst **minimal** (TZ 1): genau 2 Operanden, **expliziter** Output, keine
Diagonalen/Wiederholungen. Die schwere Verallgemeinerung (n-äres einsum,
impliziter Output, Familien-Routing, Optimizer-getriebenes fuse/split/permute)
ist TZ 6/7 und wird hier **nicht** vorgebaut — die `M/N/K/Batch`-Klassifikation
ist aber bereits die allgemeine, sodass TZ 6 nur darauf aufsetzt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from ..schema import RunConfig


def _prod(dims: list[str], sizes: dict[str, int]) -> int:
    """Produkt der Größen einer Index-Liste (leere Liste ⇒ 1)."""
    p = 1
    for d in dims:
        p *= sizes[d]
    return p


# ---------------------------------------------------------------------------
# Kontraktions-IR
# ---------------------------------------------------------------------------
@dataclass
class ContractionIR:
    """Typisierte Sicht auf eine 2-Operanden-Kontraktion.

    `m_dims`/`n_dims`/`batch_dims` in **Output-Reihenfolge**, `k_dims` in der
    Reihenfolge von Operand 0. `M`/`N`/`K`/`B` sind die *fusionierten* Größen
    (Produkt je Kategorie) — für TZ 1 (ein Index je Kategorie) trivial, für die
    spätere allgemeine Kontraktion (TZ 6) genau die kanonischen GEMM-Maße.
    """

    expr: str
    inputs: list[str]
    output: str
    m_dims: list[str]
    n_dims: list[str]
    k_dims: list[str]
    batch_dims: list[str]
    dim_sizes: dict[str, int]

    @property
    def M(self) -> int:
        return _prod(self.m_dims, self.dim_sizes)

    @property
    def N(self) -> int:
        return _prod(self.n_dims, self.dim_sizes)

    @property
    def K(self) -> int:
        return _prod(self.k_dims, self.dim_sizes)

    @property
    def B(self) -> int:
        return _prod(self.batch_dims, self.dim_sizes)

    def is_canonical_gemm(self) -> bool:
        """True ⇔ direkt als 2D-Plain-GEMM emittierbar — **ohne** jede Umformung.

        Verlangt für den strikten TZ-1-Passthrough: 2 Operanden, **kein** Batch,
        **genau ein** Index je Kategorie (M/N/K) — dann sind die Operanden schon
        2D-`(M,K)`/`(K,N)`-Tensoren, kein Reshape/Permute nötig — und kanonische
        Reihenfolge (Operand 0 = M·K, Operand 1 = K·N, Output = M·N).

        `ik,kj->ij` erfüllt das. **Nicht** erfüllt (⇒ echter B1-Reshape, TZ 6):
        `ki,kj->ij` (B transponiert), `bik,bkj->bij` (Batch), `ijk,kl->ijl`
        (mehrdim. M ⇒ Fusion `(i,j,k)→(i·j,k)`).
        """
        if len(self.inputs) != 2 or self.batch_dims:
            return False
        if not (len(self.m_dims) == 1 and len(self.n_dims) == 1 and len(self.k_dims) == 1):
            return False
        return (
            self.inputs[0] == "".join(self.m_dims + self.k_dims)
            and self.inputs[1] == "".join(self.k_dims + self.n_dims)
            and self.output == "".join(self.m_dims + self.n_dims)
        )


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def parse(config: Union[RunConfig, str], dim_sizes: dict[str, int] | None = None) -> ContractionIR:
    """`RunConfig` (oder roher Ausdruck + `dim_sizes`) → `ContractionIR`.

    Validiert streng (loud-fail statt stilles Falschergebnis): expliziter
    Output, genau 2 Operanden, keine wiederholten Indizes je Operand, jede
    Größe bekannt, kein freier Output-Index.
    """
    if isinstance(config, RunConfig):
        if config.family != "contraction":
            raise NotImplementedError(
                f"TZ 1: nur family='contraction'; '{config.family}' "
                f"(elementwise/reduction) ist TZ 7."
            )
        expr, sizes = config.expr, config.dim_sizes
    else:
        expr, sizes = config, dim_sizes
        if sizes is None:
            raise ValueError("dim_sizes muss angegeben werden, wenn expr ein String ist.")

    expr = expr.replace(" ", "")
    if "->" not in expr:
        raise ValueError(f"TZ 1 braucht einen expliziten Output ('->') in '{expr}'.")
    lhs, rhs = expr.split("->")
    inputs = [s for s in lhs.split(",") if s]
    output = rhs

    if len(inputs) != 2:
        raise NotImplementedError(
            f"TZ 1: genau 2 Operanden; '{expr}' hat {len(inputs)} "
            f"(n-äres einsum = später)."
        )

    in0, in1 = inputs
    set0, set1, set_out = set(in0), set(in1), set(output)

    # Keine Diagonalen/Wiederholungen je Operand (TZ 1).
    for name, idx in (("Operand 0", in0), ("Operand 1", in1), ("Output", output)):
        if len(set(idx)) != len(idx):
            raise NotImplementedError(
                f"Wiederholter Index in {name} ('{idx}') — Diagonalen/Spuren "
                f"sind in TZ 1 nicht unterstützt."
            )

    # Jeder Output-Index muss aus den Eingaben stammen.
    free = set_out - (set0 | set1)
    if free:
        raise ValueError(f"Output-Index/-Indizes {sorted(free)} kommen in keinem Operanden vor.")

    # Jede Größe bekannt.
    all_idx = set0 | set1 | set_out
    missing = all_idx - set(sizes)
    if missing:
        raise ValueError(f"dim_sizes fehlt für Index/Indizes {sorted(missing)}.")

    # Klassifikation (M/N in Output-Reihenfolge, K in Operand-0-Reihenfolge).
    batch_dims = [d for d in output if d in set0 and d in set1]
    m_dims = [d for d in output if d in set0 and d not in set1]
    n_dims = [d for d in output if d in set1 and d not in set0]
    k_dims = [d for d in in0 if d in set1 and d not in set_out]

    return ContractionIR(
        expr=expr, inputs=inputs, output=output,
        m_dims=m_dims, n_dims=n_dims, k_dims=k_dims, batch_dims=batch_dims,
        dim_sizes=dict(sizes),
    )
