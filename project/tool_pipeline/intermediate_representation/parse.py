"""tool_pipeline.intermediate_representation.parse — einsum-Ausdruck → typisierte Kontraktions-IR.

Klassifiziert einen **2-Operanden**-einsum-Ausdruck nach dem aus A05/06 bekannten
Schema in **M / N / K / Batch**-Dimensionen:

* **Batch (C)**: Index steht in *beiden* Operanden **und** im Output.
* **K (kontrahiert)**: Index steht in *beiden* Operanden, **nicht** im Output (summiert).
* **M**: Index steht in Operand 0 **und** im Output, nicht in Operand 1.
* **N**: Index steht in Operand 1 **und** im Output, nicht in Operand 0.

Für `ik,kj->ij` ⇒ M=[i], N=[j], K=[k], Batch=[].

Umfang: genau **2 Operanden**, Output **explizit** (`->…`) **oder implizit**
(einsum-Konvention, TZ 6), keine Diagonalen/Wiederholungen je Operand. Bewusst
draußen (später/optional): n-äres einsum (>2 Operanden), Diagonalen/Spuren. Die
`M/N/K/Batch`-Klassifikation ist die allgemeine — der echte, view-/stride-basierte
B1-Reshape (fuse/permute → kanonisches Batched-GEMM) setzt in `reshape.py`
(config/optimizer-getrieben) darauf auf.
"""

from __future__ import annotations

from collections import Counter
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

    Validiert streng (loud-fail statt stilles Falschergebnis): genau 2 Operanden,
    Output explizit **oder** implizit (einsum-Konvention), keine wiederholten
    Indizes je Operand, jede Größe bekannt, kein freier Output-Index.
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

    # Output explizit ('->…') übernehmen oder — fehlt der Pfeil — implizit ableiten
    # (nach der Operanden-Validierung, s. u.).
    if "->" in expr:
        lhs, rhs = expr.split("->")
        output = rhs
    else:
        lhs, output = expr, None
    inputs = [s for s in lhs.split(",") if s]

    if len(inputs) != 2:
        raise NotImplementedError(
            f"genau 2 Operanden; '{expr}' hat {len(inputs)} "
            f"(n-äres einsum = später/optional)."
        )

    in0, in1 = inputs

    # Keine Diagonalen/Wiederholungen je Operand.
    for name, idx in (("Operand 0", in0), ("Operand 1", in1)):
        if len(set(idx)) != len(idx):
            raise NotImplementedError(
                f"Wiederholter Index in {name} ('{idx}') — Diagonalen/Spuren "
                f"werden (bewusst) nicht unterstützt."
            )

    # Impliziten Output nach einsum-Konvention ableiten: alle Indizes, die über
    # beide Operanden GENAU EINMAL vorkommen, alphabetisch sortiert (mehrfach =
    # kontrahiert). Ein in beiden Operanden stehender Index (Batch/K) ist daher
    # NICHT im impliziten Output — Batched GEMM (Batch-Index behalten) braucht
    # deshalb einen expliziten Output (z. B. `bik,bkj->bij`).
    if output is None:
        counts = Counter(in0 + in1)
        output = "".join(sorted(i for i, c in counts.items() if c == 1))

    # Wiederholter Index im (expliziten) Output ist unzulässig (Diagonale/Spur).
    if len(set(output)) != len(output):
        raise NotImplementedError(
            f"Wiederholter Index im Output ('{output}') — Diagonalen/Spuren "
            f"werden (bewusst) nicht unterstützt."
        )

    set0, set1, set_out = set(in0), set(in1), set(output)

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
