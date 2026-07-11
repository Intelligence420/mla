"""tool_pipeline.intermediate_representation.parse — einsum-Ausdruck → typisierte IR (family-geroutet).

`parse()` routet zuerst auf die **Operations-Familie** (`config.family`, TZ 7):

* **contraction** (Default) → `ContractionIR` (die 2-Operanden-M/N/K/Batch-
  Klassifikation unten; treibt den B1-Reshape auf das kanonische Batched-GEMM).
* **elementwise** → `ElementwiseIR` (memory-bound: 1 **oder** 2 Operanden gleicher
  Form wie der Output; `ij->ij` unär/copy, `ij,ij->ij` binär add/mul — **kein**
  `ct.mma`, **kein** B1-Reshape).
* **reduction** → `ReductionIR` (memory-bound: genau 1 Operand, Achsen in
  `kept_dims`/`reduced_dims` zerlegt; `ij->i`, `ij->j`, `ij->` — `ct.sum`-Idiom).

Der Router steht **vor** dem contraction-eigenen „genau 2 Operanden"-Gate. Die
memory-bound-IRs sind bewusst leicht (nur Achsen/Größen) — sie brauchen weder die
M/N/K-Klassifikation noch den Kanonisierungs-Reshape.

Kontraktion — Klassifikation eines **2-Operanden**-Ausdrucks nach dem aus A05/06
bekannten Schema in **M / N / K / Batch**-Dimensionen:

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
# Memory-bound-IRs (TZ 7) — leicht: nur Achsen/Größen, kein M/N/K, kein Reshape
# ---------------------------------------------------------------------------
@dataclass
class ElementwiseIR:
    """Typisierte Sicht auf eine elementweise Abbildung (memory-bound).

    Jeder Operand trägt **exakt** die Output-Indizes (gleiche Form) — `ij,ij->ij`
    (binär: add/mul) oder `ij->ij` (unär: copy). Der Kernel kachelt die (row-major)
    2D-Sicht `(rows, cols)` mit `cols` = letzte (kontiguierte) Achse — das in A02
    task_03/04 als schnell bewiesene Layout.
    """

    expr: str
    inputs: list[str]
    output: str
    axes: list[str]          # Index-Reihenfolge (= Output)
    dim_sizes: dict[str, int]

    @property
    def arity(self) -> int:
        """Zahl der Operanden (1 = unär/copy, 2 = binär add/mul)."""
        return len(self.inputs)

    @property
    def shape(self) -> tuple:
        return tuple(self.dim_sizes[d] for d in self.axes)

    @property
    def num_elements(self) -> int:
        return _prod(self.axes, self.dim_sizes)

    @property
    def rows(self) -> int:
        """Fusionierte Zeilenzahl der 2D-Sicht (alle Achsen außer der letzten)."""
        return _prod(self.axes[:-1], self.dim_sizes)

    @property
    def cols(self) -> int:
        """Länge der letzten (kontiguierten) Achse — die gekachelte schnelle Dim."""
        return self.dim_sizes[self.axes[-1]] if self.axes else 1


@dataclass
class ReductionIR:
    """Typisierte Sicht auf eine Summen-Reduktion (memory-bound).

    Genau 1 Operand; der Output ist eine Teilmenge der Eingabe-Indizes. `kept_dims`
    (Output-Reihenfolge) bleiben stehen, `reduced_dims` (Operand-Reihenfolge) werden
    aufsummiert. Host-seitig wird der Operand auf `(kept_size, reduced_size)`
    permutiert+gefaltet und zeilenweise summiert (`ct.sum(axis=1)`, A02 task_02):
    `ij->i` (Zeilensumme), `ij->j` (Spaltensumme, via permute), `ij->` (volle Summe).
    """

    expr: str
    inputs: list[str]        # genau 1 Element
    output: str
    input_axes: list[str]    # Operand-Reihenfolge
    kept_dims: list[str]     # Output-Reihenfolge (bleiben stehen)
    reduced_dims: list[str]  # Operand-Reihenfolge (werden summiert)
    dim_sizes: dict[str, int]

    @property
    def kept_size(self) -> int:
        """Zeilenzahl nach permute+flatten (= Produkt der behaltenen Achsen; leer ⇒ 1)."""
        return _prod(self.kept_dims, self.dim_sizes)

    @property
    def reduced_size(self) -> int:
        """Spaltenzahl der summierten 2D-Sicht (= Produkt der reduzierten Achsen)."""
        return _prod(self.reduced_dims, self.dim_sizes)

    @property
    def out_shape(self) -> tuple:
        """Natürliche Output-Shape (leer ⇒ Skalar, `ij->`)."""
        return tuple(self.dim_sizes[d] for d in self.kept_dims)

    @property
    def in_shape(self) -> tuple:
        return tuple(self.dim_sizes[d] for d in self.input_axes)


# ---------------------------------------------------------------------------
# Parser — family-Router + gemeinsame Helfer
# ---------------------------------------------------------------------------
def _strip_split(expr: str) -> tuple[str, list[str], str | None]:
    """Roher Ausdruck → (bereinigter Ausdruck, Operanden-Liste, Output|None)."""
    expr = expr.replace(" ", "")
    if "->" in expr:
        lhs, output = expr.split("->")
    else:
        lhs, output = expr, None
    inputs = [s for s in lhs.split(",") if s]
    return expr, inputs, output


def _implicit_output(inputs: list[str]) -> str:
    """Impliziter Output nach einsum-Konvention: Indizes, die GENAU EINMAL über
    alle Operanden vorkommen, alphabetisch sortiert."""
    counts = Counter("".join(inputs))
    return "".join(sorted(i for i, c in counts.items() if c == 1))


def _check_no_repeats(inputs: list[str]) -> None:
    """Keine wiederholten Indizes je Operand (Diagonalen/Spuren bewusst draußen)."""
    for k, idx in enumerate(inputs):
        if len(set(idx)) != len(idx):
            raise NotImplementedError(
                f"Wiederholter Index in Operand {k} ('{idx}') — Diagonalen/Spuren "
                f"werden (bewusst) nicht unterstützt."
            )


def _check_sizes(indices: set[str], sizes: dict[str, int]) -> None:
    """Jede vorkommende Größe muss bekannt sein (loud-fail)."""
    missing = indices - set(sizes)
    if missing:
        raise ValueError(f"dim_sizes fehlt für Index/Indizes {sorted(missing)}.")
def parse(config: Union[RunConfig, str], dim_sizes: dict[str, int] | None = None,
          family: str | None = None) -> Union[ContractionIR, ElementwiseIR, ReductionIR]:
    """`RunConfig` (oder roher Ausdruck + `dim_sizes`) → family-typisierte IR.

    Routet zuerst auf die **Operations-Familie** und delegiert an den passenden
    Klassifikator. Bei einem `RunConfig` bestimmt `config.family` die Familie; bei
    einem rohen Ausdruck der `family`-Parameter (Default `"contraction"`, damit die
    torch-freien Controls-Helfer unverändert weiterlaufen). Jeder Zweig validiert
    streng (loud-fail statt stilles Falschergebnis).
    """
    if isinstance(config, RunConfig):
        fam = config.family
        expr, sizes = config.expr, config.dim_sizes
    else:
        expr, sizes = config, dim_sizes
        if sizes is None:
            raise ValueError("dim_sizes muss angegeben werden, wenn expr ein String ist.")
        fam = family or "contraction"

    if fam == "contraction":
        return _parse_contraction(expr, sizes)
    if fam == "elementwise":
        return _parse_elementwise(expr, sizes)
    if fam == "reduction":
        return _parse_reduction(expr, sizes)
    raise ValueError(
        f"unbekannte Operations-Familie {fam!r} "
        f"(erlaubt: 'contraction', 'elementwise', 'reduction')."
    )


def _parse_contraction(expr: str, sizes: dict[str, int]) -> ContractionIR:
    """2-Operanden-Ausdruck → `ContractionIR` (M/N/K/Batch-Klassifikation).

    Validiert streng: genau 2 Operanden, Output explizit **oder** implizit
    (einsum-Konvention), keine wiederholten Indizes je Operand, jede Größe bekannt,
    kein freier Output-Index. (Verhalten unverändert ggü. TZ 1-6.)
    """
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


def _parse_elementwise(expr: str, sizes: dict[str, int]) -> ElementwiseIR:
    """1- **oder** 2-Operanden-Ausdruck → `ElementwiseIR` (reine elementweise Abbildung).

    Verlangt, dass **jeder** Operand exakt die Output-Indizes in derselben
    Reihenfolge trägt (gleiche Form, kein Transponieren/Broadcast): `ij->ij`
    (unär/copy) oder `ij,ij->ij` (binär add/mul). Der impliziter Output nach
    einsum-Konvention deckt nur den unären Fall sinnvoll ab — binäre Ops brauchen
    einen expliziten Output (sonst würde einsum ihn kontrahieren).
    """
    expr, inputs, output = _strip_split(expr)
    if not (1 <= len(inputs) <= 2):
        raise NotImplementedError(
            f"Elementwise: 1 oder 2 Operanden erwartet; '{expr}' hat {len(inputs)}."
        )
    _check_no_repeats(inputs)
    if output is None:
        output = _implicit_output(inputs)
    if len(set(output)) != len(output):
        raise NotImplementedError(
            f"Wiederholter Index im Output ('{output}') — Diagonalen/Spuren "
            f"werden (bewusst) nicht unterstützt."
        )
    if not output:
        raise ValueError(
            "Elementwise: leerer Output — mindestens eine Achse nötig. Für binäre "
            "Ops den Output explizit angeben (z. B. ij,ij->ij)."
        )
    # Jeder Operand muss GENAU die Output-Indizes (Reihenfolge!) tragen.
    for k, idx in enumerate(inputs):
        if idx != output:
            raise ValueError(
                f"Elementwise: Operand {k} ('{idx}') muss dieselben Indizes in "
                f"derselben Reihenfolge wie der Output ('{output}') haben (reine "
                f"elementweise Abbildung; kein Transponieren/Broadcast)."
            )
    axes = list(output)
    _check_sizes(set(axes), sizes)
    return ElementwiseIR(
        expr=f"{','.join(inputs)}->{output}", inputs=inputs, output=output,
        axes=axes, dim_sizes=dict(sizes),
    )


def _parse_reduction(expr: str, sizes: dict[str, int]) -> ReductionIR:
    """1-Operanden-Ausdruck → `ReductionIR` (Summe über die weggelassenen Achsen).

    Verlangt genau 1 Operand und mindestens **eine** reduzierte Achse (Output ⊊
    Eingabe): `ij->i` (Zeilensumme), `ij->j` (Spaltensumme), `ij->` (volle Summe).
    Ein Output == Eingabe (`ij->ij`) ist keine Reduktion (das ist Elementwise/Copy)
    und wird abgelehnt.
    """
    expr, inputs, output = _strip_split(expr)
    if len(inputs) != 1:
        raise NotImplementedError(
            f"Reduktion: genau 1 Operand erwartet; '{expr}' hat {len(inputs)}."
        )
    _check_no_repeats(inputs)
    in0 = inputs[0]
    if output is None:
        output = _implicit_output(inputs)   # = in0 (jeder Index einmal) ⇒ keine Reduktion
    if len(set(output)) != len(output):
        raise NotImplementedError(
            f"Wiederholter Index im Output ('{output}') — Diagonalen/Spuren "
            f"werden (bewusst) nicht unterstützt."
        )
    set_in, set_out = set(in0), set(output)
    free = set_out - set_in
    if free:
        raise ValueError(
            f"Reduktion: Output-Index/-Indizes {sorted(free)} kommen im Operanden "
            f"('{in0}') nicht vor."
        )
    kept_dims = [d for d in output if d in set_in]        # Output-Reihenfolge
    reduced_dims = [d for d in in0 if d not in set_out]   # Operand-Reihenfolge
    if not reduced_dims:
        raise ValueError(
            f"Reduktion: keine reduzierte Achse in '{expr}' (Output = Eingabe ⇒ das "
            f"ist Elementwise/Copy). Bitte eine Achse weglassen, z. B. ij->i."
        )
    _check_sizes(set_in, sizes)
    return ReductionIR(
        expr=f"{in0}->{output}", inputs=inputs, output=output,
        input_axes=list(in0), kept_dims=kept_dims, reduced_dims=reduced_dims,
        dim_sizes=dict(sizes),
    )
