"""tool_pipeline.intermediate_representation.config — Transformations-IR (Port aus A05/06).

**Originalgetreuer Port** von `assignments/05|06/src/config.py`: die
`Config`-Dataclass als **flache, parallele Liste** über alle globalen Indizes
(je Position: `DimType` M/N/K/C, `ExecType`, Größe **und eine Stride pro
Tensor**) plus `generate_config(einsum, shapes)`. Stride 0 = Dimension fehlt in
diesem Tensor; die Strides sind die Row-Major-Strides jedes Tensors in seiner
natürlichen einsum-Achsenreihenfolge (= die realen torch-Strides eines
`contiguous` Tensors) — damit ist die Stride-Arithmetik des Optimizers zugleich
der **zero-copy-Beweis** für einen View.

Diese Transformations-IR ist **komplementär** zur `parse.ContractionIR`:
`ContractionIR` ist die strikt validierte Parser-Ausgabe (die Naht in die
Pipeline); `Config`/`Optimizer` (Port) sind die **mechanische** fuse/permute-
Maschinerie mit Stride-Beweis, auf der `reshape.to_canonical` den echten
B1-View (Kontraktion → kanonisches Batched-GEMM) baut.

Der `DataType`/`Prim*`-Meta-Anteil und `Optimizer.make_executable`/`verify`
stammen aus der A05/06-Host-Tiling-Heuristik; unser Codegen kachelt im Template
(TM/TN/TK), nicht host-seitig — sie sind **mitportiert** (voller Port), werden
vom B1-Reshape aber **nicht** benötigt. Genutzt werden: `generate_config`,
`_row_major_strides`, `DimType` und `Optimizer.fuse_dims`/`permute_dims` (+ der
Stride-Adjazenz-Test).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


# ===========================================================================
# Enumerationen (A05/06 Task 1a)
# ===========================================================================
class DimType(Enum):
    M = auto()
    N = auto()
    K = auto()
    C = auto()   # Batch (Contraction-Batch): in beiden Operanden UND im Output


class ExecType(Enum):
    SEQ = auto()
    PAR = auto()
    PRIM = auto()


class PrimType(Enum):
    GEMM = auto()
    BGEMM = auto()


class LastType(Enum):
    NONE = auto()
    ELWISE_MUL = auto()


class FirstType(Enum):
    ZERO = auto()


class DataType(Enum):
    FLOAT16 = auto()
    FLOAT32 = auto()


# ===========================================================================
# Config-Dataclass (A05/06 Task 1b)
# ===========================================================================
@dataclass
class Config:
    """Flache, parallele Sicht auf eine 2-Operanden-Kontraktion.

    Alle Listen sind gleich lang (= Anzahl globaler Dimensionen); `strides` hat
    genau 3 Einträge (Input 0, Input 1, Output), jeder eine Stride-Liste über
    dieselbe globale Dim-Reihenfolge (0 = Dim fehlt in diesem Tensor).
    """

    data_type: DataType
    prim_main: PrimType
    prim_last: LastType
    prim_first: FirstType
    dim_types: list[DimType]
    exec_types: list[ExecType]
    dim_sizes: list[int]
    strides: list[list[int]]


# ===========================================================================
# generate_config (A05/06 Task 2)
# ===========================================================================
def _parse_einsum(einsum: str) -> tuple[list[str], str]:
    """Splittet 'cmk,ckn->cmn' in (['cmk', 'ckn'], 'cmn')."""
    lhs, rhs = einsum.replace(" ", "").split("->")
    return lhs.split(","), rhs


def _classify(dim: str, inputs: list[str], output: str) -> DimType:
    in_inputs = [dim in t for t in inputs]
    in_output = dim in output
    if in_output and all(in_inputs):
        return DimType.C
    if not in_output and all(in_inputs):
        return DimType.K
    if in_output and in_inputs[0] and not in_inputs[1]:
        return DimType.M
    if in_output and in_inputs[1] and not in_inputs[0]:
        return DimType.N
    raise ValueError(f"Index {dim!r} nicht klassifizierbar")


def _row_major_strides(dim_order: list[str],
                       tensor_dims: str,
                       sizes: dict[str, int]) -> list[int]:
    """Row-Major-Strides eines Tensors, gemappt auf die globale `dim_order`.

    Stride 0 bedeutet: Dimension kommt in diesem Tensor nicht vor.
    """
    local: dict[str, int] = {}
    s = 1
    for d in reversed(tensor_dims):
        local[d] = s
        s *= sizes[d]
    return [local.get(d, 0) for d in dim_order]


def generate_config(einsum: str, shapes: list[tuple[int, ...]]) -> Config:
    """Erzeugt eine Basis-Config für die gegebene 2-Operanden-Kontraktion.

    Klassifikation: in allen Tensoren → C (Batch); nur in Inputs → K; in
    Input 0 + Output → M; in Input 1 + Output → N. Globale Dim-Reihenfolge =
    erstes Auftreten über (Inputs, Output). Strides: Row-Major pro Tensor.
    """
    inputs, output = _parse_einsum(einsum)
    if len(inputs) != 2:
        raise ValueError(f"nur 2-Operanden-Kontraktionen unterstützt, {len(inputs)} erhalten")
    if len(inputs) != len(shapes):
        raise ValueError(f"{len(inputs)} Input-Strings, aber {len(shapes)} Shapes")

    sizes: dict[str, int] = {}
    for tensor_dims, shape in zip(inputs, shapes):
        if len(tensor_dims) != len(shape):
            raise ValueError(f"Shape {shape} passt nicht zu einsum {tensor_dims!r}")
        for d, s in zip(tensor_dims, shape):
            if d in sizes and sizes[d] != s:
                raise ValueError(f"Index {d!r} hat inkonsistente Größen {sizes[d]} und {s}")
            sizes[d] = s
    for d in output:
        if d not in sizes:
            raise ValueError(f"Output-Index {d!r} kommt in keinem Input vor")

    dim_order: list[str] = []
    for source in (*inputs, output):
        for d in source:
            if d not in dim_order:
                dim_order.append(d)

    dim_types = [_classify(d, inputs, output) for d in dim_order]
    dim_sizes = [sizes[d] for d in dim_order]
    strides = [_row_major_strides(dim_order, t, sizes) for t in inputs]
    strides.append(_row_major_strides(dim_order, output, sizes))

    return Config(
        data_type=DataType.FLOAT16,
        prim_main=PrimType.GEMM,
        prim_last=LastType.NONE,
        prim_first=FirstType.ZERO,
        dim_types=dim_types,
        exec_types=[ExecType.SEQ] * len(dim_order),
        dim_sizes=dim_sizes,
        strides=strides,
    )


def pretty(cfg: Config, dim_labels: list[str] | None = None) -> str:
    """Tabellarische Darstellung einer Config (Debug/Report)."""
    n = len(cfg.dim_sizes)
    if dim_labels is None:
        dim_labels = [f"d{i}" for i in range(n)]
    header = f"{'pos':<4}{'name':<8}{'type':<6}{'exec':<6}{'size':>8}   strides"
    lines = [header, "-" * len(header)]
    for i in range(n):
        strides = " ".join(f"{s:>10}" for s in (t[i] for t in cfg.strides))
        lines.append(
            f"{i:<4}{dim_labels[i]:<8}{cfg.dim_types[i].name:<6}"
            f"{cfg.exec_types[i].name:<6}{cfg.dim_sizes[i]:>8}   {strides}")
    meta = (f"  data_type={cfg.data_type.name}  prim_main={cfg.prim_main.name}  "
            f"prim_last={cfg.prim_last.name}  prim_first={cfg.prim_first.name}")
    return "\n".join(lines) + "\n" + meta
