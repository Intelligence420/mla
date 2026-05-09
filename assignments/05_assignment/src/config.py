"""
Config-Modul: Enums, Config-Dataclass und generate_config.

Deckt Task 1 (Enums + Config-Dataclass) und Task 2 (generate_config)
des Assignments ab. Diese Definitionen werden in optimizer.py und
kernel.py wiederverwendet.
"""

from dataclasses import dataclass
from enum import Enum, auto


# ===========================================================================
# Task 1a — Enumerations
# ===========================================================================

class DimType(Enum):
    M = auto()
    N = auto()
    K = auto()
    C = auto()


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
# Task 1b — Config-Dataclass
# ===========================================================================

@dataclass
class Config:
    data_type: DataType
    prim_main: PrimType
    prim_last: LastType
    prim_first: FirstType
    dim_types: list[DimType]
    exec_types: list[ExecType]
    dim_sizes: list[int]
    strides: list[list[int]]


# ===========================================================================
# Task 2 — generate_config
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
    raise ValueError(f"cannot classify index {dim!r}")


def _row_major_strides(dim_order: list[str],
                       tensor_dims: str,
                       sizes: dict[str, int]) -> list[int]:
    """Row-Major-Strides eines Tensors, gemappt auf die globale dim_order.

    Stride 0 bedeutet: Dimension kommt in diesem Tensor nicht vor.
    """
    local: dict[str, int] = {}
    s = 1
    for d in reversed(tensor_dims):
        local[d] = s
        s *= sizes[d]
    return [local.get(d, 0) for d in dim_order]


def generate_config(einsum: str, shapes: list[tuple[int, ...]]) -> Config:
    """Erzeugt eine Basis-Config fuer die gegebene Kontraktion.

    Klassifikation:
      - in allen Tensoren (Inputs + Output) -> C
      - nur in Inputs                       -> K
      - in Input 0 + Output                 -> M
      - in Input 1 + Output                 -> N

    Globale Dim-Reihenfolge: erstes Auftreten ueber Inputs und Output.
    Strides: Row-Major pro Tensor, 0 fuer fehlende Dims.
    """
    inputs, output = _parse_einsum(einsum)
    if len(inputs) != 2:
        raise ValueError(f"only 2-input contractions supported, got {len(inputs)}")
    if len(inputs) != len(shapes):
        raise ValueError(f"got {len(inputs)} input strings but {len(shapes)} shapes")

    sizes: dict[str, int] = {}
    for tensor_dims, shape in zip(inputs, shapes):
        if len(tensor_dims) != len(shape):
            raise ValueError(f"shape {shape} does not match einsum {tensor_dims!r}")
        for d, s in zip(tensor_dims, shape):
            if d in sizes and sizes[d] != s:
                raise ValueError(
                    f"index {d!r} has inconsistent sizes {sizes[d]} and {s}")
            sizes[d] = s
    for d in output:
        if d not in sizes:
            raise ValueError(f"output index {d!r} not present in any input")

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
    """Tabellarische Darstellung einer Config (fuer Reports und __main__)."""
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
    meta = (f"  data_type={cfg.data_type.name}  "
            f"prim_main={cfg.prim_main.name}  "
            f"prim_last={cfg.prim_last.name}  "
            f"prim_first={cfg.prim_first.name}")
    return "\n".join(lines) + "\n" + meta


if __name__ == "__main__":
    # cmk, ckn -> cmn (Aufgabe Task 4a)
    cfg = generate_config("cmk,ckn->cmn", [(4, 4096, 4096), (4, 4096, 4096)])
    print("cmk,ckn->cmn  shapes (4,4096,4096), (4,4096,4096):")
    print(pretty(cfg, list("cmkn")))
    print()

    # mk, kn -> mn (klassisches GEMM)
    cfg = generate_config("mk,kn->mn", [(128, 64), (64, 256)])
    print("mk,kn->mn  shapes (128,64), (64,256):")
    print(pretty(cfg, list("mkn")))
    print()

    # bij, bjk -> bik (Batched GEMM)
    cfg = generate_config("bij,bjk->bik", [(8, 32, 64), (8, 64, 16)])
    print("bij,bjk->bik  shapes (8,32,64), (8,64,16):")
    print(pretty(cfg, list("bijk")))
