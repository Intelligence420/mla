"""tool_pipeline.intermediate_representation.optimizer — IR-Transformationen (Port aus A05/06).

**Originalgetreuer Port** von `assignments/05|06/src/optimizer.py`: ein Wrapper um
eine `Config` mit in-place-Transformationen

  a) `split_dim(dim_id, outer_size, inner_size)`  — Dimension zerlegen (Tile-Injektion)
  b) `fuse_dims(dim_id_a, dim_id_b)`              — zwei Dims verschmelzen (mit Stride-Adjazenz-Test)
  c) `permute_dims(permutation)`                  — Achsen umordnen
  d) `make_executable()` / e) `verify()`          — A05/06-Host-Tiling-Heuristik

Der **Stride-Adjazenz-Test** in `fuse_dims` (`stra == strb*size_b` oder
`stra*size_a == strb`) ist das formale **zero-copy-Kriterium**: nur benachbarte
Dims lassen sich ohne Kopie zu einer Achse zusammenfassen. `reshape.to_canonical`
nutzt daraus `fuse_dims`/`permute_dims` + den (hier ausgelagerten) Adjazenz-Test,
um den B1-View zu planen und seine zero-copy-Eigenschaft vorherzusagen.

`make_executable`/`verify` gehören zur A05/06-Host-Tiling-Heuristik (PRIM/SEQ/PAR-
Scheduling) — unser Codegen kachelt im Template, daher werden sie vom B1-Reshape
**nicht** aufgerufen; sie sind für Vollständigkeit (voller Port) mitgeführt.
"""

from __future__ import annotations

from .config import Config, DimType, ExecType


def is_row_major_contiguous_run(sizes: list[int], strides: list[int]) -> bool:
    """Ob aufeinanderfolgende Achsen (in dieser Reihenfolge) row-major zusammenhängen.

    Kriterium wie `fuse_dims`: benachbart ⇔ ``stride[i] == stride[i+1]*size[i+1]``.
    Trifft das für ALLE Nachbarpaare zu, ist das Verschmelzen dieser Achsen zu
    **einer** Achse ein zero-copy-View. Leere/Einzel-Liste ⇒ trivial ``True``.
    """
    for i in range(len(strides) - 1):
        if strides[i] != strides[i + 1] * sizes[i + 1]:
            return False
    return True


class Optimizer:
    """In-place-Transformationen auf einer `Config` (Port aus A05/06)."""

    def __init__(self, config: Config):
        self.config = config

    # -----------------------------------------------------------------
    # a) split_dim — Dimension in (outer, inner) zerlegen
    # -----------------------------------------------------------------
    def split_dim(self, dim_id: int, outer_size: int, inner_size: int) -> None:
        cfg = self.config
        original = cfg.dim_sizes[dim_id]
        if outer_size * inner_size != original:
            raise ValueError(
                f"split_dim({dim_id}, {outer_size}, {inner_size}): Produkt "
                f"{outer_size * inner_size} != ursprüngliche Größe {original}")

        dt = cfg.dim_types[dim_id]
        et = cfg.exec_types[dim_id]
        cfg.dim_types[dim_id:dim_id + 1] = [dt, dt]
        cfg.exec_types[dim_id:dim_id + 1] = [et, et]
        cfg.dim_sizes[dim_id:dim_id + 1] = [outer_size, inner_size]

        for strides in cfg.strides:
            old = strides[dim_id]
            if old == 0:
                strides[dim_id:dim_id + 1] = [0, 0]
            else:
                # Innerer Stride bleibt, äußerer steppt über inner_size Elemente.
                strides[dim_id:dim_id + 1] = [old * inner_size, old]

    # -----------------------------------------------------------------
    # b) fuse_dims — zwei Dims verschmelzen (nur wenn zero-copy zulässig)
    # -----------------------------------------------------------------
    def fuse_dims(self, dim_id_a: int, dim_id_b: int) -> None:
        cfg = self.config
        if dim_id_a == dim_id_b:
            raise ValueError("kann eine Dimension nicht mit sich selbst fusionieren")
        size_a = cfg.dim_sizes[dim_id_a]
        size_b = cfg.dim_sizes[dim_id_b]

        for t, strides in enumerate(cfg.strides):
            stra, strb = strides[dim_id_a], strides[dim_id_b]
            if stra == 0 or strb == 0:
                # Mindestens eine Dim fehlt in diesem Tensor → Adjazenz trivial.
                continue
            adjacent = (stra == strb * size_b) or (stra * size_a == strb)
            if not adjacent:
                raise ValueError(
                    f"fuse_dims({dim_id_a}, {dim_id_b}): Dims in Tensor {t} nicht "
                    f"benachbart (stride_a={stra}, size_a={size_a}, "
                    f"stride_b={strb}, size_b={size_b})")

        new_strides = []
        for strides in cfg.strides:
            stra, strb = strides[dim_id_a], strides[dim_id_b]
            if stra == 0 and strb == 0:
                new_strides.append(0)
            elif stra == 0:
                new_strides.append(strb)
            elif strb == 0:
                new_strides.append(stra)
            else:
                new_strides.append(min(stra, strb))  # innerer (kleinerer) Stride

        cfg.dim_sizes[dim_id_a] = size_a * size_b
        for t, strides in enumerate(cfg.strides):
            strides[dim_id_a] = new_strides[t]

        del cfg.dim_types[dim_id_b]
        del cfg.exec_types[dim_id_b]
        del cfg.dim_sizes[dim_id_b]
        for strides in cfg.strides:
            del strides[dim_id_b]

    # -----------------------------------------------------------------
    # c) permute_dims — Achsen umordnen
    # -----------------------------------------------------------------
    def permute_dims(self, permutation: list[int]) -> None:
        cfg = self.config
        n = len(cfg.dim_sizes)
        if sorted(permutation) != list(range(n)):
            raise ValueError(f"{permutation} ist keine Permutation von range({n})")

        cfg.dim_types = [cfg.dim_types[i] for i in permutation]
        cfg.exec_types = [cfg.exec_types[i] for i in permutation]
        cfg.dim_sizes = [cfg.dim_sizes[i] for i in permutation]
        cfg.strides = [[s[i] for i in permutation] for s in cfg.strides]

    # -----------------------------------------------------------------
    # d) make_executable — A05/06-Scheduling-Heuristik (vom B1-Reshape ungenutzt)
    # -----------------------------------------------------------------
    def make_executable(self) -> None:
        """Setzt exec_types + permutiert in cuTile-konforme [PAR…, SEQ…, PRIM…]-
        Ordnung (A05/06-Host-Tiling-Heuristik; unser Codegen kachelt im Template)."""
        cfg = self.config
        prim_picks: dict[DimType, int] = {}
        for i in range(len(cfg.dim_types) - 1, -1, -1):
            dt = cfg.dim_types[i]
            if dt in (DimType.M, DimType.N, DimType.K) and dt not in prim_picks:
                prim_picks[dt] = i
        for required in (DimType.M, DimType.N, DimType.K):
            if required not in prim_picks:
                raise ValueError(f"nicht ausführbar: keine {required.name}-Dim gefunden")

        prim_set = set(prim_picks.values())
        for i, dt in enumerate(cfg.dim_types):
            if i in prim_set:
                cfg.exec_types[i] = ExecType.PRIM
            elif dt == DimType.K:
                cfg.exec_types[i] = ExecType.SEQ
            else:
                cfg.exec_types[i] = ExecType.PAR

        order_key = {ExecType.PAR: 0, ExecType.SEQ: 1, ExecType.PRIM: 2}
        n = len(cfg.dim_sizes)
        permutation = sorted(range(n), key=lambda i: (order_key[cfg.exec_types[i]], i))
        self.permute_dims(permutation)
        self.verify()

    # -----------------------------------------------------------------
    # e) verify — Ausführbarkeits-Invarianten (zu make_executable)
    # -----------------------------------------------------------------
    def verify(self) -> None:
        cfg = self.config
        for i, (dt, et) in enumerate(zip(cfg.dim_types, cfg.exec_types)):
            if dt == DimType.K and et == ExecType.PAR:
                raise ValueError(f"Dim {i} ist K, aber exec_type=PAR (unzulässig)")

        positions: dict[ExecType, list[int]] = {
            ExecType.PAR: [], ExecType.SEQ: [], ExecType.PRIM: []
        }
        for i, et in enumerate(cfg.exec_types):
            positions[et].append(i)

        if positions[ExecType.SEQ] and positions[ExecType.PRIM]:
            if max(positions[ExecType.SEQ]) > min(positions[ExecType.PRIM]):
                raise ValueError("SEQ-Dims müssen links aller PRIM-Dims stehen")
        if positions[ExecType.PAR] and positions[ExecType.SEQ]:
            if max(positions[ExecType.PAR]) > min(positions[ExecType.SEQ]):
                raise ValueError("PAR-Dims müssen links aller SEQ-Dims stehen")
        if positions[ExecType.PAR] and positions[ExecType.PRIM]:
            if max(positions[ExecType.PAR]) > min(positions[ExecType.PRIM]):
                raise ValueError("PAR-Dims müssen links aller PRIM-Dims stehen")

        prim_positions = positions[ExecType.PRIM]
        if not prim_positions:
            raise ValueError("keine PRIM-Dims; PRIM muss M, N und K enthalten")
        n = len(cfg.exec_types)
        expected_tail = list(range(n - len(prim_positions), n))
        if prim_positions != expected_tail:
            raise ValueError("PRIM-Dims müssen den rechtesten zusammenhängenden Block bilden")
        prim_types = {cfg.dim_types[i] for i in prim_positions}
        for required in (DimType.M, DimType.N, DimType.K):
            if required not in prim_types:
                raise ValueError(f"PRIM-Dims müssen mindestens eine {required.name}-Dim enthalten")
