"""
Optimizer-Modul (Task 3).

Wrapper um eine Config mit Transformations-Methoden:

  a) split_dim(dim_id, outer_size, inner_size)
  b) fuse_dims(dim_id_a, dim_id_b)
  c) permute_dims(permutation)
  d) make_executable()
  e) verify()

Alle Methoden manipulieren die Config in-place.
"""

from config import Config, DimType, ExecType


class Optimizer:
    def __init__(self, config: Config):
        self.config = config

    # -----------------------------------------------------------------
    # Task 3a — split_dim
    # -----------------------------------------------------------------
    def split_dim(self, dim_id: int, outer_size: int, inner_size: int) -> None:
        cfg = self.config
        original = cfg.dim_sizes[dim_id]
        if outer_size * inner_size != original:
            raise ValueError(
                f"split_dim({dim_id}, {outer_size}, {inner_size}): product "
                f"{outer_size * inner_size} != original size {original}")

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
                # Inner stride bleibt gleich, outer steppt ueber inner_size Elemente.
                strides[dim_id:dim_id + 1] = [old * inner_size, old]

    # -----------------------------------------------------------------
    # Task 3b — fuse_dims
    # -----------------------------------------------------------------
    def fuse_dims(self, dim_id_a: int, dim_id_b: int) -> None:
        cfg = self.config
        if dim_id_a == dim_id_b:
            raise ValueError("cannot fuse a dimension with itself")
        size_a = cfg.dim_sizes[dim_id_a]
        size_b = cfg.dim_sizes[dim_id_b]

        for t, strides in enumerate(cfg.strides):
            stra, strb = strides[dim_id_a], strides[dim_id_b]
            if stra == 0 or strb == 0:
                # Mindestens eine Dim fehlt in diesem Tensor -> Adjacency
                # ist trivial erfuellt (sie sind nicht beide hier).
                continue
            adjacent = (stra == strb * size_b) or (stra * size_a == strb)
            if not adjacent:
                raise ValueError(
                    f"fuse_dims({dim_id_a}, {dim_id_b}): dims not adjacent "
                    f"in tensor {t} (stride_a={stra}, size_a={size_a}, "
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
                # innerer (kleinerer) Stride bleibt fuer die fusionierte Dim.
                new_strides.append(min(stra, strb))

        cfg.dim_sizes[dim_id_a] = size_a * size_b
        for t, strides in enumerate(cfg.strides):
            strides[dim_id_a] = new_strides[t]

        del cfg.dim_types[dim_id_b]
        del cfg.exec_types[dim_id_b]
        del cfg.dim_sizes[dim_id_b]
        for strides in cfg.strides:
            del strides[dim_id_b]

    # -----------------------------------------------------------------
    # Task 3c — permute_dims
    # -----------------------------------------------------------------
    def permute_dims(self, permutation: list[int]) -> None:
        cfg = self.config
        n = len(cfg.dim_sizes)
        if sorted(permutation) != list(range(n)):
            raise ValueError(
                f"permutation {permutation} is not a permutation of range({n})")

        cfg.dim_types = [cfg.dim_types[i] for i in permutation]
        cfg.exec_types = [cfg.exec_types[i] for i in permutation]
        cfg.dim_sizes = [cfg.dim_sizes[i] for i in permutation]
        cfg.strides = [[s[i] for i in permutation] for s in cfg.strides]

    # -----------------------------------------------------------------
    # Task 3d — make_executable
    # -----------------------------------------------------------------
    def make_executable(self) -> None:
        """Setzt exec_types und permutiert so, dass die Config cuTile-konform ist.

        Heuristik:
          - Pro Typ M/N/K wird die *rechteste* passende Dim als PRIM markiert.
          - Verbleibende K-Dims werden SEQ (PAR ist verboten).
          - Verbleibende M/N/C-Dims werden PAR (mehr Parallelitaet).
          - Anschliessend [PAR..., SEQ..., PRIM...] permutieren (stabil
            innerhalb eines Blocks, sodass die ursprungliche Reihenfolge
            erhalten bleibt).
        """
        cfg = self.config
        prim_picks: dict[DimType, int] = {}
        for i in range(len(cfg.dim_types) - 1, -1, -1):
            dt = cfg.dim_types[i]
            if dt in (DimType.M, DimType.N, DimType.K) and dt not in prim_picks:
                prim_picks[dt] = i
        for required in (DimType.M, DimType.N, DimType.K):
            if required not in prim_picks:
                raise ValueError(
                    f"cannot make config executable: no {required.name} "
                    f"dim found")

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
        permutation = sorted(range(n),
                             key=lambda i: (order_key[cfg.exec_types[i]], i))
        self.permute_dims(permutation)

        self.verify()

    # -----------------------------------------------------------------
    # Task 3e — verify
    # -----------------------------------------------------------------
    def verify(self) -> None:
        cfg = self.config

        for i, (dt, et) in enumerate(zip(cfg.dim_types, cfg.exec_types)):
            if dt == DimType.K and et == ExecType.PAR:
                raise ValueError(
                    f"dim {i} is K but exec_type=PAR (not allowed)")

        positions: dict[ExecType, list[int]] = {
            ExecType.PAR: [], ExecType.SEQ: [], ExecType.PRIM: []
        }
        for i, et in enumerate(cfg.exec_types):
            positions[et].append(i)

        if positions[ExecType.SEQ] and positions[ExecType.PRIM]:
            if max(positions[ExecType.SEQ]) > min(positions[ExecType.PRIM]):
                raise ValueError(
                    "SEQ dims must appear left of all PRIM dims")
        if positions[ExecType.PAR] and positions[ExecType.SEQ]:
            if max(positions[ExecType.PAR]) > min(positions[ExecType.SEQ]):
                raise ValueError(
                    "PAR dims must appear left of all SEQ dims")
        if positions[ExecType.PAR] and positions[ExecType.PRIM]:
            if max(positions[ExecType.PAR]) > min(positions[ExecType.PRIM]):
                raise ValueError(
                    "PAR dims must appear left of all PRIM dims")

        prim_positions = positions[ExecType.PRIM]
        if not prim_positions:
            raise ValueError(
                "no PRIM dims found; PRIM must include M, N, and K")
        n = len(cfg.exec_types)
        expected_tail = list(range(n - len(prim_positions), n))
        if prim_positions != expected_tail:
            raise ValueError(
                "PRIM dims must form the rightmost contiguous block")
        prim_types = {cfg.dim_types[i] for i in prim_positions}
        for required in (DimType.M, DimType.N, DimType.K):
            if required not in prim_types:
                raise ValueError(
                    f"PRIM dims must include at least one {required.name}")


if __name__ == "__main__":
    from config import generate_config, pretty

    print("=" * 70)
    print("Demo: Optimizer-Pipeline auf cmk,ckn->cmn")
    print("=" * 70)

    cfg = generate_config("cmk,ckn->cmn", [(4, 4096, 4096), (4, 4096, 4096)])
    opt = Optimizer(cfg)

    print("\n[0] Basis-Config:")
    print(pretty(cfg, list("cmkn")))

    m_id = next(i for i, t in enumerate(cfg.dim_types) if t == DimType.M)
    opt.split_dim(m_id, 4096 // 64, 64)
    print("\n[1] split_dim(m -> m_l2=64, m_prim=64):")
    print(pretty(cfg, ["c", "m_l2", "m_prim", "k", "n"]))

    n_id = next(i for i, t in enumerate(cfg.dim_types) if t == DimType.N)
    opt.split_dim(n_id, 4096 // 64, 64)
    print("\n[2] split_dim(n -> n_l2=64, n_prim=64):")
    print(pretty(cfg, ["c", "m_l2", "m_prim", "k", "n_l2", "n_prim"]))

    # Ziel-Layout: c, m_l2, n_l2, m_prim, n_prim, k
    opt.permute_dims([0, 1, 4, 2, 5, 3])
    print("\n[3] permute_dims to [c, m_l2, n_l2, m_prim, n_prim, k]:")
    print(pretty(cfg, ["c", "m_l2", "n_l2", "m_prim", "n_prim", "k"]))

    opt.make_executable()
    print("\n[4] make_executable():")
    print(pretty(cfg, ["c", "m_l2", "n_l2", "m_prim", "n_prim", "k"]))

    print("\n[5] verify(): OK")

    # ----- fuse als Identitaet zur split: split + fuse soll Original liefern
    print()
    print("=" * 70)
    print("Sanity-Check: split_dim + fuse_dims = Identitaet")
    print("=" * 70)
    cfg2 = generate_config("mk,kn->mn", [(128, 64), (64, 256)])
    opt2 = Optimizer(cfg2)
    print("\nVorher:")
    print(pretty(cfg2, list("mkn")))
    opt2.split_dim(0, 16, 8)  # m=128 -> (16, 8)
    print("\nNach split:")
    print(pretty(cfg2, ["m_outer", "m_inner", "k", "n"]))
    opt2.fuse_dims(0, 1)
    print("\nNach fuse:")
    print(pretty(cfg2, list("mkn")))

    # ----- verify-Fehlerfaelle
    print()
    print("=" * 70)
    print("verify(): erwartete ValueErrors")
    print("=" * 70)
    cfg3 = generate_config("mk,kn->mn", [(128, 64), (64, 256)])
    opt3 = Optimizer(cfg3)
    opt3.config.exec_types = [ExecType.PAR, ExecType.PAR, ExecType.PAR]
    try:
        opt3.verify()
    except ValueError as e:
        print(f"  K=PAR rejected: {e}")

    opt3.config.exec_types = [ExecType.PRIM, ExecType.SEQ, ExecType.PRIM]
    try:
        opt3.verify()
    except ValueError as e:
        print(f"  SEQ left of PRIM violated: {e}")



"""Ergebnisse
(.venv) mla08@flambe:~/MLA/mla/assignments/05_assignment/src$ python3 optimizer.py 
======================================================================
Demo: Optimizer-Pipeline auf cmk,ckn->cmn
======================================================================

[0] Basis-Config:
pos name    type  exec      size   strides
------------------------------------------
0   c       C     SEQ          4     16777216   16777216   16777216
1   m       M     SEQ       4096         4096          0       4096
2   k       K     SEQ       4096            1       4096          0
3   n       N     SEQ       4096            0          1          1
  data_type=FLOAT16  prim_main=GEMM  prim_last=NONE  prim_first=ZERO

[1] split_dim(m -> m_l2=64, m_prim=64):
pos name    type  exec      size   strides
------------------------------------------
0   c       C     SEQ          4     16777216   16777216   16777216
1   m_l2    M     SEQ         64       262144          0     262144
2   m_prim  M     SEQ         64         4096          0       4096
3   k       K     SEQ       4096            1       4096          0
4   n       N     SEQ       4096            0          1          1
  data_type=FLOAT16  prim_main=GEMM  prim_last=NONE  prim_first=ZERO

[2] split_dim(n -> n_l2=64, n_prim=64):
pos name    type  exec      size   strides
------------------------------------------
0   c       C     SEQ          4     16777216   16777216   16777216
1   m_l2    M     SEQ         64       262144          0     262144
2   m_prim  M     SEQ         64         4096          0       4096
3   k       K     SEQ       4096            1       4096          0
4   n_l2    N     SEQ         64            0         64         64
5   n_prim  N     SEQ         64            0          1          1
  data_type=FLOAT16  prim_main=GEMM  prim_last=NONE  prim_first=ZERO

[3] permute_dims to [c, m_l2, n_l2, m_prim, n_prim, k]:
pos name    type  exec      size   strides
------------------------------------------
0   c       C     SEQ          4     16777216   16777216   16777216
1   m_l2    M     SEQ         64       262144          0     262144
2   n_l2    N     SEQ         64            0         64         64
3   m_prim  M     SEQ         64         4096          0       4096
4   n_prim  N     SEQ         64            0          1          1
5   k       K     SEQ       4096            1       4096          0
  data_type=FLOAT16  prim_main=GEMM  prim_last=NONE  prim_first=ZERO

[4] make_executable():
pos name    type  exec      size   strides
------------------------------------------
0   c       C     PAR          4     16777216   16777216   16777216
1   m_l2    M     PAR         64       262144          0     262144
2   n_l2    N     PAR         64            0         64         64
3   m_prim  M     PRIM        64         4096          0       4096
4   n_prim  N     PRIM        64            0          1          1
5   k       K     PRIM      4096            1       4096          0
  data_type=FLOAT16  prim_main=GEMM  prim_last=NONE  prim_first=ZERO

[5] verify(): OK

======================================================================
Sanity-Check: split_dim + fuse_dims = Identitaet
======================================================================

Vorher:
pos name    type  exec      size   strides
------------------------------------------
0   m       M     SEQ        128           64          0        256
1   k       K     SEQ         64            1        256          0
2   n       N     SEQ        256            0          1          1
  data_type=FLOAT16  prim_main=GEMM  prim_last=NONE  prim_first=ZERO

Nach split:
pos name    type  exec      size   strides
------------------------------------------
0   m_outer M     SEQ         16          512          0       2048
1   m_inner M     SEQ          8           64          0        256
2   k       K     SEQ         64            1        256          0
3   n       N     SEQ        256            0          1          1
  data_type=FLOAT16  prim_main=GEMM  prim_last=NONE  prim_first=ZERO

Nach fuse:
pos name    type  exec      size   strides
------------------------------------------
0   m       M     SEQ        128           64          0        256
1   k       K     SEQ         64            1        256          0
2   n       N     SEQ        256            0          1          1
  data_type=FLOAT16  prim_main=GEMM  prim_last=NONE  prim_first=ZERO

======================================================================
verify(): erwartete ValueErrors
======================================================================
  K=PAR rejected: dim 1 is K but exec_type=PAR (not allowed)
  SEQ left of PRIM violated: SEQ dims must appear left of all PRIM dims
"""