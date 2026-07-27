"""
cuTile-Kernels fuer die batched Kontraktion ``cmk, ckn -> cmn``
(Task 4a–c).

Enthaelt:
  - build_basic_config()       Task 4a: ruft generate_config auf.
  - build_l2_config()          Task 4b: zwei Split-Ebenen (mma-Tile +
                               Super-Tile) + permute + make_executable.
  - kernel_baseline            Plain 3D-Grid, keine BID-Swizzle.
  - kernel_l2_optimized        Generischer, config-getriebener Kernel:
                               Grid ueber die PAR-Achsen, GEMM ueber die
                               PRIM-Achsen. Der L2-Swizzle faellt aus der
                               Enumerationsreihenfolge der gesplitteten
                               Achsen (Gruppen-Achsen innen), NICHT aus einer
                               Swizzle-Formel im Kernel.
  - _extract_l2_params         liest Super/Group/Prim-Groessen aus der Config.
  - run_baseline, run_l2       Torch-Wrapper, die einen Stream und das
                               Output-Tensor ueber ct.launch starten.
  - reference, verify_kernel   Numerische Verifikation gegen
                               torch.einsum.

FP16-Inputs/Outputs, FP32-Akkumulator.
"""

import cuda.tile as ct
import torch

from config import Config, generate_config
from optimizer import Optimizer
from config import DimType, ExecType


# ===========================================================================
# Konstanten
# ===========================================================================

DIMS = dict(C=4, M=4096, N=4096, K=4096)

# Mma-Tile-Groessen (PRIM-Achsen). Wahl matched den Peak aus Assignment 04
# Task 3 (64x64x32 auf GB10 mit ct.mma).
M_PRIM = 64
N_PRIM = 64
K_PRIM = 32

# Super-Tile-Gruppe fuer den L2-Swizzle (in mma-Tile-Einheiten). Diese Groesse
# steuert NICHT den Kernel direkt, sondern die zusaetzliche Split-Ebene in der
# Config (m_l2 -> m_super x m_group): die L2-Lokalitaet entsteht datengetrieben
# aus der Enumerationsreihenfolge der gesplitteten Achsen (Gruppen-Achsen innen),
# nicht aus einer Swizzle-Formel im Kernel.
# Working-Set pro 2D-Super-Tile (FP16, K=4096) bei GROUP=8:
#   A: GROUP_M * M_PRIM * K * 2 B = 8 * 64 * 4096 * 2 = 4 MB
#   B: GROUP_N * N_PRIM * K * 2 B = 4 MB
#   C: GROUP_M * GROUP_N * M_PRIM * N_PRIM * 2 B = 0.5 MB (Register, nicht L2)
# Summe ~8 MB bei GROUP=8; passt in den L2 (GB10 ~24 MB) und laesst Platz fuer
# mehrere gleichzeitig aktive Super-Tiles. GROUP=8 ist der gemessene Sweet Spot.
GROUP_M = 8
GROUP_N = 8


# ===========================================================================
# Task 4a — Basis-Config
# ===========================================================================

def build_basic_config() -> Config:
    """Basis-Config fuer cmk, ckn -> cmn ohne jegliche Optimierung."""
    shape_a = (DIMS["C"], DIMS["M"], DIMS["K"])
    shape_b = (DIMS["C"], DIMS["K"], DIMS["N"])
    return generate_config("cmk,ckn->cmn", [shape_a, shape_b])


# ===========================================================================
# Task 4b — L2-optimierte Config
# ===========================================================================

def _largest_divisor_leq(value: int, cap: int) -> int:
    """Groesster Teiler von *value*, der <= cap ist (fuer kleine Verify-Dims)."""
    g = min(cap, value)
    while value % g != 0:
        g -= 1
    return g


def build_l2_config(dims: dict | None = None,
                    group_m: int = GROUP_M,
                    group_n: int = GROUP_N) -> Config:
    """Baut die L2-optimierte Config rein ueber Optimizer-Operationen.

    Der Super-Tile-Swizzle entsteht *datengetrieben* aus zwei Split-Ebenen:

      1. m -> (m_l2, m_prim),  n -> (n_l2, n_prim)     (mma-Tile abspalten)
      2. m_l2 -> (m_super, m_group),  n_l2 -> (n_super, n_group)  (Super-Tile)

    Danach werden die Achsen so permutiert, dass die PAR-Ebene
    ``[c, m_super, n_super, m_group, n_group]`` lautet (Gruppen-Achsen INNEN)
    und die PRIM-Ebene ``[m_prim, n_prim, k]``. Weil die Grid-Enumeration die
    Gruppen-Achsen zuletzt (innen) durchlaeuft, fallen aufeinanderfolgende BIDs
    in ein ``group_m x group_n`` Super-Tile -> A- und B-Tiles werden ueber den
    L2 geteilt. Das ist der Swizzle -- ganz ohne Index-Arithmetik im Kernel.
    """
    dims = dims or DIMS
    cfg = build_basic_config() if dims is DIMS else \
        generate_config("cmk,ckn->cmn",
                        [(dims["C"], dims["M"], dims["K"]),
                         (dims["C"], dims["K"], dims["N"])])
    opt = Optimizer(cfg)

    # 1) mma-Tile abspalten: m -> (m_l2, m_prim), n -> (n_l2, n_prim)
    m_id = next(i for i, t in enumerate(cfg.dim_types) if t == DimType.M)
    opt.split_dim(m_id, dims["M"] // M_PRIM, M_PRIM)
    n_id = next(i for i, t in enumerate(cfg.dim_types) if t == DimType.N)
    opt.split_dim(n_id, dims["N"] // N_PRIM, N_PRIM)

    # 2) Super-Tile abspalten: m_l2 -> (m_super, m_group), n_l2 -> (n_super, n_group).
    #    Gruppengroesse an die Anzahl L2-Tiles anpassen (kleine Verify-Dims).
    m_l2 = dims["M"] // M_PRIM
    n_l2 = dims["N"] // N_PRIM
    gm = _largest_divisor_leq(m_l2, group_m)
    gn = _largest_divisor_leq(n_l2, group_n)
    m_l2_id = next(i for i, t in enumerate(cfg.dim_types) if t == DimType.M)  # linkestes M = m_l2
    opt.split_dim(m_l2_id, m_l2 // gm, gm)
    n_l2_id = next(i for i, t in enumerate(cfg.dim_types) if t == DimType.N)  # linkestes N = n_l2
    opt.split_dim(n_l2_id, n_l2 // gn, gn)

    # Reihenfolge jetzt: [c, m_super, m_group, m_prim, k, n_super, n_group, n_prim]
    # Ziel: [c, m_super, n_super, m_group, n_group, m_prim, n_prim, k]
    opt.permute_dims([0, 1, 5, 2, 6, 3, 7, 4])
    opt.make_executable()
    return opt.config


def _extract_l2_params(cfg: Config) -> tuple[int, int, int, int, int, int, int, int]:
    """Liest die Super/Group/Prim-Groessen AUS der Config (datengetrieben).

    Reihenfolge in der PAR-Ebene ist [c, m_super, n_super, m_group, n_group],
    in der PRIM-Ebene [m_prim, n_prim, k]. Rueckgabe:
    (Cd, m_super, n_super, m_group, n_group, m_prim, n_prim, k).
    """
    par = [(cfg.dim_types[i], cfg.dim_sizes[i])
           for i in range(len(cfg.dim_sizes)) if cfg.exec_types[i] == ExecType.PAR]
    prim = [(cfg.dim_types[i], cfg.dim_sizes[i])
            for i in range(len(cfg.dim_sizes)) if cfg.exec_types[i] == ExecType.PRIM]
    Cd = next(s for t, s in par if t == DimType.C)
    m_par = [s for t, s in par if t == DimType.M]   # [m_super, m_group]
    n_par = [s for t, s in par if t == DimType.N]   # [n_super, n_group]
    m_super, m_group = m_par
    n_super, n_group = n_par
    m_prim = next(s for t, s in prim if t == DimType.M)
    n_prim = next(s for t, s in prim if t == DimType.N)
    k_size = next(s for t, s in prim if t == DimType.K)
    return Cd, m_super, n_super, m_group, n_group, m_prim, n_prim, k_size


# ===========================================================================
# Task 4c — Kernels
# ===========================================================================

@ct.kernel
def kernel_baseline(A, B, C,
                    Cd: ct.Constant[int],
                    M:  ct.Constant[int],
                    N:  ct.Constant[int],
                    K:  ct.Constant[int],
                    tm: ct.Constant[int],
                    tn: ct.Constant[int],
                    tk: ct.Constant[int]):
    """Baseline: Grid (c, num_m_tiles, num_n_tiles), keine Swizzle.

    BIDs werden in der Default-Reihenfolge enumeriert (z innermost).
    Damit haben benachbarte BIDs in einer Wave gleiches (c, m_tile)
    aber unterschiedliche n_tiles -> A wird ueber L2 geteilt, B nicht.
    """
    pid_c = ct.bid(0)
    pid_m = ct.bid(1)
    pid_n = ct.bid(2)

    num_tiles_k = ct.cdiv(K, tk)
    acc = ct.full((tm, tn), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    for kk in range(num_tiles_k):
        a_tile = ct.load(A, index=(pid_c, pid_m, kk),
                         shape=(1, tm, tk), padding_mode=zero_pad)
        b_tile = ct.load(B, index=(pid_c, kk, pid_n),
                         shape=(1, tk, tn), padding_mode=zero_pad)
        a2d = ct.reshape(a_tile, (tm, tk))
        b2d = ct.reshape(b_tile, (tk, tn))
        acc = ct.mma(a2d, b2d, acc)

    out = ct.reshape(ct.astype(acc, C.dtype), (1, tm, tn))
    ct.store(C, index=(pid_c, pid_m, pid_n), tile=out)


@ct.kernel
def kernel_l2_optimized(A, B, C,
                        MS: ct.Constant[int], NS: ct.Constant[int],
                        MG: ct.Constant[int], NG: ct.Constant[int],
                        K:  ct.Constant[int],
                        tm: ct.Constant[int],
                        tn: ct.Constant[int],
                        tk: ct.Constant[int]):
    """Generischer, config-getriebener Kernel fuer die PAR/PRIM-Struktur.

    Das flache 1D-Grid enumeriert die PAR-Achsen in Config-Reihenfolge
    ``[c, m_super, n_super, m_group, n_group]`` (Gruppen-Achsen INNEN). Der
    Kernel dekodiert den BID generisch per verschachteltem divmod ueber die
    PAR-Groessen (MS, NS, MG, NG kommen aus der Config) und rekonstruiert die
    m-/n-Tile-Indizes aus Super- und Group-Anteil. Der L2-Swizzle faellt so aus
    der Enumerationsreihenfolge der gesplitteten Achsen -- KEINE
    ``// blocks_per_group``-Formel mehr, keine im Kernel verdrahtete Gruppierung.
    GEMM laeuft ueber die PRIM-Achsen (m_prim=tm, n_prim=tn, k).
    """
    bid = ct.bid(0)

    # Decode in Config-Reihenfolge, innerste PAR-Achse (n_group) zuerst.
    n_grp = bid %  NG
    t     = bid // NG
    m_grp = t %  MG
    t     = t // MG
    n_sup = t %  NS
    t     = t // NS
    m_sup = t %  MS
    pid_c = t // MS

    # m_l2/n_l2-Tile-Index = super * group_size + group_offset
    pid_m = m_sup * MG + m_grp
    pid_n = n_sup * NG + n_grp

    num_tiles_k = ct.cdiv(K, tk)
    acc = ct.full((tm, tn), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    for kk in range(num_tiles_k):
        a_tile = ct.load(A, index=(pid_c, pid_m, kk),
                         shape=(1, tm, tk), padding_mode=zero_pad)
        b_tile = ct.load(B, index=(pid_c, kk, pid_n),
                         shape=(1, tk, tn), padding_mode=zero_pad)
        a2d = ct.reshape(a_tile, (tm, tk))
        b2d = ct.reshape(b_tile, (tk, tn))
        acc = ct.mma(a2d, b2d, acc)

    out = ct.reshape(ct.astype(acc, C.dtype), (1, tm, tn))
    ct.store(C, index=(pid_c, pid_m, pid_n), tile=out)


# ===========================================================================
# Launch-Wrapper
# ===========================================================================

def run_baseline(A: torch.Tensor, B: torch.Tensor,
                 dims: dict | None = None,
                 tile: tuple[int, int, int] = (M_PRIM, N_PRIM, K_PRIM)
                 ) -> torch.Tensor:
    dims = dims or DIMS
    Cd, M, N, K = dims["C"], dims["M"], dims["N"], dims["K"]
    tm, tn, tk = tile
    Cout = torch.empty((Cd, M, N), device=A.device, dtype=A.dtype)
    grid = (Cd, ct.cdiv(M, tm), ct.cdiv(N, tn))
    ct.launch(
        torch.cuda.current_stream().cuda_stream,
        grid, kernel_baseline,
        (A, B, Cout, Cd, M, N, K, tm, tn, tk),
    )
    return Cout


def run_l2(A: torch.Tensor, B: torch.Tensor,
           dims: dict | None = None,
           tile: tuple[int, int, int] = (M_PRIM, N_PRIM, K_PRIM),
           group: tuple[int, int] = (GROUP_M, GROUP_N)
           ) -> torch.Tensor:
    """Config-getrieben: L2-Config bauen, Super/Group/Prim-Groessen daraus
    lesen und das flache PAR-Grid starten. Aendert man die Split-/Permute-
    Pipeline in ``build_l2_config``, aendert sich das Launch-Layout automatisch
    mit -- ohne den Kernel anzufassen."""
    dims = dims or DIMS
    Cd, M, N, K = dims["C"], dims["M"], dims["N"], dims["K"]
    tm, tn, tk = tile
    gm, gn = group

    cfg = build_l2_config(dims, group_m=gm, group_n=gn)
    Cd_c, MS, NS, MG, NG, m_prim, n_prim, k_size = _extract_l2_params(cfg)

    Cout = torch.empty((Cd, M, N), device=A.device, dtype=A.dtype)
    grid = (Cd * MS * NS * MG * NG, 1, 1)   # flaches PAR-Grid, Gruppen-Achsen innen
    ct.launch(
        torch.cuda.current_stream().cuda_stream,
        grid, kernel_l2_optimized,
        (A, B, Cout, MS, NS, MG, NG, k_size, m_prim, n_prim, tk),
    )
    return Cout


# ===========================================================================
# Verifikation
# ===========================================================================

VERIFY_DIMS = dict(C=2, M=128, N=128, K=128)


def reference(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """torch.einsum mit FP32-Promotion und FP16-Rueckgabe."""
    out_f32 = torch.einsum("cmk,ckn->cmn", A.float(), B.float())
    return out_f32.to(torch.float16)


def verify_kernel(run_fn, name: str,
                  dims: dict | None = None,
                  **kwargs) -> None:
    dims = dims or VERIFY_DIMS
    Cd, M, N, K = dims["C"], dims["M"], dims["N"], dims["K"]
    torch.manual_seed(0)
    A = torch.randn(Cd, M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(Cd, K, N, device="cuda", dtype=torch.float16)
    ref = reference(A, B)
    out = run_fn(A, B, dims=dims, **kwargs)
    ok = torch.allclose(out, ref, atol=2e-1, rtol=2e-2)
    err = (out.float() - ref.float()).abs().max().item()
    print(f"  {name:<10} allclose={ok}  max_abs_err={err:.4f}")
    if not ok:
        raise AssertionError(f"{name}: kernel output does not match reference")
