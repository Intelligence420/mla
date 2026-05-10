"""
cuTile-Kernels fuer die batched Kontraktion ``cmk, ckn -> cmn``
(Task 4a–c).

Enthaelt:
  - build_basic_config()       Task 4a: ruft generate_config auf.
  - build_l2_config()          Task 4b: split/permute/make_executable.
  - kernel_baseline            Plain 3D-Grid, keine BID-Swizzle.
  - kernel_l2_optimized        Gleiche per-Block-Arbeit, aber BID
                               in 2D-Super-Tiles gruppiert (klassische
                               Triton/CUTLASS L2-Swizzle).
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
from config import DimType


# ===========================================================================
# Konstanten
# ===========================================================================

DIMS = dict(C=4, M=4096, N=4096, K=4096)

# Mma-Tile-Groessen (PRIM-Achsen). Wahl matched den Peak aus Assignment 04
# Task 3 (64x64x32 auf GB10 mit ct.mma).
M_PRIM = 64
N_PRIM = 64
K_PRIM = 32

# Super-Tile-Gruppe fuer L2-Swizzle (in mma-Tile-Einheiten).
# Working-Set pro Super-Tile (FP16, K=4096):
#   A: GROUP_M * M_PRIM * K * 2 B = 8 * 64 * 4096 * 2 = 4 MB
#   B: GROUP_N * N_PRIM * K * 2 B = 4 MB
#   C: GROUP_M * GROUP_N * M_PRIM * N_PRIM * 2 B = 0.5 MB
# Summe ~8.5 MB; passt locker in L2 (DGX Spark / GB10 ~30 MB),
# laesst Platz fuer mehrere Super-Tiles in flight.
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

def build_l2_config() -> Config:
    """Pipeline: split m und n in (l2, prim), permutieren, executable machen.

    Ziel-Layout: [c, m_l2, n_l2, m_prim, n_prim, k]
    mit exec_types [PAR, PAR, PAR, PRIM, PRIM, PRIM].
    """
    cfg = build_basic_config()
    opt = Optimizer(cfg)

    m_id = next(i for i, t in enumerate(cfg.dim_types) if t == DimType.M)
    opt.split_dim(m_id, DIMS["M"] // M_PRIM, M_PRIM)

    n_id = next(i for i, t in enumerate(cfg.dim_types) if t == DimType.N)
    opt.split_dim(n_id, DIMS["N"] // N_PRIM, N_PRIM)

    # Nach den Splits ist die Reihenfolge [c, m_l2, m_prim, k, n_l2, n_prim].
    # Wir wollen [c, m_l2, n_l2, m_prim, n_prim, k] (Spec-Layout).
    opt.permute_dims([0, 1, 4, 2, 5, 3])
    opt.make_executable()
    return opt.config


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
                        Cd: ct.Constant[int],
                        M:  ct.Constant[int],
                        N:  ct.Constant[int],
                        K:  ct.Constant[int],
                        tm: ct.Constant[int],
                        tn: ct.Constant[int],
                        tk: ct.Constant[int],
                        group_m: ct.Constant[int],
                        group_n: ct.Constant[int]):
    """L2-Swizzle: gruppiert (m_tile, n_tile)-BIDs in 2D-Super-Tiles.

    Statt benachbarte BIDs ueber eine ganze N-Reihe zu enumerieren
    (Default), enumerieren wir sie ueber GROUP_M x GROUP_N Super-Tiles.
    Damit teilen sich benachbarte SMs sowohl A- als auch B-Tiles
    ueber den L2-Cache.

    Grid: (Cd, num_m_tiles * num_n_tiles, 1) - 2D linearer (m,n)-Bid.
    """
    pid_c = ct.bid(0)
    pid_id = ct.bid(1)

    num_m_tiles = ct.cdiv(M, tm)
    num_n_tiles = ct.cdiv(N, tn)

    # Klassischer L2-Swizzle:
    #   blocks_per_group = group_m * num_n_tiles   (Hoehe einer Super-Reihe in BIDs)
    #   group_id         = pid_id // blocks_per_group
    #   first_m_in_group = group_id * group_m
    #   group_size_m     = min(num_m_tiles - first_m_in_group, group_m)
    #   pid_m            = first_m_in_group + ((pid_id % blocks_per_group) % group_size_m)
    #   pid_n            = (pid_id % blocks_per_group) // group_size_m
    blocks_per_group = group_m * num_n_tiles
    group_id = pid_id // blocks_per_group
    in_group = pid_id %  blocks_per_group

    first_m_in_group = group_id * group_m
    remaining_m = num_m_tiles - first_m_in_group
    group_size_m = remaining_m if remaining_m < group_m else group_m

    pid_m = first_m_in_group + (in_group % group_size_m)
    pid_n = in_group // group_size_m

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
    dims = dims or DIMS
    Cd, M, N, K = dims["C"], dims["M"], dims["N"], dims["K"]
    tm, tn, tk = tile
    gm, gn = group
    Cout = torch.empty((Cd, M, N), device=A.device, dtype=A.dtype)
    grid = (Cd, ct.cdiv(M, tm) * ct.cdiv(N, tn), 1)
    ct.launch(
        torch.cuda.current_stream().cuda_stream,
        grid, kernel_l2_optimized,
        (A, B, Cout, Cd, M, N, K, tm, tn, tk, gm, gn),
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
