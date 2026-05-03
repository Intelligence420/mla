"""
Task 3: GEMM Dimension Size Sweep

Kontraktion: ackm, bcnk -> abnm

Feste Dimensionen: |a| = 16, |b| = 16, |c| = 32.
Variabel: |m|, |n|, |k|.

Klassifikation der Dimensionen:
  a       -> M (nur in A und C)
  b       -> N (nur in B und C)
  c       -> K (kontrahiert, in A und B, nicht in C) — Batch im *Block-Sinne*
  m       -> M (nur in A und C)
  n       -> N (nur in B und C)
  k       -> K (kontrahiert)

Wir mappen:
  GEMM-Dimensionen (ans ct.mma): m (M), k (K), n (N)
  Sequentialisiert (Schleife im Kernel): c
  Parallelisiert (Grid): a, b sowie Tiles entlang m und n

Tile-Shape (tm, tn, tk) = (64, 64, 32). Damit ist tm = |m| und tn = |n|
fuer die in der Aufgabe angegebenen Sweep-Bereiche, sodass jeweils
ein einziges Output-Tile pro (a, b)-Paar erzeugt wird (plus Zero-Padding
falls |m| oder |n| keine Vielfache von 64 sind).

Sweeps (Aufgabe b):
  1) |k| = 64, |m| = 64; |n| in 17..129
  2) |m| = 64, |n| = 64; |k| in 17..129

Verifikation gegen torch.einsum (FP16 Inputs, FP32 Akkumulator).
Benchmark mit triton.testing.do_bench.
"""

import os

import cuda.tile as ct
import matplotlib.pyplot as plt
import torch
import triton


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

A_DIM = 16
B_DIM = 16
C_DIM = 32

TILE_M = 64
TILE_N = 64
TILE_K = 32


# ===========================================================================
# Kernel: ackm, bcnk -> abnm
# ===========================================================================

@ct.kernel
def kernel_ackm_bcnk(A, B, C,
                     Cd: ct.Constant[int],
                     M:  ct.Constant[int],
                     N:  ct.Constant[int],
                     K:  ct.Constant[int],
                     tm: ct.Constant[int],
                     tn: ct.Constant[int],
                     tk: ct.Constant[int]):
    """C[a,b,n,m] = sum_{c,k} A[a,c,k,m] * B[b,c,n,k]

    Grid: (Ad, Bd, ct.cdiv(N, tn) * ct.cdiv(M, tm))
    Pro Block: ein Output-Tile (tn, tm) fuer ein (a, b)-Paar.
    K-Schleife: ueber alle c und ueber alle K-Tiles entlang k.
    """
    bid_a   = ct.bid(0)
    bid_b   = ct.bid(1)
    bid_nm  = ct.bid(2)

    num_tiles_m = ct.cdiv(M, tm)
    pid_n = bid_nm // num_tiles_m
    pid_m = bid_nm %  num_tiles_m

    num_tiles_k = ct.cdiv(K, tk)
    acc = ct.full((tn, tm), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    for cc in range(Cd):
        for kk in range(num_tiles_k):
            # A[a, c, k, m] -> 2D-Tile (tk, tm)
            a_tile = ct.load(A,
                             index=(bid_a, cc, kk, pid_m),
                             shape=(1, 1, tk, tm),
                             padding_mode=zero_pad)
            # B[b, c, n, k] -> 2D-Tile (tn, tk)
            b_tile = ct.load(B,
                             index=(bid_b, cc, pid_n, kk),
                             shape=(1, 1, tn, tk),
                             padding_mode=zero_pad)
            a2d = ct.reshape(a_tile, (tk, tm))
            b2d = ct.reshape(b_tile, (tn, tk))
            # Output (n, m): mma(B-Tile [tn, tk], A-Tile [tk, tm]) -> (tn, tm)
            acc = ct.mma(b2d, a2d, acc)

    out = ct.reshape(ct.astype(acc, C.dtype), (1, 1, tn, tm))
    ct.store(C, index=(bid_a, bid_b, pid_n, pid_m), tile=out)


def run(A, B, M, N, K, tile=(TILE_M, TILE_N, TILE_K)):
    Ad, Bd, Cd = A_DIM, B_DIM, C_DIM
    tm, tn, tk = tile

    C = torch.empty((Ad, Bd, N, M), device=A.device, dtype=A.dtype)
    grid = (Ad, Bd, ct.cdiv(N, tn) * ct.cdiv(M, tm))
    ct.launch(torch.cuda.current_stream().cuda_stream,
              grid, kernel_ackm_bcnk,
              (A, B, C, Cd, M, N, K, tm, tn, tk))
    return C


# ===========================================================================
# Verifikation / Inputs
# ===========================================================================

def make_inputs(M, N, K, dtype=torch.float16, device="cuda"):
    torch.manual_seed(0)
    A = torch.randn(A_DIM, C_DIM, K, M, dtype=dtype, device=device)
    B = torch.randn(B_DIM, C_DIM, N, K, dtype=dtype, device=device)
    return A, B


def reference(A, B):
    return torch.einsum("ackm,bcnk->abnm",
                        A.to(torch.float32),
                        B.to(torch.float32)).to(A.dtype)


def verify():
    cases = [
        (64, 64, 64),
        (64, 65, 64),
        (64, 64, 65),
        (32, 96, 33),
    ]
    for M, N, K in cases:
        A, B = make_inputs(M, N, K)
        ref = reference(A, B)
        out = run(A, B, M, N, K)
        ok = torch.allclose(out, ref, atol=2e-1, rtol=2e-2)
        err = (out.float() - ref.float()).abs().max().item()
        print(f"  (M,N,K)=({M:3d},{N:3d},{K:3d})  allclose={ok}   "
              f"max_abs_err={err:.4f}")
        assert ok, f"mismatch at (M,N,K)=({M},{N},{K})"


# ===========================================================================
# FLOPs / Helfer
# ===========================================================================

def flops_count(M, N, K):
    """2 * |a| * |b| * |c| * |m| * |n| * |k|"""
    return 2 * A_DIM * B_DIM * C_DIM * M * N * K


def tflops(M, N, K, time_ms):
    return flops_count(M, N, K) / (time_ms * 1e-3) / 1e12


def bench(fn, n_warmup=10, n_runs=50):
    return triton.testing.do_bench(fn, warmup=n_warmup, rep=n_runs)


# ===========================================================================
# Sweeps
# ===========================================================================

def sweep_n(K=64, M=64):
    """|k|=64, |m|=64; |n| von 17..129."""
    print(f"\n  Sweep n in 17..129  (K={K}, M={M})")
    ns = list(range(17, 130))
    res = []
    for N in ns:
        A, B = make_inputs(M, N, K)
        t = bench(lambda: run(A, B, M, N, K))
        f = tflops(M, N, K, t)
        res.append((N, t, f))
    return ns, res


def sweep_k(M=64, N=64):
    """|m|=64, |n|=64; |k| von 17..129."""
    print(f"\n  Sweep k in 17..129  (M={M}, N={N})")
    ks = list(range(17, 130))
    res = []
    for K in ks:
        A, B = make_inputs(M, N, K)
        t = bench(lambda: run(A, B, M, N, K))
        f = tflops(M, N, K, t)
        res.append((K, t, f))
    return ks, res


# ===========================================================================
# Plot
# ===========================================================================

def plot_sweep(xs, res, dim_name, fixed_label, path):
    """Doppel-Plot: TFLOPS und Laufzeit ueber dem variablen Sweep-Index."""
    sizes  = [r[0] for r in res]
    times  = [r[1] for r in res]
    tflops = [r[2] for r in res]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # TFLOPS
    ax1.plot(sizes, tflops, marker=".", color="#4C72B0")
    # Markiere Zweierpotenzen mit groesserem Punkt
    pow2 = [s for s in sizes if s & (s - 1) == 0]
    ax1.scatter([s for s in sizes if s in pow2],
                [t for s, t in zip(sizes, tflops) if s in pow2],
                color="#DD8452", zorder=3, label="Zweierpotenz")
    ax1.set_xlabel(f"|{dim_name}|")
    ax1.set_ylabel("TFLOPS")
    ax1.set_title(f"Durchsatz vs. |{dim_name}|  ({fixed_label})")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=8)

    # Laufzeit
    ax2.plot(sizes, times, marker=".", color="#55A868")
    ax2.scatter([s for s in sizes if s in pow2],
                [t for s, t in zip(sizes, times) if s in pow2],
                color="#DD8452", zorder=3, label="Zweierpotenz")
    ax2.set_xlabel(f"|{dim_name}|")
    ax2.set_ylabel("Laufzeit (ms)")
    ax2.set_title(f"Laufzeit vs. |{dim_name}|  ({fixed_label})")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot: {path}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("Task 3a: Verifikation")
    verify()

    # Sweep n  (|k|=64, |m|=64)
    print("\nTask 3b.1: Sweep n in 17..129 mit |k|=64, |m|=64")
    ns, res_n = sweep_n(K=64, M=64)
    print("    ein paar Punkte:")
    for n, t, f in res_n[::15]:
        print(f"      n={n:3d}  {t:7.4f} ms  {f:6.3f} TFLOPS")
    plot_sweep(ns, res_n, "n", "|k|=64, |m|=64",
               os.path.join(SCRIPT_DIR, "task03b_sweep_n.png"))

    # Sweep k  (|m|=64, |n|=64)
    print("\nTask 3b.2: Sweep k in 17..129 mit |m|=64, |n|=64")
    ks, res_k = sweep_k(M=64, N=64)
    print("    ein paar Punkte:")
    for k, t, f in res_k[::15]:
        print(f"      k={k:3d}  {t:7.4f} ms  {f:6.3f} TFLOPS")
    plot_sweep(ks, res_k, "k", "|m|=64, |n|=64",
               os.path.join(SCRIPT_DIR, "task03b_sweep_k.png"))

"""Ergebnisse
(.venv) mla07@flambe:~/mla$ python3 assignments/04_assignment/src/task_03.py
Task 3a: Verifikation
  (M,N,K)=( 64, 64, 64)  allclose=True   max_abs_err=0.1250
  (M,N,K)=( 64, 65, 64)  allclose=True   max_abs_err=0.1250
  (M,N,K)=( 64, 64, 65)  allclose=True   max_abs_err=0.1250
  (M,N,K)=( 32, 96, 33)  allclose=True   max_abs_err=0.0625

Task 3b.1: Sweep n in 17..129 mit |k|=64, |m|=64
    n= 17   0.1207 ms   9.452 TFLOPS
    n= 32   0.1200 ms  17.889 TFLOPS
    n= 47   0.1313 ms  24.027 TFLOPS
    n= 62   0.1235 ms  33.692 TFLOPS
    n= 77   0.2026 ms  25.507 TFLOPS
    n= 92   0.2127 ms  29.028 TFLOPS
    n=107   0.2253 ms  31.874 TFLOPS
    n=122   0.2143 ms  38.208 TFLOPS

Task 3b.2: Sweep k in 17..129 mit |m|=64, |n|=64
    k= 17   0.3131 ms   3.644 TFLOPS
    k= 32   0.0767 ms  27.994 TFLOPS
    k= 47   0.4258 ms   7.407 TFLOPS
    k= 62   0.5910 ms   7.040 TFLOPS
    k= 77   0.7217 ms   7.160 TFLOPS
    k= 92   0.9281 ms   6.652 TFLOPS
    k=107   1.1481 ms   6.254 TFLOPS
    k=122   1.2660 ms   6.467 TFLOPS
"""
