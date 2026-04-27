"""
Task 3: Benchmarking the Matrix Multiplication Kernel

Wir benutzen den Matmul-Kernel aus Task 2 (row-major BID-Mapping) und messen
die erreichte Performance in TFLOPS.

a) Quadratische Matmuls fuer M=N=K in {256,512,1024,2048,4096,8192} mit
   Tile-Shape (64, 64, 64) -> Plot.
b) Feste Groessen 2048**3 und 512**3, Sweep ueber alle 27 Kombinationen
   m_tile, n_tile, k_tile in {32, 64, 128} -> Heatmap (k_tile=64 fixiert)
   und beste Kombination berichten.
"""

import cuda.tile as ct
import torch
import triton
import matplotlib.pyplot as plt
import numpy as np


# ===========================================================================
# Matmul-Kernel (Task 2): row-major BID-Mapping
# ===========================================================================

@ct.kernel
def matmul_kernel(A, B, C,
                  tm: ct.Constant[int],
                  tn: ct.Constant[int],
                  tk: ct.Constant[int]):
    """C = A @ B mit row-major BID-Mapping.

    Jedes Programm berechnet ein Output-Tile (tm, tn). BID 0 erzeugt das
    Tile oben links, BID 1 rechts daneben usw. (row-major).
    """
    bid = ct.bid(0)

    M = A.shape[0]
    N = B.shape[1]

    # Anzahl Tiles entlang N (Breite). row-major Mapping:
    num_bid_n = ct.cdiv(N, tn)
    bidx = bid // num_bid_n          # Zeilen-Index des Tiles
    bidy = bid % num_bid_n           # Spalten-Index des Tiles

    # Zahl der K-Tiles: A wird in (tm, tk)-Tiles zerlegt, axis=1 -> K
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))

    # Akkumulator immer in fp32
    accumulator = ct.full((tm, tn), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    # fp32-Inputs als tf32 fuer Tensor Cores
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    for k in range(num_tiles_k):
        a = ct.load(A, index=(bidx, k), shape=(tm, tk),
                    padding_mode=zero_pad).astype(dtype)
        b = ct.load(B, index=(k, bidy), shape=(tk, tn),
                    padding_mode=zero_pad).astype(dtype)
        accumulator = ct.mma(a, b, accumulator)

    accumulator = ct.astype(accumulator, C.dtype)
    ct.store(C, index=(bidx, bidy), tile=accumulator)


# ===========================================================================
# Host-Wrapper
# ===========================================================================

def cutile_matmul(A: torch.Tensor, B: torch.Tensor,
                  tm: int, tn: int, tk: int) -> torch.Tensor:
    """Startet matmul_kernel mit den vorgegebenen Tile-Groessen."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "K-Dim stimmt nicht ueberein"

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # 1D-Grid mit ceil(M/tm) * ceil(N/tn) Bloecken
    grid = (ct.cdiv(M, tm) * ct.cdiv(N, tn), 1, 1)
    ct.launch(torch.cuda.current_stream().cuda_stream,
              grid, matmul_kernel, (A, B, C, tm, tn, tk))
    return C


# ===========================================================================
# Verifikation
# ===========================================================================

def verify():
    """Korrektheit gegen torch.matmul pruefen."""
    torch.manual_seed(0)
    M, K, N = 257, 513, 129  # absichtlich keine Zweierpotenzen
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")

    C = cutile_matmul(A, B, tm=64, tn=64, tk=64)
    ref = torch.matmul(A, B)
    assert torch.allclose(C, ref, atol=1e-1, rtol=1e-1), "Matmul falsch!"
    print("  Matmul-Kernel verifiziert.")


# ===========================================================================
# TFLOPS-Helfer
# ===========================================================================

def tflops(M: int, N: int, K: int, time_ms: float) -> float:
    """2*M*N*K FLOPs / (Sekunden * 1e12)."""
    return (2.0 * M * N * K) / (time_ms / 1000.0 * 1e12)


# ===========================================================================
# Task 3a: TFLOPS ueber quadratische Groessen
# ===========================================================================

def benchmark_square():
    """Quadratische Matmuls, Tile (64,64,64), Plot der TFLOPS."""
    sizes = [256, 512, 1024, 2048, 4096, 8192]
    tm = tn = tk = 64
    perf = []

    for n in sizes:
        A = torch.randn(n, n, dtype=torch.float16, device="cuda")
        B = torch.randn(n, n, dtype=torch.float16, device="cuda")
        time_ms = triton.testing.do_bench(
            lambda: cutile_matmul(A, B, tm, tn, tk))
        t = tflops(n, n, n, time_ms)
        perf.append(t)
        print(f"  N={n:5d}  time={time_ms:.4f} ms  {t:7.2f} TFLOPS")

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, perf, marker="o")
    plt.xscale("log", base=2)
    plt.xlabel("M = N = K")
    plt.ylabel("TFLOPS")
    plt.title("Matmul-TFLOPS, Tile (64,64,64), FP16")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig("task03a_tflops.png", dpi=150)
    plt.close()
    print("  Plot: task03a_tflops.png")
    return perf


# ===========================================================================
# Task 3b: Tile-Shape-Sweep + Heatmap
# ===========================================================================

def benchmark_tile_sweep(size: int):
    """27 Tile-Shape-Kombinationen fuer eine feste Matrixgroesse."""
    tile_values = [32, 64, 128]
    A = torch.randn(size, size, dtype=torch.float16, device="cuda")
    B = torch.randn(size, size, dtype=torch.float16, device="cuda")

    # results[tm_idx, tn_idx, tk_idx] -> TFLOPS
    results = np.zeros((3, 3, 3))

    print(f"\n  Tile-Sweep fuer {size}x{size}x{size}:")
    for i, tm in enumerate(tile_values):
        for j, tn in enumerate(tile_values):
            for kk, tk in enumerate(tile_values):
                time_ms = triton.testing.do_bench(
                    lambda: cutile_matmul(A, B, tm, tn, tk))
                t = tflops(size, size, size, time_ms)
                results[i, j, kk] = t
                print(f"    tm={tm:3d} tn={tn:3d} tk={tk:3d}  "
                      f"{time_ms:.4f} ms  {t:7.2f} TFLOPS")

    # Beste Kombination
    best_idx = np.unravel_index(np.argmax(results), results.shape)
    best = (tile_values[best_idx[0]], tile_values[best_idx[1]],
            tile_values[best_idx[2]])
    print(f"  Beste Kombination ({size}**3): "
          f"tm={best[0]} tn={best[1]} tk={best[2]}  "
          f"{results[best_idx]:.2f} TFLOPS")

    # Heatmap: tm vs tn, tk=64 (Index 1)
    heat = results[:, :, 1]
    plt.figure(figsize=(6, 5))
    plt.imshow(heat, cmap="viridis", origin="lower")
    plt.colorbar(label="TFLOPS")
    plt.xticks(range(3), tile_values)
    plt.yticks(range(3), tile_values)
    plt.xlabel("n_tile")
    plt.ylabel("m_tile")
    plt.title(f"Heatmap {size}^3, k_tile=64")
    for i in range(3):
        for j in range(3):
            plt.text(j, i, f"{heat[i, j]:.1f}",
                     ha="center", va="center", color="white")
    plt.tight_layout()
    plt.savefig(f"task03b_heatmap_{size}.png", dpi=150)
    plt.close()
    print(f"  Heatmap: task03b_heatmap_{size}.png")

    return best, results[best_idx]


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("Task 3: Verifikation")
    verify()

    print("\nTask 3a: Quadratischer Sweep")
    benchmark_square()

    print("\nTask 3b: Tile-Shape-Sweep")
    best_2048, _ = benchmark_tile_sweep(2048)
    best_512, _ = benchmark_tile_sweep(512)

    print("\n=== Zusammenfassung ===")
    print(f"  Beste Tile-Shape 2048**3: {best_2048}")
    print(f"  Beste Tile-Shape  512**3: {best_512}")
