"""
Task 4: L2 Cache Optimierung via Block Swizzling

a) Swizzled Matmul-Kernel. Statt row-major BID-Mapping wird ein
   Super-Grouping-Schema verwendet, damit benachbarte Bloecke
   raeumlich naheliegende A/B-Tiles laden -> bessere L2-Wiederverwendung.

   Mapping (Super-Grouping mit GROUP_SIZE_M):
     Bloecke werden in 2D-Gruppen der Groesse GROUP_SIZE_M x num_bid_n
     organisiert. Innerhalb einer Gruppe laeuft der BID column-major
     (zuerst alle GROUP_SIZE_M Zeilen, dann naechste Spalte).
     -> ein A-Tile-Streifen wird von GROUP_SIZE_M Bloecken
        gemeinsam wiederverwendet, ein B-Tile-Streifen ebenfalls.
   K = 4096 ist gross genug, dass jeder Block mehrere
   k-Iterationen taetigt; die wiederholten A/B-Tile-Loads
   profitieren stark vom L2.

b) Tile-Shape-Sweep fuer 2048**3 und 512**3 mit dem Swizzled-Kernel
   und Vergleich gegen den row-major Kernel aus Task 2 fuer
   8192 x 8192 x 4096.
"""

import cuda.tile as ct
import torch
import triton
import matplotlib.pyplot as plt
import numpy as np

from task_03 import cutile_matmul as cutile_matmul_rowmajor
from task_03 import tflops


# ===========================================================================
# Swizzled Matmul-Kernel
# ===========================================================================

GROUP_SIZE_M = 8


@ct.kernel
def matmul_swizzled_kernel(A, B, C,
                           tm: ct.Constant[int],
                           tn: ct.Constant[int],
                           tk: ct.Constant[int],
                           group_size_m: ct.Constant[int]):
    """C = A @ B mit super-grouping Block-Swizzle fuer bessere L2-Reuse."""
    bid = ct.bid(0)

    M = A.shape[0]
    N = B.shape[1]

    num_bid_m = ct.cdiv(M, tm)
    num_bid_n = ct.cdiv(N, tn)

    # super-grouping: Bloecke einer Gruppe teilen sich die A-Tile-Streifen
    num_bid_in_group = group_size_m * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * group_size_m
    # Letzte Gruppe kann kleiner sein
    cur_group_size_m = min(num_bid_m - first_bid_m, group_size_m)

    bidx = first_bid_m + (bid % cur_group_size_m)
    bidy = (bid % num_bid_in_group) // cur_group_size_m

    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))

    accumulator = ct.full((tm, tn), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO
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

def cutile_matmul_swizzled(A: torch.Tensor, B: torch.Tensor,
                           tm: int, tn: int, tk: int,
                           group_size_m: int = GROUP_SIZE_M) -> torch.Tensor:
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    grid = (ct.cdiv(M, tm) * ct.cdiv(N, tn), 1, 1)
    ct.launch(torch.cuda.current_stream().cuda_stream,
              grid, matmul_swizzled_kernel,
              (A, B, C, tm, tn, tk, group_size_m))
    return C


# ===========================================================================
# Verifikation
# ===========================================================================

def verify():
    torch.manual_seed(0)
    M, K, N = 257, 513, 129
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")

    C = cutile_matmul_swizzled(A, B, tm=64, tn=64, tk=64)
    ref = torch.matmul(A, B)
    assert torch.allclose(C, ref, atol=1e-1, rtol=1e-1), "Swizzled falsch!"
    print("  Swizzled-Kernel verifiziert (auch nicht-Zweierpotenz).")


# ===========================================================================
# Task 4b: Tile-Shape-Sweep
# ===========================================================================

def benchmark_tile_sweep_swizzled(size: int):
    tile_values = [32, 64, 128]
    A = torch.randn(size, size, dtype=torch.float16, device="cuda")
    B = torch.randn(size, size, dtype=torch.float16, device="cuda")

    results = np.zeros((3, 3, 3))
    print(f"\n  Swizzled Tile-Sweep fuer {size}x{size}x{size}:")
    for i, tm in enumerate(tile_values):
        for j, tn in enumerate(tile_values):
            for kk, tk in enumerate(tile_values):
                time_ms = triton.testing.do_bench(
                    lambda: cutile_matmul_swizzled(A, B, tm, tn, tk))
                t = tflops(size, size, size, time_ms)
                results[i, j, kk] = t
                print(f"    tm={tm:3d} tn={tn:3d} tk={tk:3d}  "
                      f"{time_ms:.4f} ms  {t:7.2f} TFLOPS")

    best_idx = np.unravel_index(np.argmax(results), results.shape)
    best = (tile_values[best_idx[0]], tile_values[best_idx[1]],
            tile_values[best_idx[2]])
    print(f"  Beste Kombination (swizzled, {size}**3): "
          f"tm={best[0]} tn={best[1]} tk={best[2]}  "
          f"{results[best_idx]:.2f} TFLOPS")

    heat = results[:, :, 1]
    plt.figure(figsize=(6, 5))
    plt.imshow(heat, cmap="viridis", origin="lower")
    plt.colorbar(label="TFLOPS")
    plt.xticks(range(3), tile_values)
    plt.yticks(range(3), tile_values)
    plt.xlabel("n_tile")
    plt.ylabel("m_tile")
    plt.title(f"Swizzled-Heatmap {size}^3, k_tile=64")
    for i in range(3):
        for j in range(3):
            plt.text(j, i, f"{heat[i, j]:.1f}",
                     ha="center", va="center", color="white")
    plt.tight_layout()
    plt.savefig(f"task04b_heatmap_{size}.png", dpi=150)
    plt.close()
    print(f"  Heatmap: task04b_heatmap_{size}.png")
    return best, results[best_idx]


# ===========================================================================
# Vergleich gegen Task 2 Kernel: 8192 x 8192 x 4096
# ===========================================================================

def compare_large():
    """Vergleich row-major vs swizzled fuer 8192 x 8192 x 4096."""
    M, N, K = 8192, 8192, 4096
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")

    # Beste Tile-Shape aus 2048-Sweep nehmen (typisch (128,128,64))
    tm, tn, tk = 128, 128, 64

    t_row = triton.testing.do_bench(
        lambda: cutile_matmul_rowmajor(A, B, tm, tn, tk))
    t_sw = triton.testing.do_bench(
        lambda: cutile_matmul_swizzled(A, B, tm, tn, tk))

    flops_row = tflops(M, N, K, t_row)
    flops_sw = tflops(M, N, K, t_sw)

    print(f"\n  8192 x 8192 x 4096, Tile ({tm},{tn},{tk}), FP16:")
    print(f"    row-major  : {t_row:.3f} ms  {flops_row:7.2f} TFLOPS")
    print(f"    swizzled   : {t_sw:.3f} ms  {flops_sw:7.2f} TFLOPS")
    print(f"    Speedup    : {t_row / t_sw:.2f}x")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("Task 4a: Verifikation des Swizzled-Kernels")
    verify()

    print("\nTask 4b: Swizzled Tile-Shape-Sweep")
    best_2048, _ = benchmark_tile_sweep_swizzled(2048)
    best_512, _ = benchmark_tile_sweep_swizzled(512)

    print("\nVergleich row-major vs swizzled (8192 x 8192 x 4096):")
    compare_large()

    print("\n=== Zusammenfassung ===")
    print(f"  Beste Tile-Shape swizzled 2048**3: {best_2048}")
    print(f"  Beste Tile-Shape swizzled  512**3: {best_512}")
