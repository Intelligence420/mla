"""
Task 4: Benchmarking Bandwidth

a) cuTile copy kernel for 2D matrix (M, N) with tile (tile_M, tile_N).
   Verify correctness.
b) M=2048 fixed, N ∈ {16 … 128}, tile_M=64, tile_N=N.
   Measure runtime, compute effective bandwidth, plot results.
"""

import cuda.tile as ct
import cupy as cp
import torch


# ===========================================================================
# Copy Kernel
# ===========================================================================

@ct.kernel
def copy_kernel(src, dst,
                tile_m: ct.Constant[int],
                tile_n: ct.Constant[int]):
    """Copy a 2D tile of size (tile_m, tile_n) from src to dst."""
    # Block-IDs für 2D-Grid
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)

    # Tile laden und speichern
    tile = ct.load(src, index=(pid_m, pid_n), shape=(tile_m, tile_n))
    ct.store(dst, index=(pid_m, pid_n), tile=tile)


# ===========================================================================
# Host / Launch
# ===========================================================================

def copy_matrix(src: torch.Tensor, tile_m: int, tile_n: int) -> torch.Tensor:
    """Launch copy_kernel and return the copied matrix."""
    M, N = src.shape
    dst = torch.empty_like(src)

    # Grid berechnen + Kernel starten
    grid = (ct.cdiv(M, tile_m), ct.cdiv(N, tile_n), 1)
    ct.launch(torch.cuda.current_stream().cuda_stream,
              grid, copy_kernel, (src, dst, tile_m, tile_n))

    return dst


# ===========================================================================
# Verifikation
# ===========================================================================

def verify():
    """Verify that the copy kernel produces an exact copy."""
    M, N = 2048, 64
    src = torch.randn(M, N, dtype=torch.float16, device="cuda")
    dst = copy_matrix(src, tile_m=64, tile_n=N)

    assert torch.equal(src, dst), "Copy mismatch!"
    print("  Copy kernel verified.")


# ===========================================================================
# Bandwidth Benchmark + Plot
# ===========================================================================

def bandwidth_benchmark():
    """
    M=2048 fixed, N from 16 to 128 (step 16).
    tile_M=64, tile_N=N (full width).

    bandwidth (GB/s) = 2 * M * N * sizeof(element) / (time_s * 1e9)

    Factor 2: one read + one write.
    """
    import triton
    import matplotlib.pyplot as plt

    M = 2048
    tile_m = 64
    element_size = 2                          # FP16 = 2 Bytes

    ns = list(range(16, 129, 16))             # [16, 32, 48, …, 128]
    bandwidths = []

    for N in ns:
        tile_n = N                            # tile_N = N (full width)
        src = torch.randn(M, N, dtype=torch.float16, device="cuda")

        # Runtime messen
        time_ms = triton.testing.do_bench(
            lambda: copy_matrix(src, tile_m, tile_n))
        time_s = time_ms / 1000.0

        # Bandwidth berechnen
        bw = 2 * M * N * element_size / (time_s * 1e9)
        bandwidths.append(bw)
        print(f"  N={N:4d}  time={time_ms:.4f} ms  BW={bw:.2f} GB/s")

    # Plot erstellen
    plt.figure(figsize=(8, 5))
    plt.plot(ns, bandwidths, marker="o")
    plt.xlabel("N (last dimension)")
    plt.ylabel("Effective Bandwidth (GB/s)")
    plt.title("Copy Kernel Bandwidth (M=2048, tile_M=64, tile_N=N)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("bandwidth_plot.png", dpi=150)
    plt.show()
    print("  Plot saved to bandwidth_plot.png")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("Task 4a: Copy Kernel — Verification")
    verify()
    print("Task 4b: Bandwidth Benchmark")
    bandwidth_benchmark()
