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
    # padding_mode=ZERO nötig, da tile_n auf Zweierpotenz aufgerundet wird
    tile = ct.load(src, index=(pid_m, pid_n), shape=(tile_m, tile_n),
                   padding_mode=ct.PaddingMode.ZERO)
    ct.store(dst, index=(pid_m, pid_n), tile=tile)


# ===========================================================================
# Helper
# ===========================================================================

def next_power_of_2(n: int) -> int:
    """Nächste Zweierpotenz >= n."""
    p = 1
    while p < n:
        p *= 2
    return p


# ===========================================================================
# Host / Launch
# ===========================================================================

def copy_matrix(src: torch.Tensor, tile_m: int, tile_n: int) -> torch.Tensor:
    """Launch copy_kernel and return the copied matrix."""
    M, N = src.shape
    dst = torch.empty_like(src)

    # Tile-Dimensionen müssen Zweierpotenzen sein
    tile_m_pow2 = next_power_of_2(tile_m)
    tile_n_pow2 = next_power_of_2(tile_n)

    # Grid berechnen + Kernel starten
    grid = (ct.cdiv(M, tile_m_pow2), ct.cdiv(N, tile_n_pow2), 1)
    ct.launch(torch.cuda.current_stream().cuda_stream,
              grid, copy_kernel, (src, dst, tile_m_pow2, tile_n_pow2))

    return dst


# ===========================================================================
# Verifikation
# ===========================================================================

def verify():
    """Verify that the copy kernel produces an exact copy.

    Prueft neben der Zweierpotenz N=64 (kein Padding) auch N=65 und N=100,
    bei denen tile_n auf die naechste Zweierpotenz (128) aufgerundet wird und
    damit der Padding-/OOB-Maskierungspfad von ct.store ausgeuebt wird.
    """
    for N in (64, 65, 100):
        src = torch.randn(2048, N, dtype=torch.float16, device="cuda")
        dst = copy_matrix(src, tile_m=64, tile_n=N)
        assert torch.equal(src, dst), f"Copy mismatch for N={N}!"
    print("  Copy kernel verified.")


# ===========================================================================
# Bandwidth Benchmark + Plot
# ===========================================================================

def bandwidth_benchmark():
    """
    M=2048 fixed, N from 16 to 128 (step 1, volle Range).
    tile_M=64, tile_N=N (full width).

    bandwidth (GB/s) = 2 * M * N * sizeof(element) / (time_s * 1e9)

    Factor 2: one read + one write.
    """
    import triton
    import matplotlib.pyplot as plt

    M = 2048
    tile_m = 64
    element_size = 2                          # FP16 = 2 Bytes

    ns = list(range(16, 129))                 # [16, 17, 18, …, 128] – ganze Range
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

"""Ergebnisse (volle Range N=16..128, Schritt 1 -> 113 Messpunkte;
repräsentativer Auszug: Peaks an Zweierpotenzen + Einbrüche bei 2^+1)
Task 4a: Copy Kernel — Verification
  Copy kernel verified.
Task 4b: Bandwidth Benchmark
  N=  16  time=0.0047 ms  BW=27.91 GB/s   (2er-Potenz, kein Padding)
  N=  17  time=0.0065 ms  BW=21.52 GB/s   (tile_N 16->32, ~1/2 Padding)
  N=  32  time=0.0069 ms  BW=38.18 GB/s   (2er-Potenz)
  N=  33  time=0.0085 ms  BW=31.65 GB/s   (tile_N 32->64)
  N=  64  time=0.0082 ms  BW=64.06 GB/s   (2er-Potenz)
  N=  65  time=0.0116 ms  BW=45.81 GB/s   (tile_N 64->128)
  N= 128  time=0.0119 ms  BW=88.15 GB/s   (2er-Potenz, kein Padding)
  Plot saved to bandwidth_plot.png
  """