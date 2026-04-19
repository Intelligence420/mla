"""
Task 3: 4D Tensor Elementwise Addition

a) Two cuTile kernels for C = A + B, shape (M, N, K, L):
     Variante 1 — Tile über (K, L), Grid über (M, N)
     Variante 2 — Tile über (M, N), Grid über (K, L)
   Verify both against PyTorch A + B via torch.allclose.

b) Benchmark with triton.testing.do_bench.
   Dimensions: M=16, N=128, K=16, L=128.
   Report runtime differences and explain why.
"""

import cuda.tile as ct
import cupy as cp
import torch


# ===========================================================================
# Variante 1: Tile über (K, L), Grid über (M, N)
# ===========================================================================

@ct.kernel
def add_4d_tile_KL(A, B, C,
                   tile_k: ct.Constant[int],
                   tile_l: ct.Constant[int]):
    """Each block handles one (K, L) slice for a fixed (m, n) position."""
    # Block-IDs → Indizes in M- und N-Dimension
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)

    # Tiles laden — ganzes (K, L)-Slice pro Block
    a_tile = ct.load(A, index=(pid_m, pid_n, 0, 0),
                     shape=(1, 1, tile_k, tile_l),
                     padding_mode=ct.PaddingMode.ZERO)
    b_tile = ct.load(B, index=(pid_m, pid_n, 0, 0),
                     shape=(1, 1, tile_k, tile_l),
                     padding_mode=ct.PaddingMode.ZERO)

    # Elementweise addieren + speichern
    ct.store(C, index=(pid_m, pid_n, 0, 0), tile=a_tile + b_tile)


# ===========================================================================
# Variante 2: Tile über (M, N), Grid über (K, L)
# ===========================================================================

@ct.kernel
def add_4d_tile_MN(A, B, C,
                   tile_m: ct.Constant[int],
                   tile_n: ct.Constant[int]):
    """Each block handles one (M, N) slice for a fixed (k, l) position."""
    # Block-IDs → Indizes in K- und L-Dimension
    pid_k = ct.bid(0)
    pid_l = ct.bid(1)

    # Tiles laden — ganzes (M, N)-Slice pro Block
    a_tile = ct.load(A, index=(0, 0, pid_k, pid_l),
                     shape=(tile_m, tile_n, 1, 1),
                     padding_mode=ct.PaddingMode.ZERO)
    b_tile = ct.load(B, index=(0, 0, pid_k, pid_l),
                     shape=(tile_m, tile_n, 1, 1),
                     padding_mode=ct.PaddingMode.ZERO)

    # Elementweise addieren + speichern
    ct.store(C, index=(0, 0, pid_k, pid_l), tile=a_tile + b_tile)


# ===========================================================================
# Host / Launch
# ===========================================================================

def add_4d_variant1(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Launch add_4d_tile_KL — Grid über (M, N)."""
    M, N, K, L = A.shape
    C = torch.empty_like(A)

    # Grid + Tile-Größen setzen + Kernel starten
    tile_k, tile_l = K, L  # Zweierpotenzen bei M=16, N=128, K=16, L=128
    grid = (M, N, 1)
    ct.launch(torch.cuda.current_stream().cuda_stream,
              grid, add_4d_tile_KL, (A, B, C, tile_k, tile_l))

    return C


def add_4d_variant2(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Launch add_4d_tile_MN — Grid über (K, L)."""
    M, N, K, L = A.shape
    C = torch.empty_like(A)

    # Grid + Tile-Größen setzen + Kernel starten
    tile_m, tile_n = M, N  # Zweierpotenzen bei M=16, N=128, K=16, L=128
    grid = (K, L, 1)
    ct.launch(torch.cuda.current_stream().cuda_stream,
              grid, add_4d_tile_MN, (A, B, C, tile_m, tile_n))

    return C


# ===========================================================================
# Verification
# ===========================================================================

def verify():
    """Check both variants against PyTorch native A + B."""
    M, N, K, L = 16, 128, 16, 128
    A = torch.randn(M, N, K, L, dtype=torch.float16, device="cuda") # zufällige zahlen
    B = torch.randn(M, N, K, L, dtype=torch.float16, device="cuda")
    expected = A + B

    C1 = add_4d_variant1(A, B)
    C2 = add_4d_variant2(A, B)

    assert torch.allclose(C1, expected), "Variante 1 mismatch!"
    assert torch.allclose(C2, expected), "Variante 2 mismatch!"
    print("  Variante 1 (tile KL, grid MN) passed")
    print("  Variante 2 (tile MN, grid KL) passed")


# ===========================================================================
# Benchmark
# ===========================================================================

def benchmark():
    """Benchmark both variants via triton.testing.do_bench."""
    import triton

    M, N, K, L = 16, 128, 16, 128
    A = torch.randn(M, N, K, L, dtype=torch.float16, device="cuda")
    B = torch.randn(M, N, K, L, dtype=torch.float16, device="cuda")

    # Laufzeit messen: https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html 
    t1 = triton.testing.do_bench(lambda: add_4d_variant1(A, B))
    t2 = triton.testing.do_bench(lambda: add_4d_variant2(A, B))
    print(f"  Variante 1 (tile KL): {t1:.4f} ms")
    print(f"  Variante 2 (tile MN): {t2:.4f} ms")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("Task 3a: 4D Elementwise Addition — Verifikation")
    verify()
    print("Task 3b: Benchmark")
    benchmark()
