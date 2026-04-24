"""
Task 2: Simple Matrix Multiplication Kernel

cuTile kernel computing C = A @ B for
  A of shape (M, K), B of shape (K, N), C of shape (M, N).

Requirements:
  * Each program produces one (m_tile, n_tile) output tile
  * Tile sizes are parameters of the launch function
  * Block IDs are mapped in row-major order:
      BID 0 → top-left tile, BID 1 → next tile to the right,
      wrapping to the next row when the current row is exhausted
  * Support shapes that are not powers of 2
  * Use ct.mma for the inner accumulation

Verify correctness via torch.allclose against torch.matmul.
"""

import cuda.tile as ct
import cupy as cp
import torch


# ===========================================================================
# Kernel
# ===========================================================================

@ct.kernel
def matmul_kernel(A, B, C,
                  M: ct.Constant[int],
                  N: ct.Constant[int],
                  K: ct.Constant[int],
                  tile_m: ct.Constant[int],
                  tile_n: ct.Constant[int],
                  tile_k: ct.Constant[int]):
    """Compute one (tile_m, tile_n) output tile of C = A @ B."""
    # Row-major BID mapping: flache BID → 2D-Tile-Position
    #   bid 0 → (0, 0), bid 1 → (0, 1), ..., wrap auf (1, 0) am Zeilenende
    bid = ct.bid(0)
    num_tiles_n = ct.cdiv(N, tile_n)
    pid_m = bid // num_tiles_n
    pid_n = bid % num_tiles_n

    # FP32-Akkumulator unabhängig vom Input-dtype (Standardmuster aus cuTile)
    acc = ct.full((tile_m, tile_n), 0, dtype=ct.float32)

    # K-Schleife: läuft über ceil(K / tile_k) K-Tiles, Padding-Zeros
    # am Rand sind für MAC neutral (0 * x + acc == acc).
    num_tiles_k = ct.cdiv(K, tile_k)
    for k in range(num_tiles_k):
        a_tile = ct.load(A, index=(pid_m, k), shape=(tile_m, tile_k),
                         padding_mode=ct.PaddingMode.ZERO)
        b_tile = ct.load(B, index=(k, pid_n), shape=(tile_k, tile_n),
                         padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a_tile, b_tile, acc)

    # ct.store schneidet out-of-bounds Elemente am Rand automatisch ab,
    # daher ist kein explizites Masking für Nicht-Zweierpotenzen nötig.
    ct.store(C, index=(pid_m, pid_n), tile=acc)


# ===========================================================================
# Host / Launch
# ===========================================================================

def matmul(A: torch.Tensor, B: torch.Tensor,
           tile_m: int, tile_n: int, tile_k: int) -> torch.Tensor:
    """Launch matmul_kernel and return C = A @ B."""
    M, K = A.shape
    _, N = B.shape
    C = torch.empty(M, N, dtype=torch.float32, device=A.device)

    # Row-major flat layout: grid = num_tiles_m * num_tiles_n Blöcke
    num_tiles_m = (M + tile_m - 1) // tile_m
    num_tiles_n = (N + tile_n - 1) // tile_n
    grid = (num_tiles_m * num_tiles_n, 1, 1)

    ct.launch(torch.cuda.current_stream().cuda_stream,
              grid, matmul_kernel,
              (A, B, C, M, N, K, tile_m, tile_n, tile_k))
    return C


# ===========================================================================
# Verifikation
# ===========================================================================

def check(M: int, N: int, K: int,
          tile_m: int, tile_n: int, tile_k: int,
          dtype: torch.dtype = torch.float16):
    """Run the kernel once and compare against torch.matmul."""
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=dtype, device="cuda")
    B = torch.randn(K, N, dtype=dtype, device="cuda")

    C = matmul(A, B, tile_m, tile_n, tile_k)
    expected = torch.matmul(A.float(), B.float())

    # FP16-Inputs mit FP32-Akkumulator → großzügigere Toleranzen
    atol, rtol = (1e-1, 1e-2) if dtype == torch.float16 else (1e-3, 1e-3)
    ok = torch.allclose(C, expected, atol=atol, rtol=rtol)
    tag = f"(M,N,K)=({M},{N},{K}), tile=({tile_m},{tile_n},{tile_k})"
    print(f"  {tag} → allclose={ok}")
    assert ok, f"matmul mismatch for {tag}"


def verify():
    """Compare matmul kernel against torch.matmul for several shapes."""
    # Zweierpotenzen — Basisfall
    check(256, 256, 256, 64, 64, 64)
    # Rechteckig, Zweierpotenzen
    check(512, 256, 128, 64, 64, 64)
    # Nicht-Zweierpotenzen — letzte Zeile/Spalte/K-Iteration nur teilweise belegt
    check(300, 200, 100, 64, 64, 64)
    check(129, 257, 65, 32, 64, 32)


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("Task 2: Simple Matmul Kernel — Verifikation")
    verify()

"""Ergebnisse
(wird nach Messung auf dem Uni-Rechner ausgefüllt)
"""
