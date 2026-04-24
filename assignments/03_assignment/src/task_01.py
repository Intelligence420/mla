"""
Task 1: FP32 vs FP16 Performance

Two cuTile kernels computing A @ B = C with
  shape(A) = (64, 4096), shape(B) = (4096, 64), shape(C) = (64, 64).

  1. kernel_fp16 — A, B in FP16; accumulator/output C in FP32
  2. kernel_fp32 — A, B, C in FP32

Both kernels use ct.mma, a single CTA (grid = (1, 1, 1)) and a fixed
tile shape (m_tile=64, n_tile=64, k_tile=64).

a) Verify both kernels against torch.matmul via torch.allclose.
b) Benchmark with triton.testing.do_bench and report the FP16 speedup.
"""

import cuda.tile as ct
import cupy as cp
import torch


# ===========================================================================
# Konstanten — feste Tile-/Matrix-Shapes 
# ===========================================================================

M, K, N = 64, 4096, 64
TILE_M, TILE_N, TILE_K = 64, 64, 64
NUM_K_TILES = K // TILE_K   # 4096 / 64 = 64 Iterationen ... "//" für Floor-Division (ganze Tiles)


# ===========================================================================
# Kernel: FP16 Inputs, FP32 Akkumulator
# ===========================================================================

@ct.kernel
def kernel_fp16(A, B, C):
    """A, B in FP16 → accumulate in FP32 → store FP32 result to C."""
    # Akkumulator-Tile in FP32 mit 0 initialisieren
    acc = ct.full((TILE_M, TILE_N), 0, dtype=ct.float32)

    # Schleife über die K-Dimension in Tiles der Größe TILE_K
    for k in range(NUM_K_TILES):
        a_tile = ct.load(A, index=(0, k), shape=(TILE_M, TILE_K),
                         padding_mode=ct.PaddingMode.ZERO)
        b_tile = ct.load(B, index=(k, 0), shape=(TILE_K, TILE_N),
                         padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a_tile, b_tile, acc)

    # Ergebnis direkt als FP32 speichern (C hat bereits dtype=float32)
    ct.store(C, index=(0, 0), tile=acc)


# ===========================================================================
# Kernel: FP32 Inputs + FP32 Akkumulator
# ===========================================================================

@ct.kernel
def kernel_fp32(A, B, C):
    """A, B, C alle in FP32."""
    acc = ct.full((TILE_M, TILE_N), 0, dtype=ct.float32)

    for k in range(NUM_K_TILES):
        a_tile = ct.load(A, index=(0, k), shape=(TILE_M, TILE_K),
                         padding_mode=ct.PaddingMode.ZERO)
        b_tile = ct.load(B, index=(k, 0), shape=(TILE_K, TILE_N),
                         padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a_tile, b_tile, acc)

    ct.store(C, index=(0, 0), tile=acc)


# ===========================================================================
# Host / Launch
# ===========================================================================

def run_fp16(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Launch kernel_fp16 and return the FP32 result C."""
    C = torch.empty(M, N, dtype=torch.float32, device=A.device)
    ct.launch(torch.cuda.current_stream().cuda_stream,
              (1, 1, 1), kernel_fp16, (A, B, C))
    return C


def run_fp32(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Launch kernel_fp32 and return the FP32 result C."""
    C = torch.empty(M, N, dtype=torch.float32, device=A.device)
    ct.launch(torch.cuda.current_stream().cuda_stream,
              (1, 1, 1), kernel_fp32, (A, B, C))
    return C


# ===========================================================================
# Verifikation
# ===========================================================================

def verify():
    """Compare both kernels against torch.matmul."""
    torch.manual_seed(0)

    # FP16-Variante
    A16 = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B16 = torch.randn(K, N, dtype=torch.float16, device="cuda")
    C16 = run_fp16(A16, B16)
    expected16 = torch.matmul(A16.float(), B16.float())
    # FP16-Inputs + FP32-Akkumulator → leicht großzügigere Toleranz
    ok16 = torch.allclose(C16, expected16, atol=1e-1, rtol=1e-2)
    print(f"  kernel_fp16 → allclose={ok16}")
    assert ok16, "kernel_fp16 mismatch"

    # FP32-Variante
    A32 = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B32 = torch.randn(K, N, dtype=torch.float32, device="cuda")
    C32 = run_fp32(A32, B32)
    expected32 = torch.matmul(A32, B32)
    ok32 = torch.allclose(C32, expected32, atol=1e-3, rtol=1e-3)
    print(f"  kernel_fp32 → allclose={ok32}")
    assert ok32, "kernel_fp32 mismatch"


# ===========================================================================
# Benchmark
# ===========================================================================

def benchmark():
    """Measure average runtime for both kernels and report the speedup."""
    import triton

    torch.manual_seed(0)
    A16 = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B16 = torch.randn(K, N, dtype=torch.float16, device="cuda")
    A32 = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B32 = torch.randn(K, N, dtype=torch.float32, device="cuda")

    t_fp16 = triton.testing.do_bench(lambda: run_fp16(A16, B16))
    t_fp32 = triton.testing.do_bench(lambda: run_fp32(A32, B32))
    speedup = t_fp32 / t_fp16 

    print(f"  kernel_fp16: {t_fp16:.4f} ms")
    print(f"  kernel_fp32: {t_fp32:.4f} ms")
    print(f"  Speedup FP16 über FP32: {speedup:.2f}x")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("Task 1a: FP16 vs FP32 — Verifikation")
    verify()
    print("Task 1b: FP16 vs FP32 — Benchmark")
    benchmark()

"""Ergebnisse
(.venv) mla08@flambe:~/MLA/mla$ python3 assignments/03_assignment/src/task_01.py 
Task 1a: FP16 vs FP32 — Verifikation
  kernel_fp16 → allclose=True
  kernel_fp32 → allclose=True
Task 1b: FP16 vs FP32 — Benchmark
  kernel_fp16: 0.0336 ms
  kernel_fp32: 1.7305 ms
  Speedup FP16 über FP32: 51.52x
"""
