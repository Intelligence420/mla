"""
Task 3: Benchmarking the Matrix Multiplication Kernel

Reuse the Task 2 kernel and report TFLOPS:

  TFLOPS = (2 * M * N * K) / (t_s * 1e12)

a) Square matmuls with tile (64, 64, 64) for
     M = N = K ∈ {256, 512, 1024, 2048, 4096, 8192}.
   Plot TFLOPS over matrix size and report observations.

b) All 27 tile combinations with m_tile, n_tile, k_tile ∈ {32, 64, 128}
   for the fixed sizes 2048³ and 512³.
   Visualise as a heatmap (m_tile × n_tile, k_tile = 64 fixed).
   Report the best tile combination.
"""

import cuda.tile as ct
import cupy as cp
import torch

from task_02 import matmul


# ===========================================================================
# Helpers
# ===========================================================================

def tflops(M: int, N: int, K: int, time_s: float) -> float:
    """2 * M * N * K FLOPs divided by runtime in seconds → TFLOPS."""
    return (2.0 * M * N * K) / (time_s * 1e12)


def bench_matmul(M: int, N: int, K: int,
                 tile_m: int, tile_n: int, tile_k: int) -> float:
    """Benchmark a single (shape, tile) combination and return TFLOPS."""
    import triton

    # TODO:
    #   - A, B auf CUDA erzeugen (Datentyp je nach gewählter Präzision)
    #   - triton.testing.do_bench(lambda: matmul(A, B, tile_m, tile_n, tile_k))
    #   - Rückgabe: tflops(M, N, K, time_ms / 1000)
    return 0.0


# ===========================================================================
# Task 3a: Sweep über Matrixgröße
# ===========================================================================

def sweep_matrix_size():
    """Benchmark square matmuls with tile (64, 64, 64)."""
    import matplotlib.pyplot as plt

    sizes = [256, 512, 1024, 2048, 4096, 8192]
    results = []

    # TODO: Für jede Größe M=N=K TFLOPS messen und in `results` sammeln.
    #   results.append((size, tflops))

    # TODO: Plot TFLOPS vs. Matrixgröße erzeugen und als PNG speichern
    #   (z. B. tflops_vs_size.png), log-x-Achse macht hier meistens Sinn.


# ===========================================================================
# Task 3b: Sweep über Tile-Shapes
# ===========================================================================

def sweep_tile_shapes(size: int):
    """Benchmark all 27 tile combinations for a fixed square matrix size."""
    import numpy as np
    import matplotlib.pyplot as plt

    tiles = [32, 64, 128]
    results = np.zeros((len(tiles), len(tiles), len(tiles)), dtype=float)

    # TODO: Drei verschachtelte Schleifen über tile_m, tile_n, tile_k.
    #   results[i, j, k] = bench_matmul(size, size, size, m, n, k)

    # TODO: Beste Kombination (argmax) bestimmen und ausgeben.

    # TODO: Heatmap für k_tile = 64 plotten
    #   - axes: m_tile (y) × n_tile (x)
    #   - plt.imshow / sns.heatmap
    #   - PNG speichern, Dateiname z. B. f"heatmap_{size}.png"


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("Task 3a: TFLOPS vs. Matrixgröße (tile 64×64×64)")
    sweep_matrix_size()

    print("Task 3b: Tile-Shape-Sweep — 2048³")
    sweep_tile_shapes(2048)

    print("Task 3b: Tile-Shape-Sweep — 512³")
    sweep_tile_shapes(512)

"""Ergebnisse
(wird nach Implementierung ausgefüllt)
"""
