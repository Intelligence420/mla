"""
Task 1: FP32 vs FP16 Performance (+ FP8 und FP64 aus Interesse)

cuTile-Kernels, die A @ B = C berechnen mit
  shape(A) = (64, 4096), shape(B) = (4096, 64), shape(C) = (64, 64).

Pflicht-Varianten:
  1. kernel_fp16 — A, B in FP16; accumulator/output C in FP32
  2. kernel_fp32 — A, B, C in FP32

Zusätzliche Vergleichsvarianten:
  3. kernel_fp8  — A, B in FP8 (E4M3); accumulator/output C in FP32
  4. kernel_fp64 — A, B, C in FP64 (Blackwell-FP64-Tensor-Core)

Alle Kernels nutzen ct.mma, ein einziges CTA (grid = (1, 1, 1)) und die
feste Tile-Größe (m_tile=64, n_tile=64, k_tile=64).

a) Verifikation beider Pflicht-Kernels gegen torch.matmul per torch.allclose.
b) Benchmark mit triton.testing.do_bench + Plot (tflops_vs_dtype.png).
"""

import cuda.tile as ct
import cupy as cp
import torch


# ===========================================================================
# Konstanten — feste Tile-/Matrix-Shapes
# ===========================================================================

M, K, N = 64, 4096, 64
TILE_M, TILE_N, TILE_K = 64, 64, 64
NUM_K_TILES = K // TILE_K   # 4096 / 64 = 64 Iterationen ("//" = Floor-Division)

FLOPS = 2 * M * N * K       # pro Matmul-Aufruf

# Ausgabeverzeichnis für den Plot = dieses Skript-Verzeichnis
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH = os.path.join(SCRIPT_DIR, "tflops_vs_dtype.png")


# ===========================================================================
# Kernel: FP16 Inputs, FP32 Akkumulator
# ===========================================================================

@ct.kernel
def kernel_fp16(A, B, C):
    """A, B in FP16 → accumulate in FP32 → store FP32 result to C."""
    acc = ct.full((TILE_M, TILE_N), 0, dtype=ct.float32)
    for k in range(NUM_K_TILES):
        a_tile = ct.load(A, index=(0, k), shape=(TILE_M, TILE_K),
                         padding_mode=ct.PaddingMode.ZERO)
        b_tile = ct.load(B, index=(k, 0), shape=(TILE_K, TILE_N),
                         padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a_tile, b_tile, acc)
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
# Kernel: FP8 (E4M3) Inputs, FP32 Akkumulator
# ===========================================================================

@ct.kernel
def kernel_fp8(A, B, C):
    """A, B in FP8 E4M3 → accumulate in FP32 → store FP32 result to C."""
    acc = ct.full((TILE_M, TILE_N), 0, dtype=ct.float32)
    for k in range(NUM_K_TILES):
        a_tile = ct.load(A, index=(0, k), shape=(TILE_M, TILE_K),
                         padding_mode=ct.PaddingMode.ZERO)
        b_tile = ct.load(B, index=(k, 0), shape=(TILE_K, TILE_N),
                         padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a_tile, b_tile, acc)
    ct.store(C, index=(0, 0), tile=acc)


# ===========================================================================
# Kernel: FP64 Inputs + FP64 Akkumulator
# ===========================================================================

@ct.kernel
def kernel_fp64(A, B, C):
    """A, B, C alle in FP64 — ct.mma nutzt den FP64-Tensor-Core-Pfad."""
    acc = ct.full((TILE_M, TILE_N), 0, dtype=ct.float64)
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

def _launch(kernel, A, B, out_dtype):
    C = torch.empty(M, N, dtype=out_dtype, device=A.device)
    ct.launch(torch.cuda.current_stream().cuda_stream,
              (1, 1, 1), kernel, (A, B, C))
    return C


def run_fp16(A, B): return _launch(kernel_fp16, A, B, torch.float32)
def run_fp32(A, B): return _launch(kernel_fp32, A, B, torch.float32)
def run_fp8(A, B):  return _launch(kernel_fp8,  A, B, torch.float32)
def run_fp64(A, B): return _launch(kernel_fp64, A, B, torch.float64)


# ===========================================================================
# Verifikation
# ===========================================================================

def verify():
    """Compare both Pflicht-Kernels against torch.matmul."""
    torch.manual_seed(0)

    # FP16-Variante
    A16 = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B16 = torch.randn(K, N, dtype=torch.float16, device="cuda")
    C16 = run_fp16(A16, B16)
    expected16 = torch.matmul(A16.float(), B16.float())
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
# Zusatz-Verifikation für FP8 und FP64
# ===========================================================================

def verify_extras():
    """Lockere Verifikation für die Zusatzvarianten FP8 und FP64."""
    torch.manual_seed(0)

    # FP8 E4M3 — nur 3 Mantissa-Bits → sehr großzügige Toleranz nötig.
    # Inputs werden aus FP16-Random-Werten gecastet (torch.randn kennt kein FP8).
    A8 = torch.randn(M, K, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
    B8 = torch.randn(K, N, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
    C8 = run_fp8(A8, B8)
    expected8 = torch.matmul(A8.float(), B8.float())
    # FP8 akkumuliert 4096 grob quantisierte Produkte → Toleranz im Einheitenbereich
    ok8 = torch.allclose(C8, expected8, atol=2.0, rtol=1e-1)
    print(f"  kernel_fp8  → allclose={ok8}")
    if not ok8:
        print(f"    max abs err = {(C8 - expected8).abs().max().item():.3f}")

    # FP64 — Referenz ebenfalls in FP64
    A64 = torch.randn(M, K, dtype=torch.float64, device="cuda")
    B64 = torch.randn(K, N, dtype=torch.float64, device="cuda")
    C64 = run_fp64(A64, B64)
    expected64 = torch.matmul(A64, B64)
    ok64 = torch.allclose(C64, expected64, atol=1e-10, rtol=1e-10)
    print(f"  kernel_fp64 → allclose={ok64}")
    assert ok64, "kernel_fp64 mismatch"


# ===========================================================================
# Benchmark
# ===========================================================================

def benchmark():
    """Measure average runtime for all variants and return {dtype_name: ms}."""
    import triton

    torch.manual_seed(0)
    A16 = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B16 = torch.randn(K, N, dtype=torch.float16, device="cuda")
    A32 = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B32 = torch.randn(K, N, dtype=torch.float32, device="cuda")
    A8  = A16.to(torch.float8_e4m3fn)
    B8  = B16.to(torch.float8_e4m3fn)
    A64 = torch.randn(M, K, dtype=torch.float64, device="cuda")
    B64 = torch.randn(K, N, dtype=torch.float64, device="cuda")

    times = {
        "FP8":  triton.testing.do_bench(lambda: run_fp8(A8,  B8)),
        "FP16": triton.testing.do_bench(lambda: run_fp16(A16, B16)),
        "FP32": triton.testing.do_bench(lambda: run_fp32(A32, B32)),
        "FP64": triton.testing.do_bench(lambda: run_fp64(A64, B64)),
    }

    # Report
    speedup_base = times["FP32"]
    for name, t in times.items():
        tflops = FLOPS / (t * 1e-3) / 1e12
        print(f"  kernel_{name.lower():<4}: {t:.4f} ms  ({tflops:>7.3f} TFLOPS, "
              f"{speedup_base / t:>6.2f}x vs FP32)")

    return times


# ===========================================================================
# Plot
# ===========================================================================

def plot(times: dict, path: str = PLOT_PATH):
    """Two-panel bar plot: Laufzeit (ms) und Durchsatz (TFLOPS) pro dtype."""
    import matplotlib.pyplot as plt

    # Von grob nach fein: FP8 → FP16 → FP32 → FP64
    order = ["FP8", "FP16", "FP32", "FP64"]
    names = [n for n in order if n in times]
    ms = [times[n] for n in names]
    tflops = [FLOPS / (t * 1e-3) / 1e12 for t in ms]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    bars1 = ax1.bar(names, ms, color="#4C72B0")
    ax1.set_yscale("log")
    ax1.set_ylabel("Laufzeit [ms] (log)")
    ax1.set_title("Kernel-Laufzeit pro dtype")
    for b, v in zip(bars1, ms):
        ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=9)

    bars2 = ax2.bar(names, tflops, color="#55A868")
    ax2.set_yscale("log")
    ax2.set_ylabel("Durchsatz [TFLOPS] (log)")
    ax2.set_title("Rechendurchsatz pro dtype (single CTA)")
    for b, v in zip(bars2, tflops):
        ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"cuTile-Matmul {M}×{K} · {K}×{N}, Tile "
                 f"({TILE_M},{TILE_N},{TILE_K}), grid=(1,1,1)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  Plot gespeichert: {path}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("Task 1a: FP16 vs FP32 — Verifikation")
    verify()
    print("Task 1a-extra: FP8 und FP64 — Verifikation")
    verify_extras()
    print("Task 1b: FP8 / FP16 / FP32 / FP64 — Benchmark")
    times = benchmark()
    print("Task 1b: Plot")
    plot(times)

"""Ergebnisse
(.venv) mla08@flambe:~/MLA/mla$ python3 assignments/03_assignment/src/task_01.py 
Task 1a: FP16 vs FP32 — Verifikation
  kernel_fp16 → allclose=True
  kernel_fp32 → allclose=True
Task 1a-extra: FP8 und FP64 — Verifikation
  kernel_fp8  → allclose=True
  kernel_fp64 → allclose=True
Task 1b: FP8 / FP16 / FP32 / FP64 — Benchmark
  kernel_fp8 : 0.0235 ms  (  1.425 TFLOPS,  72.75x vs FP32)
  kernel_fp16: 0.0274 ms  (  1.225 TFLOPS,  62.54x vs FP32)
  kernel_fp32: 1.7128 ms  (  0.020 TFLOPS,   1.00x vs FP32)
  kernel_fp64: 9.5736 ms  (  0.004 TFLOPS,   0.18x vs FP32)
Task 1b: Plot
  Plot gespeichert: /home/mla08/MLA/mla/assignments/03_assignment/src/tflops_vs_dtype.png
"""
