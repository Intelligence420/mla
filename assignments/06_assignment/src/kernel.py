"""
Kernel-Modul (Task 4).

cuTile-Kernel fuer die Kontraktion ``acspx, bspy -> abcyx`` gemaess
der optimierten Config aus Task 3:

  PAR-Dims:  a, c, x_seq, b, y_seq       (Grid)
  SEQ-Dim:   sp_seq                       (innere Schleife)
  PRIM-Dims: x_prim (M), y_prim (N),      (ct.mma)
             sp_prim (K)

Tensor-Layouts nach Reshape (alle contiguous, kein Daten-Kopieren):
  A : (a, c, sp_seq, sp_prim, x_seq, x_prim)
  B : (b,    sp_seq, sp_prim, y_seq, y_prim)
  C : (a, b, c, y_seq, y_prim, x_seq, x_prim)

Zwei Kernel-Varianten:

* ``kernel_baseline``  — direkte Umsetzung der Config: 3D-Grid
  ``(a*c, b, x_seq*y_seq)``, pro Block ein Output-Tile, K-Schleife
  ueber ``sp_seq`` mit einem ``ct.mma`` pro Iteration.
* ``kernel_l2``        — identische Arithmetik, aber BID-Swizzle in
  ``(x_seq, y_seq)``-Gruppen fuer L2-Reuse (Pattern aus Assignment 05).
"""

import cuda.tile as ct
import torch
import triton

from config import generate_config, pretty, DimType, ExecType
from optimizer import Optimizer


# ===========================================================================
# Konstanten
# ===========================================================================

PRIM_M = 64    # x_prim
PRIM_N = 64    # y_prim
PRIM_K = 32    # sp_prim


# ===========================================================================
# Baseline-Kernel (direkte Umsetzung der Task-3-Config)
# ===========================================================================

@ct.kernel
def kernel_baseline(A, B, C,
                    Ad:    ct.Constant[int],
                    Bd:    ct.Constant[int],
                    Cd:    ct.Constant[int],
                    XSEQ:  ct.Constant[int],
                    YSEQ:  ct.Constant[int],
                    SPSEQ: ct.Constant[int],
                    tx:    ct.Constant[int],
                    ty:    ct.Constant[int],
                    tk:    ct.Constant[int]):
    """A : (Ad, Cd, SPSEQ, tk, XSEQ, tx)
       B : (Bd,     SPSEQ, tk, YSEQ, ty)
       C : (Ad, Bd, Cd, YSEQ, ty, XSEQ, tx)

       Grid: (Ad*Cd, Bd, XSEQ*YSEQ).
    """
    bid_ac = ct.bid(0)    # range Ad*Cd
    bid_b  = ct.bid(1)    # range Bd
    bid_xy = ct.bid(2)    # range XSEQ*YSEQ

    pid_a = bid_ac // Cd
    pid_c = bid_ac %  Cd
    pid_b = bid_b
    pid_x = bid_xy // YSEQ
    pid_y = bid_xy %  YSEQ

    acc = ct.full((ty, tx), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    for sp_seq in range(SPSEQ):
        a_tile = ct.load(A,
                         index=(pid_a, pid_c, sp_seq, 0, pid_x, 0),
                         shape=(1, 1, 1, tk, 1, tx),
                         padding_mode=zero_pad)
        b_tile = ct.load(B,
                         index=(pid_b, sp_seq, 0, pid_y, 0),
                         shape=(1, 1, tk, 1, ty),
                         padding_mode=zero_pad)
        a_kx = ct.reshape(a_tile, (tk, tx))   # (sp_prim, x_prim)
        b_ky = ct.reshape(b_tile, (tk, ty))   # (sp_prim, y_prim)
        b_yk = ct.permute(b_ky, (1, 0))       # (y_prim, sp_prim)
        # mma(M', K') @ (K', N') -> (M', N') mit M'=y_prim, N'=x_prim
        acc = ct.mma(b_yk, a_kx, acc)

    out = ct.reshape(ct.astype(acc, C.dtype), (1, 1, 1, 1, ty, 1, tx))
    ct.store(C, index=(pid_a, pid_b, pid_c, pid_y, 0, pid_x, 0), tile=out)


# ===========================================================================
# L2-optimierter Kernel (Super-Tile-Swizzle in y_seq-Richtung).
#
# Im Baseline-Kernel mit ``pid_x = bid_xy // YSEQ`` teilen YSEQ
# konsekutive BIDs bereits dieselbe A-Spalte (selbes x_seq) — A-Reuse
# entlang y kommt also "umsonst". Was fehlt, ist *B*-Reuse: benachbarte
# Bloecke variieren y_seq und laden so immer neue B-Streifen.
#
# Swizzle hier: Bloecke in GY-Streifen entlang y_seq stapeln. Innerhalb
# einer Gruppe sind GY konsekutive Bloecke auf dasselbe y_seq fixiert,
# danach wechseln x_seq und y_seq gemeinsam in einer Wave.
# ===========================================================================

@ct.kernel
def kernel_l2(A, B, C,
              Ad:    ct.Constant[int],
              Bd:    ct.Constant[int],
              Cd:    ct.Constant[int],
              XSEQ:  ct.Constant[int],
              YSEQ:  ct.Constant[int],
              SPSEQ: ct.Constant[int],
              tx:    ct.Constant[int],
              ty:    ct.Constant[int],
              tk:    ct.Constant[int],
              GY:    ct.Constant[int]):
    bid_ac = ct.bid(0)
    bid_b  = ct.bid(1)
    bid_xy = ct.bid(2)

    pid_a = bid_ac // Cd
    pid_c = bid_ac %  Cd
    pid_b = bid_b

    blocks_per_group = GY * XSEQ
    group_id = bid_xy // blocks_per_group
    in_group = bid_xy %  blocks_per_group
    first_y  = group_id * GY
    cur_gy   = min(YSEQ - first_y, GY)
    pid_y    = first_y + (in_group % cur_gy)
    pid_x    = in_group // cur_gy

    acc = ct.full((ty, tx), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    for sp_seq in range(SPSEQ):
        a_tile = ct.load(A,
                         index=(pid_a, pid_c, sp_seq, 0, pid_x, 0),
                         shape=(1, 1, 1, tk, 1, tx),
                         padding_mode=zero_pad)
        b_tile = ct.load(B,
                         index=(pid_b, sp_seq, 0, pid_y, 0),
                         shape=(1, 1, tk, 1, ty),
                         padding_mode=zero_pad)
        a_kx = ct.reshape(a_tile, (tk, tx))
        b_ky = ct.reshape(b_tile, (tk, ty))
        b_yk = ct.permute(b_ky, (1, 0))
        acc = ct.mma(b_yk, a_kx, acc)

    out = ct.reshape(ct.astype(acc, C.dtype), (1, 1, 1, 1, ty, 1, tx))
    ct.store(C, index=(pid_a, pid_b, pid_c, pid_y, 0, pid_x, 0), tile=out)


# ===========================================================================
# Kernel mit groesseren PRIM-Tiles (128 x 128 x 32)
#
# Same arithmetic, just |x_prim| = |y_prim| = 128 — vier Mal mehr FLOPs
# pro mma, deutlich weniger Bloecke und damit weniger Launch-/Load-Overhead.
# Setzt voraus: |x| % 128 == 0 und |y| % 128 == 0 (gegeben: 1536, 1152).
# ===========================================================================

@ct.kernel
def kernel_big(A, B, C,
               Ad:    ct.Constant[int],
               Bd:    ct.Constant[int],
               Cd:    ct.Constant[int],
               XSEQ:  ct.Constant[int],
               YSEQ:  ct.Constant[int],
               SPSEQ: ct.Constant[int],
               tx:    ct.Constant[int],
               ty:    ct.Constant[int],
               tk:    ct.Constant[int]):
    """Identische Logik zum Baseline, aber mit groesseren tx/ty."""
    bid_ac = ct.bid(0)
    bid_b  = ct.bid(1)
    bid_xy = ct.bid(2)

    pid_a = bid_ac // Cd
    pid_c = bid_ac %  Cd
    pid_b = bid_b
    pid_x = bid_xy // YSEQ
    pid_y = bid_xy %  YSEQ

    acc = ct.full((ty, tx), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    for sp_seq in range(SPSEQ):
        a_tile = ct.load(A,
                         index=(pid_a, pid_c, sp_seq, 0, pid_x, 0),
                         shape=(1, 1, 1, tk, 1, tx),
                         padding_mode=zero_pad)
        b_tile = ct.load(B,
                         index=(pid_b, sp_seq, 0, pid_y, 0),
                         shape=(1, 1, tk, 1, ty),
                         padding_mode=zero_pad)
        a_kx = ct.reshape(a_tile, (tk, tx))
        b_ky = ct.reshape(b_tile, (tk, ty))
        b_yk = ct.permute(b_ky, (1, 0))
        acc = ct.mma(b_yk, a_kx, acc)

    out = ct.reshape(ct.astype(acc, C.dtype), (1, 1, 1, 1, ty, 1, tx))
    ct.store(C, index=(pid_a, pid_b, pid_c, pid_y, 0, pid_x, 0), tile=out)


# ===========================================================================
# Host-Funktionen (Reshape + Launch)
# ===========================================================================

def _views(tensor_acspx: torch.Tensor,
           tensor_bspy:  torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                tuple[int, int, int, int, int, int]]:
    """Bringt die Eingabe-Tensoren in die 6D/5D-Sicht, die der Kernel
    indiziert. Liefert (A_view, B_view, (Ad, Bd, Cd, XSEQ, YSEQ, SPSEQ))."""
    Ad, Cd, S, P, X = tensor_acspx.shape
    Bd, S2, P2, Y = tensor_bspy.shape
    assert (S, P) == (S2, P2), f"s/p mismatch: {(S, P)} vs {(S2, P2)}"
    SP = S * P
    assert X % PRIM_M == 0, f"x={X} not divisible by PRIM_M={PRIM_M}"
    assert Y % PRIM_N == 0, f"y={Y} not divisible by PRIM_N={PRIM_N}"
    assert SP % PRIM_K == 0, f"s*p={SP} not divisible by PRIM_K={PRIM_K}"
    XSEQ, YSEQ, SPSEQ = X // PRIM_M, Y // PRIM_N, SP // PRIM_K

    A = tensor_acspx.contiguous().view(Ad, Cd, SPSEQ, PRIM_K, XSEQ, PRIM_M)
    B = tensor_bspy.contiguous().view(Bd, SPSEQ, PRIM_K, YSEQ, PRIM_N)
    return A, B, (Ad, Bd, Cd, XSEQ, YSEQ, SPSEQ)


def run_baseline(tensor_acspx: torch.Tensor,
                 tensor_bspy:  torch.Tensor) -> torch.Tensor:
    """Berechnet abcyx = einsum(acspx,bspy) per Baseline-Kernel.
    Output ist FP16, Akku FP32."""
    A, B, (Ad, Bd, Cd, XSEQ, YSEQ, SPSEQ) = _views(tensor_acspx, tensor_bspy)
    Y, X = YSEQ * PRIM_N, XSEQ * PRIM_M

    C = torch.empty((Ad, Bd, Cd, Y, X),
                    device=tensor_acspx.device, dtype=tensor_acspx.dtype)
    C_view = C.view(Ad, Bd, Cd, YSEQ, PRIM_N, XSEQ, PRIM_M)

    grid = (Ad * Cd, Bd, XSEQ * YSEQ)
    ct.launch(torch.cuda.current_stream().cuda_stream,
              grid, kernel_baseline,
              (A, B, C_view, Ad, Bd, Cd, XSEQ, YSEQ, SPSEQ,
               PRIM_M, PRIM_N, PRIM_K))
    return C


def run_l2(tensor_acspx: torch.Tensor,
           tensor_bspy:  torch.Tensor,
           group_y: int = 4) -> torch.Tensor:
    """Wie run_baseline, aber mit Super-Tile-Swizzle der Breite group_y
    in y_seq-Richtung (B-Tile-Reuse innerhalb einer Gruppe)."""
    A, B, (Ad, Bd, Cd, XSEQ, YSEQ, SPSEQ) = _views(tensor_acspx, tensor_bspy)
    Y, X = YSEQ * PRIM_N, XSEQ * PRIM_M

    C = torch.empty((Ad, Bd, Cd, Y, X),
                    device=tensor_acspx.device, dtype=tensor_acspx.dtype)
    C_view = C.view(Ad, Bd, Cd, YSEQ, PRIM_N, XSEQ, PRIM_M)

    grid = (Ad * Cd, Bd, XSEQ * YSEQ)
    ct.launch(torch.cuda.current_stream().cuda_stream,
              grid, kernel_l2,
              (A, B, C_view, Ad, Bd, Cd, XSEQ, YSEQ, SPSEQ,
               PRIM_M, PRIM_N, PRIM_K, group_y))
    return C


def run_big(tensor_acspx: torch.Tensor,
            tensor_bspy:  torch.Tensor,
            prim_m: int = 128, prim_n: int = 128, prim_k: int = 32) -> torch.Tensor:
    """Baseline-Layout mit groesseren PRIM-Tiles. |x| und |y| muessen
    durch prim_m bzw. prim_n teilbar sein."""
    Ad, Cd, S, P, X = tensor_acspx.shape
    Bd, _, _, Y = tensor_bspy.shape
    SP = S * P
    assert X % prim_m == 0 and Y % prim_n == 0 and SP % prim_k == 0
    XSEQ, YSEQ, SPSEQ = X // prim_m, Y // prim_n, SP // prim_k

    A = tensor_acspx.contiguous().view(Ad, Cd, SPSEQ, prim_k, XSEQ, prim_m)
    B = tensor_bspy.contiguous().view(Bd, SPSEQ, prim_k, YSEQ, prim_n)

    C = torch.empty((Ad, Bd, Cd, Y, X),
                    device=tensor_acspx.device, dtype=tensor_acspx.dtype)
    C_view = C.view(Ad, Bd, Cd, YSEQ, prim_n, XSEQ, prim_m)

    grid = (Ad * Cd, Bd, XSEQ * YSEQ)
    ct.launch(torch.cuda.current_stream().cuda_stream,
              grid, kernel_big,
              (A, B, C_view, Ad, Bd, Cd, XSEQ, YSEQ, SPSEQ,
               prim_m, prim_n, prim_k))
    return C


# ===========================================================================
# Verifikation und Benchmark
# ===========================================================================

def reference(tensor_acspx: torch.Tensor,
              tensor_bspy:  torch.Tensor) -> torch.Tensor:
    """``torch.einsum``-Referenz; FP32-Akku, Rueckgabe in Input-dtype."""
    out = torch.einsum("acspx,bspy->abcyx",
                       tensor_acspx.to(torch.float32),
                       tensor_bspy.to(torch.float32))
    return out.to(tensor_acspx.dtype)


def verify_kernel(tensor_acspx: torch.Tensor,
                  tensor_bspy:  torch.Tensor,
                  atol: float = 2e-1, rtol: float = 2e-2) -> None:
    """Vergleicht alle Kernel-Varianten gegen ``torch.einsum``."""
    ref = reference(tensor_acspx, tensor_bspy)
    print(f"  Shapes: A={tuple(tensor_acspx.shape)}, "
          f"B={tuple(tensor_bspy.shape)}, ref={tuple(ref.shape)}")

    out_b = run_baseline(tensor_acspx, tensor_bspy)
    err_b = (out_b.float() - ref.float()).abs().max().item()
    ok_b = torch.allclose(out_b, ref, atol=atol, rtol=rtol)
    print(f"  baseline      allclose={ok_b}   max_abs_err={err_b:.4f}")
    assert ok_b, "baseline mismatch"

    out_l = run_l2(tensor_acspx, tensor_bspy)
    err_l = (out_l.float() - ref.float()).abs().max().item()
    ok_l = torch.allclose(out_l, ref, atol=atol, rtol=rtol)
    print(f"  l2-swizzle    allclose={ok_l}   max_abs_err={err_l:.4f}")
    assert ok_l, "l2-swizzle mismatch"

    out_big = run_big(tensor_acspx, tensor_bspy)
    err_big = (out_big.float() - ref.float()).abs().max().item()
    ok_big = torch.allclose(out_big, ref, atol=atol, rtol=rtol)
    print(f"  big-prim 128  allclose={ok_big}   max_abs_err={err_big:.4f}")
    assert ok_big, "big-prim mismatch"


def flops_count(tensor_acspx: torch.Tensor,
                tensor_bspy:  torch.Tensor) -> int:
    """``2 * |a| * |b| * |c| * |s| * |p| * |x| * |y|``"""
    a, c, s, p, x = tensor_acspx.shape
    b, _, _, y = tensor_bspy.shape
    return 2 * a * b * c * s * p * x * y


def benchmark(tensor_acspx: torch.Tensor,
              tensor_bspy:  torch.Tensor,
              group_y_sweep: tuple[int, ...] = (2, 3, 4, 6, 9)) -> dict:
    """Bencht alle Varianten plus ``torch.einsum``."""
    flops = flops_count(tensor_acspx, tensor_bspy)
    bench = lambda fn: triton.testing.do_bench(fn, warmup=10, rep=50)
    a16, b16 = tensor_acspx, tensor_bspy

    print(f"  FLOPs = {flops:.3e}")

    t_torch = bench(lambda: torch.einsum("acspx,bspy->abcyx", a16, b16))
    t_base  = bench(lambda: run_baseline(a16, b16))
    t_big   = bench(lambda: run_big(a16, b16))
    t_big_k64 = bench(lambda: run_big(a16, b16, prim_m=128, prim_n=128, prim_k=64))
    t_l2s   = {g: bench(lambda g=g: run_l2(a16, b16, group_y=g))
               for g in group_y_sweep}

    def tflops(t_ms):
        return flops / (t_ms * 1e-3) / 1e12

    print(f"  torch.einsum (FP16)        {t_torch:8.4f} ms   "
          f"{tflops(t_torch):7.3f} TFLOPS")
    print(f"  baseline (64x64x32)        {t_base:8.4f} ms   "
          f"{tflops(t_base):7.3f} TFLOPS   ({t_torch/t_base:5.2f}x vs torch)")
    print(f"  big-prim (128x128x32)      {t_big:8.4f} ms   "
          f"{tflops(t_big):7.3f} TFLOPS   ({t_torch/t_big:5.2f}x vs torch)")
    print(f"  big-prim (128x128x64)      {t_big_k64:8.4f} ms   "
          f"{tflops(t_big_k64):7.3f} TFLOPS   ({t_torch/t_big_k64:5.2f}x vs torch)")
    for g, t in t_l2s.items():
        print(f"  l2-swizzle GY={g:<2}            {t:8.4f} ms   "
              f"{tflops(t):7.3f} TFLOPS   ({t_torch/t:5.2f}x vs torch)")

    return {
        "flops": flops,
        "torch": t_torch,
        "baseline": t_base,
        "big": t_big,
        "l2": t_l2s,
    }


# ===========================================================================
# __main__: laeuft auf synthetischen Tensoren (Datei-unabhaengig)
# ===========================================================================

if __name__ == "__main__":
    # Form-Parameter aus dem real-Dataset (s. main.py / aufgabe_06.rst)
    SHAPE_A = (4, 3, 64, 64, 1536)
    SHAPE_B = (4,    64, 64, 1152)

    print("Task 4 — Kernel auf synthetischen Tensoren")
    print(f"  A shape = {SHAPE_A}")
    print(f"  B shape = {SHAPE_B}")
    torch.manual_seed(0)
    a = torch.randn(*SHAPE_A, dtype=torch.float16, device="cuda")
    b = torch.randn(*SHAPE_B, dtype=torch.float16, device="cuda")

    print("\nVerifikation (gegen torch.einsum, FP32-Akku):")
    verify_kernel(a, b)

    print("\nBenchmark:")
    benchmark(a, b)

"""Ergebnisse (synthetische Tensoren, Shapes wie lf_tr_64_intermediate.npz)
(.venv) mla07@flambe:~/mla$ python3 assignments/06_assignment/src/kernel.py
Task 4 — Kernel auf synthetischen Tensoren
  A shape = (4, 3, 64, 64, 1536)
  B shape = (4, 64, 64, 1152)

Verifikation (gegen torch.einsum, FP32-Akku):
  Shapes: A=(4, 3, 64, 64, 1536), B=(4, 64, 64, 1152), ref=(4, 4, 3, 1152, 1536)
  baseline      allclose=True   max_abs_err=0.2500
  l2-swizzle    allclose=True   max_abs_err=0.2500
  big-prim 128  allclose=True   max_abs_err=0.2500

Benchmark:
  FLOPs = 6.958e+11
  torch.einsum (FP16)         11.3598 ms    61.250 TFLOPS
  baseline (64x64x32)         42.4653 ms    16.385 TFLOPS   ( 0.27x vs torch)
  big-prim (128x128x32)       24.5135 ms    28.384 TFLOPS   ( 0.46x vs torch)
  big-prim (128x128x64)       25.5447 ms    27.238 TFLOPS   ( 0.44x vs torch)
  l2-swizzle GY=2              62.2469 ms    11.178 TFLOPS   ( 0.18x vs torch)
  l2-swizzle GY=3              54.4952 ms    12.768 TFLOPS   ( 0.21x vs torch)
  l2-swizzle GY=4              51.8564 ms    13.418 TFLOPS   ( 0.22x vs torch)
  l2-swizzle GY=6              46.7180 ms    14.893 TFLOPS   ( 0.24x vs torch)
  l2-swizzle GY=9              45.0970 ms    15.429 TFLOPS   ( 0.25x vs torch)
"""
