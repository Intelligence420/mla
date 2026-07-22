"""
Kernel-Modul (Task 4).

cuTile-Kernel fuer die Kontraktion ``acspx, bspy -> abcyx`` gemaess
der optimierten Config aus Task 3 (M/N-PAR-Achsen verschachtelt):

  PAR-Dims:  a, c, b, x_super, y_super, x_group, y_group   (Grid)
  SEQ-Dim:   sp_seq                                        (innere Schleife)
  PRIM-Dims: x_prim (M), y_prim (N), sp_prim (K)           (ct.mma)

Tensor-Layouts nach Reshape (alle contiguous, kein Daten-Kopieren):
  A : (a, c, sp_seq, sp_prim, x_seq, x_prim)
  B : (b,    sp_seq, sp_prim, y_seq, y_prim)
  C : (a, b, c, y_seq, y_prim, x_seq, x_prim)

Kernel-Varianten:

* ``kernel_baseline``  — direkte Umsetzung der Config: 3D-Grid
  ``(a*c, b, x_seq*y_seq)``, pro Block ein Output-Tile, K-Schleife
  ueber ``sp_seq`` mit einem ``ct.mma`` pro Iteration.
* ``kernel_generic``   — generischer, CONFIG-GETRIEBENER Kernel: Grid ueber
  die PAR-Achsen, GEMM ueber die PRIM-Achsen. Der L2-Super-Tile-Swizzle
  entsteht aus der Dimensionsstruktur der Config (``build_optimized_config``:
  x_seq/y_seq -> (super, group), M-/N-Gruppen verschachtelt), NICHT aus einer
  ``// blocks_per_group``-Formel im Kernel. ``group=None`` (super=1) ergibt die
  natuerliche Enumeration; ``group=(gx, gy)`` ein GX x GY 2D-Super-Tile.
* ``kernel_big``       — Baseline-Layout mit groesseren PRIM-Tiles (128x128).
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
# Optimierte Config (Task 3) + generischer, config-getriebener Kernel (Task 4a)
#
# Die L2-Optimierung wird DATENGETRIEBEN ueber die Config ausgedrueckt, nicht
# per Hand im Kernel: x_seq und y_seq werden je in (super, group) gesplittet und
# die M-/N-Gruppen-Achsen so permutiert, dass benachbarte BIDs ein
# GX x GY 2D-Super-Tile abdecken. Der Kernel dekodiert nur generisch (Grid ueber
# die PAR-Achsen, GEMM ueber die PRIM-Achsen) — keine ``// blocks_per_group``-
# Formel mehr. GX=x_seq / GY=y_seq bedeutet "keine Verschachtelung" (super=1) und
# reproduziert die natuerliche Enumeration.
# ===========================================================================

def build_optimized_config(shape_acspx: tuple, shape_bspy: tuple,
                           prim: tuple[int, int, int] = (PRIM_M, PRIM_N, PRIM_K),
                           group: tuple[int, int] | None = None) -> "object":
    """Baut die optimierte Config rein ueber Optimizer-Operationen.

    Splits: sp=fuse(s,p) -> (sp_seq, sp_prim); x -> (x_seq, x_prim);
    y -> (y_seq, y_prim); dann x_seq -> (x_super, x_group),
    y_seq -> (y_super, y_group). Permutation verschachtelt die M-/N-PAR-Achsen
    zu ``[a, c, b, x_super, y_super, x_group, y_group]`` (Gruppen-Achsen innen).
    group=(gx, gy); None -> gx=x_seq, gy=y_seq (natuerliche Reihenfolge).
    """
    pm, pn, pk = prim
    a, c, s, p, x = shape_acspx
    b, _, _, y = shape_bspy
    x_seq, y_seq = x // pm, y // pn
    gx, gy = (x_seq, y_seq) if group is None else group

    cfg = generate_config("acspx,bspy->abcyx", [shape_acspx, shape_bspy])
    opt = Optimizer(cfg)
    opt.fuse_dims(2, 3)                       # s,p -> sp   [a,c,sp,x,b,y]
    opt.split_dim(2, (s * p) // pk, pk)       # sp -> sp_seq, sp_prim
    opt.split_dim(4, x_seq, pm)               # x  -> x_seq,  x_prim
    opt.split_dim(7, y_seq, pn)               # y  -> y_seq,  y_prim
    # [a,c,sp_seq,sp_prim,x_seq,x_prim,b,y_seq,y_prim]
    opt.split_dim(4, x_seq // gx, gx)         # x_seq -> x_super, x_group
    opt.split_dim(8, y_seq // gy, gy)         # y_seq -> y_super, y_group
    # [a,c,sp_seq,sp_prim,x_super,x_group,x_prim,b,y_super,y_group,y_prim]
    #  0 1   2      3        4       5      6    7    8       9      10
    # Ziel: PAR=[a,c,b,x_super,y_super,x_group,y_group], SEQ=[sp_seq],
    #       PRIM=[x_prim,y_prim,sp_prim]
    opt.permute_dims([0, 1, 7, 4, 8, 5, 9, 2, 6, 10, 3])
    opt.make_executable()
    return cfg


def _extract_par(cfg) -> tuple[int, int, int, int, int, int, int]:
    """Liest die PAR-Groessen AUS der Config (datengetrieben). PAR-Reihenfolge
    der Pipeline: [a, c, b, x_super, y_super, x_group, y_group]."""
    par = [cfg.dim_sizes[i] for i in range(len(cfg.dim_sizes))
           if cfg.exec_types[i] == ExecType.PAR]
    Ad, Cd, Bd, XS, YS, GX, GY = par
    return Ad, Cd, Bd, XS, YS, GX, GY


@ct.kernel
def kernel_generic(A, B, C,
                   Cd: ct.Constant[int],
                   XS: ct.Constant[int], YS: ct.Constant[int],
                   GX: ct.Constant[int], GY: ct.Constant[int],
                   SPSEQ: ct.Constant[int],
                   tx: ct.Constant[int], ty: ct.Constant[int], tk: ct.Constant[int]):
    """Generisch: Grid (a*c, b, x_super*y_super*x_group*y_group), GEMM ueber
    PRIM. Das bid(2)-Decode dekodiert die gesplitteten x/y-Achsen mit den
    GRUPPEN-Achsen innen -> aufeinanderfolgende BIDs bilden ein GX x GY
    2D-Super-Tile. Der Swizzle faellt aus dieser Enumeration, nicht aus einer
    Formel im Kernel. GX=x_seq/GY=y_seq (super=1) ergibt die natuerliche Ordnung.
    """
    pid_a = ct.bid(0) // Cd
    pid_c = ct.bid(0) %  Cd
    pid_b = ct.bid(1)

    g = ct.bid(2)
    y_grp = g %  GY; g = g // GY
    x_grp = g %  GX; g = g // GX
    y_sup = g %  YS
    x_sup = g // YS
    pid_x = x_sup * GX + x_grp
    pid_y = y_sup * GY + y_grp

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


def run_generic(tensor_acspx: torch.Tensor,
                tensor_bspy:  torch.Tensor,
                prim: tuple[int, int, int] = (PRIM_M, PRIM_N, PRIM_K),
                group: tuple[int, int] | None = None) -> torch.Tensor:
    """Config-getrieben: baut die optimierte Config, liest Super/Group-Groessen
    daraus und startet ``kernel_generic``. ``group=(gx, gy)`` waehlt das
    2D-Super-Tile; ``None`` -> natuerliche Reihenfolge (super=1). Aendert man die
    Split-/Permute-Pipeline in ``build_optimized_config``, aendert sich das
    Launch-Layout automatisch mit — ohne den Kernel anzufassen."""
    pm, pn, pk = prim
    Ad, Cd, S, P, X = tensor_acspx.shape
    Bd, _, _, Y = tensor_bspy.shape
    SP = S * P
    assert X % pm == 0 and Y % pn == 0 and SP % pk == 0
    XSEQ, YSEQ, SPSEQ = X // pm, Y // pn, SP // pk

    A = tensor_acspx.contiguous().view(Ad, Cd, SPSEQ, pk, XSEQ, pm)
    B = tensor_bspy.contiguous().view(Bd, SPSEQ, pk, YSEQ, pn)

    cfg = build_optimized_config(tuple(tensor_acspx.shape),
                                 tuple(tensor_bspy.shape), prim=prim, group=group)
    _, _, _, XS, YS, GX, GY = _extract_par(cfg)

    C = torch.empty((Ad, Bd, Cd, Y, X),
                    device=tensor_acspx.device, dtype=tensor_acspx.dtype)
    C_view = C.view(Ad, Bd, Cd, YSEQ, pn, XSEQ, pm)

    grid = (Ad * Cd, Bd, XS * YS * GX * GY)
    ct.launch(torch.cuda.current_stream().cuda_stream,
              grid, kernel_generic,
              (A, B, C_view, Cd, XS, YS, GX, GY, SPSEQ, pm, pn, pk))
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

    out_g = run_generic(tensor_acspx, tensor_bspy, group=(4, 3))
    err_g = (out_g.float() - ref.float()).abs().max().item()
    ok_g = torch.allclose(out_g, ref, atol=atol, rtol=rtol)
    print(f"  generic(ilv)  allclose={ok_g}   max_abs_err={err_g:.4f}")
    assert ok_g, "generic (interleaved) mismatch"

    out_gn = run_generic(tensor_acspx, tensor_bspy, prim=(128, 128, 32))
    err_gn = (out_gn.float() - ref.float()).abs().max().item()
    ok_gn = torch.allclose(out_gn, ref, atol=atol, rtol=rtol)
    print(f"  generic(128)  allclose={ok_gn}   max_abs_err={err_gn:.4f}")
    assert ok_gn, "generic (128, natural) mismatch"

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
              group_sweep: tuple[tuple[int, int], ...] = ((4, 3), (6, 6),
                                                          (8, 9), (12, 9))) -> dict:
    """Bencht Baseline, den generischen (config-getriebenen) Kernel und
    ``torch.einsum``. Alle nicht-Referenz-Varianten laufen ueber
    ``kernel_generic`` — nur die Config unterscheidet sie."""
    flops = flops_count(tensor_acspx, tensor_bspy)
    bench = lambda fn: triton.testing.do_bench(fn, warmup=10, rep=50)
    a16, b16 = tensor_acspx, tensor_bspy

    def tflops(t_ms):
        return flops / (t_ms * 1e-3) / 1e12

    print(f"  FLOPs = {flops:.3e}")

    t_torch    = bench(lambda: torch.einsum("acspx,bspy->abcyx", a16, b16))
    t_base     = bench(lambda: run_baseline(a16, b16))
    t_gen64    = bench(lambda: run_generic(a16, b16))                     # 64, natural
    t_gen128   = bench(lambda: run_generic(a16, b16, prim=(128, 128, 32)))  # 128, natural
    t_big      = bench(lambda: run_big(a16, b16))
    t_ilv = {gp: bench(lambda gp=gp: run_generic(a16, b16, group=gp))
             for gp in group_sweep}                                       # 64, interleaved

    print(f"  torch.einsum (FP16)             {t_torch:8.4f} ms   "
          f"{tflops(t_torch):7.3f} TFLOPS   1.00x")
    print(f"  baseline 3D (64x64x32)          {t_base:8.4f} ms   "
          f"{tflops(t_base):7.3f} TFLOPS   ({t_torch/t_base:5.2f}x vs torch)")
    print(f"  generic natural (64x64x32)      {t_gen64:8.4f} ms   "
          f"{tflops(t_gen64):7.3f} TFLOPS   ({t_torch/t_gen64:5.2f}x vs torch)")
    print(f"  generic natural (128x128x32)    {t_gen128:8.4f} ms   "
          f"{tflops(t_gen128):7.3f} TFLOPS   ({t_torch/t_gen128:5.2f}x vs torch)")
    print(f"  big-prim (128x128x32)           {t_big:8.4f} ms   "
          f"{tflops(t_big):7.3f} TFLOPS   ({t_torch/t_big:5.2f}x vs torch)")
    for gp, t in t_ilv.items():
        print(f"  generic interleaved GX,GY={gp!s:7s}  {t:8.4f} ms   "
              f"{tflops(t):7.3f} TFLOPS   ({t_torch/t:5.2f}x vs torch)")

    return {
        "flops": flops,
        "torch": t_torch,
        "baseline": t_base,
        "generic64": t_gen64,
        "generic128": t_gen128,
        "big": t_big,
        "interleaved": t_ilv,
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
(.venv) mla08@flambe:~/MLA/mla/assignments/06_assignment/src$ python3 kernel.py
Verifikation (gegen torch.einsum, FP32-Akku):
  baseline      allclose=True   max_abs_err=0.2500
  generic(ilv)  allclose=True   max_abs_err=0.2500
  generic(128)  allclose=True   max_abs_err=0.2500
  big-prim 128  allclose=True   max_abs_err=0.2500

Benchmark (FLOPs=6.958e+11):
  torch.einsum (FP16)            12.86 ms   54.12 TFLOPS   1.00x
  baseline 3D (64x64x32)         44.27 ms   15.72 TFLOPS   0.29x
  generic natural (64x64x32)     43.68 ms   15.93 TFLOPS   0.29x   (== baseline: generischer Decode
                                                                    ohne Overhead)
  generic natural (128x128x32)   26.28 ms   26.47 TFLOPS   0.49x   (bester Custom-Kernel)
  big-prim (128x128x32)          28.68 ms   24.26 TFLOPS   0.45x
  generic interleaved (4,3)      57.53 ms   12.10 TFLOPS   0.22x   (2D-Super-Tile hier langsamer:
  generic interleaved (6,6)      48.67 ms   14.30 TFLOPS   0.26x    B (~9 MB/Batch) passt in den 24 MB
  generic interleaved (8,9)      46.86 ms   14.85 TFLOPS   0.27x    L2 -> Swizzle spart nichts, bricht
  generic interleaved (12,9)     47.33 ms   14.70 TFLOPS   0.27x    nur die natuerliche A-Reuse)
"""
