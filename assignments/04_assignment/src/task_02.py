"""
Task 2: Kernel Fusion

Einsum: eabklxy, ecklyz -> eabcxz, plus eine elementwise Multiplikation
mit einem Tensor D der Form (e, a, b, c, x, z).

Wir vergleichen drei Varianten:
  - kernel_b              : Kontraktion ohne Fusion (Variante b) aus Task 1)
  - kernel_fused          : Kontraktion + elementwise mul mit D in einem Pass
                            (D wird im Akkumulator-Tile multipliziert, bevor
                             zurueckgeschrieben wird)
  - kernel_elemwise       : reine elementwise mul auf vorhandenem C-Tensor

Vergleich:
  - Korrektheit gegen torch.einsum(...) * D
  - Laufzeit fused vs. (kernel_b + kernel_elemwise)

Tensor-Groessen so gewaehlt, dass die FLOPs der Kontraktion etwa
2 * 2048^3 betragen (~1.7e10).

Alle Kernels: FP16 Inputs/Outputs, FP32 Akkumulator.
"""

import os

import cuda.tile as ct
import matplotlib.pyplot as plt
import torch
import triton


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ===========================================================================
# Kernel 1: Kontraktion (wie Task 1b))
# ===========================================================================

@ct.kernel
def kernel_contract(A, B, C,
                    E:  ct.Constant[int], Ad: ct.Constant[int],
                    Bd: ct.Constant[int], Cd: ct.Constant[int],
                    K:  ct.Constant[int], L:  ct.Constant[int],
                    X:  ct.Constant[int], Y:  ct.Constant[int], Z: ct.Constant[int],
                    tx: ct.Constant[int], ty: ct.Constant[int], tz: ct.Constant[int]):
    """C[e,a,b,c,x,z] = sum_{k,l,y} A[e,a,b,k,l,x,y] * B[e,c,k,l,y,z]"""
    bid_eA = ct.bid(0)
    bid_BC = ct.bid(1)
    bid_xz = ct.bid(2)

    pid_e = bid_eA // Ad
    pid_a = bid_eA %  Ad
    pid_b = bid_BC // Cd
    pid_c = bid_BC %  Cd

    num_tiles_z = ct.cdiv(Z, tz)
    pid_x = bid_xz // num_tiles_z
    pid_z = bid_xz %  num_tiles_z

    acc = ct.full((tx, tz), 0, dtype=ct.float32)
    num_tiles_y = ct.cdiv(Y, ty)
    zero_pad = ct.PaddingMode.ZERO

    for kk in range(K):
        for ll in range(L):
            for yy in range(num_tiles_y):
                a_tile = ct.load(A,
                                 index=(pid_e, pid_a, pid_b, kk, ll, pid_x, yy),
                                 shape=(1, 1, 1, 1, 1, tx, ty),
                                 padding_mode=zero_pad)
                b_tile = ct.load(B,
                                 index=(pid_e, pid_c, kk, ll, yy, pid_z),
                                 shape=(1, 1, 1, 1, ty, tz),
                                 padding_mode=zero_pad)
                a2d = ct.reshape(a_tile, (tx, ty))
                b2d = ct.reshape(b_tile, (ty, tz))
                acc = ct.mma(a2d, b2d, acc)

    out = ct.reshape(ct.astype(acc, C.dtype), (1, 1, 1, 1, tx, tz))
    ct.store(C, index=(pid_e, pid_a, pid_b, pid_c, pid_x, pid_z), tile=out)


# ===========================================================================
# Kernel 2: fused (Kontraktion + elementwise Multiplikation mit D)
# ===========================================================================

@ct.kernel
def kernel_fused(A, B, D, C,
                 E:  ct.Constant[int], Ad: ct.Constant[int],
                 Bd: ct.Constant[int], Cd: ct.Constant[int],
                 K:  ct.Constant[int], L:  ct.Constant[int],
                 X:  ct.Constant[int], Y:  ct.Constant[int], Z: ct.Constant[int],
                 tx: ct.Constant[int], ty: ct.Constant[int], tz: ct.Constant[int]):
    """C[e,a,b,c,x,z] = D[e,a,b,c,x,z] * sum_{k,l,y} A[..] * B[..]"""
    bid_eA = ct.bid(0)
    bid_BC = ct.bid(1)
    bid_xz = ct.bid(2)

    pid_e = bid_eA // Ad
    pid_a = bid_eA %  Ad
    pid_b = bid_BC // Cd
    pid_c = bid_BC %  Cd

    num_tiles_z = ct.cdiv(Z, tz)
    pid_x = bid_xz // num_tiles_z
    pid_z = bid_xz %  num_tiles_z

    acc = ct.full((tx, tz), 0, dtype=ct.float32)
    num_tiles_y = ct.cdiv(Y, ty)
    zero_pad = ct.PaddingMode.ZERO

    for kk in range(K):
        for ll in range(L):
            for yy in range(num_tiles_y):
                a_tile = ct.load(A,
                                 index=(pid_e, pid_a, pid_b, kk, ll, pid_x, yy),
                                 shape=(1, 1, 1, 1, 1, tx, ty),
                                 padding_mode=zero_pad)
                b_tile = ct.load(B,
                                 index=(pid_e, pid_c, kk, ll, yy, pid_z),
                                 shape=(1, 1, 1, 1, ty, tz),
                                 padding_mode=zero_pad)
                a2d = ct.reshape(a_tile, (tx, ty))
                b2d = ct.reshape(b_tile, (ty, tz))
                acc = ct.mma(a2d, b2d, acc)

    # Fusion: D einmal als Tile laden und mit acc (FP32) multiplizieren
    d_tile = ct.load(D,
                     index=(pid_e, pid_a, pid_b, pid_c, pid_x, pid_z),
                     shape=(1, 1, 1, 1, tx, tz),
                     padding_mode=zero_pad)
    d2d = ct.astype(ct.reshape(d_tile, (tx, tz)), ct.float32)
    fused = acc * d2d

    out = ct.reshape(ct.astype(fused, C.dtype), (1, 1, 1, 1, tx, tz))
    ct.store(C, index=(pid_e, pid_a, pid_b, pid_c, pid_x, pid_z), tile=out)


# ===========================================================================
# Kernel 3: reine elementwise Multiplikation
# ===========================================================================

@ct.kernel
def kernel_elemwise(C, D,
                    E:  ct.Constant[int], Ad: ct.Constant[int],
                    Bd: ct.Constant[int], Cd: ct.Constant[int],
                    X:  ct.Constant[int], Z: ct.Constant[int],
                    tx: ct.Constant[int], tz: ct.Constant[int]):
    """C[e,a,b,c,x,z] *= D[e,a,b,c,x,z]"""
    bid_eA = ct.bid(0)
    bid_BC = ct.bid(1)
    bid_xz = ct.bid(2)

    pid_e = bid_eA // Ad
    pid_a = bid_eA %  Ad
    pid_b = bid_BC // Cd
    pid_c = bid_BC %  Cd

    num_tiles_z = ct.cdiv(Z, tz)
    pid_x = bid_xz // num_tiles_z
    pid_z = bid_xz %  num_tiles_z

    zero_pad = ct.PaddingMode.ZERO

    c_tile = ct.load(C,
                     index=(pid_e, pid_a, pid_b, pid_c, pid_x, pid_z),
                     shape=(1, 1, 1, 1, tx, tz),
                     padding_mode=zero_pad)
    d_tile = ct.load(D,
                     index=(pid_e, pid_a, pid_b, pid_c, pid_x, pid_z),
                     shape=(1, 1, 1, 1, tx, tz),
                     padding_mode=zero_pad)
    c2d = ct.reshape(c_tile, (tx, tz))
    d2d = ct.reshape(d_tile, (tx, tz))
    res = ct.astype(c2d, ct.float32) * ct.astype(d2d, ct.float32)

    out = ct.reshape(ct.astype(res, C.dtype), (1, 1, 1, 1, tx, tz))
    ct.store(C, index=(pid_e, pid_a, pid_b, pid_c, pid_x, pid_z), tile=out)


# ===========================================================================
# Host-Funktionen
# ===========================================================================

def _grid(dims, tx, tz):
    E, Ad, Bd, Cd = dims["E"], dims["A"], dims["B"], dims["C"]
    X, Z = dims["X"], dims["Z"]
    return (E * Ad, Bd * Cd, ct.cdiv(X, tx) * ct.cdiv(Z, tz))


def run_contract(A, B, dims, tile=(64, 32, 64)):
    E, Ad, Bd, Cd = dims["E"], dims["A"], dims["B"], dims["C"]
    K, L = dims["K"], dims["L"]
    X, Y, Z = dims["X"], dims["Y"], dims["Z"]
    tx, ty, tz = tile

    C = torch.empty((E, Ad, Bd, Cd, X, Z), device=A.device, dtype=A.dtype)
    ct.launch(torch.cuda.current_stream().cuda_stream,
              _grid(dims, tx, tz), kernel_contract,
              (A, B, C, E, Ad, Bd, Cd, K, L, X, Y, Z, tx, ty, tz))
    return C


def run_fused(A, B, D, dims, tile=(64, 32, 64)):
    E, Ad, Bd, Cd = dims["E"], dims["A"], dims["B"], dims["C"]
    K, L = dims["K"], dims["L"]
    X, Y, Z = dims["X"], dims["Y"], dims["Z"]
    tx, ty, tz = tile

    C = torch.empty((E, Ad, Bd, Cd, X, Z), device=A.device, dtype=A.dtype)
    ct.launch(torch.cuda.current_stream().cuda_stream,
              _grid(dims, tx, tz), kernel_fused,
              (A, B, D, C, E, Ad, Bd, Cd, K, L, X, Y, Z, tx, ty, tz))
    return C


def run_elemwise(C, D, dims, tile=(64, 64)):
    E, Ad, Bd, Cd = dims["E"], dims["A"], dims["B"], dims["C"]
    X, Z = dims["X"], dims["Z"]
    tx, tz = tile

    ct.launch(torch.cuda.current_stream().cuda_stream,
              _grid(dims, tx, tz), kernel_elemwise,
              (C, D, E, Ad, Bd, Cd, X, Z, tx, tz))
    return C


def run_sequential(A, B, D, dims):
    """Naiver Pfad: erst Kontraktion, dann elementwise Mul (zwei Kernel-Launches)."""
    C = run_contract(A, B, dims)
    C = run_elemwise(C, D, dims)
    return C


# ===========================================================================
# Inputs / Verifikation
# ===========================================================================

def make_inputs(dims, dtype=torch.float16, device="cuda"):
    torch.manual_seed(0)
    E, Ad, Bd, Cd = dims["E"], dims["A"], dims["B"], dims["C"]
    K, L = dims["K"], dims["L"]
    X, Y, Z = dims["X"], dims["Y"], dims["Z"]
    A = torch.randn(E, Ad, Bd, K, L, X, Y, dtype=dtype, device=device)
    B = torch.randn(E, Cd,    K, L, Y, Z, dtype=dtype, device=device)
    D = torch.randn(E, Ad, Bd, Cd, X, Z, dtype=dtype, device=device)
    return A, B, D


def reference(A, B, D):
    Cf = torch.einsum("eabklxy,ecklyz->eabcxz",
                      A.to(torch.float32), B.to(torch.float32))
    return (Cf * D.to(torch.float32)).to(A.dtype)


def verify():
    dims = dict(E=2, A=2, B=2, C=2, K=2, L=2, X=64, Y=64, Z=64)
    A, B, D = make_inputs(dims)
    ref = reference(A, B, D)
    print(f"  Shapes: A={tuple(A.shape)}, B={tuple(B.shape)}, "
          f"D={tuple(D.shape)}, ref={tuple(ref.shape)}")

    # fused
    out_fused = run_fused(A, B, D, dims)
    ok_f = torch.allclose(out_fused, ref, atol=2e-1, rtol=2e-2)
    err_f = (out_fused.float() - ref.float()).abs().max().item()
    print(f"  fused        allclose={ok_f}   max_abs_err={err_f:.4f}")
    assert ok_f, "fused mismatch"

    # sequentiell (kernel_contract + kernel_elemwise)
    out_seq = run_sequential(A, B, D, dims)
    ok_s = torch.allclose(out_seq, ref, atol=2e-1, rtol=2e-2)
    err_s = (out_seq.float() - ref.float()).abs().max().item()
    print(f"  sequentiell  allclose={ok_s}   max_abs_err={err_s:.4f}")
    assert ok_s, "sequentiell mismatch"


# ===========================================================================
# FLOPs / Helfer
# ===========================================================================

def contract_flops(dims):
    E, Ad, Bd, Cd = dims["E"], dims["A"], dims["B"], dims["C"]
    K, L = dims["K"], dims["L"]
    X, Y, Z = dims["X"], dims["Y"], dims["Z"]
    return 2 * E * Ad * Bd * Cd * K * L * X * Y * Z


def out_size(dims):
    E, Ad, Bd, Cd = dims["E"], dims["A"], dims["B"], dims["C"]
    return E * Ad * Bd * Cd * dims["X"] * dims["Z"]


def bench(fn, n_warmup=10, n_runs=50):
    return triton.testing.do_bench(fn, warmup=n_warmup, rep=n_runs)


# ===========================================================================
# Benchmark: fused vs. sequentiell, FLOPs ~ 2 * 2048^3
# ===========================================================================

def benchmark():
    """Tensor-Groessen so, dass die Kontraktion ~ 2 * 2048^3 FLOPs hat.
       2 * 2048^3 ≈ 1.717e10 FLOPs.
       Hier: E=2, A=B=C=2, K=8, L=4, X=Y=Z=256 → 2*2*2*2*2*8*4*256^3
                                              ≈ 1.72e10 FLOPs ≈ 2 * 2048^3
       Speicher (FP16): A ≈ 67 MB, B ≈ 33 MB, C/D ≈ 4 MB jeweils → unkritisch.
    """
    dims = dict(E=2, A=2, B=2, C=2, K=8, L=4, X=256, Y=256, Z=256)
    A, B, D = make_inputs(dims)

    print(f"\n  dims = {dims}")
    print(f"  Kontraktions-FLOPs = {contract_flops(dims):.3e}")
    print(f"  (Referenz: 2 * 2048^3 = {2 * 2048**3:.3e})")
    print(f"  Output-Elemente C = {out_size(dims):.3e}")

    # Vorab-allokierter Output fuer kernel_elemwise
    C_pre = run_contract(A, B, dims)

    t_contract = bench(lambda: run_contract(A, B, dims))
    t_elem     = bench(lambda: run_elemwise(C_pre.clone(), D, dims))
    t_seq      = bench(lambda: run_sequential(A, B, D, dims))
    t_fused    = bench(lambda: run_fused(A, B, D, dims))

    print(f"\n  kernel_contract    {t_contract:8.4f} ms")
    print(f"  kernel_elemwise    {t_elem:8.4f} ms")
    print(f"  sequentiell        {t_seq:8.4f} ms   (= contract + elemwise)")
    print(f"  fused              {t_fused:8.4f} ms")

    speedup = t_seq / t_fused
    print(f"\n  Speedup fused vs. sequentiell: {speedup:.3f}x")

    return dims, dict(contract=t_contract, elemwise=t_elem,
                      sequentiell=t_seq, fused=t_fused)


# ===========================================================================
# Plot
# ===========================================================================

def plot_results(dims, results, path=None):
    if path is None:
        path = SCRIPT_DIR

    names = ["contract", "elemwise", "sequentiell", "fused"]
    times = [results[n] for n in names]
    colors = ["#4C72B0", "#DD8452", "#8172B2", "#55A868"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, times, color=colors)
    ax.set_ylabel("Laufzeit (ms)")
    ax.set_title(f"Task 2: Fusion vs. sequentiell\n"
                 f"FLOPs ≈ {contract_flops(dims):.2e}")
    ax.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars, times):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()

    p = os.path.join(path, "task02_fusion.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"\n  Plot: {p}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("Task 2: Verifikation")
    verify()

    print("\nTask 2: Benchmark fused vs. sequentiell")
    dims, results = benchmark()

    print("\nTask 2: Plot")
    plot_results(dims, results)

"""Ergebnisse
(.venv) mla07@flambe:~/mla$ python3 assignments/04_assignment/src/task_02.py
Task 2: Verifikation
  Shapes: A=(2, 2, 2, 2, 2, 64, 64), B=(2, 2, 2, 2, 64, 64), D=(2, 2, 2, 2, 64, 64), ref=(2, 2, 2, 2, 64, 64)
  fused        allclose=True   max_abs_err=0.0312
  sequentiell  allclose=True   max_abs_err=0.0625

Task 2: Benchmark fused vs. sequentiell

  dims = {'E': 2, 'A': 2, 'B': 2, 'C': 2, 'K': 8, 'L': 4, 'X': 256, 'Y': 256, 'Z': 256}
  Kontraktions-FLOPs = 1.718e+10
  (Referenz: 2 * 2048^3 = 1.718e+10)
  Output-Elemente C = 1.049e+06

  kernel_contract     12.8287 ms
  kernel_elemwise      0.0665 ms
  sequentiell         12.8383 ms   (= contract + elemwise)
  fused               13.0479 ms

  Speedup fused vs. sequentiell: 0.984x
"""
