"""
Task 1: Tiled Contraction Kernel Variants

Einsum: eabklxy, ecklyz -> eabcxz

Dimensions-Klassifikation
-------------------------
  e        -> Batch (B-Typ)   : in A, B und C
  a, b, x  -> M-Typ           : nur in A und C
  c, z     -> N-Typ           : nur in B und C
  k, l, y  -> K-Typ           : nur in A und B (kontrahiert)

Varianten
---------
  b) GEMM = (x, y, z); sequentialisiere K-Dims (k, l);
     parallelisiere (e, a, b, c).
  c) wie b), aber zusaetzlich auch b-Dim sequentialisiert;
     parallelisiere (e, a, c).
  d) GEMM = (x, y*l, z); l und y per Permute/Reshape gemerged;
     sequentialisiere k; parallelisiere (e, a, b, c).
  e) GEMM = (e, x, y, z) als 3D-mma; sequentialisiere (k, l);
     parallelisiere (a, b, c).

Alle Kernels: FP16-Inputs, FP32-Akkumulator, Output FP16.
Verifikation gegen torch.einsum().
Benchmark mit triton.testing.do_bench.
"""

import cuda.tile as ct
import torch
import triton


# ===========================================================================
# Konstanten — kleine Default-Shapes fuer Verifikation
# ===========================================================================

# Achtung: alle Tensoren < 32 GiB halten.
# Verifikations-Konfiguration: bewusst klein und schief.
DIMS_VERIFY = dict(E=2, A=2, B=2, C=2, K=2, L=2, X=64, Y=64, Z=64)

# GEMM-Tile-Groessen
TILE_X, TILE_Y, TILE_Z = 32, 32, 32

# Tile entlang e fuer Variante e) (3D-mma)
TILE_E = 2


# ===========================================================================
# Variante b) GEMM=(x,y,z), seq (k,l), parallel (e,a,b,c)
# ===========================================================================

@ct.kernel
def kernel_b(A, B, C,
             E:  ct.Constant[int], Ad: ct.Constant[int],
             Bd: ct.Constant[int], Cd: ct.Constant[int],
             K:  ct.Constant[int], L:  ct.Constant[int],
             X:  ct.Constant[int], Y:  ct.Constant[int], Z: ct.Constant[int],
             tx: ct.Constant[int], ty: ct.Constant[int], tz: ct.Constant[int]):
    """C[e,a,b,c,x,z] = sum_{k,l,y} A[e,a,b,k,l,x,y] * B[e,c,k,l,y,z]"""
    # Grid: (E*Ad, Bd*Cd, num_tiles_x * num_tiles_z)
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
                a_tile = ct.load(
                    A,
                    index=(pid_e, pid_a, pid_b, kk, ll, pid_x, yy),
                    shape=(1, 1, 1, 1, 1, tx, ty),
                    padding_mode=zero_pad,
                )
                b_tile = ct.load(
                    B,
                    index=(pid_e, pid_c, kk, ll, yy, pid_z),
                    shape=(1, 1, 1, 1, ty, tz),
                    padding_mode=zero_pad,
                )
                a2d = ct.reshape(a_tile, (tx, ty))
                b2d = ct.reshape(b_tile, (ty, tz))
                acc = ct.mma(a2d, b2d, acc)

    out = ct.reshape(ct.astype(acc, C.dtype), (1, 1, 1, 1, tx, tz))
    ct.store(C, index=(pid_e, pid_a, pid_b, pid_c, pid_x, pid_z), tile=out)


def run_b(A, B, dims, tile=(TILE_X, TILE_Y, TILE_Z)):
    E, Ad, Bd, Cd = dims["E"], dims["A"], dims["B"], dims["C"]
    K, L = dims["K"], dims["L"]
    X, Y, Z = dims["X"], dims["Y"], dims["Z"]
    tx, ty, tz = tile

    C = torch.empty((E, Ad, Bd, Cd, X, Z), device=A.device, dtype=A.dtype)
    grid = (E * Ad, Bd * Cd, ct.cdiv(X, tx) * ct.cdiv(Z, tz))
    ct.launch(
        torch.cuda.current_stream().cuda_stream,
        grid, kernel_b,
        (A, B, C, E, Ad, Bd, Cd, K, L, X, Y, Z, tx, ty, tz),
    )
    return C


# ===========================================================================
# Variante c) GEMM=(x,y,z), seq (k,l,b), parallel (e,a,c)
# ===========================================================================

@ct.kernel
def kernel_c(A, B, C,
             E:  ct.Constant[int], Ad: ct.Constant[int],
             Bd: ct.Constant[int], Cd: ct.Constant[int],
             K:  ct.Constant[int], L:  ct.Constant[int],
             X:  ct.Constant[int], Y:  ct.Constant[int], Z: ct.Constant[int],
             tx: ct.Constant[int], ty: ct.Constant[int], tz: ct.Constant[int]):
    """Wie kernel_b, aber b-Dim als sequentielle Innenschleife pro Block.
    Block produziert Bd Output-Tiles entlang b. B-Tile haengt nicht von b ab
    -> L2-Reuse zwischen den b-Iterationen.
    """
    # Grid: (E*Ad, Cd, num_tiles_x * num_tiles_z)
    bid_eA = ct.bid(0)
    bid_c  = ct.bid(1)
    bid_xz = ct.bid(2)

    pid_e = bid_eA // Ad
    pid_a = bid_eA %  Ad
    pid_c = bid_c

    num_tiles_z = ct.cdiv(Z, tz)
    pid_x = bid_xz // num_tiles_z
    pid_z = bid_xz %  num_tiles_z

    num_tiles_y = ct.cdiv(Y, ty)
    zero_pad = ct.PaddingMode.ZERO

    for bb in range(Bd):
        acc = ct.full((tx, tz), 0, dtype=ct.float32)
        for kk in range(K):
            for ll in range(L):
                for yy in range(num_tiles_y):
                    a_tile = ct.load(
                        A,
                        index=(pid_e, pid_a, bb, kk, ll, pid_x, yy),
                        shape=(1, 1, 1, 1, 1, tx, ty),
                        padding_mode=zero_pad,
                    )
                    b_tile = ct.load(
                        B,
                        index=(pid_e, pid_c, kk, ll, yy, pid_z),
                        shape=(1, 1, 1, 1, ty, tz),
                        padding_mode=zero_pad,
                    )
                    a2d = ct.reshape(a_tile, (tx, ty))
                    b2d = ct.reshape(b_tile, (ty, tz))
                    acc = ct.mma(a2d, b2d, acc)

        out = ct.reshape(ct.astype(acc, C.dtype), (1, 1, 1, 1, tx, tz))
        ct.store(C, index=(pid_e, pid_a, bb, pid_c, pid_x, pid_z), tile=out)


def run_c(A, B, dims, tile=(TILE_X, TILE_Y, TILE_Z)):
    E, Ad, Bd, Cd = dims["E"], dims["A"], dims["B"], dims["C"]
    K, L = dims["K"], dims["L"]
    X, Y, Z = dims["X"], dims["Y"], dims["Z"]
    tx, ty, tz = tile

    C = torch.empty((E, Ad, Bd, Cd, X, Z), device=A.device, dtype=A.dtype)
    grid = (E * Ad, Cd, ct.cdiv(X, tx) * ct.cdiv(Z, tz))
    ct.launch(
        torch.cuda.current_stream().cuda_stream,
        grid, kernel_c,
        (A, B, C, E, Ad, Bd, Cd, K, L, X, Y, Z, tx, ty, tz),
    )
    return C


# ===========================================================================
# Variante d) GEMM=(x, y*l, z), l und y gemerged
# ===========================================================================

@ct.kernel
def kernel_d(A, B, C,
             E:  ct.Constant[int], Ad: ct.Constant[int],
             Bd: ct.Constant[int], Cd: ct.Constant[int],
             K:  ct.Constant[int], L:  ct.Constant[int],
             X:  ct.Constant[int], Y:  ct.Constant[int], Z: ct.Constant[int],
             tx: ct.Constant[int], ty: ct.Constant[int], tz: ct.Constant[int]):
    """GEMM-K = y*l: pro K-Schleifen-Iteration werden L Y-Slices gleichzeitig
    in den mma gepackt. Permutation (L, X, Y) -> (X, L, Y) auf A noetig,
    weil A's Layout L vor X hat; B ist schon (L, Y, Z) und kann direkt
    flach reshaped werden.
    """
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
    tly = L * ty  # gemergte K-Dim Groesse pro mma

    for kk in range(K):
        for yy in range(num_tiles_y):
            # A: load (1,1,1,1, L, tx, ty) -> reshape (L, tx, ty)
            #    -> permute (tx, L, ty) -> reshape (tx, L*ty)
            a_tile = ct.load(
                A,
                index=(pid_e, pid_a, pid_b, kk, 0, pid_x, yy),
                shape=(1, 1, 1, 1, L, tx, ty),
                padding_mode=zero_pad,
            )
            a_tile = ct.reshape(a_tile, (L, tx, ty))
            a_tile = ct.permute(a_tile, (1, 0, 2))
            a_tile = ct.reshape(a_tile, (tx, tly))

            # B: load (1,1,1, L, ty, tz) -> reshape (L*ty, tz) (kein Permute noetig)
            b_tile = ct.load(
                B,
                index=(pid_e, pid_c, kk, 0, yy, pid_z),
                shape=(1, 1, 1, L, ty, tz),
                padding_mode=zero_pad,
            )
            b_tile = ct.reshape(b_tile, (tly, tz))

            acc = ct.mma(a_tile, b_tile, acc)

    out = ct.reshape(ct.astype(acc, C.dtype), (1, 1, 1, 1, tx, tz))
    ct.store(C, index=(pid_e, pid_a, pid_b, pid_c, pid_x, pid_z), tile=out)


def run_d(A, B, dims, tile=(TILE_X, TILE_Y, TILE_Z)):
    E, Ad, Bd, Cd = dims["E"], dims["A"], dims["B"], dims["C"]
    K, L = dims["K"], dims["L"]
    X, Y, Z = dims["X"], dims["Y"], dims["Z"]
    tx, ty, tz = tile

    C = torch.empty((E, Ad, Bd, Cd, X, Z), device=A.device, dtype=A.dtype)
    grid = (E * Ad, Bd * Cd, ct.cdiv(X, tx) * ct.cdiv(Z, tz))
    ct.launch(
        torch.cuda.current_stream().cuda_stream,
        grid, kernel_d,
        (A, B, C, E, Ad, Bd, Cd, K, L, X, Y, Z, tx, ty, tz),
    )
    return C


# ===========================================================================
# Variante e) GEMM=(e,x,y,z) als 3D-mma
# ===========================================================================

@ct.kernel
def kernel_e(A, B, C,
             E:  ct.Constant[int], Ad: ct.Constant[int],
             Bd: ct.Constant[int], Cd: ct.Constant[int],
             K:  ct.Constant[int], L:  ct.Constant[int],
             X:  ct.Constant[int], Y:  ct.Constant[int], Z: ct.Constant[int],
             te: ct.Constant[int],
             tx: ct.Constant[int], ty: ct.Constant[int], tz: ct.Constant[int]):
    """3D-mma: e wird als Batch-Dim direkt in ct.mma uebergeben.
    Akkumulator hat shape (te, tx, tz).
    """
    # Grid: (num_tiles_e * Ad, Bd*Cd, num_tiles_x * num_tiles_z)
    num_tiles_e = ct.cdiv(E, te)
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

    acc = ct.full((te, tx, tz), 0, dtype=ct.float32)
    num_tiles_y = ct.cdiv(Y, ty)
    zero_pad = ct.PaddingMode.ZERO

    for kk in range(K):
        for ll in range(L):
            for yy in range(num_tiles_y):
                a_tile = ct.load(
                    A,
                    index=(pid_e, pid_a, pid_b, kk, ll, pid_x, yy),
                    shape=(te, 1, 1, 1, 1, tx, ty),
                    padding_mode=zero_pad,
                )
                b_tile = ct.load(
                    B,
                    index=(pid_e, pid_c, kk, ll, yy, pid_z),
                    shape=(te, 1, 1, 1, ty, tz),
                    padding_mode=zero_pad,
                )
                a3d = ct.reshape(a_tile, (te, tx, ty))
                b3d = ct.reshape(b_tile, (te, ty, tz))
                acc = ct.mma(a3d, b3d, acc)

    out = ct.reshape(ct.astype(acc, C.dtype), (te, 1, 1, 1, tx, tz))
    ct.store(C, index=(pid_e, pid_a, pid_b, pid_c, pid_x, pid_z), tile=out)


def run_e(A, B, dims, tile=(TILE_X, TILE_Y, TILE_Z), te=TILE_E):
    E, Ad, Bd, Cd = dims["E"], dims["A"], dims["B"], dims["C"]
    K, L = dims["K"], dims["L"]
    X, Y, Z = dims["X"], dims["Y"], dims["Z"]
    tx, ty, tz = tile

    C = torch.empty((E, Ad, Bd, Cd, X, Z), device=A.device, dtype=A.dtype)
    grid = (ct.cdiv(E, te) * Ad, Bd * Cd, ct.cdiv(X, tx) * ct.cdiv(Z, tz))
    ct.launch(
        torch.cuda.current_stream().cuda_stream,
        grid, kernel_e,
        (A, B, C, E, Ad, Bd, Cd, K, L, X, Y, Z, te, tx, ty, tz),
    )
    return C


# ===========================================================================
# Verifikation
# ===========================================================================

def _make_inputs(dims, dtype=torch.float16, device="cuda"):
    torch.manual_seed(0)
    E, Ad, Bd, Cd = dims["E"], dims["A"], dims["B"], dims["C"]
    K, L = dims["K"], dims["L"]
    X, Y, Z = dims["X"], dims["Y"], dims["Z"]
    A = torch.randn(E, Ad, Bd, K, L, X, Y, dtype=dtype, device=device)
    B = torch.randn(E, Cd,    K, L, Y, Z, dtype=dtype, device=device)
    return A, B


def reference(A, B):
    """torch.einsum-Referenz; Akkumulation in FP32, Result auf Input-dtype."""
    Af = A.to(torch.float32)
    Bf = B.to(torch.float32)
    Cf = torch.einsum("eabklxy,ecklyz->eabcxz", Af, Bf)
    return Cf.to(A.dtype)


def _check(name, runner, A, B, ref, dims, tol=(2e-1, 2e-2)):
    out = runner(A, B, dims)
    atol, rtol = tol
    ok = torch.allclose(out, ref, atol=atol, rtol=rtol)
    err = (out.float() - ref.float()).abs().max().item()
    print(f"  {name:<10}  allclose={ok}   max_abs_err={err:.4f}")
    assert ok, f"{name} mismatch"


def verify():
    dims = DIMS_VERIFY
    A, B = _make_inputs(dims)
    ref = reference(A, B)
    print(f"  Shapes: A={tuple(A.shape)}, B={tuple(B.shape)}, "
          f"ref={tuple(ref.shape)}")
    _check("kernel_b", run_b, A, B, ref, dims)
    _check("kernel_c", run_c, A, B, ref, dims)
    _check("kernel_d", run_d, A, B, ref, dims)
    _check("kernel_e", run_e, A, B, ref, dims)


# ===========================================================================
# FLOPs / Hilfsfunktionen
# ===========================================================================

def flops_count(dims):
    """2 * (Produkt aller einzigartigen Indizes) FLOPs pro Kontraktion."""
    E, Ad, Bd, Cd = dims["E"], dims["A"], dims["B"], dims["C"]
    K, L = dims["K"], dims["L"]
    X, Y, Z = dims["X"], dims["Y"], dims["Z"]
    return 2 * E * Ad * Bd * Cd * K * L * X * Y * Z


def tflops(dims, time_ms):
    return flops_count(dims) / (time_ms * 1e-3) / 1e12


def _bench(runner, A, B, dims, n_warmup=10, n_runs=50):
    return triton.testing.do_bench(
        lambda: runner(A, B, dims),
        warmup=n_warmup, rep=n_runs,
    )


# ===========================================================================
# Benchmark
# ===========================================================================

def bench_compare(label, dims, runners):
    """Run alle uebergebenen runner-Tupel (name, fn) auf dims und drucke
    Laufzeiten/TFLOPS."""
    A, B = _make_inputs(dims)
    print(f"\n  {label}")
    print(f"    dims = {dims}")
    print(f"    FLOPs = {flops_count(dims):.3e}")
    results = {}
    for name, fn in runners:
        t = _bench(fn, A, B, dims)
        f = tflops(dims, t)
        results[name] = (t, f)
        print(f"    {name:<10} {t:8.4f} ms   {f:7.3f} TFLOPS")
    return results


def benchmark():
    """b) vs c): Ein Setting wo b gewinnt, eines wo c gewinnt.
       b) vs d): Ein Setting wo b gewinnt, eines wo d gewinnt.
    """
    # b vs c: c gewinnt, wenn |b| gross ist und |a|*|c| klein, sodass
    # B-Tile-Reuse durch Sequentialisierung der b-Schleife die L2-Hits
    # erhoeht UND das Grid in b) zu fragmentiert waere.
    # b vs c: b gewinnt, wenn |b| klein ist (kaum Reuse) und |a|*|c|
    # gross genug, sodass die zusaetzliche Parallelitaet hilft.
    print("\n=== Vergleich b) vs c) ===")
    cfg_b_wins = dict(E=2, A=8, B=2, C=8, K=2, L=2, X=128, Y=64, Z=128)
    cfg_c_wins = dict(E=1, A=2, B=16, C=2, K=2, L=2, X=128, Y=64, Z=128)
    bench_compare("Setting wo b) vorne ist (b klein, a*c gross)",
                  cfg_b_wins, [("b)", run_b), ("c)", run_c)])
    bench_compare("Setting wo c) vorne ist (b gross, a*c klein)",
                  cfg_c_wins, [("b)", run_b), ("c)", run_c)])

    # b vs d: d gewinnt, wenn L gross ist (mehr Arbeit pro mma; groessere
    # K-Dim erhoeht arithmetic intensity).
    # b vs d: b gewinnt, wenn L = 1 (kein Vorteil durch Merge, aber Permute
    # kostet) bzw. L sehr klein.
    print("\n=== Vergleich b) vs d) ===")
    cfg_d_wins = dict(E=2, A=4, B=2, C=4, K=2, L=8, X=128, Y=32, Z=128)
    cfg_b_wins_vs_d = dict(E=2, A=4, B=2, C=4, K=8, L=1, X=128, Y=64, Z=128)
    bench_compare("Setting wo d) vorne ist (L gross, Y klein)",
                  cfg_d_wins, [("b)", run_b), ("d)", run_d)])
    bench_compare("Setting wo b) vorne ist (L=1, Y gross)",
                  cfg_b_wins_vs_d, [("b)", run_b), ("d)", run_d)])

    # e) - Quervergleich auf einer mittleren Konfiguration
    print("\n=== Variante e) ===")
    cfg_e = dict(E=4, A=2, B=2, C=2, K=2, L=2, X=128, Y=64, Z=128)
    bench_compare("Quervergleich b) / d) / e)",
                  cfg_e,
                  [("b)", run_b), ("d)", run_d), ("e)", run_e)])


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("Task 1a: Dimension-Klassifikation")
    print("  e            -> Batch (B-Typ)")
    print("  a, b, x      -> M-Typ")
    print("  c, z         -> N-Typ")
    print("  k, l, y      -> K-Typ (kontrahiert)")

    print("\nTask 1 b)/c)/d)/e): Verifikation gegen torch.einsum")
    verify()

    print("\nTask 1: Benchmark")
    benchmark()
