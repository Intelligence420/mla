"""
Analyse-Testing: which numeric formats actually work through ct.mma on a
batched GEMM on THIS machine (NVIDIA GB10, sm_121).

For each candidate dtype we build a batched GEMM (B,M,K)x(B,K,N)->(B,M,N)
through ct.mma and report:
  dtype | compiles? | runs? | correct vs fp32 | rough TFLOP/s | acc dtype | notes

Reference: fp32 torch.bmm (verify-before-trust).
Orientation mirrors the A05 *baseline* straight-GEMM form:
  a2d=(tm,tk), b2d=(tk,tn), acc=ct.mma(a2d,b2d,acc)  -> (tm,tn).
(A06 permutes B + swaps operands only because its output layout is yx; for a
plain m,n output the baseline orientation is the correct one.)
"""

import traceback
import cuda.tile as ct
import torch
import triton

# ---------------------------------------------------------------------------
# Problem size: small, tile-divisible (shared machine -> keep it tiny)
# ---------------------------------------------------------------------------
B, M, K, N = 4, 512, 512, 512
TM, TN, TK = 128, 128, 64          # tile-divisible: 512 % 128 == 0, 512 % 64 == 0
FLOPS = 2 * B * M * N * K


# ---------------------------------------------------------------------------
# Kernels. One per "compute -> accumulate" combination needed.
# Grid = (B, cdiv(M,TM), cdiv(N,TN)); each block makes one (TM,TN) output tile.
# ---------------------------------------------------------------------------

def _make_kernel(acc_dt, cast_dt=None):
    """Build a batched-GEMM kernel.

    acc_dt : cuTile dtype of the accumulator (f16/f32/f64).
    cast_dt: if set, loaded tiles are ct.astype'd to this dtype before mma
             (used for tf32: load f32, cast to tfloat32).
    """
    @ct.kernel
    def k(A, B_, C,
          Md: ct.Constant[int], Nd: ct.Constant[int], Kd: ct.Constant[int],
          tm: ct.Constant[int], tn: ct.Constant[int], tk: ct.Constant[int]):
        pid_b = ct.bid(0)
        pid_m = ct.bid(1)
        pid_n = ct.bid(2)
        zero = ct.PaddingMode.ZERO
        acc = ct.full((tm, tn), 0, dtype=acc_dt)
        for kk in range(ct.cdiv(Kd, tk)):
            a_t = ct.load(A,  index=(pid_b, pid_m, kk),
                          shape=(1, tm, tk), padding_mode=zero)
            b_t = ct.load(B_, index=(pid_b, kk, pid_n),
                          shape=(1, tk, tn), padding_mode=zero)
            a2 = ct.reshape(a_t, (tm, tk))
            b2 = ct.reshape(b_t, (tk, tn))
            if cast_dt is not None:
                a2 = ct.astype(a2, cast_dt)
                b2 = ct.astype(b2, cast_dt)
            acc = ct.mma(a2, b2, acc)
        out = ct.reshape(ct.astype(acc, C.dtype), (1, tm, tn))
        ct.store(C, index=(pid_b, pid_m, pid_n), tile=out)
    return k


# Pre-build the kernel objects (compilation happens lazily at first launch).
K_ACC_F32 = _make_kernel(ct.float32)
K_ACC_F16 = _make_kernel(ct.float16)
K_ACC_F64 = _make_kernel(ct.float64)
K_TF32    = _make_kernel(ct.float32, cast_dt=ct.tfloat32)


def _launch(kernel, A, B_, out_dtype):
    C = torch.empty((B, M, N), device=A.device, dtype=out_dtype)
    grid = (B, ct.cdiv(M, TM), ct.cdiv(N, TN))
    ct.launch(torch.cuda.current_stream().cuda_stream, grid, kernel,
              (A, B_, C, M, N, K, TM, TN, TK))
    return C


# ---------------------------------------------------------------------------
# Candidate specifications. Each builds inputs + picks a kernel + out dtype.
# ---------------------------------------------------------------------------
def _randn(dtype):
    torch.manual_seed(0)
    a = torch.randn(B, M, K, device="cuda", dtype=dtype)
    b = torch.randn(B, K, N, device="cuda", dtype=dtype)
    return a, b


def _randn_fp8(e4m3=True):
    torch.manual_seed(0)
    fp8 = torch.float8_e4m3fn if e4m3 else torch.float8_e5m2
    a = torch.randn(B, M, K, device="cuda", dtype=torch.float16).to(fp8)
    b = torch.randn(B, K, N, device="cuda", dtype=torch.float16).to(fp8)
    return a, b


CANDIDATES = []  # (label, make_inputs, kernel, out_dtype, acc_label, atol, rtol, ref_fp64)

def add(label, mk, kernel, out_dt, acc, atol, rtol, ref_fp64=False):
    CANDIDATES.append((label, mk, kernel, out_dt, acc, atol, rtol, ref_fp64))

add("fp16 -> fp32",       lambda: _randn(torch.float16),  K_ACC_F32, torch.float32, "fp32", 2e-1, 1e-2)
add("bf16 -> fp32",       lambda: _randn(torch.bfloat16), K_ACC_F32, torch.float32, "fp32", 1.0,  2e-2)
add("tf32 -> fp32",       lambda: _randn(torch.float32),  K_TF32,    torch.float32, "fp32", 1.0,  2e-2)
add("fp8e4m3 -> fp32",    lambda: _randn_fp8(True),       K_ACC_F32, torch.float32, "fp32", 8.0,  2e-1)
add("fp8e4m3 -> fp16",    lambda: _randn_fp8(True),       K_ACC_F16, torch.float16, "fp16", 8.0,  2e-1)
add("fp8e5m2 -> fp32",    lambda: _randn_fp8(False),      K_ACC_F32, torch.float32, "fp32", 16.0, 3e-1)
add("fp32 -> fp32",       lambda: _randn(torch.float32),  K_ACC_F32, torch.float32, "fp32", 1e-2, 1e-3)
# fp64 verified against an fp64 reference (a fp32 reference floors at ~1e-4 and
# would spuriously fail a correct fp64 kernel).
add("fp64 -> fp64",       lambda: _randn(torch.float64),  K_ACC_F64, torch.float64, "fp64", 1e-9, 1e-9, ref_fp64=True)


# ---------------------------------------------------------------------------
# Reference + run one candidate
# ---------------------------------------------------------------------------
def reference(a, b, fp64=False):
    if fp64:
        return torch.bmm(a.double(), b.double())
    return torch.bmm(a.float(), b.float())


def run_one(label, mk, kernel, out_dt, acc_label, atol, rtol, ref_fp64=False):
    row = dict(dtype=label, compiles="-", runs="-", correct="-",
               max_err="-", tflops="-", acc=acc_label, note="")
    try:
        a, b = mk()
    except Exception as e:
        row["note"] = f"input build failed: {type(e).__name__}: {e}"
        return row

    ref = reference(a, b, fp64=ref_fp64)

    # compile+run once
    try:
        C = _launch(kernel, a, b, out_dt)
        torch.cuda.synchronize()
        row["compiles"] = "yes"
        row["runs"] = "yes"
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        # Heuristic: a TileError before launch == compile failure.
        if isinstance(e, ct.TileError):
            row["compiles"] = "no"
        else:
            row["compiles"] = "yes"
            row["runs"] = "no"
        row["note"] = msg.replace("\n", " ")[:600]
        return row

    # correctness (keep fp64 precision when the ref is fp64)
    try:
        Cf = C.double() if ref_fp64 else C.float()
        err = (Cf - ref).abs().max().item()
        ok = torch.allclose(Cf, ref, atol=atol, rtol=rtol)
        row["correct"] = "PASS" if ok else "FAIL"
        row["max_err"] = f"{err:.4g}"
        if not ok:
            row["note"] = f"allclose fail (atol={atol},rtol={rtol})"
    except Exception as e:
        row["correct"] = "ERR"
        row["note"] = f"verify err: {type(e).__name__}: {e}"

    # benchmark
    try:
        t_ms = triton.testing.do_bench(lambda: _launch(kernel, a, b, out_dt),
                                       warmup=10, rep=30)
        row["tflops"] = f"{FLOPS / (t_ms * 1e-3) / 1e12:.1f}"
    except Exception as e:
        row["tflops"] = "ERR"
        row["note"] = (row["note"] + f" | bench err: {e}").strip(" |")[:600]

    return row


# ---------------------------------------------------------------------------
def main():
    print(f"# GB10 ct.mma dtype feasibility  |  GEMM B={B} M={M} K={K} N={N}  "
          f"tile=({TM},{TN},{TK})\n")
    dev = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"device={dev} sm_{cap[0]}{cap[1]} torch={torch.__version__} "
          f"cuda_rt={torch.version.cuda} triton={triton.__version__}\n")

    rows = []
    for spec in CANDIDATES:
        print(f">>> {spec[0]}")
        r = run_one(*spec)
        rows.append(r)
        print(f"    compiles={r['compiles']} runs={r['runs']} "
              f"correct={r['correct']} max_err={r['max_err']} "
              f"tflops={r['tflops']} acc={r['acc']}")
        if r["note"]:
            print(f"    note: {r['note']}")

    # table
    hdr = ("dtype", "comp", "run", "correct", "max_err", "TFLOP/s", "acc", "note")
    w = (18, 5, 4, 8, 10, 9, 6, 40)
    line = lambda c: "  ".join(str(x)[:wi].ljust(wi) for x, wi in zip(c, w))
    print("\n" + line(hdr))
    print("-" * 110)
    for r in rows:
        print(line((r["dtype"], r["compiles"], r["runs"], r["correct"],
                    r["max_err"], r["tflops"], r["acc"], r["note"])))


if __name__ == "__main__":
    main()
