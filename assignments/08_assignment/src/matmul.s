  .file "matmul.s"
  .section .text.matmul,"ax",@progbits
  .globl matmul
  .p2align 4
  .type matmul,@function
matmul:
// Computes out += in0 * in1
//
// L1 tensor views (zero-initialised at NPU setup):
//   p=2, q=2, r=8, m=8, n=8, k=8
//   in0: prmk   (BF16, stride bytes: p=1024, r=128, m=16, k=2)
//   in1: rqkn   (BF16, stride bytes: r=256,  q=128, k=16, n=2)
//   out: pqmn   (BF16, stride bytes: p=256,  q=128, m=16, n=2)
//
// Calling convention (cf. A07/custom_vadd.s):
//   p0 = &in0,  p1 = &in1,  p2 = &out
//
// Register allocation (Task 3):
//   dm0..dm3 : four 8x8 FP32 output accumulators (one per (p,q))
//   dm4      : BF16->FP32 staging accumulator for the BF16->BFP16 path
//   ex0, ex1 : BFP16 operands for in0 at p=0 / p=1 (current r-step)
//   ex2, ex3 : BFP16 operands for in1 at q=0 / q=1 (current r-step)
//   p0       : streams in0[p=0, *], post-increments by +64 per half-load
//   p3       : streams in0[p=1, *], post-increments by +64 per half-load
//   p1       : streams in1, post-increments by +64 per half-load
//   p2       : out base (static)
//   r0       : vmac.f modifier (#780 selects bfp16ebs8 8x8x8 mode, Folie 5)
//
// Indexed-offset range observed empirically (Peano error):
//   getSImmOpValueXStep<4, 64u>  →  4-bit signed × stride 64  →  [-512, 511].
// We therefore avoid offsets ≥ 512; every block walks the pointer via
// two consecutive +64 post-increments.
//
// Latency assumptions (verify against compiled aie::mmul reference):
//   vlda.conv = 4   (from A07, Task 4)
//   vconv     = 2   (M-slot vector move-with-conversion, conservative guess)
//   vmac.f    = 6   (assumed identical to vadd.f from A07)
//
// Schedule style: one slot active per VLIW cycle (other slots auto-nop).
// Explicit `nop` lines pad latencies. Correctness-first draft; see
// Task 6 for overlap-based optimisation potential.

  // ── Setup ──────────────────────────────────────────────────────────
  // Compute p3 = p0 + 1024 (start of in0[p=1, *]).
  mov     r1, p0
  movxm   r2, #1024
  add     r1, r1, r2
  mov     p3, r1

  movxm   r0, #780                        // BFP16 8x8x8 mode modifier
                                          // (mova has only 8-bit imm; 780 needs movxm)

  // ── Out initialisation: load BF16 zeros from L1, convert to FP32 ──
  // Scratchpad is zero-initialised (A08, Data Layout section). Offsets
  // here stay ≤ #448 → within indexed-offset range.
  vlda.conv.fp32.bf16  cml0, [p2, #0]
  vlda.conv.fp32.bf16  cmh0, [p2, #64]
  vlda.conv.fp32.bf16  cml1, [p2, #128]
  vlda.conv.fp32.bf16  cmh1, [p2, #192]
  vlda.conv.fp32.bf16  cml2, [p2, #256]
  vlda.conv.fp32.bf16  cmh2, [p2, #320]
  vlda.conv.fp32.bf16  cml3, [p2, #384]
  vlda.conv.fp32.bf16  cmh3, [p2, #448]
  // vlda latency 4 elapses well before first vmac reads dm0..dm3.

  // ── Main reduction loop, fully unrolled (8 × r-step) ──────────────
  //
  // Per r-step (28 cycles):
  //   4 × (2 vlda + 3 nop + 1 vconv) = 24 cycles  (block load + convert)
  //   4 × vmac                                      4 cycles
  // Each block uses two +64 post-increments; the pointer naturally
  // ends up at the start of the next block (no extra arithmetic).

  // ════════════════════════════════════════════════════════════════
  // r = 0
  // ════════════════════════════════════════════════════════════════
  // Block A: in0[0,0] -> ex0, advance p0 by +128
  vlda.conv.fp32.bf16  cml4, [p0], #64
  vlda.conv.fp32.bf16  cmh4, [p0], #64
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex0, dm4
  // Block B: in0[1,0] -> ex1, advance p3 by +128
  vlda.conv.fp32.bf16  cml4, [p3], #64
  vlda.conv.fp32.bf16  cmh4, [p3], #64
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex1, dm4
  // Block C: in1[0,0] -> ex2, advance p1 by +128
  vlda.conv.fp32.bf16  cml4, [p1], #64
  vlda.conv.fp32.bf16  cmh4, [p1], #64
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex2, dm4
  // Block D: in1[0,1] -> ex3, advance p1 by another +128 (total +256/r)
  vlda.conv.fp32.bf16  cml4, [p1], #64
  vlda.conv.fp32.bf16  cmh4, [p1], #64
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex3, dm4
  // 4 MACs for r=0
  vmac.f  dm0, dm0, ex0, ex2, r0
  vmac.f  dm1, dm1, ex0, ex3, r0
  vmac.f  dm2, dm2, ex1, ex2, r0
  vmac.f  dm3, dm3, ex1, ex3, r0

  // ════════════════════════════════════════════════════════════════
  // r = 1 .. 7 — identical pattern, pointers continue streaming
  // ════════════════════════════════════════════════════════════════
  // r = 1
  vlda.conv.fp32.bf16  cml4, [p0], #64
  vlda.conv.fp32.bf16  cmh4, [p0], #64
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex0, dm4
  vlda.conv.fp32.bf16  cml4, [p3], #64
  vlda.conv.fp32.bf16  cmh4, [p3], #64
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex1, dm4
  vlda.conv.fp32.bf16  cml4, [p1], #64
  vlda.conv.fp32.bf16  cmh4, [p1], #64
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex2, dm4
  vlda.conv.fp32.bf16  cml4, [p1], #64
  vlda.conv.fp32.bf16  cmh4, [p1], #64
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex3, dm4
  vmac.f  dm0, dm0, ex0, ex2, r0
  vmac.f  dm1, dm1, ex0, ex3, r0
  vmac.f  dm2, dm2, ex1, ex2, r0
  vmac.f  dm3, dm3, ex1, ex3, r0

  // r = 2
  vlda.conv.fp32.bf16  cml4, [p0], #64
  vlda.conv.fp32.bf16  cmh4, [p0], #64
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex0, dm4
  vlda.conv.fp32.bf16  cml4, [p3], #64
  vlda.conv.fp32.bf16  cmh4, [p3], #64
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex1, dm4
  vlda.conv.fp32.bf16  cml4, [p1], #64
  vlda.conv.fp32.bf16  cmh4, [p1], #64
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex2, dm4
  vlda.conv.fp32.bf16  cml4, [p1], #64
  vlda.conv.fp32.bf16  cmh4, [p1], #64
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex3, dm4
  vmac.f  dm0, dm0, ex0, ex2, r0
  vmac.f  dm1, dm1, ex0, ex3, r0
  vmac.f  dm2, dm2, ex1, ex2, r0
  vmac.f  dm3, dm3, ex1, ex3, r0

  // r = 3
  vlda.conv.fp32.bf16  cml4, [p0], #64
  vlda.conv.fp32.bf16  cmh4, [p0], #64
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex0, dm4
  vlda.conv.fp32.bf16  cml4, [p3], #64
  vlda.conv.fp32.bf16  cmh4, [p3], #64
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex1, dm4
  vlda.conv.fp32.bf16  cml4, [p1], #64
  vlda.conv.fp32.bf16  cmh4, [p1], #64
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex2, dm4
  vlda.conv.fp32.bf16  cml4, [p1], #64
  vlda.conv.fp32.bf16  cmh4, [p1], #64
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex3, dm4
  vmac.f  dm0, dm0, ex0, ex2, r0
  vmac.f  dm1, dm1, ex0, ex3, r0
  vmac.f  dm2, dm2, ex1, ex2, r0
  vmac.f  dm3, dm3, ex1, ex3, r0

  // r = 4
  vlda.conv.fp32.bf16  cml4, [p0], #64
  vlda.conv.fp32.bf16  cmh4, [p0], #64
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex0, dm4
  vlda.conv.fp32.bf16  cml4, [p3], #64
  vlda.conv.fp32.bf16  cmh4, [p3], #64
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex1, dm4
  vlda.conv.fp32.bf16  cml4, [p1], #64
  vlda.conv.fp32.bf16  cmh4, [p1], #64
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex2, dm4
  vlda.conv.fp32.bf16  cml4, [p1], #64
  vlda.conv.fp32.bf16  cmh4, [p1], #64
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex3, dm4
  vmac.f  dm0, dm0, ex0, ex2, r0
  vmac.f  dm1, dm1, ex0, ex3, r0
  vmac.f  dm2, dm2, ex1, ex2, r0
  vmac.f  dm3, dm3, ex1, ex3, r0

  // r = 5
  vlda.conv.fp32.bf16  cml4, [p0], #64
  vlda.conv.fp32.bf16  cmh4, [p0], #64
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex0, dm4
  vlda.conv.fp32.bf16  cml4, [p3], #64
  vlda.conv.fp32.bf16  cmh4, [p3], #64
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex1, dm4
  vlda.conv.fp32.bf16  cml4, [p1], #64
  vlda.conv.fp32.bf16  cmh4, [p1], #64
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex2, dm4
  vlda.conv.fp32.bf16  cml4, [p1], #64
  vlda.conv.fp32.bf16  cmh4, [p1], #64
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex3, dm4
  vmac.f  dm0, dm0, ex0, ex2, r0
  vmac.f  dm1, dm1, ex0, ex3, r0
  vmac.f  dm2, dm2, ex1, ex2, r0
  vmac.f  dm3, dm3, ex1, ex3, r0

  // r = 6
  vlda.conv.fp32.bf16  cml4, [p0], #64
  vlda.conv.fp32.bf16  cmh4, [p0], #64
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex0, dm4
  vlda.conv.fp32.bf16  cml4, [p3], #64
  vlda.conv.fp32.bf16  cmh4, [p3], #64
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex1, dm4
  vlda.conv.fp32.bf16  cml4, [p1], #64
  vlda.conv.fp32.bf16  cmh4, [p1], #64
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex2, dm4
  vlda.conv.fp32.bf16  cml4, [p1], #64
  vlda.conv.fp32.bf16  cmh4, [p1], #64
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex3, dm4
  vmac.f  dm0, dm0, ex0, ex2, r0
  vmac.f  dm1, dm1, ex0, ex3, r0
  vmac.f  dm2, dm2, ex1, ex2, r0
  vmac.f  dm3, dm3, ex1, ex3, r0

  // r = 7
  vlda.conv.fp32.bf16  cml4, [p0], #64
  vlda.conv.fp32.bf16  cmh4, [p0], #64
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex0, dm4
  vlda.conv.fp32.bf16  cml4, [p3], #64
  vlda.conv.fp32.bf16  cmh4, [p3], #64
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex1, dm4
  vlda.conv.fp32.bf16  cml4, [p1], #64
  vlda.conv.fp32.bf16  cmh4, [p1], #64
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex2, dm4
  vlda.conv.fp32.bf16  cml4, [p1], #64
  vlda.conv.fp32.bf16  cmh4, [p1], #64
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex3, dm4
  vmac.f  dm0, dm0, ex0, ex2, r0
  vmac.f  dm1, dm1, ex0, ex3, r0
  vmac.f  dm2, dm2, ex1, ex2, r0
  vmac.f  dm3, dm3, ex1, ex3, r0

  // ── Out store: convert FP32 acc -> BF16, write back to L1 ─────────
  // vmac latency 6 must elapse before vst reads dm3.
  nop
  nop
  nop
  nop
  nop
  vst.conv.bf16.fp32   cml0, [p2, #0]
  vst.conv.bf16.fp32   cmh0, [p2, #64]
  vst.conv.bf16.fp32   cml1, [p2, #128]
  vst.conv.bf16.fp32   cmh1, [p2, #192]
  vst.conv.bf16.fp32   cml2, [p2, #256]
  vst.conv.bf16.fp32   cmh2, [p2, #320]
  vst.conv.bf16.fp32   cml3, [p2, #384]
  vst.conv.bf16.fp32   cmh3, [p2, #448]

  ret lr
  nop                                 // Delay Slot 5
  nop                                 // Delay Slot 4
  nop                                 // Delay Slot 3
  nop                                 // Delay Slot 2
  nop                                 // Delay Slot 1
.Lfunc_end0:
  .size matmul, .Lfunc_end0-matmul
