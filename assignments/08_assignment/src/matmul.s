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
//   p0       : walks in0[p=0, *], post-incremented by +128 per r-step
//   p3       : walks in0[p=1, *], post-incremented by +128 per r-step
//   p1       : walks in1, post-incremented by +256 per r-step
//   p2       : out base (static)
//   r0       : vmac.f modifier (#780 selects bfp16ebs8 8x8x8 mode, Folie 5)
//
// Latency assumptions (verify against compiled aie::mmul reference):
//   vlda.conv = 4   (from A07, Task 4)
//   vconv     = 2   (M-slot vector move-with-conversion, conservative guess)
//   vmac.f    = 6   (assumed identical to vadd.f from A07)
//
// Schedule style: one slot active per VLIW cycle (other slots auto-nop).
// Explicit `nop` lines pad latencies. This is a correctness-first draft;
// see Task 6 for overlap-based optimisation potential.

  // ── Setup ──────────────────────────────────────────────────────────
  // Compute p3 = p0 + 1024 (start of in0[p=1, *]).
  mov     r1, p0                          // M : scalar = base(in0)
  movxm   r2, #1024                       // XM: 11-bit immediate via 32-bit form
  add     r1, r1, r2                      // X : r1 = base + 1024
  mov     p3, r1                          // M : p3 = base + 1024

  mova    r0, #780                        // A : BFP16 8x8x8 mode modifier

  // ── Out initialisation: load BF16 zeros from L1, convert to FP32 ──
  // Scratchpad is zero-initialised (A08, Data Layout section).
  vlda.conv.fp32.bf16  cml0, [p2, #0]     // dm0.lo = 0
  vlda.conv.fp32.bf16  cmh0, [p2, #64]    // dm0.hi = 0
  vlda.conv.fp32.bf16  cml1, [p2, #128]   // dm1.lo = 0
  vlda.conv.fp32.bf16  cmh1, [p2, #192]   // dm1.hi = 0
  vlda.conv.fp32.bf16  cml2, [p2, #256]   // dm2.lo = 0
  vlda.conv.fp32.bf16  cmh2, [p2, #320]   // dm2.hi = 0
  vlda.conv.fp32.bf16  cml3, [p2, #384]   // dm3.lo = 0
  vlda.conv.fp32.bf16  cmh3, [p2, #448]   // dm3.hi = 0
  // vlda latency 4 expires before first vmac reads dm0..dm3 (well below).

  // ── Main reduction loop, fully unrolled (8 × r-step) ──────────────
  //
  // Per r-step (28 cycles in this naive layout):
  //   1) load 4 BFP16 blocks (in0[0,r], in0[1,r], in1[r,0], in1[r,1])
  //      each via 2 vlda halves + 3 nops + 1 vconv  =  6 cycles
  //   2) 4 vmacs (one per (p,q)) in 4 cycles
  //   Pointer post-increments embedded in the last vlda of each block.

  // ════════════════════════════════════════════════════════════════
  // r = 0
  // ════════════════════════════════════════════════════════════════
  // Block A: in0[0,0] -> ex0
  vlda.conv.fp32.bf16  cml4, [p0, #0]
  vlda.conv.fp32.bf16  cmh4, [p0, #64]
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex0, dm4
  // Block B: in0[1,0] -> ex1
  vlda.conv.fp32.bf16  cml4, [p3, #0]
  vlda.conv.fp32.bf16  cmh4, [p3, #64]
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex1, dm4
  // Block C: in1[0,0] -> ex2
  vlda.conv.fp32.bf16  cml4, [p1, #0]
  vlda.conv.fp32.bf16  cmh4, [p1, #64]
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex2, dm4
  // Block D: in1[0,1] -> ex3  (last in-block of this r-step → no post-inc yet)
  vlda.conv.fp32.bf16  cml4, [p1, #128]
  vlda.conv.fp32.bf16  cmh4, [p1, #192]
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
  // r = 1  (advance pointers by their per-r increments first)
  // ════════════════════════════════════════════════════════════════
  // Advance p0 +128, p3 +128, p1 +256 — embedded as immediate offsets
  // below (we shift all in-iteration offsets by r*stride).
  // Block A: in0[0,1] -> ex0
  vlda.conv.fp32.bf16  cml4, [p0, #128]
  vlda.conv.fp32.bf16  cmh4, [p0, #192]
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex0, dm4
  // Block B: in0[1,1] -> ex1
  vlda.conv.fp32.bf16  cml4, [p3, #128]
  vlda.conv.fp32.bf16  cmh4, [p3, #192]
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex1, dm4
  // Block C: in1[1,0] -> ex2
  vlda.conv.fp32.bf16  cml4, [p1, #256]
  vlda.conv.fp32.bf16  cmh4, [p1, #320]
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex2, dm4
  // Block D: in1[1,1] -> ex3
  vlda.conv.fp32.bf16  cml4, [p1, #384]
  vlda.conv.fp32.bf16  cmh4, [p1, #448]
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex3, dm4
  vmac.f  dm0, dm0, ex0, ex2, r0
  vmac.f  dm1, dm1, ex0, ex3, r0
  vmac.f  dm2, dm2, ex1, ex2, r0
  vmac.f  dm3, dm3, ex1, ex3, r0

  // ════════════════════════════════════════════════════════════════
  // r = 2
  // ════════════════════════════════════════════════════════════════
  vlda.conv.fp32.bf16  cml4, [p0, #256]
  vlda.conv.fp32.bf16  cmh4, [p0, #320]
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex0, dm4
  vlda.conv.fp32.bf16  cml4, [p3, #256]
  vlda.conv.fp32.bf16  cmh4, [p3, #320]
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex1, dm4
  vlda.conv.fp32.bf16  cml4, [p1, #512]
  vlda.conv.fp32.bf16  cmh4, [p1, #576]
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex2, dm4
  vlda.conv.fp32.bf16  cml4, [p1, #640]
  vlda.conv.fp32.bf16  cmh4, [p1, #704]
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex3, dm4
  vmac.f  dm0, dm0, ex0, ex2, r0
  vmac.f  dm1, dm1, ex0, ex3, r0
  vmac.f  dm2, dm2, ex1, ex2, r0
  vmac.f  dm3, dm3, ex1, ex3, r0

  // ════════════════════════════════════════════════════════════════
  // r = 3
  // ════════════════════════════════════════════════════════════════
  vlda.conv.fp32.bf16  cml4, [p0, #384]
  vlda.conv.fp32.bf16  cmh4, [p0, #448]
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex0, dm4
  vlda.conv.fp32.bf16  cml4, [p3, #384]
  vlda.conv.fp32.bf16  cmh4, [p3, #448]
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex1, dm4
  vlda.conv.fp32.bf16  cml4, [p1, #768]
  vlda.conv.fp32.bf16  cmh4, [p1, #832]
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex2, dm4
  vlda.conv.fp32.bf16  cml4, [p1, #896]
  vlda.conv.fp32.bf16  cmh4, [p1, #960]
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex3, dm4
  vmac.f  dm0, dm0, ex0, ex2, r0
  vmac.f  dm1, dm1, ex0, ex3, r0
  vmac.f  dm2, dm2, ex1, ex2, r0
  vmac.f  dm3, dm3, ex1, ex3, r0

  // ════════════════════════════════════════════════════════════════
  // r = 4  — vlda immediate offsets keep growing; check the encoded
  //         range on the Peano assembler (max possibly ±2048 B).
  //         If the assembler rejects, post-increment pX once per r
  //         instead of using accumulated offsets.
  // ════════════════════════════════════════════════════════════════
  vlda.conv.fp32.bf16  cml4, [p0, #512]
  vlda.conv.fp32.bf16  cmh4, [p0, #576]
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex0, dm4
  vlda.conv.fp32.bf16  cml4, [p3, #512]
  vlda.conv.fp32.bf16  cmh4, [p3, #576]
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex1, dm4
  vlda.conv.fp32.bf16  cml4, [p1, #1024]
  vlda.conv.fp32.bf16  cmh4, [p1, #1088]
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex2, dm4
  vlda.conv.fp32.bf16  cml4, [p1, #1152]
  vlda.conv.fp32.bf16  cmh4, [p1, #1216]
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex3, dm4
  vmac.f  dm0, dm0, ex0, ex2, r0
  vmac.f  dm1, dm1, ex0, ex3, r0
  vmac.f  dm2, dm2, ex1, ex2, r0
  vmac.f  dm3, dm3, ex1, ex3, r0

  // ════════════════════════════════════════════════════════════════
  // r = 5
  // ════════════════════════════════════════════════════════════════
  vlda.conv.fp32.bf16  cml4, [p0, #640]
  vlda.conv.fp32.bf16  cmh4, [p0, #704]
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex0, dm4
  vlda.conv.fp32.bf16  cml4, [p3, #640]
  vlda.conv.fp32.bf16  cmh4, [p3, #704]
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex1, dm4
  vlda.conv.fp32.bf16  cml4, [p1, #1280]
  vlda.conv.fp32.bf16  cmh4, [p1, #1344]
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex2, dm4
  vlda.conv.fp32.bf16  cml4, [p1, #1408]
  vlda.conv.fp32.bf16  cmh4, [p1, #1472]
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex3, dm4
  vmac.f  dm0, dm0, ex0, ex2, r0
  vmac.f  dm1, dm1, ex0, ex3, r0
  vmac.f  dm2, dm2, ex1, ex2, r0
  vmac.f  dm3, dm3, ex1, ex3, r0

  // ════════════════════════════════════════════════════════════════
  // r = 6
  // ════════════════════════════════════════════════════════════════
  vlda.conv.fp32.bf16  cml4, [p0, #768]
  vlda.conv.fp32.bf16  cmh4, [p0, #832]
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex0, dm4
  vlda.conv.fp32.bf16  cml4, [p3, #768]
  vlda.conv.fp32.bf16  cmh4, [p3, #832]
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex1, dm4
  vlda.conv.fp32.bf16  cml4, [p1, #1536]
  vlda.conv.fp32.bf16  cmh4, [p1, #1600]
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex2, dm4
  vlda.conv.fp32.bf16  cml4, [p1, #1664]
  vlda.conv.fp32.bf16  cmh4, [p1, #1728]
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex3, dm4
  vmac.f  dm0, dm0, ex0, ex2, r0
  vmac.f  dm1, dm1, ex0, ex3, r0
  vmac.f  dm2, dm2, ex1, ex2, r0
  vmac.f  dm3, dm3, ex1, ex3, r0

  // ════════════════════════════════════════════════════════════════
  // r = 7
  // ════════════════════════════════════════════════════════════════
  vlda.conv.fp32.bf16  cml4, [p0, #896]
  vlda.conv.fp32.bf16  cmh4, [p0, #960]
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex0, dm4
  vlda.conv.fp32.bf16  cml4, [p3, #896]
  vlda.conv.fp32.bf16  cmh4, [p3, #960]
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex1, dm4
  vlda.conv.fp32.bf16  cml4, [p1, #1792]
  vlda.conv.fp32.bf16  cmh4, [p1, #1856]
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex2, dm4
  vlda.conv.fp32.bf16  cml4, [p1, #1920]
  vlda.conv.fp32.bf16  cmh4, [p1, #1984]
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex3, dm4
  vmac.f  dm0, dm0, ex0, ex2, r0
  vmac.f  dm1, dm1, ex0, ex3, r0
  vmac.f  dm2, dm2, ex1, ex2, r0
  vmac.f  dm3, dm3, ex1, ex3, r0

  // ── Out store: convert FP32 acc -> BF16, write back to L1 ─────────
  // vmac latency 6 must elapse before vst can read the final dm3.
  // We pad explicitly here; later passes can fold these into the
  // ret delay slots.
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
