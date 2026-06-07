  .file "zero.s"
  .section .text.zero,"ax",@progbits
  .globl zero
  .p2align 4
  .type zero,@function
zero:
  // Sets 512 bytes (the full 2x2x8x8 bf16 out tile) to zero.
  // vst x writes 64 B, so 8 stores are needed: the original 4 stores zeroed only
  // the first 256 B (p=0 half), leaving the p=1 half stale -> it accumulated
  // garbage and ~half the output rows were wrong.
  mov r0, #0
  vbcst.16 x0, r0
  vst x0, [p0], #64                   // 1
  vst x0, [p0], #64                   // 2
  vst x0, [p0], #64                   // 3
  vst x0, [p0], #64                   // 4
  vst x0, [p0], #64                   // 5
  vst x0, [p0], #64                   // 6
  vst x0, [p0], #64                   // 7
  vst x0, [p0], #64                   // 8  -> 8*64 = 512 B
  ret lr
  nop                                 // Delay Slot 5
  nop                                 // Delay Slot 4
  nop                                 // Delay Slot 3
  nop                                 // Delay Slot 2
  nop                                 // Delay Slot 1
.Lfunc_end0:
  .size zero, .Lfunc_end0-zero
