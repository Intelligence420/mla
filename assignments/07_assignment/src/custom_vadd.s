  .file "custom_vadd.s"
  .section .text.custom_vadd,"ax",@progbits
  .globl custom_vadd
  .p2align 4
  .type custom_vadd,@function
custom_vadd:
// Computes C = A + B + B
// Calling convention: p0 = ptr_in0, p1 = ptr_in1, p2 = ptr_out
  vlda.conv.fp32.bf16  cml0, [p0, #0]      // c1:  A low  -> dm0.lo
  vlda.conv.fp32.bf16  cmh0, [p0, #64]     // c2:  A high -> dm0.hi
  vlda.conv.fp32.bf16  cml1, [p1, #0]      // c3:  B low  -> dm1.lo
  vlda.conv.fp32.bf16  cmh1, [p1, #64]     // c4:  B high -> dm1.hi
  nop                                      // c5:  vlda latency = 4
  nop                                      // c6
  mova                 r0, #60             // c7:  shift modifier (mova latency = 1)
  vadd.f               dm0, dm0, dm1, r0   // c8:  dm0 = A + B
  nop                                      // c9:  vadd.f latency = 6
  nop                                      // c10
  nop                                      // c11
  nop                                      // c12
  nop                                      // c13
  vadd.f               dm0, dm0, dm1, r0   // c14: dm0 = (A + B) + B
  nop                                      // c15: pad so the 5 ret delay slots reach c21
  ret lr                                   // c16: function return
  nop                                      // c17: Delay Slot 5
  nop                                      // c18: Delay Slot 4
  nop                                      // c19: Delay Slot 3
  vst.conv.bf16.fp32   cml0, [p2, #0]      // c20: Delay Slot 2  (vadd.f latency = 6 ok)
  vst.conv.bf16.fp32   cmh0, [p2, #64]     // c21: Delay Slot 1
.Lfunc_end0:
  .size custom_vadd, .Lfunc_end0-custom_vadd
