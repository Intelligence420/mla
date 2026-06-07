	.file	"matmul_ref.cpp"
	.section	.text.matmul,"ax",@progbits
	.globl	matmul                          // -- Begin function matmul
	.p2align	4
	.type	matmul,@function
matmul:                                 // @matmul
// %bb.0:                               // %entry
	mova	r0, #29;		nopb	;		nops	;		nopxm	;		nopv	
	mova	r3, #1;		nopb	;		movx	r2, #0;		mov	r1, #7
	mova	r6, #64;		movx	r5, #128;		mov	r4, #60
	mova	r16, #2;		movx	r7, #10;		mov	r23, #1
	mova	r19, #5;		movx	r18, #4;		mov	r17, #3
	mova	r20, #6;		paddxm	 [sp], #64
	mova	r29, #28;		st	 p6, [sp, #-64];		movx	r24, #0;		mov	r21, #8 // 4-byte Folded Spill
.LBB0_1:                                // %for.cond1.preheader
                                        // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_2 Depth 2
                                        //       Child Loop BB0_3 Depth 3
	nopa	;		nopb	;		nops	;		lshl	 r22, r24, r7;		nopm	;		nopv	
	movs	p3, p0;		nopx	;		mov	m0, r22
	padda	 [p3], m0
	vldb	 x0, [p3, #0]
	vldb	 x2, [p3, #64]
	nop	
	nop	
	nop	
	nop	
	nop	
	vshuffle	x0, x0, x0, r0
	vshuffle	x6, x2, x2, r0
	vextbcstshfl.64	 x4, x0, r2, r29
	vextbcstshfl.64	 x2, x0, r16, r29
	vextbcstshfl.64	 x7, x6, r3, r29
	vextbcstshfl.64	 x9, x6, r16, r29
	vextbcstshfl.64	 x11, x6, r17, r29
	vmov	bmll3, x4
	vextbcstshfl.64	 x4, x0, r3, r29
	vmov	bmlh3, x2
	vextbcstshfl.64	 x2, x0, r17, r29
	vmov	bmlh4, x2
	vextbcstshfl.64	 x2, x0, r18, r29
	vmov	bmll4, x4
	vextbcstshfl.64	 x4, x6, r20, r29
	vmov	bmhl3, x2
	vextbcstshfl.64	 x2, x0, r19, r29
	vmov	bmhl4, x2
	vextbcstshfl.64	 x2, x0, r20, r29
	vextbcstshfl.64	 x0, x0, r1, r29
	vmov	bmhh4, x0
	vextbcstshfl.64	 x0, x6, r2, r29
	vmov	bmhh3, x2
	vextbcstshfl.64	 x2, x6, r19, r29
	mova	dc0, #0;		movs	p4, p2;		lshl	 r22, r24, r21;		vmov	lfh0, x0
	mova	r24, #1;		movs	m0, r22;		vextbcstshfl.64	 x0, x6, r18, r29
	padda	 [p4], m0;		or	 r22, r23, r23;		vextbcstshfl.64	 x6, x6, r1, r29
.LBB0_2:                                // %for.body4
                                        //   Parent Loop BB0_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB0_3 Depth 3
	nopa	;		nopb	;		nops	;		nopx	;		mov	r23, dc0;		nopv	
	nopa	;		lshl	 r23, r23, r3
	movs	p5, p1;		mov	m0, r23
	padda	 [p5], m0
	vldb	 x8, [p5, #0]
	nop	
	nop	
	nop	
	nop	
	nop	
	vmov	x1, bmll3
	vextbcst.128	 x10, x8, #0
	vmov	x1, lfh0
	vextbcst.128	 x10, x8, #1;		vmul.f	dm0, x1, x10, r4
	vmov	x1, bmll4;		vmul.f	dm1, x1, x10, r4
	nop	
	vmov	x1, bmlh3;		vmac.f	dm0, dm0, x1, x10, r4
	vldb	 x8, [p5, #64];		vextbcst.128	 x10, x8, #2;		vmac.f	dm1, dm1, x7, x10, r4
	nop	
	vextbcst.128	 x8, x8, #3;		vmac.f	dm0, dm0, x1, x10, r4
	vmov	x10, bmlh4;		vmac.f	dm1, dm1, x9, x10, r4
	nop	
	vmac.f	dm0, dm0, x10, x8, r4
	vmov	x1, bmhl3;		vmac.f	dm1, dm1, x11, x8, r4
	vextbcst.128	 x10, x8, #0
	nop	
	vextbcst.128	 x10, x8, #1;		vmac.f	dm0, dm0, x1, x10, r4
	mova	r24, #64;		or	 r23, r24, r24;		vmov	x1, bmhl4;		vmac.f	dm1, dm1, x0, x10, r4
	mova	r25, #128;		movs	p6, p3;		lshl	 r26, r24, r3
	movs	m0, r26;		lshl	 r26, r25, r3;		vextbcst.128	 x10, x8, #2;		vmac.f	dm0, dm0, x1, x10, r4
	padda	 [p6], m0;		nopb	;		movs	m0, r26;		add	 r25, r25, r5;		vmov	x1, bmhh3;		vmac.f	dm1, dm1, x2, x10, r4
	vldb	 x8, [p6, #0];		add	 r24, r24, r6
	vlda	 x10, [p6, #64];		movs	p6, p5;		vextbcst.128	 x8, x8, #3;		vmac.f	dm0, dm0, x1, x10, r4
	paddb	 [p6], m0;		vmov	x10, bmhh4;		vmac.f	dm2, dm1, x4, x10, r4
	vldb	 x1, [p6, #0]
	add.nc	lc, r1, #-1;		vmac.f	dm1, dm0, x10, x8, r4
	movxm	ls, #.LBB0_3;		vmac.f	dm0, dm2, x6, x8, r4
	movxm	le, #.L_LEnd0
	vshuffle	x3, x8, x8, r0
	vshuffle	x5, x10, x10, r0
	vextbcstshfl.64	 x8, x3, r2, r29
	vextbcst.128	 x10, x1, #0
	vextbcstshfl.64	 x8, x5, r2, r29
	vextbcstshfl.64	 x10, x3, r3, r29;		vmac.f	dm2, dm1, x8, x10, r4
	vextbcst.128	 x8, x1, #1;		vmac.f	dm0, dm0, x8, x10, r4
	vextbcstshfl.64	 x10, x5, r3, r29
	vlda	 x10, [p6, #64];		vextbcstshfl.64	 x8, x3, r16, r29;		vmac.f	dm1, dm2, x10, x8, r4
	vextbcst.128	 x10, x1, #2;		vmac.f	dm0, dm0, x10, x8, r4
	vextbcstshfl.64	 x8, x5, r16, r29
	vextbcstshfl.64	 x10, x3, r17, r29;		vmac.f	dm1, dm1, x8, x10, r4
	vextbcst.128	 x1, x1, #3;		vmac.f	dm0, dm0, x8, x10, r4
	vextbcstshfl.64	 x8, x5, r17, r29
	vextbcstshfl.64	 x1, x3, r18, r29;		vmac.f	dm1, dm1, x10, x1, r4
	vextbcst.128	 x8, x10, #0;		vmac.f	dm0, dm0, x8, x1, r4
	vextbcstshfl.64	 x1, x5, r18, r29
.LBB0_3:                                // %for.body14
                                        //   Parent Loop BB0_1 Depth=1
                                        //     Parent Loop BB0_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	nopa	;		nopb	;		movs	p6, p3;		lshl	 r26, r24, r3;		vextbcstshfl.64	 x8, x3, r19, r29;		vmac.f	dm1, dm1, x1, x8, r4
	nopa	;		nopb	;		movs	m0, r26;		lshl	 r26, r25, r3;		vextbcst.128	 x1, x10, #1;		vmac.f	dm0, dm0, x1, x8, r4
	padda	 [p6], m0;		nopb	;		movs	m0, r26;		add	 r25, r25, r5;		vextbcstshfl.64	 x8, x5, r19, r29;		nopv	
	vldb	 x8, [p6, #0];		add	 r24, r24, r6;		vextbcstshfl.64	 x1, x3, r20, r29;		vmac.f	dm1, dm1, x8, x1, r4
	vlda	 x10, [p6, #64];		movs	p6, p5;		vextbcst.128	 x8, x10, #2;		vmac.f	dm0, dm0, x8, x1, r4
	paddb	 [p6], m0;		vextbcstshfl.64	 x1, x5, r20, r29
	vldb	 x1, [p6, #0];		vextbcstshfl.64	 x3, x3, r1, r29;		vmac.f	dm1, dm1, x1, x8, r4
	vextbcst.128	 x10, x10, #3;		vmac.f	dm0, dm0, x1, x8, r4
	vextbcstshfl.64	 x5, x5, r1, r29
	vmac.f	dm1, dm1, x3, x10, r4
	vshuffle	x3, x8, x8, r0;		vmac.f	dm0, dm0, x5, x10, r4
	vshuffle	x5, x10, x10, r0
	vextbcstshfl.64	 x8, x3, r2, r29
	vextbcst.128	 x10, x1, #0
	vextbcstshfl.64	 x8, x5, r2, r29
	vextbcstshfl.64	 x10, x3, r3, r29;		vmac.f	dm2, dm1, x8, x10, r4
	vextbcst.128	 x8, x1, #1;		vmac.f	dm0, dm0, x8, x10, r4
	vextbcstshfl.64	 x10, x5, r3, r29
	vlda	 x10, [p6, #64];		vextbcstshfl.64	 x8, x3, r16, r29;		vmac.f	dm1, dm2, x10, x8, r4
	vextbcst.128	 x10, x1, #2;		vmac.f	dm0, dm0, x10, x8, r4
	vextbcstshfl.64	 x8, x5, r16, r29
	vextbcstshfl.64	 x10, x3, r17, r29;		vmac.f	dm1, dm1, x8, x10, r4
	vextbcst.128	 x1, x1, #3;		vmac.f	dm0, dm0, x8, x10, r4
	vextbcstshfl.64	 x8, x5, r17, r29
	vextbcstshfl.64	 x1, x3, r18, r29;		vmac.f	dm1, dm1, x10, x1, r4
	vextbcst.128	 x8, x10, #0;		vmac.f	dm0, dm0, x8, x1, r4
.L_LEnd0:
	nopa	;		nopb	;		nops	;		nopx	;		vextbcstshfl.64	 x1, x5, r18, r29;		nopv	
// %bb.4:                               // %for.cond.cleanup13
                                        //   in Loop: Header=BB0_2 Depth=2
	nopa	;		nopb	;		movs	p5, p4;		nopx	;		vextbcstshfl.64	 x8, x3, r19, r29;		vmac.f	dm1, dm1, x1, x8, r4
	nopa	;		nopb	;		nopx	;		vextbcst.128	 x1, x10, #1;		vmac.f	dm0, dm0, x1, x8, r4
	vextbcstshfl.64	 x8, x5, r19, r29
	vextbcstshfl.64	 x1, x3, r20, r29;		vmac.f	dm1, dm1, x8, x1, r4
	vextbcst.128	 x8, x10, #2;		vmac.f	dm0, dm0, x8, x1, r4
	vextbcstshfl.64	 x1, x5, r20, r29
	vextbcstshfl.64	 x3, x3, r1, r29;		vmac.f	dm1, dm1, x1, x8, r4
	vextbcst.128	 x10, x10, #3;		vmac.f	dm0, dm0, x1, x8, r4
	vextbcstshfl.64	 x5, x5, r1, r29
	vmac.f	dm1, dm1, x3, x10, r4
	mova	dc0, #64;		mov	r24, dc0;		vmac.f	dm0, dm0, x5, x10, r4
	lshl	 r24, r24, r3
	mova	r24, #0;		mov	m0, r24
	padda	 [p5], m0
	vlda.conv.fp32.bf16	 cml2, [p5, #0]
	vlda.conv.fp32.bf16	 cmh2, [p5, #64];		vconv.bf16.fp32	 x8, cml1
	vconv.bf16.fp32	 x10, cml0
	vconv.fp32.bf16	cml0, x8
	vconv.fp32.bf16	cmh0, x10
	vadd.f	dm0, dm0, dm2, r4
	nop	
	jnz	 r23, #.LBB0_2
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	vst.conv.bf16.fp32	 cml0, [p5, #0] //  Delay Slot 2
	vst.conv.bf16.fp32	 cmh0, [p5, #64] //  Delay Slot 1
// %bb.5:                               // %for.cond.cleanup3
                                        //   in Loop: Header=BB0_1 Depth=1
	jnz	 r22, #.LBB0_1
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	mova	r24, #1;		movx	r23, #0         //  Delay Slot 1
// %bb.6:                               // %for.cond.cleanup
	lda	 p6, [sp, #-64]                 // 4-byte Folded Reload
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	paddxm	 [sp], #-64                     //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end0:
	.size	matmul, .Lfunc_end0-matmul
                                        // -- End function
	.section	".linker-options","e",@llvm_linker_options
	.ident	"clang version 21.0.0 (https://github.com/Xilinx/llvm-aie 9e603b765b27cae1566a02965eb0152640199850)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
