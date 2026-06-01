	.file	"matmul_ref.cpp"
	.section	.text.matmul,"ax",@progbits
	.globl	matmul                          // -- Begin function matmul
	.p2align	4
	.type	matmul,@function
matmul:                                 // @matmul
// %bb.0:                               // %entry
	mova	r2, #53;		nopb	;		movx	r1, #52;		mov	r0, #7
	mova	r6, #64;		movx	r5, #128;		mov	r4, #780
	mova	r7, #1;		movxm	r3, #16256
	mova	r19, #1;		paddxm	 [sp], #64
	mova	r16, #10;		movx	r20, #0;		vbcst.16	 x0, r3
	mova	r3, #60;		st	 p6, [sp, #-64];		movx	r17, #8;		vmov	x1, x0 // 4-byte Folded Spill
.LBB0_1:                                // %for.cond1.preheader
                                        // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_2 Depth 2
                                        //       Child Loop BB0_3 Depth 3
	nopa	;		nopb	;		lshl	 r18, r20, r16
	movs	p3, p0;		mov	m0, r18
	padda	 [p3], m0;		lshl	 r18, r20, r17;		mov	dc0, #0
	vlda.conv.fp32.bf16	 cml0, [p3, #0];		movs	p4, p2;		mov	m0, r18
	vlda.conv.fp32.bf16	 cmh0, [p3, #64];		paddb	 [p4], m0;		movx	r20, #1;		mov	r18, r19
.LBB0_2:                                // %for.body4
                                        //   Parent Loop BB0_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB0_3 Depth 3
	nopa	;		nopb	;		nopx	;		mov	r19, dc0
	lshl	 r19, r19, r7
	movs	p5, p1;		mov	m0, r19
	padda	 [p5], m0
	vldb	 x2, [p5, #0]
	vldb	 x4, [p5, #64]
	nop	
	nop	
	nop	
	nop	
	nop	
	nop	
	mova	r21, #128;		or	 r19, r20, r20;		vshuffle	x6, x2, x4, r1
	mova	r20, #64;		lshl	 r22, r21, r7;		vshuffle	x7, x2, x4, r2
	movs	m0, r22;		lshl	 r22, r20, r7;		mov	p6, p5
	padda	 [p6], m0;		add	 r21, r21, r5;		mov	m0, r22
	vldb	 x3, [p6, #0];		add	 r20, r20, r6;		vmul.f	dm1, y3, y0, r3
	vldb	 x5, [p6, #64];		movs	p6, p3
	paddb	 [p6], m0
	vlda.conv.fp32.bf16	 cml4, [p6, #0];		lshl	 r22, r21, r7
	vlda.conv.fp32.bf16	 cmh4, [p6, #64];		movs	m0, r22;		lshl	 r22, r20, r7;		mov	p6, p5
	padda	 [p6], m0;		vconv.bfp16ebs8.fp32	 ex2, dm0;		add	 r21, r21, r5;		mov	m0, r22
	vconv.bfp16ebs8.fp32	 ex4, dm1;		vldb	 x3, [p6, #0];		add	 r20, r20, r6
	vldb	 x5, [p6, #64];		movs	p6, p3
	paddb	 [p6], m0;		vshuffle	x6, x3, x5, r1
	vlda.conv.fp32.bf16	 cml4, [p6, #0];		lshl	 r22, r21, r7;		vshuffle	x7, x3, x5, r2
	vlda.conv.fp32.bf16	 cmh4, [p6, #64];		nopb	;		movs	m0, r22;		lshl	 r22, r20, r7;		mov	p6, p5;		vmul.f	dm2, ex2, ex4, r4
	padda	 [p6], m0;		add	 r21, r21, r5;		mov	m0, r22;		vmul.f	dm1, y3, y0, r3
	vconv.bfp16ebs8.fp32	 ex9, dm4;		vldb	 x3, [p6, #0];		add	 r20, r20, r6
	movs	p6, p3;		vldb	 x5, [p6, #64];		movxm	ls, #.LBB0_3
	paddb	 [p6], m0;		vshuffle	x6, x3, x5, r1
	vlda.conv.fp32.bf16	 cml4, [p6, #0];		lshl	 r22, r21, r7;		vshuffle	x7, x3, x5, r2
	vlda.conv.fp32.bf16	 cmh4, [p6, #64];		movs	m0, r22;		lshl	 r22, r20, r7;		mov	p6, p5
	padda	 [p6], m0;		nopb	;		vconv.bfp16ebs8.fp32	 ex11, dm1;		add	 r21, r21, r5;		mov	m0, r22;		vmul.f	dm1, y3, y0, r3
	vconv.bfp16ebs8.fp32	 ex9, dm4;		vldb	 x3, [p6, #0];		add	 r20, r20, r6;		add.nc	lc, r0, #-4
	nopa	;		vldb	 x5, [p6, #64];		movs	p6, p3;		movxm	le, #.L_LEnd0;		nopv	
	nopa	;		paddb	 [p6], m0;		nops	;		nopx	;		vshuffle	x6, x3, x5, r1;		nopv	
.LBB0_3:                                // %for.body13
                                        //   Parent Loop BB0_1 Depth=1
                                        //     Parent Loop BB0_2 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	vlda.conv.fp32.bf16	 cml4, [p6, #0];		nopb	;		nops	;		lshl	 r22, r21, r7;		vshuffle	x7, x3, x5, r2;		vmac.f	dm2, dm2, ex9, ex11, r4
	vlda.conv.fp32.bf16	 cmh4, [p6, #64];		nopb	;		movs	m0, r22;		lshl	 r22, r20, r7;		mov	p6, p5;		nopv	
	padda	 [p6], m0;		nopb	;		vconv.bfp16ebs8.fp32	 ex11, dm1;		add	 r21, r21, r5;		mov	m0, r22;		vmul.f	dm1, y3, y0, r3
	nopa	;		vldb	 x3, [p6, #0];		vconv.bfp16ebs8.fp32	 ex9, dm4;		add	 r20, r20, r6;		nopm	;		nopv	
	nopa	;		vldb	 x5, [p6, #64];		movs	p6, p3;		nopxm	;		nopv	
.L_LEnd0:
	nopa	;		paddb	 [p6], m0;		nops	;		nopx	;		vshuffle	x6, x3, x5, r1;		nopv	
// %bb.4:                               // %for.cond.cleanup12
                                        //   in Loop: Header=BB0_2 Depth=2
	vlda.conv.fp32.bf16	 cml4, [p6, #0];		movs	p5, p4;		vshuffle	x7, x3, x5, r2;		vmac.f	dm2, dm2, ex9, ex11, r4
	vlda.conv.fp32.bf16	 cmh4, [p6, #64]
	mova	dc0, #64;		vconv.bfp16ebs8.fp32	 ex11, dm1;		mov	r20, dc0;		vmul.f	dm1, y3, y0, r3
	vconv.bfp16ebs8.fp32	 ex9, dm4;		lshl	 r20, r20, r7
	mova	r20, #0;		mov	m0, r20
	padda	 [p5], m0;		vshuffle	x6, x3, x5, r1
	vshuffle	x7, x3, x5, r2;		vmac.f	dm2, dm2, ex9, ex11, r4
	nop	
	vconv.bfp16ebs8.fp32	 ex11, dm1;		vmul.f	dm1, y3, y0, r3
	vconv.bfp16ebs8.fp32	 ex9, dm4
	nop	
	nop	
	vmac.f	dm2, dm2, ex9, ex11, r4
	nop	
	vconv.bfp16ebs8.fp32	 ex11, dm1
	nop	
	nop	
	nop	
	vmac.f	dm2, dm2, ex9, ex11, r4
	nop	
	jnz	 r19, #.LBB0_2
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	vst.conv.bf16.fp32	 cml2, [p5, #0] //  Delay Slot 2
	vst.conv.bf16.fp32	 cmh2, [p5, #64] //  Delay Slot 1
// %bb.5:                               // %for.cond.cleanup3
                                        //   in Loop: Header=BB0_1 Depth=1
	jnz	 r18, #.LBB0_1
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	mova	r20, #1;		movx	r19, #0         //  Delay Slot 1
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
