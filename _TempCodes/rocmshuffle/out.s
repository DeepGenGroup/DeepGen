
# __CLANG_OFFLOAD_BUNDLE____START__ hip-amdgcn-amd-amdhsa-gfx906
	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx906:sramecc+"
	.protected	_Z6reducePfS_           ; -- Begin function _Z6reducePfS_
	.globl	_Z6reducePfS_
	.p2align	8
	.type	_Z6reducePfS_,@function
_Z6reducePfS_:                          ; @_Z6reducePfS_
; %bb.0:
	s_load_dwordx2 s[0:1], s[4:5], 0x8
	v_cvt_f32_ubyte0_e32 v1, v0
	v_lshlrev_b32_e32 v0, 2, v0
	s_waitcnt lgkmcnt(0)
	global_store_dword v0, v1, s[0:1]
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel _Z6reducePfS_
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 6
		.amdhsa_reserve_vcc 0
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	_Z6reducePfS_, .Lfunc_end0-_Z6reducePfS_
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 32
; NumSgprs: 6
; NumVgprs: 2
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 6
; NumVGPRsForWavesPerEU: 2
; Occupancy: 10
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.ident	"clang version 15.0.0"
	.section	".note.GNU-stack"
	.addrsig
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
    .fp64_status:    0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z6reducePfS_
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .sgpr_spill_count: 0
    .symbol:         _Z6reducePfS_.kd
    .vgpr_count:     2
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   'amdgcn-amd-amdhsa--gfx906:sramecc+'
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata

# __CLANG_OFFLOAD_BUNDLE____END__ hip-amdgcn-amd-amdhsa-gfx906

# __CLANG_OFFLOAD_BUNDLE____START__ host-x86_64-unknown-linux-gnu
	.text
	.file	"reduce_4.cpp"
	.globl	_Z21__device_stub__reducePfS_   # -- Begin function _Z21__device_stub__reducePfS_
	.p2align	4, 0x90
	.type	_Z21__device_stub__reducePfS_,@function
_Z21__device_stub__reducePfS_:          # @_Z21__device_stub__reducePfS_
	.cfi_startproc
# %bb.0:
	subq	$88, %rsp
	.cfi_def_cfa_offset 96
	movq	%rdi, 56(%rsp)
	movq	%rsi, 48(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 64(%rsp)
	leaq	48(%rsp), %rax
	movq	%rax, 72(%rsp)
	leaq	32(%rsp), %rdi
	leaq	16(%rsp), %rsi
	leaq	8(%rsp), %rdx
	movq	%rsp, %rcx
	callq	__hipPopCallConfiguration
	movq	32(%rsp), %rsi
	movl	40(%rsp), %edx
	movq	16(%rsp), %rcx
	movl	24(%rsp), %r8d
	leaq	64(%rsp), %r9
	movl	$_Z6reducePfS_, %edi
	pushq	(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$104, %rsp
	.cfi_adjust_cfa_offset -104
	retq
.Lfunc_end0:
	.size	_Z21__device_stub__reducePfS_, .Lfunc_end0-_Z21__device_stub__reducePfS_
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2                               # -- Begin function main
.LCPI1_0:
	.long	0x3c23d70a                      # float 0.00999999977
	.text
	.globl	main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r12
	.cfi_def_cfa_offset 32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	subq	$120, %rsp
	.cfi_def_cfa_offset 160
	.cfi_offset %rbx, -40
	.cfi_offset %r12, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	leaq	28(%rsp), %rdi
	callq	hipGetDeviceCount
	cmpl	$0, 28(%rsp)
	je	.LBB1_1
# %bb.6:
	leaq	24(%rsp), %rdi
	callq	hipGetDevice
	movl	24(%rsp), %edi
	callq	hipSetDevice
	movl	$256, %edi                      # imm = 0x100
	callq	_Znam
	movq	%rax, %r12
	movl	$256, %edi                      # imm = 0x100
	callq	_Znam
	movq	%rax, %r14
	movl	$1, %edi
	callq	srand
	xorl	%ebx, %ebx
	.p2align	4, 0x90
.LBB1_7:                                # =>This Inner Loop Header: Depth=1
	callq	rand
	cltq
	imulq	$274877907, %rax, %rcx          # imm = 0x10624DD3
	movq	%rcx, %rdx
	shrq	$63, %rdx
	sarq	$38, %rcx
	addl	%edx, %ecx
	imull	$1000, %ecx, %ecx               # imm = 0x3E8
	subl	%ecx, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%eax, %xmm0
	mulss	.LCPI1_0(%rip), %xmm0
	movss	%xmm0, (%r12,%rbx,4)
	incq	%rbx
	cmpq	$64, %rbx
	jne	.LBB1_7
# %bb.8:
	movl	$12, %ebx
	.p2align	4, 0x90
.LBB1_9:                                # =>This Inner Loop Header: Depth=1
	movss	-12(%r12,%rbx), %xmm0           # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movl	$.L.str.1, %edi
	movb	$1, %al
	callq	printf
	movss	-8(%r12,%rbx), %xmm0            # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movl	$.L.str.1, %edi
	movb	$1, %al
	callq	printf
	movss	-4(%r12,%rbx), %xmm0            # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movl	$.L.str.1, %edi
	movb	$1, %al
	callq	printf
	movss	(%r12,%rbx), %xmm0              # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movl	$.L.str.1, %edi
	movb	$1, %al
	callq	printf
	movl	$10, %edi
	callq	putchar@PLT
	addq	$16, %rbx
	cmpq	$268, %rbx                      # imm = 0x10C
	jne	.LBB1_9
# %bb.10:
	movl	$.Lstr.4, %edi
	callq	puts@PLT
	leaq	16(%rsp), %rdi
	movl	$256, %esi                      # imm = 0x100
	callq	hipMalloc
	leaq	8(%rsp), %rdi
	movl	$256, %esi                      # imm = 0x100
	callq	hipMalloc
	movq	16(%rsp), %rdi
	movl	$256, %edx                      # imm = 0x100
	movq	%r12, %rsi
	movl	$1, %ecx
	callq	hipMemcpy
	movabsq	$4294967297, %rdi               # imm = 0x100000001
	leaq	63(%rdi), %rdx
	xorl	%r15d, %r15d
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB1_12
# %bb.11:
	movq	16(%rsp), %rax
	movq	8(%rsp), %rcx
	movq	%rax, 88(%rsp)
	movq	%rcx, 80(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	64(%rsp), %rdi
	leaq	48(%rsp), %rsi
	leaq	40(%rsp), %rdx
	leaq	32(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	64(%rsp), %rsi
	movl	72(%rsp), %edx
	movq	48(%rsp), %rcx
	movl	56(%rsp), %r8d
	leaq	96(%rsp), %r9
	movl	$_Z6reducePfS_, %edi
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	48(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB1_12:
	movq	8(%rsp), %rsi
	movl	$256, %edx                      # imm = 0x100
	movq	%r14, %rdi
	movl	$2, %ecx
	callq	hipMemcpy
	movq	8(%rsp), %rax
	movss	(%rax), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movl	$.L.str.1, %edi
	movb	$1, %al
	callq	printf
	movq	8(%rsp), %rax
	movss	16(%rax), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movl	$.L.str.1, %edi
	movb	$1, %al
	callq	printf
	movq	8(%rsp), %rax
	movss	32(%rax), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movl	$.L.str.1, %edi
	movb	$1, %al
	callq	printf
	movq	8(%rsp), %rax
	movss	48(%rax), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movl	$.L.str.1, %edi
	movb	$1, %al
	callq	printf
	movq	8(%rsp), %rax
	movss	64(%rax), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movl	$.L.str.1, %edi
	movb	$1, %al
	callq	printf
	movq	8(%rsp), %rax
	movss	80(%rax), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movl	$.L.str.1, %edi
	movb	$1, %al
	callq	printf
	movq	8(%rsp), %rax
	movss	96(%rax), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movl	$.L.str.1, %edi
	movb	$1, %al
	callq	printf
	movq	8(%rsp), %rax
	movss	112(%rax), %xmm0                # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movl	$.L.str.1, %edi
	movb	$1, %al
	callq	printf
	movq	8(%rsp), %rax
	movss	128(%rax), %xmm0                # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movl	$.L.str.1, %edi
	movb	$1, %al
	callq	printf
	movq	8(%rsp), %rax
	movss	144(%rax), %xmm0                # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movl	$.L.str.1, %edi
	movb	$1, %al
	callq	printf
	movq	8(%rsp), %rax
	movss	160(%rax), %xmm0                # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movl	$.L.str.1, %edi
	movb	$1, %al
	callq	printf
	movq	8(%rsp), %rax
	movss	176(%rax), %xmm0                # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movl	$.L.str.1, %edi
	movb	$1, %al
	callq	printf
	movq	8(%rsp), %rax
	movss	192(%rax), %xmm0                # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movl	$.L.str.1, %edi
	movb	$1, %al
	callq	printf
	movq	8(%rsp), %rax
	movss	208(%rax), %xmm0                # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movl	$.L.str.1, %edi
	movb	$1, %al
	callq	printf
	movq	8(%rsp), %rax
	movss	224(%rax), %xmm0                # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movl	$.L.str.1, %edi
	movb	$1, %al
	callq	printf
	movq	8(%rsp), %rax
	movss	240(%rax), %xmm0                # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movl	$.L.str.1, %edi
	movb	$1, %al
	callq	printf
	movl	$.Lstr.4, %edi
	callq	puts@PLT
	movq	%r12, %rdi
	callq	_ZdaPv
	movq	16(%rsp), %rdi
	callq	hipFree
	movq	%r14, %rdi
	callq	_ZdaPv
	movq	8(%rsp), %rdi
	callq	hipFree
	jmp	.LBB1_13
.LBB1_1:
	movl	$_ZSt4cerr, %edi
	movl	$.L.str, %esi
	movl	$30, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	_ZSt4cerr(%rip), %rax
	movq	-24(%rax), %rax
	movq	_ZSt4cerr+240(%rax), %rbx
	testq	%rbx, %rbx
	je	.LBB1_14
# %bb.2:
	cmpb	$0, 56(%rbx)
	je	.LBB1_4
# %bb.3:
	movzbl	67(%rbx), %eax
	jmp	.LBB1_5
.LBB1_4:
	movq	%rbx, %rdi
	callq	_ZNKSt5ctypeIcE13_M_widen_initEv
	movq	(%rbx), %rax
	movq	%rbx, %rdi
	movl	$10, %esi
	callq	*48(%rax)
.LBB1_5:
	movsbl	%al, %esi
	movl	$_ZSt4cerr, %edi
	callq	_ZNSo3putEc
	movq	%rax, %rdi
	callq	_ZNSo5flushEv
	movl	$1, %r15d
.LBB1_13:
	movl	%r15d, %eax
	addq	$120, %rsp
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.LBB1_14:
	.cfi_def_cfa_offset 160
	callq	_ZSt16__throw_bad_castv
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
	.cfi_endproc
                                        # -- End function
	.section	.text.startup,"ax",@progbits
	.p2align	4, 0x90                         # -- Begin function _GLOBAL__sub_I_reduce_4.cpp
	.type	_GLOBAL__sub_I_reduce_4.cpp,@function
_GLOBAL__sub_I_reduce_4.cpp:            # @_GLOBAL__sub_I_reduce_4.cpp
	.cfi_startproc
# %bb.0:
	pushq	%rax
	.cfi_def_cfa_offset 16
	movl	$_ZStL8__ioinit, %edi
	callq	_ZNSt8ios_base4InitC1Ev
	movl	$_ZNSt8ios_base4InitD1Ev, %edi
	movl	$_ZStL8__ioinit, %esi
	movl	$__dso_handle, %edx
	popq	%rax
	.cfi_def_cfa_offset 8
	jmp	__cxa_atexit                    # TAILCALL
.Lfunc_end2:
	.size	_GLOBAL__sub_I_reduce_4.cpp, .Lfunc_end2-_GLOBAL__sub_I_reduce_4.cpp
	.cfi_endproc
                                        # -- End function
	.text
	.p2align	4, 0x90                         # -- Begin function __hip_module_ctor
	.type	__hip_module_ctor,@function
__hip_module_ctor:                      # @__hip_module_ctor
	.cfi_startproc
# %bb.0:
	subq	$40, %rsp
	.cfi_def_cfa_offset 48
	movq	__hip_gpubin_handle(%rip), %rdi
	testq	%rdi, %rdi
	jne	.LBB3_2
# %bb.1:
	movl	$__hip_fatbin_wrapper, %edi
	callq	__hipRegisterFatBinary
	movq	%rax, %rdi
	movq	%rax, __hip_gpubin_handle(%rip)
.LBB3_2:
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z6reducePfS_, %esi
	movl	$.L__unnamed_1, %edx
	movl	$.L__unnamed_1, %ecx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	movl	$__hip_module_dtor, %edi
	addq	$40, %rsp
	.cfi_def_cfa_offset 8
	jmp	atexit                          # TAILCALL
.Lfunc_end3:
	.size	__hip_module_ctor, .Lfunc_end3-__hip_module_ctor
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function __hip_module_dtor
	.type	__hip_module_dtor,@function
__hip_module_dtor:                      # @__hip_module_dtor
	.cfi_startproc
# %bb.0:
	pushq	%rax
	.cfi_def_cfa_offset 16
	movq	__hip_gpubin_handle(%rip), %rdi
	testq	%rdi, %rdi
	je	.LBB4_2
# %bb.1:
	callq	__hipUnregisterFatBinary
	movq	$0, __hip_gpubin_handle(%rip)
.LBB4_2:
	popq	%rax
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end4:
	.size	__hip_module_dtor, .Lfunc_end4-__hip_module_dtor
	.cfi_endproc
                                        # -- End function
	.type	_ZStL8__ioinit,@object          # @_ZStL8__ioinit
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.hidden	__dso_handle
	.type	_Z6reducePfS_,@object           # @_Z6reducePfS_
	.section	.rodata,"a",@progbits
	.globl	_Z6reducePfS_
	.p2align	3
_Z6reducePfS_:
	.quad	_Z21__device_stub__reducePfS_
	.size	_Z6reducePfS_, 8

	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"\346\262\241\346\234\211\346\211\276\345\210\260\345\217\257\347\224\250\347\232\204HIP\350\256\276\345\244\207"
	.size	.L.str, 31

	.type	.L.str.1,@object                # @.str.1
.L.str.1:
	.asciz	"%.7f "
	.size	.L.str.1, 6

	.type	.L__unnamed_1,@object           # @0
.L__unnamed_1:
	.asciz	"_Z6reducePfS_"
	.size	.L__unnamed_1, 14

	.type	__hip_fatbin_wrapper,@object    # @__hip_fatbin_wrapper
	.section	.hipFatBinSegment,"a",@progbits
	.p2align	3
__hip_fatbin_wrapper:
	.long	1212764230                      # 0x48495046
	.long	1                               # 0x1
	.quad	__hip_fatbin
	.quad	0
	.size	__hip_fatbin_wrapper, 24

	.hidden	__hip_gpubin_handle             # @__hip_gpubin_handle
	.type	__hip_gpubin_handle,@object
	.section	.bss.__hip_gpubin_handle,"aGw",@nobits,__hip_gpubin_handle,comdat
	.weak	__hip_gpubin_handle
	.p2align	3
__hip_gpubin_handle:
	.quad	0
	.size	__hip_gpubin_handle, 8

	.section	.init_array,"aw",@init_array
	.p2align	3
	.quad	_GLOBAL__sub_I_reduce_4.cpp
	.quad	__hip_module_ctor
	.type	.Lstr.4,@object                 # @str.4
	.section	.rodata.str1.1,"aMS",@progbits,1
.Lstr.4:
	.asciz	"\n"
	.size	.Lstr.4, 2

	.ident	"clang version 15.0.0"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Z21__device_stub__reducePfS_
	.addrsig_sym _GLOBAL__sub_I_reduce_4.cpp
	.addrsig_sym __hip_module_ctor
	.addrsig_sym __hip_module_dtor
	.addrsig_sym _ZStL8__ioinit
	.addrsig_sym __dso_handle
	.addrsig_sym _Z6reducePfS_
	.addrsig_sym _ZSt4cerr
	.addrsig_sym __hip_fatbin
	.addrsig_sym __hip_fatbin_wrapper

# __CLANG_OFFLOAD_BUNDLE____END__ host-x86_64-unknown-linux-gnu
