
# __CLANG_OFFLOAD_BUNDLE____START__ hip-amdgcn-amd-amdhsa-gfx906
	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx906:sramecc+"
	.protected	_Z6reducePfS_           ; -- Begin function _Z6reducePfS_
	.globl	_Z6reducePfS_
	.p2align	8
	.type	_Z6reducePfS_,@function
_Z6reducePfS_:                          ; @_Z6reducePfS_
; %bb.0:
	s_load_dwordx4 s[0:3], s[4:5], 0x0
	v_lshl_add_u32 v0, s6, 6, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v3, s1
	v_add_co_u32_e32 v2, vcc, s0, v0
	v_addc_co_u32_e32 v3, vcc, v3, v1, vcc
	global_load_dword v2, v[2:3], off
	v_mbcnt_lo_u32_b32 v3, -1, 0
	v_mbcnt_hi_u32_b32 v3, -1, v3  // v3 = laneId
	v_cmp_ne_u32_e32 vcc, 63, v3  // vcc = (63!=v3)
	v_addc_co_u32_e32 v4, vcc, 0, v3, vcc  // v4 = 0+v3+ (63!=v3) ,并设置进位 vcc(计算新低32bit) : v4 = v3 + 1
	v_lshlrev_b32_e32 v4, 2, v4   // v4 *= 4
	v_cmp_gt_u32_e32 vcc, 62, v3 // vcc = 62 >v3 
	v_cndmask_b32_e64 v5, 0, 1, vcc  // if 62 >v3 then v5 = 1 else v5 = 0 
	v_lshlrev_b32_e32 v5, 1, v5  // v5*=2 (v5 = 2)
	v_add_lshl_u32 v5, v5, v3, 2  // v5 = (v5 + v3)*4  即 v5 = (2+v3)*4
	v_cmp_gt_u32_e32 vcc, 60, v3  // vcc = (60 > v3)
	s_waitcnt vmcnt(0)
	ds_bpermute_b32 v4, v4, v2  // 取地址v4=(v3+1)*4 的v2寄存器的值，放入v4
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v2, v2, v4  // v2 += v4
	ds_bpermute_b32 v4, v5, v2  // 取地址v5=(2+v3)*4 的v2寄存器的值，放入v4
	v_cndmask_b32_e64 v5, 0, 1, vcc  // if  (60 > v3) then v5 = 1 else v5 = 0
	v_lshlrev_b32_e32 v5, 2, v5  // v5 *= 4 : 即 v5 = 4
	v_add_lshl_u32 v5, v5, v3, 2  // v5 = (v5+v3)*4
	v_cmp_gt_u32_e32 vcc, 56, v3  // vcc = 56 > v3
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v2, v2, v4  // v2 += v4
	ds_bpermute_b32 v4, v5, v2  // 取地址v5=(4+v3)*4 的v2寄存器的值，放入v4
	v_cndmask_b32_e64 v5, 0, 1, vcc
	v_lshlrev_b32_e32 v5, 3, v5
	v_add_lshl_u32 v3, v5, v3, 2
	v_add_co_u32_e32 v0, vcc, s2, v0
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v2, v2, v4
	ds_bpermute_b32 v3, v3, v2
	v_mov_b32_e32 v4, s3
	v_addc_co_u32_e32 v1, vcc, v4, v1, vcc
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v2, v2, v3
	global_store_dword v[0:1], v2, off  // store v2 to addr(v[0:1])
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
		.amdhsa_next_free_vgpr 6
		.amdhsa_next_free_sgpr 7
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
; codeLenInByte = 256
; NumSgprs: 9
; NumVgprs: 6
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 9
; NumVGPRsForWavesPerEU: 6
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
    .sgpr_count:     9
    .sgpr_spill_count: 0
    .symbol:         _Z6reducePfS_.kd
    .vgpr_count:     6
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   'amdgcn-amd-amdhsa--gfx906:sramecc+'
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata

# __CLANG_OFFLOAD_BUNDLE____END__ hip-amdgcn-amd-amdhsa-gfx906

# __CLANG_OFFLOAD_BUNDLE____START__ hip-amdgcn-amd-amdhsa-gfx926
	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx926:sramecc+"
	.protected	_Z6reducePfS_           ; -- Begin function _Z6reducePfS_
	.globl	_Z6reducePfS_
	.p2align	8
	.type	_Z6reducePfS_,@function
_Z6reducePfS_:                          ; @_Z6reducePfS_
; %bb.0:
	s_load_dwordx4 s[0:3], s[4:5], 0x0
	v_lshl_add_u32 v0, s6, 6, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v3, s1
	v_add_co_u32_e32 v2, vcc, s0, v0
	v_addc_co_u32_e32 v3, vcc, v3, v1, vcc
	global_load_dword v2, v[2:3], off
	v_mbcnt_lo_u32_b32 v3, -1, 0
	v_mbcnt_hi_u32_b32 v3, -1, v3
	v_cmp_ne_u32_e32 vcc, 63, v3
	v_addc_co_u32_e32 v4, vcc, 0, v3, vcc
	v_lshlrev_b32_e32 v4, 2, v4
	v_cmp_gt_u32_e32 vcc, 62, v3
	v_cndmask_b32_e64 v5, 0, 1, vcc
	v_lshlrev_b32_e32 v5, 1, v5
	v_add_lshl_u32 v5, v5, v3, 2
	v_cmp_gt_u32_e32 vcc, 60, v3
	s_waitcnt vmcnt(0)
	ds_bpermute_b32 v4, v4, v2
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v2, v2, v4
	ds_bpermute_b32 v4, v5, v2
	v_cndmask_b32_e64 v5, 0, 1, vcc
	v_lshlrev_b32_e32 v5, 2, v5
	v_add_lshl_u32 v5, v5, v3, 2
	v_cmp_gt_u32_e32 vcc, 56, v3
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v2, v2, v4
	ds_bpermute_b32 v4, v5, v2
	v_cndmask_b32_e64 v5, 0, 1, vcc
	v_lshlrev_b32_e32 v5, 3, v5
	v_add_lshl_u32 v3, v5, v3, 2
	v_add_co_u32_e32 v0, vcc, s2, v0
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v2, v2, v4
	ds_bpermute_b32 v3, v3, v2
	v_mov_b32_e32 v4, s3
	v_addc_co_u32_e32 v1, vcc, v4, v1, vcc
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v2, v2, v3
	global_store_dword v[0:1], v2, off
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
		.amdhsa_next_free_vgpr 6
		.amdhsa_next_free_sgpr 7
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
; codeLenInByte = 256
; NumSgprs: 9
; NumVgprs: 6
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 9
; NumVGPRsForWavesPerEU: 6
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
    .sgpr_count:     9
    .sgpr_spill_count: 0
    .symbol:         _Z6reducePfS_.kd
    .vgpr_count:     6
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   'amdgcn-amd-amdhsa--gfx926:sramecc+'
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata

# __CLANG_OFFLOAD_BUNDLE____END__ hip-amdgcn-amd-amdhsa-gfx926

# __CLANG_OFFLOAD_BUNDLE____START__ hip-amdgcn-amd-amdhsa-gfx928
	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx928:sramecc+"
	.protected	_Z6reducePfS_           ; -- Begin function _Z6reducePfS_
	.globl	_Z6reducePfS_
	.p2align	8
	.type	_Z6reducePfS_,@function
_Z6reducePfS_:                          ; @_Z6reducePfS_
; %bb.0:
	s_load_dwordx4 s[0:3], s[4:5], 0x0
	v_lshl_add_u32 v0, s6, 6, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v3, s1
	v_add_co_u32_e32 v2, vcc, s0, v0
	v_addc_co_u32_e32 v3, vcc, v3, v1, vcc
	global_load_dword v2, v[2:3], off
	v_mbcnt_lo_u32_b32 v3, -1, 0
	v_mbcnt_hi_u32_b32 v3, -1, v3
	v_cmp_ne_u32_e32 vcc, 63, v3
	v_addc_co_u32_e32 v4, vcc, 0, v3, vcc
	v_lshlrev_b32_e32 v4, 2, v4
	v_cmp_gt_u32_e32 vcc, 62, v3
	v_cndmask_b32_e64 v5, 0, 1, vcc
	v_lshlrev_b32_e32 v5, 1, v5
	v_add_lshl_u32 v5, v5, v3, 2
	v_cmp_gt_u32_e32 vcc, 60, v3
	s_waitcnt vmcnt(0)
	ds_bpermute_b32 v4, v4, v2
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v2, v2, v4
	ds_bpermute_b32 v4, v5, v2
	v_cndmask_b32_e64 v5, 0, 1, vcc
	v_lshlrev_b32_e32 v5, 2, v5
	v_add_lshl_u32 v5, v5, v3, 2
	v_cmp_gt_u32_e32 vcc, 56, v3
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v2, v2, v4
	ds_bpermute_b32 v4, v5, v2
	v_cndmask_b32_e64 v5, 0, 1, vcc
	v_lshlrev_b32_e32 v5, 3, v5
	v_add_lshl_u32 v3, v5, v3, 2
	v_add_co_u32_e32 v0, vcc, s2, v0
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v2, v2, v4
	ds_bpermute_b32 v3, v3, v2
	v_mov_b32_e32 v4, s3
	v_addc_co_u32_e32 v1, vcc, v4, v1, vcc
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v2, v2, v3
	global_store_dword v[0:1], v2, off
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
		.amdhsa_next_free_vgpr 6
		.amdhsa_next_free_sgpr 7
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
; codeLenInByte = 256
; NumSgprs: 9
; NumVgprs: 6
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 9
; NumVGPRsForWavesPerEU: 6
; Occupancy: 10
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.p2align	2                               ; -- Begin function __softfloat_f64_add
	.type	__softfloat_f64_add,@function
__softfloat_f64_add:                    ; @__softfloat_f64_add
; %bb.0:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_xor_saveexec_b64 s[4:5], -1
	buffer_store_dword v4, off, s[0:3], s32 ; 4-byte Folded Spill
	s_mov_b64 exec, s[4:5]
	v_writelane_b32 v4, s34, 0
	v_writelane_b32 v4, s35, 1
	v_writelane_b32 v4, s30, 2
	v_writelane_b32 v4, s31, 3
	v_mov_b32_e32 v12, 0
	v_bfe_u32 v11, v1, 20, 11
	v_bfe_u32 v13, v3, 20, 11
	v_mov_b32_e32 v14, v12
	v_mov_b32_e32 v6, v1
	v_cmp_gt_i64_e64 s[8:9], 0, v[0:1]
	v_lshrrev_b32_e32 v7, 31, v1
	v_lshrrev_b32_e32 v8, 31, v3
	v_cmp_ne_u64_e64 s[4:5], v[11:12], v[13:14]
	v_sub_co_u32_e64 v15, s[6:7], v11, v13
	v_mov_b32_e32 v5, v0
	s_mov_b64 s[14:15], 0
	v_cmp_ne_u32_e32 vcc, v7, v8
	v_and_b32_e32 v10, 0xfffff, v1
	v_mov_b32_e32 v9, v0
	v_and_b32_e32 v8, 0xfffff, v3
	v_mov_b32_e32 v7, v2
	v_subb_co_u32_e64 v16, s[6:7], 0, 0, s[6:7]
	s_and_saveexec_b64 s[6:7], vcc
	s_xor_b64 s[10:11], exec, s[6:7]
	s_cbranch_execz .LBB1_6
; %bb.1:
	s_and_saveexec_b64 s[6:7], s[4:5]
	s_xor_b64 s[12:13], exec, s[6:7]
	s_cbranch_execnz .LBB1_16
; %bb.2:
	s_andn2_saveexec_b64 s[6:7], s[12:13]
	s_cbranch_execnz .LBB1_102
.LBB1_3:
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[12:13], s[14:15]
.LBB1_4:
	s_mov_b32 s6, 0
	s_mov_b32 s7, 0x7ff00000
	v_and_b32_e32 v6, 0x7ff00000, v1
	v_mov_b32_e32 v5, 0
	v_cmp_ne_u64_e32 vcc, s[6:7], v[5:6]
	v_cmp_eq_u64_e64 s[6:7], 0, v[9:10]
	s_or_b64 vcc, vcc, s[6:7]
	v_cndmask_b32_e32 v6, v1, v3, vcc
	v_cndmask_b32_e32 v5, v0, v2, vcc
.LBB1_5:
	s_or_b64 exec, exec, s[12:13]
                                        ; implicit-def: $vgpr11_vgpr12
                                        ; implicit-def: $vgpr13_vgpr14
                                        ; implicit-def: $vgpr15_vgpr16
                                        ; implicit-def: $vgpr2_vgpr3
                                        ; implicit-def: $vgpr0_vgpr1
                                        ; implicit-def: $vgpr9_vgpr10
                                        ; implicit-def: $vgpr7_vgpr8
.LBB1_6:
	s_andn2_saveexec_b64 s[6:7], s[10:11]
	s_cbranch_execz .LBB1_15
; %bb.7:
	s_mov_b64 s[10:11], 0
	s_mov_b64 s[14:15], 0
	s_mov_b64 s[16:17], 0
                                        ; implicit-def: $vgpr17_vgpr18
                                        ; implicit-def: $vgpr19_vgpr20
	s_and_saveexec_b64 s[12:13], s[4:5]
	s_xor_b64 s[12:13], exec, s[12:13]
	s_cbranch_execnz .LBB1_20
; %bb.8:
	s_andn2_saveexec_b64 s[4:5], s[12:13]
	s_cbranch_execnz .LBB1_45
.LBB1_9:
	s_or_b64 exec, exec, s[4:5]
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_cbranch_execnz .LBB1_48
.LBB1_10:
	s_or_b64 exec, exec, s[12:13]
	s_mov_b64 s[12:13], 0
	s_and_saveexec_b64 s[4:5], s[10:11]
	s_cbranch_execnz .LBB1_49
.LBB1_11:
	s_or_b64 exec, exec, s[4:5]
	s_and_saveexec_b64 s[4:5], s[16:17]
	s_xor_b64 s[10:11], exec, s[4:5]
	s_cbranch_execnz .LBB1_52
.LBB1_12:
	s_or_b64 exec, exec, s[10:11]
	s_and_saveexec_b64 s[4:5], s[12:13]
.LBB1_13:
	v_add_co_u32_e32 v5, vcc, v7, v0
	v_addc_co_u32_e32 v6, vcc, v8, v1, vcc
.LBB1_14:
	s_or_b64 exec, exec, s[4:5]
.LBB1_15:
	s_or_b64 exec, exec, s[6:7]
	v_readlane_b32 s30, v4, 2
	v_mov_b32_e32 v0, v5
	v_mov_b32_e32 v1, v6
	v_readlane_b32 s31, v4, 3
	v_readlane_b32 s35, v4, 1
	v_readlane_b32 s34, v4, 0
	s_xor_saveexec_b64 s[4:5], -1
	buffer_load_dword v4, off, s[0:3], s32  ; 4-byte Folded Reload
	s_mov_b64 exec, s[4:5]
	s_waitcnt vmcnt(0)
	s_setpc_b64 s[30:31]
.LBB1_16:
	v_lshlrev_b64 v[21:22], 10, v[9:10]
	v_lshlrev_b64 v[19:20], 10, v[7:8]
	v_cmp_lt_i64_e32 vcc, -1, v[15:16]
	s_mov_b64 s[16:17], 0
	s_mov_b64 s[20:21], 0
	s_mov_b64 s[14:15], 0
                                        ; implicit-def: $vgpr17_vgpr18
	s_and_saveexec_b64 s[6:7], vcc
	s_xor_b64 s[18:19], exec, s[6:7]
	s_cbranch_execnz .LBB1_63
; %bb.17:
	s_or_saveexec_b64 s[22:23], s[18:19]
	s_mov_b64 s[18:19], s[8:9]
	s_xor_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB1_72
.LBB1_18:
	s_or_b64 exec, exec, s[22:23]
	s_and_saveexec_b64 s[6:7], s[20:21]
	s_xor_b64 s[20:21], exec, s[6:7]
	s_cbranch_execnz .LBB1_79
.LBB1_19:
	s_or_b64 exec, exec, s[20:21]
	s_and_saveexec_b64 s[6:7], s[16:17]
	s_cbranch_execnz .LBB1_98
	s_branch .LBB1_101
.LBB1_20:
	v_lshlrev_b64 v[21:22], 9, v[7:8]
	v_cmp_lt_i64_e32 vcc, -1, v[15:16]
	s_mov_b64 s[16:17], 0
	s_mov_b64 s[18:19], 0
	s_mov_b64 s[14:15], 0
	s_and_saveexec_b64 s[4:5], vcc
	s_xor_b64 s[20:21], exec, s[4:5]
	s_cbranch_execz .LBB1_30
; %bb.21:
	s_mov_b64 s[4:5], 0x7ff
	v_cmp_ne_u64_e32 vcc, s[4:5], v[11:12]
	s_mov_b64 s[14:15], 0
	s_mov_b64 s[4:5], 0
	s_and_saveexec_b64 s[18:19], vcc
	s_xor_b64 s[18:19], exec, s[18:19]
	s_cbranch_execz .LBB1_27
; %bb.22:
	v_lshlrev_b64 v[17:18], 10, v[7:8]
	v_cmp_eq_u64_e32 vcc, 0, v[13:14]
	v_or_b32_e32 v19, 0x20000000, v22
	v_cmp_lt_u64_e64 s[4:5], 62, v[15:16]
	v_cndmask_b32_e32 v14, v19, v18, vcc
	v_cndmask_b32_e32 v13, v21, v17, vcc
                                        ; implicit-def: $vgpr21_vgpr22
	s_and_saveexec_b64 s[22:23], s[4:5]
	s_xor_b64 s[4:5], exec, s[22:23]
; %bb.23:
	v_cmp_ne_u64_e32 vcc, 0, v[13:14]
	s_mov_b32 s22, 0
	v_cndmask_b32_e64 v21, 0, 1, vcc
	v_mov_b32_e32 v22, s22
                                        ; implicit-def: $vgpr15
                                        ; implicit-def: $vgpr13_vgpr14
; %bb.24:
	s_andn2_saveexec_b64 s[4:5], s[4:5]
; %bb.25:
	v_sub_u32_e32 v16, 0, v15
	v_lshlrev_b64 v[16:17], v16, v[13:14]
	v_lshrrev_b64 v[21:22], v15, v[13:14]
	v_cmp_ne_u64_e32 vcc, 0, v[16:17]
	v_cndmask_b32_e64 v13, 0, 1, vcc
	v_or_b32_e32 v21, v21, v13
; %bb.26:
	s_or_b64 exec, exec, s[4:5]
	s_mov_b64 s[4:5], exec
.LBB1_27:
	s_andn2_saveexec_b64 s[18:19], s[18:19]
; %bb.28:
	v_cmp_ne_u64_e32 vcc, 0, v[9:10]
	s_and_b64 s[14:15], vcc, exec
; %bb.29:
	s_or_b64 exec, exec, s[18:19]
	s_and_b64 s[14:15], s[14:15], exec
	s_and_b64 s[18:19], s[4:5], exec
                                        ; implicit-def: $vgpr13_vgpr14
                                        ; implicit-def: $vgpr15
.LBB1_30:
	s_or_saveexec_b64 s[20:21], s[20:21]
	v_lshlrev_b64 v[23:24], 9, v[9:10]
	v_mov_b32_e32 v26, v12
	v_mov_b32_e32 v25, v11
	s_xor_b64 exec, exec, s[20:21]
	s_cbranch_execz .LBB1_38
; %bb.31:
	s_mov_b64 s[4:5], 0x7ff
	v_cmp_ne_u64_e32 vcc, s[4:5], v[13:14]
	v_mov_b32_e32 v26, v12
	s_mov_b64 s[22:23], -1
	s_mov_b64 s[4:5], s[18:19]
	v_mov_b32_e32 v25, v11
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB1_37
; %bb.32:
	v_sub_co_u32_e64 v18, s[4:5], 0, v15
	v_or_b32_e32 v17, 0x20000000, v24
	v_lshlrev_b64 v[24:25], 10, v[9:10]
	v_cmp_eq_u64_e32 vcc, 0, v[11:12]
	v_subb_co_u32_e64 v19, s[4:5], 0, v16, s[4:5]
	v_cmp_lt_u64_e64 s[4:5], 62, v[18:19]
	v_cndmask_b32_e32 v17, v17, v25, vcc
	v_cndmask_b32_e32 v16, v23, v24, vcc
                                        ; implicit-def: $vgpr23_vgpr24
	s_and_saveexec_b64 s[22:23], s[4:5]
	s_xor_b64 s[4:5], exec, s[22:23]
; %bb.33:
	v_cmp_ne_u64_e32 vcc, 0, v[16:17]
	s_mov_b32 s22, 0
	v_cndmask_b32_e64 v23, 0, 1, vcc
	v_mov_b32_e32 v24, s22
                                        ; implicit-def: $vgpr18
                                        ; implicit-def: $vgpr16_vgpr17
                                        ; implicit-def: $vgpr15
; %bb.34:
	s_andn2_saveexec_b64 s[4:5], s[4:5]
; %bb.35:
	v_lshlrev_b64 v[19:20], v15, v[16:17]
	v_lshrrev_b64 v[23:24], v18, v[16:17]
	v_cmp_ne_u64_e32 vcc, 0, v[19:20]
	v_cndmask_b32_e64 v15, 0, 1, vcc
	v_or_b32_e32 v23, v23, v15
; %bb.36:
	s_or_b64 exec, exec, s[4:5]
	v_mov_b32_e32 v26, v14
	s_xor_b64 s[22:23], exec, -1
	s_or_b64 s[4:5], s[18:19], exec
	v_mov_b32_e32 v25, v13
.LBB1_37:
	s_or_b64 exec, exec, s[16:17]
	s_andn2_b64 s[18:19], s[18:19], exec
	s_and_b64 s[4:5], s[4:5], exec
	s_and_b64 s[16:17], s[22:23], exec
	s_or_b64 s[18:19], s[18:19], s[4:5]
.LBB1_38:
	s_or_b64 exec, exec, s[20:21]
	s_mov_b64 s[4:5], 0
                                        ; implicit-def: $vgpr17_vgpr18
                                        ; implicit-def: $vgpr19_vgpr20
	s_and_saveexec_b64 s[20:21], s[18:19]
	s_xor_b64 s[18:19], exec, s[20:21]
; %bb.39:
	v_add_co_u32_e32 v13, vcc, v21, v23
	v_addc_co_u32_e32 v14, vcc, v22, v24, vcc
	v_bfrev_b32_e32 v15, 4
	v_add_co_u32_e32 v13, vcc, 0, v13
	v_addc_co_u32_e32 v14, vcc, v14, v15, vcc
	v_cmp_gt_u64_e32 vcc, 2.0, v[13:14]
	s_mov_b64 s[4:5], exec
	v_cndmask_b32_e64 v15, 0, 1, vcc
	v_lshlrev_b64 v[17:18], v15, v[13:14]
	v_sub_co_u32_e32 v19, vcc, v25, v15
	v_subbrev_co_u32_e32 v20, vcc, 0, v26, vcc
; %bb.40:
	s_or_b64 exec, exec, s[18:19]
	s_and_saveexec_b64 s[18:19], s[16:17]
	s_cbranch_execz .LBB1_44
; %bb.41:
	v_cmp_eq_u64_e32 vcc, 0, v[7:8]
	s_mov_b64 s[16:17], -1
	s_and_saveexec_b64 s[20:21], vcc
	s_xor_b64 s[20:21], exec, s[20:21]
; %bb.42:
	v_mov_b32_e32 v5, 0x7ff00000
	v_mov_b32_e32 v6, 0xfff00000
	v_cndmask_b32_e64 v6, v5, v6, s[8:9]
	v_mov_b32_e32 v5, 0
	s_xor_b64 s[16:17], exec, -1
; %bb.43:
	s_or_b64 exec, exec, s[20:21]
	s_andn2_b64 s[14:15], s[14:15], exec
	s_and_b64 s[16:17], s[16:17], exec
	s_or_b64 s[14:15], s[14:15], s[16:17]
.LBB1_44:
	s_or_b64 exec, exec, s[18:19]
	s_and_b64 s[16:17], s[4:5], exec
	s_and_b64 s[14:15], s[14:15], exec
	s_andn2_saveexec_b64 s[4:5], s[12:13]
	s_cbranch_execz .LBB1_9
.LBB1_45:
	s_mov_b64 s[10:11], 0x7fe
	v_cmp_lt_i64_e32 vcc, s[10:11], v[11:12]
	s_mov_b64 s[10:11], -1
	s_mov_b64 s[12:13], s[14:15]
	s_and_saveexec_b64 s[18:19], vcc
; %bb.46:
	v_or_b32_e32 v14, v8, v10
	v_or_b32_e32 v13, v7, v9
	v_cmp_ne_u64_e32 vcc, 0, v[13:14]
	s_andn2_b64 s[12:13], s[14:15], exec
	s_and_b64 s[20:21], vcc, exec
	s_xor_b64 s[10:11], exec, -1
	s_or_b64 s[12:13], s[12:13], s[20:21]
; %bb.47:
	s_or_b64 exec, exec, s[18:19]
	s_andn2_b64 s[14:15], s[14:15], exec
	s_and_b64 s[12:13], s[12:13], exec
	s_and_b64 s[10:11], s[10:11], exec
	s_or_b64 s[14:15], s[14:15], s[12:13]
	s_or_b64 exec, exec, s[4:5]
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_cbranch_execz .LBB1_10
.LBB1_48:
	s_mov_b32 s4, 0
	s_mov_b32 s5, 0x7ff00000
	v_and_b32_e32 v6, 0x7ff00000, v1
	v_mov_b32_e32 v5, 0
	v_cmp_ne_u64_e32 vcc, s[4:5], v[5:6]
	v_cmp_eq_u64_e64 s[4:5], 0, v[9:10]
	s_or_b64 vcc, vcc, s[4:5]
	v_cndmask_b32_e32 v6, v1, v3, vcc
	v_cndmask_b32_e32 v5, v0, v2, vcc
	s_or_b64 exec, exec, s[12:13]
	s_mov_b64 s[12:13], 0
	s_and_saveexec_b64 s[4:5], s[10:11]
	s_cbranch_execz .LBB1_11
.LBB1_49:
	v_cmp_ne_u64_e32 vcc, 0, v[11:12]
	s_mov_b64 s[14:15], -1
	s_mov_b64 s[10:11], s[16:17]
	s_and_saveexec_b64 s[12:13], vcc
; %bb.50:
	v_add_co_u32_e32 v2, vcc, v7, v9
	v_addc_co_u32_e32 v3, vcc, v8, v10, vcc
	v_lshlrev_b64 v[17:18], 9, v[2:3]
	v_mov_b32_e32 v20, v12
	v_or_b32_e32 v18, 2.0, v18
	s_xor_b64 s[14:15], exec, -1
	s_or_b64 s[10:11], s[16:17], exec
	v_mov_b32_e32 v19, v11
; %bb.51:
	s_or_b64 exec, exec, s[12:13]
	s_and_b64 s[12:13], s[14:15], exec
	s_andn2_b64 s[14:15], s[16:17], exec
	s_and_b64 s[10:11], s[10:11], exec
	s_or_b64 s[16:17], s[14:15], s[10:11]
	s_or_b64 exec, exec, s[4:5]
	s_and_saveexec_b64 s[4:5], s[16:17]
	s_xor_b64 s[10:11], exec, s[4:5]
	s_cbranch_execz .LBB1_12
.LBB1_52:
	v_and_b32_e32 v2, 0xffff, v19
	s_mov_b64 s[4:5], 0x7fc
	v_mov_b32_e32 v3, 0
	v_cmp_lt_u64_e32 vcc, s[4:5], v[2:3]
	s_mov_b64 s[16:17], -1
                                        ; implicit-def: $vgpr5_vgpr6
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB1_60
; %bb.53:
	v_cmp_lt_i64_e32 vcc, -1, v[19:20]
	s_mov_b64 s[18:19], -1
	s_mov_b64 s[4:5], 0
                                        ; implicit-def: $vgpr5_vgpr6
	s_and_saveexec_b64 s[16:17], vcc
	s_xor_b64 s[16:17], exec, s[16:17]
	s_cbranch_execz .LBB1_57
; %bb.54:
	v_add_co_u32_e32 v2, vcc, 0x200, v17
	s_mov_b64 s[20:21], 0x7fd
	v_addc_co_u32_e32 v3, vcc, 0, v18, vcc
	v_cmp_lt_u64_e64 s[4:5], s[20:21], v[19:20]
	v_cmp_gt_i64_e32 vcc, 0, v[2:3]
                                        ; implicit-def: $vgpr5_vgpr6
	s_or_b64 s[4:5], s[4:5], vcc
	s_and_saveexec_b64 s[22:23], s[4:5]
	s_xor_b64 s[4:5], exec, s[22:23]
; %bb.55:
	v_mov_b32_e32 v2, 0x7ff00000
	v_mov_b32_e32 v3, 0xfff00000
	v_cndmask_b32_e64 v6, v2, v3, s[8:9]
	v_mov_b32_e32 v5, 0
	s_xor_b64 s[18:19], exec, -1
; %bb.56:
	s_or_b64 exec, exec, s[4:5]
	v_mov_b32_e32 v19, s20
	v_mov_b32_e32 v20, s21
	s_and_b64 s[4:5], s[18:19], exec
.LBB1_57:
	s_andn2_saveexec_b64 s[8:9], s[16:17]
; %bb.58:
	v_lshrrev_b64 v[2:3], 1, v[17:18]
	v_and_b32_e32 v9, 1, v17
	v_or_b32_e32 v2, v2, v9
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v18, v3
	v_mov_b32_e32 v20, 0
	s_or_b64 s[4:5], s[4:5], exec
	v_mov_b32_e32 v17, v2
; %bb.59:
	s_or_b64 exec, exec, s[8:9]
	s_orn2_b64 s[16:17], s[4:5], exec
.LBB1_60:
	s_or_b64 exec, exec, s[14:15]
	s_and_saveexec_b64 s[4:5], s[16:17]
	s_cbranch_execz .LBB1_62
; %bb.61:
	v_add_co_u32_e32 v5, vcc, 0x200, v17
	v_and_b32_e32 v2, 0x3ff, v17
	s_mov_b64 s[8:9], 0x200
	v_mov_b32_e32 v3, 0
	v_addc_co_u32_e32 v6, vcc, 0, v18, vcc
	v_cmp_eq_u64_e32 vcc, s[8:9], v[2:3]
	v_lshrrev_b64 v[5:6], 10, v[5:6]
	v_cndmask_b32_e64 v2, 0, 1, vcc
	v_not_b32_e32 v2, v2
	v_and_b32_e32 v5, v5, v2
	v_cmp_ne_u64_e32 vcc, 0, v[5:6]
	v_and_b32_e32 v2, 0x80000000, v1
	v_lshlrev_b32_e32 v3, 20, v19
	v_cndmask_b32_e32 v3, 0, v3, vcc
	v_or_b32_e32 v2, v6, v2
	v_add_co_u32_e32 v5, vcc, 0, v5
	v_addc_co_u32_e32 v6, vcc, v2, v3, vcc
.LBB1_62:
	s_or_b64 exec, exec, s[4:5]
	s_or_b64 exec, exec, s[10:11]
	s_and_saveexec_b64 s[4:5], s[12:13]
	s_cbranch_execnz .LBB1_13
	s_branch .LBB1_14
.LBB1_63:
	s_mov_b64 s[22:23], 0x7ff
	v_cmp_ne_u64_e32 vcc, s[22:23], v[11:12]
	s_mov_b64 s[14:15], 0
	s_mov_b64 s[6:7], 0
                                        ; implicit-def: $vgpr17_vgpr18
	s_and_saveexec_b64 s[20:21], vcc
	s_xor_b64 s[20:21], exec, s[20:21]
	s_cbranch_execz .LBB1_69
; %bb.64:
	v_cmp_eq_u64_e32 vcc, 0, v[13:14]
	v_cndmask_b32_e32 v14, 0, v19, vcc
	v_cndmask_b32_e32 v13, 2.0, v20, vcc
	v_cmp_lt_u64_e32 vcc, 62, v[15:16]
	v_add_co_u32_e64 v16, s[6:7], v14, v19
	v_addc_co_u32_e64 v17, s[6:7], v13, v20, s[6:7]
                                        ; implicit-def: $vgpr13_vgpr14
	s_and_saveexec_b64 s[6:7], vcc
	s_xor_b64 s[6:7], exec, s[6:7]
; %bb.65:
	v_cmp_ne_u64_e32 vcc, 0, v[16:17]
	v_mov_b32_e32 v14, s23
	v_cndmask_b32_e64 v13, 0, 1, vcc
                                        ; implicit-def: $vgpr15
                                        ; implicit-def: $vgpr16_vgpr17
; %bb.66:
	s_andn2_saveexec_b64 s[6:7], s[6:7]
; %bb.67:
	v_sub_u32_e32 v13, 0, v15
	v_lshlrev_b64 v[18:19], v13, v[16:17]
	v_lshrrev_b64 v[13:14], v15, v[16:17]
	v_cmp_ne_u64_e32 vcc, 0, v[18:19]
	v_cndmask_b32_e64 v15, 0, 1, vcc
	v_or_b32_e32 v13, v13, v15
; %bb.68:
	s_or_b64 exec, exec, s[6:7]
	v_sub_co_u32_e32 v17, vcc, v21, v13
	s_mov_b64 s[6:7], exec
	v_subb_co_u32_e32 v18, vcc, v22, v14, vcc
.LBB1_69:
	s_andn2_saveexec_b64 s[20:21], s[20:21]
; %bb.70:
	v_cmp_ne_u64_e32 vcc, 0, v[9:10]
	s_and_b64 s[14:15], vcc, exec
; %bb.71:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[14:15], s[14:15], exec
	s_and_b64 s[20:21], s[6:7], exec
                                        ; implicit-def: $vgpr13_vgpr14
                                        ; implicit-def: $vgpr19_vgpr20
                                        ; implicit-def: $vgpr15
                                        ; implicit-def: $vgpr21_vgpr22
	s_or_saveexec_b64 s[22:23], s[18:19]
	s_mov_b64 s[18:19], s[8:9]
	s_xor_b64 exec, exec, s[22:23]
	s_cbranch_execz .LBB1_18
.LBB1_72:
	s_mov_b64 s[18:19], 0x7ff
	v_cmp_ne_u64_e32 vcc, s[18:19], v[13:14]
	s_mov_b64 s[24:25], -1
	s_mov_b64 s[6:7], s[20:21]
	s_mov_b64 s[26:27], s[8:9]
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB1_78
; %bb.73:
	v_cmp_eq_u64_e32 vcc, 0, v[11:12]
	v_cndmask_b32_e32 v11, 2.0, v22, vcc
	v_cndmask_b32_e32 v12, 0, v21, vcc
	v_sub_co_u32_e32 v17, vcc, 0, v15
	v_subb_co_u32_e32 v18, vcc, 0, v16, vcc
	v_cmp_lt_u64_e32 vcc, 62, v[17:18]
	v_add_co_u32_e64 v21, s[6:7], v12, v21
	v_addc_co_u32_e64 v22, s[6:7], v11, v22, s[6:7]
                                        ; implicit-def: $vgpr11_vgpr12
	s_and_saveexec_b64 s[6:7], vcc
	s_xor_b64 s[6:7], exec, s[6:7]
; %bb.74:
	v_cmp_ne_u64_e32 vcc, 0, v[21:22]
	v_mov_b32_e32 v12, s19
	v_cndmask_b32_e64 v11, 0, 1, vcc
                                        ; implicit-def: $vgpr17
                                        ; implicit-def: $vgpr21_vgpr22
                                        ; implicit-def: $vgpr15
; %bb.75:
	s_andn2_saveexec_b64 s[6:7], s[6:7]
; %bb.76:
	v_lshlrev_b64 v[15:16], v15, v[21:22]
	v_lshrrev_b64 v[11:12], v17, v[21:22]
	v_cmp_ne_u64_e32 vcc, 0, v[15:16]
	v_cndmask_b32_e64 v15, 0, 1, vcc
	v_or_b32_e32 v11, v11, v15
; %bb.77:
	s_or_b64 exec, exec, s[6:7]
	s_xor_b64 s[6:7], s[8:9], -1
	v_sub_co_u32_e32 v17, vcc, v19, v11
	v_subb_co_u32_e32 v18, vcc, v20, v12, vcc
	s_andn2_b64 s[18:19], s[8:9], exec
	s_and_b64 s[6:7], s[6:7], exec
	v_mov_b32_e32 v11, v13
	s_or_b64 s[26:27], s[18:19], s[6:7]
	s_xor_b64 s[24:25], exec, -1
	s_or_b64 s[6:7], s[20:21], exec
	v_mov_b32_e32 v12, v14
.LBB1_78:
	s_or_b64 exec, exec, s[16:17]
	s_andn2_b64 s[16:17], s[8:9], exec
	s_and_b64 s[18:19], s[26:27], exec
	s_andn2_b64 s[20:21], s[20:21], exec
	s_and_b64 s[6:7], s[6:7], exec
	s_or_b64 s[18:19], s[16:17], s[18:19]
	s_and_b64 s[16:17], s[24:25], exec
	s_or_b64 s[20:21], s[20:21], s[6:7]
	s_or_b64 exec, exec, s[22:23]
	s_and_saveexec_b64 s[6:7], s[20:21]
	s_xor_b64 s[20:21], exec, s[6:7]
	s_cbranch_execz .LBB1_19
.LBB1_79:
	v_add_co_u32_e32 v15, vcc, 0, v17
	v_addc_co_u32_e32 v16, vcc, 2.0, v18, vcc
	v_cmp_eq_u32_e32 vcc, 0, v16
	v_cndmask_b32_e64 v5, 0, 1, vcc
	v_cndmask_b32_e32 v6, v16, v17, vcc
	s_mov_b32 s6, 0x10000
	v_lshlrev_b16_e32 v5, 5, v5
	v_lshlrev_b32_e32 v14, 16, v6
	v_cmp_gt_u32_e32 vcc, s6, v6
	v_or_b32_e32 v13, 16, v5
	v_cndmask_b32_e32 v6, v6, v14, vcc
	s_mov_b32 s6, 0x1000000
	v_cndmask_b32_e32 v5, v5, v13, vcc
	v_lshlrev_b32_e32 v14, 8, v6
	v_cmp_gt_u32_e32 vcc, s6, v6
	v_cndmask_b32_e32 v6, v6, v14, vcc
	v_lshrrev_b32_e32 v6, 24, v6
	s_getpc_b64 s[6:7]
	s_add_u32 s6, s6, softfloat_countLeadingZeros8@rel32@lo+4
	s_addc_u32 s7, s7, softfloat_countLeadingZeros8@rel32@hi+12
	global_load_ubyte v6, v6, s[6:7]
	v_or_b32_e32 v13, 8, v5
	v_cndmask_b32_e32 v5, v5, v13, vcc
	v_mov_b32_e32 v13, 10
	s_mov_b64 s[6:7], 0x7fc
	s_waitcnt vmcnt(0)
	v_add_u16_e32 v5, v6, v5
	v_add_u16_e32 v6, -1, v5
	v_sub_u16_e32 v5, 0, v5
	v_bfe_i32 v5, v5, 0, 8
	v_and_b32_e32 v17, 0xff, v6
	v_cmp_lt_i16_sdwa s[22:23], sext(v6), v13 src0_sel:BYTE_0 src1_sel:DWORD
	v_ashrrev_i32_e32 v6, 31, v5
	v_add_co_u32_e32 v13, vcc, v11, v5
	v_addc_co_u32_e32 v14, vcc, v12, v6, vcc
	v_cmp_lt_u32_e32 vcc, s6, v13
	s_or_b64 s[22:23], s[22:23], vcc
                                        ; implicit-def: $vgpr5_vgpr6
	s_and_saveexec_b64 s[24:25], s[22:23]
	s_xor_b64 s[22:23], exec, s[24:25]
	s_cbranch_execz .LBB1_95
; %bb.80:
	v_and_b32_e32 v5, 0xffff, v13
	v_mov_b32_e32 v6, 0
	v_lshlrev_b64 v[11:12], v17, v[15:16]
	v_cmp_lt_u64_e32 vcc, s[6:7], v[5:6]
	s_mov_b64 s[26:27], -1
                                        ; implicit-def: $vgpr5_vgpr6
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB1_92
; %bb.81:
	v_cmp_lt_i64_e32 vcc, -1, v[13:14]
	s_mov_b64 s[28:29], -1
	s_mov_b64 s[6:7], 0
                                        ; implicit-def: $vgpr5_vgpr6
	s_and_saveexec_b64 s[26:27], vcc
	s_xor_b64 s[26:27], exec, s[26:27]
	s_cbranch_execz .LBB1_85
; %bb.82:
	v_add_co_u32_e32 v5, vcc, 0x200, v11
	s_mov_b64 s[30:31], 0x7fd
	v_addc_co_u32_e32 v6, vcc, 0, v12, vcc
	v_cmp_lt_u64_e64 s[6:7], s[30:31], v[13:14]
	v_cmp_gt_i64_e32 vcc, 0, v[5:6]
                                        ; implicit-def: $vgpr5_vgpr6
	s_or_b64 s[6:7], s[6:7], vcc
	s_and_saveexec_b64 vcc, s[6:7]
	s_xor_b64 s[6:7], exec, vcc
; %bb.83:
	v_mov_b32_e32 v5, 0x7ff00000
	v_mov_b32_e32 v6, 0xfff00000
	v_cndmask_b32_e64 v6, v5, v6, s[18:19]
	v_mov_b32_e32 v5, 0
	s_xor_b64 s[28:29], exec, -1
; %bb.84:
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v13, s30
	v_mov_b32_e32 v14, s31
	s_and_b64 s[6:7], s[28:29], exec
.LBB1_85:
	s_andn2_saveexec_b64 s[26:27], s[26:27]
	s_cbranch_execz .LBB1_91
; %bb.86:
	v_sub_co_u32_e32 v15, vcc, 0, v13
	v_subb_co_u32_e32 v16, vcc, 0, v14, vcc
	v_cmp_lt_u64_e32 vcc, 62, v[15:16]
	s_and_saveexec_b64 s[28:29], vcc
	s_xor_b64 s[28:29], exec, s[28:29]
; %bb.87:
	v_cmp_ne_u64_e32 vcc, 0, v[11:12]
                                        ; implicit-def: $vgpr15
                                        ; implicit-def: $vgpr13
	v_cndmask_b32_e64 v11, 0, 1, vcc
	s_mov_b32 vcc_lo, 0
	v_mov_b32_e32 v12, vcc_lo
; %bb.88:
	s_andn2_saveexec_b64 s[28:29], s[28:29]
; %bb.89:
	v_lshlrev_b64 v[13:14], v13, v[11:12]
	v_lshrrev_b64 v[11:12], v15, v[11:12]
	v_cmp_ne_u64_e32 vcc, 0, v[13:14]
	v_cndmask_b32_e64 v13, 0, 1, vcc
	v_or_b32_e32 v11, v11, v13
; %bb.90:
	s_or_b64 exec, exec, s[28:29]
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v14, 0
	s_or_b64 s[6:7], s[6:7], exec
.LBB1_91:
	s_or_b64 exec, exec, s[26:27]
	s_orn2_b64 s[26:27], s[6:7], exec
.LBB1_92:
	s_or_b64 exec, exec, s[24:25]
	s_and_saveexec_b64 s[6:7], s[26:27]
	s_cbranch_execz .LBB1_94
; %bb.93:
	v_and_b32_e32 v5, 0x3ff, v11
	v_add_co_u32_e32 v11, vcc, 0x200, v11
	s_mov_b64 s[24:25], 0x200
	v_mov_b32_e32 v6, 0
	v_addc_co_u32_e32 v12, vcc, 0, v12, vcc
	v_cmp_eq_u64_e32 vcc, s[24:25], v[5:6]
	v_lshrrev_b64 v[11:12], 10, v[11:12]
	v_cndmask_b32_e64 v5, 0, 1, vcc
	v_not_b32_e32 v5, v5
	v_and_b32_e32 v11, v11, v5
	v_bfrev_b32_e32 v5, 1
	v_cmp_ne_u64_e32 vcc, 0, v[11:12]
	v_cndmask_b32_e64 v5, 0, v5, s[18:19]
	v_lshlrev_b32_e32 v6, 20, v13
	v_cndmask_b32_e32 v6, 0, v6, vcc
	v_or_b32_e32 v12, v12, v5
	v_add_co_u32_e32 v5, vcc, 0, v11
	v_addc_co_u32_e32 v6, vcc, v12, v6, vcc
.LBB1_94:
	s_or_b64 exec, exec, s[6:7]
                                        ; implicit-def: $vgpr13
                                        ; implicit-def: $vgpr15_vgpr16
                                        ; implicit-def: $vgpr17
.LBB1_95:
	s_andn2_saveexec_b64 s[6:7], s[22:23]
; %bb.96:
	v_cmp_ne_u64_e32 vcc, 0, v[15:16]
	v_bfrev_b32_e32 v5, 1
	v_lshlrev_b32_e32 v6, 20, v13
	v_cndmask_b32_e64 v5, 0, v5, s[18:19]
	v_cndmask_b32_e32 v6, 0, v6, vcc
	v_add_co_u32_e64 v11, vcc, 0, 0
	v_addc_co_u32_e32 v12, vcc, v6, v5, vcc
	v_add_u32_e32 v5, 54, v17
	v_lshlrev_b64 v[5:6], v5, v[15:16]
	v_add_co_u32_e32 v5, vcc, v11, v5
	v_addc_co_u32_e32 v6, vcc, v12, v6, vcc
; %bb.97:
	s_or_b64 exec, exec, s[6:7]
	s_or_b64 exec, exec, s[20:21]
	s_and_saveexec_b64 s[6:7], s[16:17]
	s_cbranch_execz .LBB1_101
.LBB1_98:
	v_cmp_eq_u64_e32 vcc, 0, v[7:8]
	s_mov_b64 s[16:17], -1
	s_and_saveexec_b64 s[18:19], vcc
	s_xor_b64 s[18:19], exec, s[18:19]
; %bb.99:
	v_mov_b32_e32 v5, 0xfff00000
	v_mov_b32_e32 v6, 0x7ff00000
	v_cndmask_b32_e64 v6, v5, v6, s[8:9]
	v_mov_b32_e32 v5, 0
	s_xor_b64 s[16:17], exec, -1
; %bb.100:
	s_or_b64 exec, exec, s[18:19]
	s_andn2_b64 s[14:15], s[14:15], exec
	s_and_b64 s[16:17], s[16:17], exec
	s_or_b64 s[14:15], s[14:15], s[16:17]
.LBB1_101:
	s_or_b64 exec, exec, s[6:7]
	s_and_b64 s[14:15], s[14:15], exec
                                        ; implicit-def: $vgpr7_vgpr8
                                        ; implicit-def: $vgpr11_vgpr12
	s_andn2_saveexec_b64 s[6:7], s[12:13]
	s_cbranch_execz .LBB1_3
.LBB1_102:
	s_mov_b64 s[12:13], 0x7ff
	v_cmp_ne_u64_e32 vcc, s[12:13], v[11:12]
	s_and_saveexec_b64 s[12:13], vcc
	s_xor_b64 s[12:13], exec, s[12:13]
	s_cbranch_execz .LBB1_106
; %bb.103:
	v_cmp_ne_u64_e32 vcc, v[9:10], v[7:8]
	v_mov_b32_e32 v5, 0
	v_mov_b32_e32 v6, 0
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB1_105
; %bb.104:
	v_sub_co_u32_e32 v5, vcc, v9, v7
	v_subb_co_u32_e32 v6, vcc, v10, v8, vcc
	v_ashrrev_i32_e32 v8, 31, v6
	v_xor_b32_e32 v5, v5, v8
	v_xor_b32_e32 v13, v6, v8
	v_sub_co_u32_e32 v7, vcc, v5, v8
	v_subb_co_u32_e32 v8, vcc, v13, v8, vcc
	v_cmp_eq_u32_e32 vcc, 0, v8
	v_cndmask_b32_e64 v5, 0, 1, vcc
	v_cndmask_b32_e32 v13, v8, v7, vcc
	s_mov_b32 s18, 0x10000
	v_lshlrev_b16_e32 v5, 5, v5
	v_lshlrev_b32_e32 v15, 16, v13
	v_cmp_gt_u32_e32 vcc, s18, v13
	v_or_b32_e32 v14, 16, v5
	v_cndmask_b32_e32 v13, v13, v15, vcc
	s_mov_b32 s18, 0x1000000
	v_cndmask_b32_e32 v5, v5, v14, vcc
	v_lshlrev_b32_e32 v15, 8, v13
	v_cmp_gt_u32_e32 vcc, s18, v13
	v_cndmask_b32_e32 v13, v13, v15, vcc
	v_lshrrev_b32_e32 v13, 24, v13
	s_getpc_b64 s[18:19]
	s_add_u32 s18, s18, softfloat_countLeadingZeros8@rel32@lo+4
	s_addc_u32 s19, s19, softfloat_countLeadingZeros8@rel32@hi+12
	global_load_ubyte v15, v13, s[18:19]
	v_or_b32_e32 v14, 8, v5
	v_cndmask_b32_e32 v5, v5, v14, vcc
	v_add_co_u32_e32 v13, vcc, -1, v11
	v_addc_co_u32_e64 v14, s[18:19], 0, -1, vcc
	v_cmp_gt_u64_e32 vcc, v[13:14], v[11:12]
	v_xor_b32_e32 v6, v6, v1
	v_cndmask_b32_e64 v11, v14, 0, vcc
	v_cndmask_b32_e64 v13, v13, 0, vcc
	v_and_b32_e32 v12, 0x80000000, v6
	s_waitcnt vmcnt(0)
	v_add_u16_e32 v5, v5, v15
	v_add_u16_e32 v14, -11, v5
	v_bfe_i32 v5, v14, 0, 8
	v_ashrrev_i32_e32 v6, 31, v5
	v_sub_co_u32_e32 v5, vcc, v13, v5
	v_subb_co_u32_e32 v6, vcc, v11, v6, vcc
	v_cmp_gt_i64_e32 vcc, 0, v[5:6]
	v_cndmask_b32_e32 v11, v14, v13, vcc
	v_cmp_lt_i64_e32 vcc, 0, v[5:6]
	v_and_b32_e32 v6, 63, v11
	v_cndmask_b32_e32 v5, 0, v5, vcc
	v_lshlrev_b32_e32 v5, 20, v5
	v_add_co_u32_e64 v11, vcc, 0, 0
	v_addc_co_u32_e32 v12, vcc, v5, v12, vcc
	v_and_b32_e32 v5, 0xffff, v6
	v_lshlrev_b64 v[5:6], v5, v[7:8]
	v_add_co_u32_e32 v5, vcc, v11, v5
	v_addc_co_u32_e32 v6, vcc, v12, v6, vcc
.LBB1_105:
	s_or_b64 exec, exec, s[16:17]
                                        ; implicit-def: $vgpr7_vgpr8
.LBB1_106:
	s_or_saveexec_b64 s[12:13], s[12:13]
	s_mov_b64 s[16:17], s[14:15]
	s_xor_b64 exec, exec, s[12:13]
; %bb.107:
	v_or_b32_e32 v6, v8, v10
	v_or_b32_e32 v5, v7, v9
	v_cmp_ne_u64_e32 vcc, 0, v[5:6]
	s_mov_b32 s16, 0
	s_mov_b32 s17, 0xfff80000
	v_mov_b32_e32 v5, s16
	v_mov_b32_e32 v6, s17
	s_andn2_b64 s[16:17], s[14:15], exec
	s_and_b64 s[18:19], vcc, exec
	s_or_b64 s[16:17], s[16:17], s[18:19]
; %bb.108:
	s_or_b64 exec, exec, s[12:13]
	s_andn2_b64 s[12:13], s[14:15], exec
	s_and_b64 s[14:15], s[16:17], exec
	s_or_b64 s[14:15], s[12:13], s[14:15]
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_cbranch_execnz .LBB1_4
	s_branch .LBB1_5
.Lfunc_end1:
	.size	__softfloat_f64_add, .Lfunc_end1-__softfloat_f64_add
                                        ; -- End function
	.section	.AMDGPU.csdata
; Function info:
; codeLenInByte = 3344
; NumSgprs: 40
; NumVgprs: 27
; ScratchSize: 8
; MemoryBound: 0
	.text
	.p2align	2                               ; -- Begin function __softfloat_f64_div
	.type	__softfloat_f64_div,@function
__softfloat_f64_div:                    ; @__softfloat_f64_div
; %bb.0:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_xor_saveexec_b64 s[4:5], -1
	buffer_store_dword v4, off, s[0:3], s32 ; 4-byte Folded Spill
	s_mov_b64 exec, s[4:5]
	v_writelane_b32 v4, s34, 0
	v_writelane_b32 v4, s35, 1
	v_writelane_b32 v4, s30, 2
	v_writelane_b32 v4, s31, 3
	v_bfe_u32 v11, v1, 20, 11
	s_mov_b64 s[4:5], 0x7ff
	v_mov_b32_e32 v12, 0
	v_cmp_ne_u64_e32 vcc, s[4:5], v[11:12]
	v_and_b32_e32 v8, 0xfffff, v1
	v_mov_b32_e32 v7, v0
	v_bfe_u32 v13, v3, 20, 11
	v_mov_b32_e32 v14, v12
	v_and_b32_e32 v10, 0xfffff, v3
	v_mov_b32_e32 v9, v2
	v_xor_b32_e32 v21, v3, v1
	s_mov_b64 s[8:9], 0
	s_mov_b64 s[6:7], 0
	s_mov_b64 s[4:5], 0
                                        ; implicit-def: $vgpr5_vgpr6
	s_and_saveexec_b64 s[10:11], vcc
	s_xor_b64 s[10:11], exec, s[10:11]
	s_cbranch_execnz .LBB2_7
; %bb.1:
	s_andn2_saveexec_b64 s[10:11], s[10:11]
	s_cbranch_execnz .LBB2_14
.LBB2_2:
	s_or_b64 exec, exec, s[10:11]
	s_and_saveexec_b64 s[10:11], s[6:7]
	s_cbranch_execnz .LBB2_17
.LBB2_3:
	s_or_b64 exec, exec, s[10:11]
	s_and_saveexec_b64 s[6:7], s[8:9]
	s_cbranch_execnz .LBB2_18
.LBB2_4:
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[6:7], s[4:5]
.LBB2_5:
	s_mov_b32 s4, 0
	s_mov_b32 s5, 0x7ff00000
	v_and_b32_e32 v6, 0x7ff00000, v1
	v_mov_b32_e32 v5, 0
	v_cmp_ne_u64_e32 vcc, s[4:5], v[5:6]
	v_cmp_eq_u64_e64 s[4:5], 0, v[7:8]
	s_or_b64 vcc, vcc, s[4:5]
	v_cndmask_b32_e32 v6, v1, v3, vcc
	v_cndmask_b32_e32 v5, v0, v2, vcc
.LBB2_6:
	s_or_b64 exec, exec, s[6:7]
	v_readlane_b32 s30, v4, 2
	v_mov_b32_e32 v0, v5
	v_mov_b32_e32 v1, v6
	v_readlane_b32 s31, v4, 3
	v_readlane_b32 s35, v4, 1
	v_readlane_b32 s34, v4, 0
	s_xor_saveexec_b64 s[4:5], -1
	buffer_load_dword v4, off, s[0:3], s32  ; 4-byte Folded Reload
	s_mov_b64 exec, s[4:5]
	s_waitcnt vmcnt(0)
	s_setpc_b64 s[30:31]
.LBB2_7:
	s_mov_b64 s[4:5], 0x7fe
	v_mov_b32_e32 v16, v12
	v_mov_b32_e32 v20, v8
	v_mov_b32_e32 v18, v10
	v_cmp_lt_i64_e32 vcc, s[4:5], v[13:14]
	v_mov_b32_e32 v15, v11
	v_mov_b32_e32 v19, v7
	v_mov_b32_e32 v17, v9
	s_mov_b64 s[4:5], 0
	s_mov_b64 s[14:15], 0
	s_mov_b64 s[12:13], 0
	s_and_saveexec_b64 s[6:7], vcc
	s_xor_b64 s[6:7], exec, s[6:7]
	s_cbranch_execnz .LBB2_19
; %bb.8:
	s_or_saveexec_b64 s[16:17], s[6:7]
                                        ; implicit-def: $vgpr5_vgpr6
	s_xor_b64 exec, exec, s[16:17]
	s_cbranch_execnz .LBB2_20
.LBB2_9:
	s_or_b64 exec, exec, s[16:17]
	s_and_saveexec_b64 s[6:7], s[14:15]
.LBB2_10:
	v_and_b32_e32 v6, 0x80000000, v21
	v_mov_b32_e32 v5, 0
	s_andn2_b64 s[12:13], s[12:13], exec
.LBB2_11:
	s_or_b64 exec, exec, s[6:7]
	s_mov_b64 s[6:7], 0
	s_and_saveexec_b64 s[14:15], s[4:5]
	s_xor_b64 s[4:5], exec, s[14:15]
; %bb.12:
	v_or_b32_e32 v6, 0, v8
	v_or_b32_e32 v5, v11, v7
	v_cmp_ne_u64_e32 vcc, 0, v[5:6]
	s_mov_b32 s6, 0
	s_mov_b32 s7, 0xfff80000
	v_mov_b32_e32 v5, s6
	v_mov_b32_e32 v6, s7
	s_and_b64 s[6:7], vcc, exec
; %bb.13:
	s_or_b64 exec, exec, s[4:5]
	s_and_b64 s[4:5], s[12:13], exec
	s_and_b64 s[6:7], s[6:7], exec
                                        ; implicit-def: $vgpr13_vgpr14
	s_andn2_saveexec_b64 s[10:11], s[10:11]
	s_cbranch_execz .LBB2_2
.LBB2_14:
	v_cmp_eq_u64_e32 vcc, 0, v[7:8]
	s_mov_b64 s[8:9], 0
	s_mov_b64 s[14:15], -1
	s_mov_b64 s[12:13], s[6:7]
	s_and_saveexec_b64 s[16:17], vcc
; %bb.15:
	s_mov_b64 s[12:13], 0x7ff
	v_cmp_ne_u64_e32 vcc, s[12:13], v[13:14]
	s_andn2_b64 s[12:13], s[6:7], exec
	s_and_b64 s[18:19], vcc, exec
	s_mov_b64 s[8:9], exec
	s_xor_b64 s[14:15], exec, -1
	s_or_b64 s[12:13], s[12:13], s[18:19]
; %bb.16:
	s_or_b64 exec, exec, s[16:17]
	s_andn2_b64 s[4:5], s[4:5], exec
	s_and_b64 s[14:15], s[14:15], exec
	s_andn2_b64 s[6:7], s[6:7], exec
	s_and_b64 s[12:13], s[12:13], exec
	s_or_b64 s[4:5], s[4:5], s[14:15]
	s_and_b64 s[8:9], s[8:9], exec
	s_or_b64 s[6:7], s[6:7], s[12:13]
	s_or_b64 exec, exec, s[10:11]
	s_and_saveexec_b64 s[10:11], s[6:7]
	s_cbranch_execz .LBB2_3
.LBB2_17:
	v_and_b32_e32 v5, 0x80000000, v21
	v_or_b32_e32 v6, 0x7ff00000, v5
	v_mov_b32_e32 v5, 0
	s_andn2_b64 s[8:9], s[8:9], exec
	s_or_b64 exec, exec, s[10:11]
	s_and_saveexec_b64 s[6:7], s[8:9]
	s_cbranch_execz .LBB2_4
.LBB2_18:
	v_cmp_ne_u64_e32 vcc, 0, v[9:10]
	s_mov_b32 s8, 0
	s_mov_b32 s9, 0xfff80000
	v_mov_b32_e32 v5, s8
	v_mov_b32_e32 v6, s9
	s_andn2_b64 s[4:5], s[4:5], exec
	s_and_b64 s[8:9], vcc, exec
	s_or_b64 s[4:5], s[4:5], s[8:9]
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[6:7], s[4:5]
	s_cbranch_execnz .LBB2_5
	s_branch .LBB2_6
.LBB2_19:
	v_cmp_eq_u64_e32 vcc, 0, v[9:10]
	s_mov_b64 s[12:13], exec
	s_and_b64 s[14:15], vcc, exec
                                        ; implicit-def: $vgpr13_vgpr14
                                        ; implicit-def: $vgpr17_vgpr18
                                        ; implicit-def: $vgpr15_vgpr16
                                        ; implicit-def: $vgpr19_vgpr20
	s_or_saveexec_b64 s[16:17], s[6:7]
                                        ; implicit-def: $vgpr5_vgpr6
	s_xor_b64 exec, exec, s[16:17]
	s_cbranch_execz .LBB2_9
.LBB2_20:
	v_cmp_eq_u64_e32 vcc, 0, v[13:14]
	s_mov_b64 s[18:19], 0
	s_mov_b64 s[6:7], -1
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB2_24
; %bb.21:
	v_cmp_ne_u64_e32 vcc, 0, v[9:10]
	s_mov_b64 s[6:7], 0
	s_mov_b64 s[20:21], -1
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execz .LBB2_23
; %bb.22:
	v_cmp_eq_u32_e32 vcc, 0, v10
	v_cndmask_b32_e64 v5, 0, 1, vcc
	v_cndmask_b32_e32 v6, v10, v2, vcc
	s_mov_b32 s20, 0x10000
	v_lshlrev_b16_e32 v5, 5, v5
	v_lshlrev_b32_e32 v14, 16, v6
	v_cmp_gt_u32_e32 vcc, s20, v6
	v_or_b32_e32 v13, 16, v5
	v_cndmask_b32_e32 v6, v6, v14, vcc
	s_mov_b32 s20, 0x1000000
	v_cndmask_b32_e32 v5, v5, v13, vcc
	v_lshlrev_b32_e32 v14, 8, v6
	v_cmp_gt_u32_e32 vcc, s20, v6
	v_cndmask_b32_e32 v6, v6, v14, vcc
	v_lshrrev_b32_e32 v6, 24, v6
	s_getpc_b64 s[20:21]
	s_add_u32 s20, s20, softfloat_countLeadingZeros8@rel32@lo+4
	s_addc_u32 s21, s21, softfloat_countLeadingZeros8@rel32@hi+12
	global_load_ubyte v6, v6, s[20:21]
	v_or_b32_e32 v13, 8, v5
	v_cndmask_b32_e32 v5, v5, v13, vcc
	s_mov_b64 s[6:7], exec
	s_xor_b64 s[20:21], exec, -1
	s_waitcnt vmcnt(0)
	v_add_u16_e32 v5, v5, v6
	v_add_u16_e32 v5, -11, v5
	v_bfe_i32 v5, v5, 0, 8
	v_sub_u32_e32 v13, 1, v5
	v_lshlrev_b64 v[17:18], v5, v[9:10]
	v_ashrrev_i32_e32 v14, 31, v13
.LBB2_23:
	s_or_b64 exec, exec, s[18:19]
	s_and_b64 s[18:19], s[20:21], exec
	s_orn2_b64 s[6:7], s[6:7], exec
.LBB2_24:
	s_or_b64 exec, exec, s[4:5]
	s_mov_b64 s[22:23], s[14:15]
                                        ; implicit-def: $vgpr5_vgpr6
	s_and_saveexec_b64 s[20:21], s[6:7]
	s_cbranch_execz .LBB2_52
; %bb.25:
	v_cmp_eq_u64_e32 vcc, 0, v[11:12]
	s_mov_b64 s[6:7], 0
	s_mov_b64 s[26:27], -1
	s_mov_b64 s[22:23], s[14:15]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB2_29
; %bb.26:
	v_cmp_ne_u64_e32 vcc, 0, v[7:8]
	s_mov_b64 s[24:25], -1
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB2_28
; %bb.27:
	v_cmp_eq_u32_e32 vcc, 0, v8
	v_cndmask_b32_e64 v5, 0, 1, vcc
	v_cndmask_b32_e32 v6, v8, v0, vcc
	s_mov_b32 s24, 0x10000
	v_lshlrev_b16_e32 v5, 5, v5
	v_lshlrev_b32_e32 v15, 16, v6
	v_cmp_gt_u32_e32 vcc, s24, v6
	v_or_b32_e32 v12, 16, v5
	v_cndmask_b32_e32 v6, v6, v15, vcc
	s_mov_b32 s24, 0x1000000
	v_cndmask_b32_e32 v5, v5, v12, vcc
	v_lshlrev_b32_e32 v15, 8, v6
	v_cmp_gt_u32_e32 vcc, s24, v6
	v_cndmask_b32_e32 v6, v6, v15, vcc
	v_lshrrev_b32_e32 v6, 24, v6
	s_getpc_b64 s[24:25]
	s_add_u32 s24, s24, softfloat_countLeadingZeros8@rel32@lo+4
	s_addc_u32 s25, s25, softfloat_countLeadingZeros8@rel32@hi+12
	global_load_ubyte v6, v6, s[24:25]
	v_or_b32_e32 v12, 8, v5
	v_cndmask_b32_e32 v5, v5, v12, vcc
	s_mov_b64 s[6:7], exec
	s_xor_b64 s[24:25], exec, -1
	s_waitcnt vmcnt(0)
	v_add_u16_e32 v5, v5, v6
	v_add_u16_e32 v5, -11, v5
	v_bfe_i32 v5, v5, 0, 8
	v_sub_u32_e32 v15, 1, v5
	v_lshlrev_b64 v[19:20], v5, v[7:8]
	v_ashrrev_i32_e32 v16, 31, v15
.LBB2_28:
	s_or_b64 exec, exec, s[22:23]
	s_andn2_b64 s[22:23], s[14:15], exec
	s_and_b64 s[24:25], s[24:25], exec
	s_or_b64 s[22:23], s[22:23], s[24:25]
	s_orn2_b64 s[26:27], s[6:7], exec
.LBB2_29:
	s_or_b64 exec, exec, s[4:5]
                                        ; implicit-def: $vgpr5_vgpr6
	s_and_saveexec_b64 s[24:25], s[26:27]
	s_cbranch_execz .LBB2_51
; %bb.30:
	v_or_b32_e32 v18, 0x100000, v18
	v_alignbit_b32 v12, v18, v17, 21
	v_cvt_f32_u32_e32 v5, v12
	v_mov_b32_e32 v6, 0x4f800000
	v_sub_u32_e32 v27, 0, v12
	s_brev_b32 s6, -2
	v_mac_f32_e32 v5, 0, v6
	v_or_b32_e32 v20, 0x100000, v20
	v_rcp_f32_e32 v5, v5
	s_mov_b32 s27, 0
	v_mul_f32_e32 v5, 0x5f7ffffc, v5
	v_mul_f32_e32 v6, 0x2f800000, v5
	v_trunc_f32_e32 v22, v6
	v_mac_f32_e32 v5, 0xcf800000, v22
	v_cvt_u32_f32_e32 v26, v5
	v_cvt_u32_f32_e32 v28, v22
	v_mad_u64_u32 v[5:6], s[4:5], v27, v26, 0
	v_mad_u64_u32 v[22:23], s[4:5], v27, v28, v[6:7]
	v_mul_hi_u32 v29, v26, v5
	v_mad_u64_u32 v[5:6], s[4:5], v28, v5, 0
	v_sub_u32_e32 v24, v22, v26
	v_mad_u64_u32 v[22:23], s[4:5], v26, v24, 0
	v_mad_u64_u32 v[24:25], s[4:5], v28, v24, 0
	v_add_co_u32_e32 v22, vcc, v29, v22
	v_addc_co_u32_e32 v23, vcc, 0, v23, vcc
	v_add_co_u32_e32 v5, vcc, v22, v5
	v_addc_co_u32_e32 v5, vcc, v23, v6, vcc
	v_addc_co_u32_e32 v6, vcc, 0, v25, vcc
	v_add_co_u32_e32 v5, vcc, v5, v24
	v_addc_co_u32_e32 v6, vcc, 0, v6, vcc
	v_add_co_u32_e32 v29, vcc, v26, v5
	v_addc_co_u32_e32 v28, vcc, v28, v6, vcc
	v_mad_u64_u32 v[5:6], s[4:5], v27, v29, 0
	v_mad_u64_u32 v[22:23], s[4:5], v27, v28, v[6:7]
	v_mad_u64_u32 v[23:24], s[4:5], v28, v5, 0
	v_sub_u32_e32 v6, v22, v29
	v_mad_u64_u32 v[25:26], s[4:5], v29, v6, 0
	v_mul_hi_u32 v22, v29, v5
	v_mad_u64_u32 v[5:6], s[4:5], v28, v6, 0
	v_add_co_u32_e32 v22, vcc, v22, v25
	v_addc_co_u32_e32 v25, vcc, 0, v26, vcc
	v_add_co_u32_e32 v22, vcc, v22, v23
	v_addc_co_u32_e32 v22, vcc, v25, v24, vcc
	v_addc_co_u32_e32 v6, vcc, 0, v6, vcc
	v_add_co_u32_e32 v5, vcc, v22, v5
	v_addc_co_u32_e32 v6, vcc, 0, v6, vcc
	v_add_co_u32_e32 v24, vcc, v29, v5
	v_mad_u64_u32 v[22:23], s[4:5], v24, s6, 0
	v_mul_hi_u32 v26, v24, -1
	v_addc_co_u32_e32 v25, vcc, v28, v6, vcc
	v_mad_u64_u32 v[5:6], s[4:5], v25, -1, 0
	v_mad_u64_u32 v[24:25], s[4:5], v25, s6, 0
	v_add_co_u32_e32 v22, vcc, v26, v22
	v_addc_co_u32_e32 v23, vcc, 0, v23, vcc
	v_add_co_u32_e32 v5, vcc, v22, v5
	v_addc_co_u32_e32 v5, vcc, v23, v6, vcc
	v_addc_co_u32_e32 v6, vcc, 0, v25, vcc
	v_add_co_u32_e32 v25, vcc, v5, v24
	v_addc_co_u32_e32 v22, vcc, 0, v6, vcc
	v_mad_u64_u32 v[5:6], s[4:5], v12, v25, 0
	v_mad_u64_u32 v[22:23], s[4:5], v12, v22, v[6:7]
	v_cmp_lt_u64_e64 s[4:5], v[19:20], v[17:18]
	v_not_b32_e32 v5, v5
	v_cndmask_b32_e64 v6, 10, 11, s[4:5]
	v_lshlrev_b64 v[23:24], v6, v[19:20]
	v_xor_b32_e32 v6, 0x7fffffff, v22
	v_sub_co_u32_e32 v18, vcc, v5, v12
	v_subbrev_co_u32_e32 v19, vcc, 0, v6, vcc
	v_cmp_ge_u32_e32 vcc, v18, v12
	v_cndmask_b32_e64 v18, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, 0, v19
	v_cndmask_b32_e32 v18, -1, v18, vcc
	v_add_u32_e32 v19, 2, v25
	v_add_u32_e32 v20, 1, v25
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_cndmask_b32_e32 v18, v20, v19, vcc
	v_cmp_ge_u32_e32 vcc, v5, v12
	v_cndmask_b32_e64 v5, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, 0, v6
	v_cndmask_b32_e32 v5, -1, v5, vcc
	v_cmp_ne_u32_e32 vcc, 0, v5
	v_cndmask_b32_e32 v5, v25, v18, vcc
	v_add_u32_e32 v27, -2, v5
	v_mad_u64_u32 v[25:26], s[6:7], v24, v27, 0
	v_lshlrev_b32_e32 v17, 7, v17
	v_and_b32_e32 v19, 0xfffff80, v17
	v_alignbit_b32 v5, v26, v25, 31
	v_and_b32_e32 v18, -2, v5
	v_mad_u64_u32 v[5:6], s[6:7], v18, v12, 0
	v_mad_u64_u32 v[17:18], s[6:7], v18, v19, 0
	v_sub_co_u32_e32 v5, vcc, v23, v5
	v_subb_co_u32_e32 v6, vcc, v24, v6, vcc
	v_lshlrev_b64 v[5:6], 28, v[5:6]
	v_sub_co_u32_e32 v20, vcc, v5, v17
	v_subb_co_u32_e32 v22, vcc, v6, v18, vcc
	v_mul_hi_u32 v5, v22, v27
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v24, v6
	v_add_u32_e32 v5, 4, v5
	v_lshlrev_b64 v[17:18], 4, v[5:6]
	v_and_b32_e32 v23, 28, v5
	v_cmp_eq_u64_e32 vcc, 0, v[23:24]
	v_add_co_u32_e64 v17, s[6:7], 0, v17
	v_addc_co_u32_e64 v18, s[6:7], v18, v26, s[6:7]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB2_36
; %bb.31:
	v_lshlrev_b32_e32 v5, 1, v5
	v_and_b32_e32 v5, -16, v5
	v_mad_u64_u32 v[23:24], s[28:29], v12, v5, 0
	v_mad_u64_u32 v[5:6], s[28:29], v19, v5, 0
	v_sub_co_u32_e32 v19, vcc, v20, v23
	v_subb_co_u32_e32 v20, vcc, v22, v24, vcc
	v_lshlrev_b64 v[19:20], 28, v[19:20]
	v_and_b32_e32 v12, 0xffffff80, v17
	v_sub_co_u32_e32 v22, vcc, v19, v5
	v_subb_co_u32_e32 v23, vcc, v20, v6, vcc
	v_cmp_gt_i64_e32 vcc, 0, v[22:23]
	s_and_saveexec_b64 s[28:29], vcc
	s_xor_b64 s[28:29], exec, s[28:29]
; %bb.32:
	v_add_co_u32_e32 v17, vcc, 0xffffff80, v12
	v_addc_co_u32_e32 v18, vcc, -1, v18, vcc
                                        ; implicit-def: $vgpr19_vgpr20
                                        ; implicit-def: $vgpr5_vgpr6
                                        ; implicit-def: $vgpr12
; %bb.33:
	s_andn2_saveexec_b64 s[28:29], s[28:29]
; %bb.34:
	v_cmp_ne_u64_e32 vcc, v[19:20], v[5:6]
	v_cndmask_b32_e64 v5, 0, 1, vcc
	v_or_b32_e32 v17, v12, v5
; %bb.35:
	s_or_b64 exec, exec, s[28:29]
.LBB2_36:
	s_or_b64 exec, exec, s[6:7]
	v_sub_co_u32_e32 v5, vcc, v15, v13
	v_mov_b32_e32 v6, 0x3fe
	v_mov_b32_e32 v12, 0x3fd
	v_subb_co_u32_e32 v13, vcc, v16, v14, vcc
	v_cndmask_b32_e64 v12, v6, v12, s[4:5]
	v_add_co_u32_e32 v12, vcc, v12, v5
	v_mov_b32_e32 v6, 0
	v_addc_co_u32_e32 v13, vcc, 0, v13, vcc
	v_and_b32_e32 v5, 0xffff, v12
	s_movk_i32 s26, 0x7fc
	v_cmp_lt_u64_e32 vcc, s[26:27], v[5:6]
	s_mov_b64 s[26:27], -1
                                        ; implicit-def: $vgpr5_vgpr6
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB2_48
; %bb.37:
	v_cmp_lt_i64_e32 vcc, -1, v[12:13]
	s_mov_b64 s[28:29], -1
	s_mov_b64 s[4:5], 0
                                        ; implicit-def: $vgpr5_vgpr6
	s_and_saveexec_b64 s[26:27], vcc
	s_xor_b64 s[26:27], exec, s[26:27]
	s_cbranch_execz .LBB2_41
; %bb.38:
	v_add_co_u32_e32 v5, vcc, 0x200, v17
	s_mov_b64 s[30:31], 0x7fd
	v_addc_co_u32_e32 v6, vcc, 0, v18, vcc
	v_cmp_lt_u64_e64 s[4:5], s[30:31], v[12:13]
	v_cmp_gt_i64_e32 vcc, 0, v[5:6]
                                        ; implicit-def: $vgpr5_vgpr6
	s_or_b64 s[4:5], s[4:5], vcc
	s_and_saveexec_b64 vcc, s[4:5]
	s_xor_b64 s[4:5], exec, vcc
; %bb.39:
	v_and_b32_e32 v5, 0x80000000, v21
	v_or_b32_e32 v6, 0x7ff00000, v5
	v_mov_b32_e32 v5, 0
	s_xor_b64 s[28:29], exec, -1
; %bb.40:
	s_or_b64 exec, exec, s[4:5]
	v_mov_b32_e32 v12, s30
	v_mov_b32_e32 v13, s31
	s_and_b64 s[4:5], s[28:29], exec
.LBB2_41:
	s_andn2_saveexec_b64 s[26:27], s[26:27]
	s_cbranch_execz .LBB2_47
; %bb.42:
	v_sub_co_u32_e32 v14, vcc, 0, v12
	v_subb_co_u32_e32 v15, vcc, 0, v13, vcc
	v_cmp_lt_u64_e32 vcc, 62, v[14:15]
	s_and_saveexec_b64 s[28:29], vcc
	s_xor_b64 s[28:29], exec, s[28:29]
; %bb.43:
	v_cmp_ne_u64_e32 vcc, 0, v[17:18]
                                        ; implicit-def: $vgpr14
                                        ; implicit-def: $vgpr12
	v_cndmask_b32_e64 v17, 0, 1, vcc
	s_mov_b32 vcc_lo, 0
	v_mov_b32_e32 v18, vcc_lo
; %bb.44:
	s_andn2_saveexec_b64 s[28:29], s[28:29]
; %bb.45:
	v_lshlrev_b64 v[12:13], v12, v[17:18]
	v_lshrrev_b64 v[17:18], v14, v[17:18]
	v_cmp_ne_u64_e32 vcc, 0, v[12:13]
	v_cndmask_b32_e64 v12, 0, 1, vcc
	v_or_b32_e32 v17, v17, v12
; %bb.46:
	s_or_b64 exec, exec, s[28:29]
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v13, 0
	s_or_b64 s[4:5], s[4:5], exec
.LBB2_47:
	s_or_b64 exec, exec, s[26:27]
	s_orn2_b64 s[26:27], s[4:5], exec
.LBB2_48:
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[4:5], s[26:27]
	s_cbranch_execz .LBB2_50
; %bb.49:
	v_add_co_u32_e32 v13, vcc, 0x200, v17
	v_and_b32_e32 v5, 0x3ff, v17
	s_mov_b64 s[6:7], 0x200
	v_mov_b32_e32 v6, 0
	v_addc_co_u32_e32 v14, vcc, 0, v18, vcc
	v_cmp_eq_u64_e32 vcc, s[6:7], v[5:6]
	v_lshrrev_b64 v[13:14], 10, v[13:14]
	v_cndmask_b32_e64 v5, 0, 1, vcc
	v_not_b32_e32 v5, v5
	v_and_b32_e32 v13, v13, v5
	v_cmp_ne_u64_e32 vcc, 0, v[13:14]
	v_and_b32_e32 v5, 0x80000000, v21
	v_lshlrev_b32_e32 v6, 20, v12
	v_cndmask_b32_e32 v6, 0, v6, vcc
	v_or_b32_e32 v12, v14, v5
	v_add_co_u32_e32 v5, vcc, 0, v13
	v_addc_co_u32_e32 v6, vcc, v12, v6, vcc
.LBB2_50:
	s_or_b64 exec, exec, s[4:5]
.LBB2_51:
	s_or_b64 exec, exec, s[24:25]
	s_andn2_b64 s[4:5], s[14:15], exec
	s_and_b64 s[6:7], s[22:23], exec
	s_or_b64 s[22:23], s[4:5], s[6:7]
.LBB2_52:
	s_or_b64 exec, exec, s[20:21]
	s_andn2_b64 s[6:7], s[14:15], exec
	s_and_b64 s[14:15], s[22:23], exec
	s_and_b64 s[4:5], s[18:19], exec
	s_or_b64 s[14:15], s[6:7], s[14:15]
	s_or_b64 exec, exec, s[16:17]
	s_and_saveexec_b64 s[6:7], s[14:15]
	s_cbranch_execnz .LBB2_10
	s_branch .LBB2_11
.Lfunc_end2:
	.size	__softfloat_f64_div, .Lfunc_end2-__softfloat_f64_div
                                        ; -- End function
	.section	.AMDGPU.csdata
; Function info:
; codeLenInByte = 2392
; NumSgprs: 40
; NumVgprs: 30
; ScratchSize: 8
; MemoryBound: 0
	.text
	.p2align	2                               ; -- Begin function __softfloat_f64_mul
	.type	__softfloat_f64_mul,@function
__softfloat_f64_mul:                    ; @__softfloat_f64_mul
; %bb.0:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	v_bfe_u32 v12, v1, 20, 11
	s_mov_b64 s[4:5], 0x7ff
	v_mov_b32_e32 v13, 0
	v_cmp_ne_u64_e32 vcc, s[4:5], v[12:13]
	v_and_b32_e32 v7, 0xfffff, v1
	v_mov_b32_e32 v6, v0
	v_bfe_u32 v10, v3, 20, 11
	v_mov_b32_e32 v11, v13
	v_and_b32_e32 v9, 0xfffff, v3
	v_mov_b32_e32 v8, v2
	v_xor_b32_e32 v16, v3, v1
	s_mov_b64 s[10:11], 0
	s_mov_b64 s[8:9], 0
                                        ; implicit-def: $vgpr4_vgpr5
                                        ; implicit-def: $vgpr14_vgpr15
	s_and_saveexec_b64 s[6:7], vcc
	s_xor_b64 s[6:7], exec, s[6:7]
	s_cbranch_execnz .LBB3_5
; %bb.1:
	s_andn2_saveexec_b64 s[12:13], s[6:7]
	s_cbranch_execnz .LBB3_16
.LBB3_2:
	s_or_b64 exec, exec, s[12:13]
	s_and_saveexec_b64 s[4:5], s[10:11]
	s_cbranch_execnz .LBB3_21
.LBB3_3:
	s_or_b64 exec, exec, s[4:5]
	s_and_saveexec_b64 s[6:7], s[8:9]
	s_cbranch_execnz .LBB3_22
.LBB3_4:
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v0, v4
	v_mov_b32_e32 v1, v5
	s_setpc_b64 s[30:31]
.LBB3_5:
	v_cmp_ne_u64_e32 vcc, s[4:5], v[10:11]
                                        ; implicit-def: $vgpr4_vgpr5
	s_and_saveexec_b64 s[4:5], vcc
	s_xor_b64 s[8:9], exec, s[4:5]
	s_cbranch_execz .LBB3_11
; %bb.6:
	v_cmp_eq_u64_e32 vcc, 0, v[12:13]
	v_mov_b32_e32 v15, v7
	s_mov_b64 s[10:11], 0
	s_mov_b64 s[14:15], -1
	v_mov_b32_e32 v14, v6
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execnz .LBB3_23
; %bb.7:
	s_or_b64 exec, exec, s[4:5]
                                        ; implicit-def: $vgpr4_vgpr5
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_cbranch_execnz .LBB3_26
.LBB3_8:
	s_or_b64 exec, exec, s[12:13]
	s_and_saveexec_b64 s[4:5], s[10:11]
.LBB3_9:
	v_and_b32_e32 v5, 0x80000000, v16
	v_mov_b32_e32 v4, 0
.LBB3_10:
	s_or_b64 exec, exec, s[4:5]
                                        ; implicit-def: $vgpr8_vgpr9
                                        ; implicit-def: $vgpr12_vgpr13
.LBB3_11:
	s_or_saveexec_b64 s[4:5], s[8:9]
	s_mov_b64 s[10:11], 0
	s_mov_b64 s[12:13], 0
                                        ; implicit-def: $vgpr14_vgpr15
	s_xor_b64 exec, exec, s[4:5]
	s_cbranch_execz .LBB3_15
; %bb.12:
	v_cmp_eq_u64_e32 vcc, 0, v[8:9]
	s_mov_b64 s[8:9], 0
	s_mov_b64 s[10:11], -1
                                        ; implicit-def: $vgpr14_vgpr15
	s_and_saveexec_b64 s[12:13], vcc
	s_xor_b64 s[12:13], exec, s[12:13]
; %bb.13:
	s_mov_b64 s[8:9], exec
	v_or_b32_e32 v15, 0, v7
	v_or_b32_e32 v14, v12, v6
	s_xor_b64 s[10:11], exec, -1
; %bb.14:
	s_or_b64 exec, exec, s[12:13]
	s_and_b64 s[12:13], s[10:11], exec
	s_and_b64 s[10:11], s[8:9], exec
.LBB3_15:
	s_or_b64 exec, exec, s[4:5]
	s_and_b64 s[8:9], s[12:13], exec
	s_and_b64 s[10:11], s[10:11], exec
                                        ; implicit-def: $vgpr10_vgpr11
                                        ; implicit-def: $vgpr8_vgpr9
	s_andn2_saveexec_b64 s[12:13], s[6:7]
	s_cbranch_execz .LBB3_2
.LBB3_16:
	v_cmp_ne_u64_e32 vcc, 0, v[6:7]
	v_cmp_eq_u64_e64 s[4:5], 0, v[6:7]
	s_mov_b64 s[6:7], s[10:11]
	s_and_saveexec_b64 s[14:15], s[4:5]
	s_cbranch_execz .LBB3_20
; %bb.17:
	s_mov_b64 s[4:5], 0x7ff
	v_cmp_ne_u64_e64 s[4:5], s[4:5], v[10:11]
	v_cmp_eq_u64_e64 s[6:7], 0, v[8:9]
	s_or_b64 s[18:19], s[6:7], s[4:5]
	s_mov_b64 s[6:7], -1
	s_mov_b64 s[4:5], s[10:11]
	s_and_saveexec_b64 s[16:17], s[18:19]
; %bb.18:
	v_or_b32_e32 v15, 0, v9
	v_or_b32_e32 v14, v10, v8
	s_xor_b64 s[6:7], exec, -1
	s_or_b64 s[4:5], s[10:11], exec
; %bb.19:
	s_or_b64 exec, exec, s[16:17]
	s_andn2_b64 s[16:17], vcc, exec
	s_and_b64 s[6:7], s[6:7], exec
	s_or_b64 vcc, s[16:17], s[6:7]
	s_andn2_b64 s[6:7], s[10:11], exec
	s_and_b64 s[4:5], s[4:5], exec
	s_or_b64 s[6:7], s[6:7], s[4:5]
.LBB3_20:
	s_or_b64 exec, exec, s[14:15]
	s_andn2_b64 s[4:5], s[8:9], exec
	s_and_b64 s[8:9], vcc, exec
	s_or_b64 s[8:9], s[4:5], s[8:9]
	s_andn2_b64 s[4:5], s[10:11], exec
	s_and_b64 s[6:7], s[6:7], exec
	s_or_b64 s[10:11], s[4:5], s[6:7]
	s_or_b64 exec, exec, s[12:13]
	s_and_saveexec_b64 s[4:5], s[10:11]
	s_cbranch_execz .LBB3_3
.LBB3_21:
	v_and_b32_e32 v4, 0x80000000, v16
	v_cmp_ne_u64_e32 vcc, 0, v[14:15]
	v_or_b32_e32 v4, 0x7ff00000, v4
	v_mov_b32_e32 v5, 0xfff80000
	v_cndmask_b32_e32 v5, v5, v4, vcc
	v_mov_b32_e32 v4, 0
	s_or_b64 exec, exec, s[4:5]
	s_and_saveexec_b64 s[6:7], s[8:9]
	s_cbranch_execz .LBB3_4
.LBB3_22:
	s_mov_b32 s4, 0
	s_mov_b32 s5, 0x7ff00000
	v_and_b32_e32 v5, 0x7ff00000, v1
	v_mov_b32_e32 v4, 0
	v_cmp_ne_u64_e32 vcc, s[4:5], v[4:5]
	v_cmp_eq_u64_e64 s[4:5], 0, v[6:7]
	s_or_b64 vcc, vcc, s[4:5]
	v_cndmask_b32_e32 v5, v1, v3, vcc
	v_cndmask_b32_e32 v4, v0, v2, vcc
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v0, v4
	v_mov_b32_e32 v1, v5
	s_setpc_b64 s[30:31]
.LBB3_23:
	v_cmp_ne_u64_e32 vcc, 0, v[6:7]
	v_mov_b32_e32 v15, v7
	s_mov_b64 s[12:13], 0
	s_mov_b64 s[14:15], -1
	v_mov_b32_e32 v14, v6
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB3_25
; %bb.24:
	v_cmp_eq_u32_e32 vcc, 0, v7
	v_cndmask_b32_e64 v4, 0, 1, vcc
	v_cndmask_b32_e32 v5, v7, v0, vcc
	s_mov_b32 s14, 0x10000
	v_lshlrev_b16_e32 v4, 5, v4
	v_lshlrev_b32_e32 v13, 16, v5
	v_cmp_gt_u32_e32 vcc, s14, v5
	v_or_b32_e32 v12, 16, v4
	v_cndmask_b32_e32 v5, v5, v13, vcc
	s_mov_b32 s14, 0x1000000
	v_cndmask_b32_e32 v4, v4, v12, vcc
	v_lshlrev_b32_e32 v13, 8, v5
	v_cmp_gt_u32_e32 vcc, s14, v5
	v_cndmask_b32_e32 v5, v5, v13, vcc
	v_lshrrev_b32_e32 v5, 24, v5
	s_getpc_b64 s[14:15]
	s_add_u32 s14, s14, softfloat_countLeadingZeros8@rel32@lo+4
	s_addc_u32 s15, s15, softfloat_countLeadingZeros8@rel32@hi+12
	global_load_ubyte v5, v5, s[14:15]
	v_or_b32_e32 v12, 8, v4
	v_cndmask_b32_e32 v4, v4, v12, vcc
	s_mov_b64 s[12:13], exec
	s_xor_b64 s[14:15], exec, -1
	s_waitcnt vmcnt(0)
	v_add_u16_e32 v4, v4, v5
	v_add_u16_e32 v4, -11, v4
	v_bfe_i32 v4, v4, 0, 8
	v_sub_u32_e32 v12, 1, v4
	v_lshlrev_b64 v[14:15], v4, v[6:7]
	v_ashrrev_i32_e32 v13, 31, v12
.LBB3_25:
	s_or_b64 exec, exec, s[10:11]
	s_and_b64 s[10:11], s[14:15], exec
	s_orn2_b64 s[14:15], s[12:13], exec
	s_or_b64 exec, exec, s[4:5]
                                        ; implicit-def: $vgpr4_vgpr5
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_cbranch_execz .LBB3_8
.LBB3_26:
	v_cmp_eq_u64_e32 vcc, 0, v[10:11]
	s_mov_b64 s[16:17], 0
	s_mov_b64 s[18:19], -1
	s_mov_b64 s[14:15], s[10:11]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB3_30
; %bb.27:
	v_cmp_ne_u64_e32 vcc, 0, v[8:9]
	s_mov_b64 s[18:19], -1
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB3_29
; %bb.28:
	v_cmp_eq_u32_e32 vcc, 0, v9
	v_cndmask_b32_e64 v4, 0, 1, vcc
	v_cndmask_b32_e32 v5, v9, v2, vcc
	s_mov_b32 s18, 0x10000
	v_lshlrev_b16_e32 v4, 5, v4
	v_lshlrev_b32_e32 v11, 16, v5
	v_cmp_gt_u32_e32 vcc, s18, v5
	v_or_b32_e32 v10, 16, v4
	v_cndmask_b32_e32 v5, v5, v11, vcc
	s_mov_b32 s18, 0x1000000
	v_cndmask_b32_e32 v4, v4, v10, vcc
	v_lshlrev_b32_e32 v11, 8, v5
	v_cmp_gt_u32_e32 vcc, s18, v5
	v_cndmask_b32_e32 v5, v5, v11, vcc
	v_lshrrev_b32_e32 v5, 24, v5
	s_getpc_b64 s[18:19]
	s_add_u32 s18, s18, softfloat_countLeadingZeros8@rel32@lo+4
	s_addc_u32 s19, s19, softfloat_countLeadingZeros8@rel32@hi+12
	global_load_ubyte v5, v5, s[18:19]
	v_or_b32_e32 v10, 8, v4
	v_cndmask_b32_e32 v4, v4, v10, vcc
	s_mov_b64 s[16:17], exec
	s_xor_b64 s[18:19], exec, -1
	s_waitcnt vmcnt(0)
	v_add_u16_e32 v4, v4, v5
	v_add_u16_e32 v4, -11, v4
	v_bfe_i32 v4, v4, 0, 8
	v_sub_u32_e32 v10, 1, v4
	v_lshlrev_b64 v[8:9], v4, v[8:9]
	v_ashrrev_i32_e32 v11, 31, v10
.LBB3_29:
	s_or_b64 exec, exec, s[14:15]
	s_andn2_b64 s[14:15], s[10:11], exec
	s_and_b64 s[18:19], s[18:19], exec
	s_or_b64 s[14:15], s[14:15], s[18:19]
	s_orn2_b64 s[18:19], s[16:17], exec
.LBB3_30:
	s_or_b64 exec, exec, s[4:5]
                                        ; implicit-def: $vgpr4_vgpr5
	s_and_saveexec_b64 s[16:17], s[18:19]
	s_cbranch_execz .LBB3_46
; %bb.31:
	v_lshlrev_b32_e32 v19, 10, v14
	v_lshlrev_b32_e32 v20, 11, v8
	v_mad_u64_u32 v[4:5], s[4:5], v20, v19, 0
	v_alignbit_b32 v8, v9, v8, 21
	v_or_b32_e32 v21, 0x80000000, v8
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v17, v5
	v_mad_u64_u32 v[8:9], s[4:5], v21, v19, v[17:18]
	v_alignbit_b32 v14, v15, v14, 22
	v_or_b32_e32 v14, 2.0, v14
	v_mov_b32_e32 v5, v9
	v_mov_b32_e32 v9, v18
	v_mad_u64_u32 v[8:9], s[4:5], v20, v14, v[8:9]
	v_add_co_u32_e32 v15, vcc, v10, v12
	v_addc_co_u32_e32 v11, vcc, v11, v13, vcc
	v_add_co_u32_e32 v9, vcc, v5, v9
	v_addc_co_u32_e64 v10, s[4:5], 0, 0, vcc
	v_mad_u64_u32 v[12:13], s[4:5], v21, v14, v[9:10]
	v_or_b32_e32 v17, v4, v8
	v_cmp_ne_u64_e32 vcc, 0, v[17:18]
	v_mov_b32_e32 v5, 0xfffffc00
	v_cndmask_b32_e64 v4, 0, 1, vcc
	v_or_b32_e32 v12, v4, v12
	v_cmp_gt_u64_e32 vcc, 2.0, v[12:13]
	v_mov_b32_e32 v4, 0xfffffc01
	v_cndmask_b32_e32 v4, v4, v5, vcc
	v_add_co_u32_e64 v10, s[4:5], v15, v4
	s_mov_b64 s[18:19], 0x7fc
	v_cndmask_b32_e64 v4, 0, 1, vcc
	v_and_b32_e32 v17, 0xffff, v10
	v_lshlrev_b64 v[8:9], v4, v[12:13]
	v_cmp_lt_u64_e32 vcc, s[18:19], v[17:18]
	s_mov_b64 s[20:21], 0
	v_addc_co_u32_e64 v11, s[4:5], -1, v11, s[4:5]
	s_mov_b64 s[22:23], -1
                                        ; implicit-def: $vgpr4_vgpr5
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execz .LBB3_43
; %bb.32:
	v_cmp_lt_i64_e32 vcc, -1, v[10:11]
	s_mov_b64 s[24:25], -1
                                        ; implicit-def: $vgpr4_vgpr5
	s_and_saveexec_b64 s[4:5], vcc
	s_xor_b64 s[22:23], exec, s[4:5]
	s_cbranch_execz .LBB3_36
; %bb.33:
	v_add_co_u32_e32 v4, vcc, 0x200, v8
	s_mov_b64 s[20:21], 0x7fd
	v_addc_co_u32_e32 v5, vcc, 0, v9, vcc
	v_cmp_lt_u64_e64 s[4:5], s[20:21], v[10:11]
	v_cmp_gt_i64_e32 vcc, 0, v[4:5]
                                        ; implicit-def: $vgpr4_vgpr5
	s_or_b64 s[4:5], s[4:5], vcc
	s_and_saveexec_b64 s[26:27], s[4:5]
	s_xor_b64 s[4:5], exec, s[26:27]
; %bb.34:
	v_and_b32_e32 v4, 0x80000000, v16
	v_or_b32_e32 v5, 0x7ff00000, v4
	v_mov_b32_e32 v4, 0
	s_xor_b64 s[24:25], exec, -1
; %bb.35:
	s_or_b64 exec, exec, s[4:5]
	v_mov_b32_e32 v10, s20
	v_mov_b32_e32 v11, s21
	s_and_b64 s[20:21], s[24:25], exec
.LBB3_36:
	s_andn2_saveexec_b64 s[4:5], s[22:23]
	s_cbranch_execz .LBB3_42
; %bb.37:
	v_sub_co_u32_e32 v12, vcc, 0, v10
	v_subb_co_u32_e32 v13, vcc, 0, v11, vcc
	v_cmp_lt_u64_e32 vcc, 62, v[12:13]
	s_and_saveexec_b64 s[22:23], vcc
	s_xor_b64 s[22:23], exec, s[22:23]
; %bb.38:
	v_cmp_ne_u64_e32 vcc, 0, v[8:9]
	s_mov_b32 s24, 0
	v_cndmask_b32_e64 v8, 0, 1, vcc
	v_mov_b32_e32 v9, s24
                                        ; implicit-def: $vgpr12
                                        ; implicit-def: $vgpr10
; %bb.39:
	s_andn2_saveexec_b64 s[22:23], s[22:23]
; %bb.40:
	v_lshlrev_b64 v[10:11], v10, v[8:9]
	v_lshrrev_b64 v[8:9], v12, v[8:9]
	v_cmp_ne_u64_e32 vcc, 0, v[10:11]
	v_cndmask_b32_e64 v10, 0, 1, vcc
	v_or_b32_e32 v8, v8, v10
; %bb.41:
	s_or_b64 exec, exec, s[22:23]
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v11, 0
	s_or_b64 s[20:21], s[20:21], exec
.LBB3_42:
	s_or_b64 exec, exec, s[4:5]
	s_orn2_b64 s[22:23], s[20:21], exec
.LBB3_43:
	s_or_b64 exec, exec, s[18:19]
	s_and_saveexec_b64 s[4:5], s[22:23]
	s_cbranch_execz .LBB3_45
; %bb.44:
	v_and_b32_e32 v4, 0x3ff, v8
	v_add_co_u32_e32 v8, vcc, 0x200, v8
	s_mov_b64 s[18:19], 0x200
	v_mov_b32_e32 v5, 0
	v_addc_co_u32_e32 v9, vcc, 0, v9, vcc
	v_cmp_eq_u64_e32 vcc, s[18:19], v[4:5]
	v_lshrrev_b64 v[8:9], 10, v[8:9]
	v_cndmask_b32_e64 v4, 0, 1, vcc
	v_not_b32_e32 v4, v4
	v_and_b32_e32 v8, v8, v4
	v_cmp_ne_u64_e32 vcc, 0, v[8:9]
	v_and_b32_e32 v4, 0x80000000, v16
	v_lshlrev_b32_e32 v5, 20, v10
	v_cndmask_b32_e32 v5, 0, v5, vcc
	v_or_b32_e32 v9, v9, v4
	v_add_co_u32_e32 v4, vcc, 0, v8
	v_addc_co_u32_e32 v5, vcc, v9, v5, vcc
.LBB3_45:
	s_or_b64 exec, exec, s[4:5]
.LBB3_46:
	s_or_b64 exec, exec, s[16:17]
	s_andn2_b64 s[4:5], s[10:11], exec
	s_and_b64 s[10:11], s[14:15], exec
	s_or_b64 s[10:11], s[4:5], s[10:11]
	s_or_b64 exec, exec, s[12:13]
	s_and_saveexec_b64 s[4:5], s[10:11]
	s_cbranch_execnz .LBB3_9
	s_branch .LBB3_10
.Lfunc_end3:
	.size	__softfloat_f64_mul, .Lfunc_end3-__softfloat_f64_mul
                                        ; -- End function
	.section	.AMDGPU.csdata
; Function info:
; codeLenInByte = 1616
; NumSgprs: 36
; NumVgprs: 22
; ScratchSize: 0
; MemoryBound: 0
	.text
	.p2align	2                               ; -- Begin function __softfloat_f64_sub
	.type	__softfloat_f64_sub,@function
__softfloat_f64_sub:                    ; @__softfloat_f64_sub
; %bb.0:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_xor_saveexec_b64 s[4:5], -1
	buffer_store_dword v4, off, s[0:3], s32 ; 4-byte Folded Spill
	s_mov_b64 exec, s[4:5]
	v_writelane_b32 v4, s30, 0
	v_writelane_b32 v4, s31, 1
	v_mov_b32_e32 v12, 0
	v_bfe_u32 v11, v1, 20, 11
	v_bfe_u32 v13, v3, 20, 11
	v_mov_b32_e32 v14, v12
	v_mov_b32_e32 v6, v1
	v_cmp_gt_i64_e64 s[8:9], 0, v[0:1]
	v_lshrrev_b32_e32 v7, 31, v1
	v_lshrrev_b32_e32 v8, 31, v3
	v_cmp_ne_u64_e64 s[4:5], v[11:12], v[13:14]
	v_sub_co_u32_e64 v15, s[6:7], v11, v13
	v_mov_b32_e32 v5, v0
	s_mov_b64 s[14:15], 0
	v_cmp_ne_u32_e32 vcc, v7, v8
	v_and_b32_e32 v10, 0xfffff, v1
	v_mov_b32_e32 v9, v0
	v_and_b32_e32 v8, 0xfffff, v3
	v_mov_b32_e32 v7, v2
	v_subb_co_u32_e64 v16, s[6:7], 0, 0, s[6:7]
	s_and_saveexec_b64 s[6:7], vcc
	s_xor_b64 s[10:11], exec, s[6:7]
	s_cbranch_execz .LBB4_9
; %bb.1:
	s_mov_b64 s[16:17], 0
                                        ; implicit-def: $vgpr17_vgpr18
                                        ; implicit-def: $vgpr19_vgpr20
	s_and_saveexec_b64 s[6:7], s[4:5]
	s_xor_b64 s[12:13], exec, s[6:7]
	s_cbranch_execnz .LBB4_16
; %bb.2:
	s_or_saveexec_b64 s[6:7], s[12:13]
	s_mov_b64 s[18:19], 0
	s_xor_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB4_41
.LBB4_3:
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_cbranch_execnz .LBB4_44
.LBB4_4:
	s_or_b64 exec, exec, s[12:13]
	s_mov_b64 s[12:13], 0
	s_and_saveexec_b64 s[6:7], s[18:19]
	s_cbranch_execnz .LBB4_45
.LBB4_5:
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[6:7], s[16:17]
	s_xor_b64 s[14:15], exec, s[6:7]
	s_cbranch_execnz .LBB4_48
.LBB4_6:
	s_or_b64 exec, exec, s[14:15]
	s_and_saveexec_b64 s[6:7], s[12:13]
.LBB4_7:
	v_add_co_u32_e32 v5, vcc, v7, v0
	v_addc_co_u32_e32 v6, vcc, v8, v1, vcc
.LBB4_8:
	s_or_b64 exec, exec, s[6:7]
                                        ; implicit-def: $vgpr11_vgpr12
                                        ; implicit-def: $vgpr13_vgpr14
                                        ; implicit-def: $vgpr15_vgpr16
                                        ; implicit-def: $vgpr2_vgpr3
                                        ; implicit-def: $vgpr0_vgpr1
                                        ; implicit-def: $vgpr9_vgpr10
                                        ; implicit-def: $vgpr7_vgpr8
.LBB4_9:
	s_andn2_saveexec_b64 s[6:7], s[10:11]
	s_cbranch_execz .LBB4_15
; %bb.10:
	s_mov_b64 s[12:13], 0
	s_and_saveexec_b64 s[10:11], s[4:5]
	s_xor_b64 s[10:11], exec, s[10:11]
	s_cbranch_execnz .LBB4_59
; %bb.11:
	s_andn2_saveexec_b64 s[4:5], s[10:11]
	s_cbranch_execnz .LBB4_102
.LBB4_12:
	s_or_b64 exec, exec, s[4:5]
	s_and_saveexec_b64 s[8:9], s[12:13]
.LBB4_13:
	s_mov_b32 s4, 0
	s_mov_b32 s5, 0x7ff00000
	v_and_b32_e32 v6, 0x7ff00000, v1
	v_mov_b32_e32 v5, 0
	v_cmp_ne_u64_e32 vcc, s[4:5], v[5:6]
	v_cmp_eq_u64_e64 s[4:5], 0, v[9:10]
	s_or_b64 vcc, vcc, s[4:5]
	v_cndmask_b32_e32 v6, v1, v3, vcc
	v_cndmask_b32_e32 v5, v0, v2, vcc
.LBB4_14:
	s_or_b64 exec, exec, s[8:9]
.LBB4_15:
	s_or_b64 exec, exec, s[6:7]
	v_readlane_b32 s30, v4, 0
	v_mov_b32_e32 v0, v5
	v_mov_b32_e32 v1, v6
	v_readlane_b32 s31, v4, 1
	s_xor_saveexec_b64 s[4:5], -1
	buffer_load_dword v4, off, s[0:3], s32  ; 4-byte Folded Reload
	s_mov_b64 exec, s[4:5]
	s_waitcnt vmcnt(0)
	s_setpc_b64 s[30:31]
.LBB4_16:
	v_lshlrev_b64 v[21:22], 9, v[7:8]
	v_cmp_lt_i64_e32 vcc, -1, v[15:16]
	s_mov_b64 s[16:17], 0
	s_mov_b64 s[18:19], 0
	s_mov_b64 s[14:15], 0
	s_and_saveexec_b64 s[6:7], vcc
	s_xor_b64 s[20:21], exec, s[6:7]
	s_cbranch_execz .LBB4_26
; %bb.17:
	s_mov_b64 s[6:7], 0x7ff
	v_cmp_ne_u64_e32 vcc, s[6:7], v[11:12]
	s_mov_b64 s[14:15], 0
	s_mov_b64 s[6:7], 0
	s_and_saveexec_b64 s[18:19], vcc
	s_xor_b64 s[18:19], exec, s[18:19]
	s_cbranch_execz .LBB4_23
; %bb.18:
	v_lshlrev_b64 v[17:18], 10, v[7:8]
	v_cmp_eq_u64_e32 vcc, 0, v[13:14]
	v_or_b32_e32 v19, 0x20000000, v22
	v_cmp_lt_u64_e64 s[6:7], 62, v[15:16]
	v_cndmask_b32_e32 v14, v19, v18, vcc
	v_cndmask_b32_e32 v13, v21, v17, vcc
                                        ; implicit-def: $vgpr21_vgpr22
	s_and_saveexec_b64 s[22:23], s[6:7]
	s_xor_b64 s[6:7], exec, s[22:23]
; %bb.19:
	v_cmp_ne_u64_e32 vcc, 0, v[13:14]
	s_mov_b32 s22, 0
	v_cndmask_b32_e64 v21, 0, 1, vcc
	v_mov_b32_e32 v22, s22
                                        ; implicit-def: $vgpr15
                                        ; implicit-def: $vgpr13_vgpr14
; %bb.20:
	s_andn2_saveexec_b64 s[6:7], s[6:7]
; %bb.21:
	v_sub_u32_e32 v16, 0, v15
	v_lshlrev_b64 v[16:17], v16, v[13:14]
	v_lshrrev_b64 v[21:22], v15, v[13:14]
	v_cmp_ne_u64_e32 vcc, 0, v[16:17]
	v_cndmask_b32_e64 v13, 0, 1, vcc
	v_or_b32_e32 v21, v21, v13
; %bb.22:
	s_or_b64 exec, exec, s[6:7]
	s_mov_b64 s[6:7], exec
.LBB4_23:
	s_andn2_saveexec_b64 s[18:19], s[18:19]
; %bb.24:
	v_cmp_ne_u64_e32 vcc, 0, v[9:10]
	s_and_b64 s[14:15], vcc, exec
; %bb.25:
	s_or_b64 exec, exec, s[18:19]
	s_and_b64 s[14:15], s[14:15], exec
	s_and_b64 s[18:19], s[6:7], exec
                                        ; implicit-def: $vgpr13_vgpr14
                                        ; implicit-def: $vgpr15
.LBB4_26:
	s_or_saveexec_b64 s[20:21], s[20:21]
	v_lshlrev_b64 v[23:24], 9, v[9:10]
	v_mov_b32_e32 v26, v12
	v_mov_b32_e32 v25, v11
	s_xor_b64 exec, exec, s[20:21]
	s_cbranch_execz .LBB4_34
; %bb.27:
	s_mov_b64 s[6:7], 0x7ff
	v_cmp_ne_u64_e32 vcc, s[6:7], v[13:14]
	v_mov_b32_e32 v26, v12
	s_mov_b64 s[22:23], -1
	s_mov_b64 s[6:7], s[18:19]
	v_mov_b32_e32 v25, v11
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB4_33
; %bb.28:
	v_sub_co_u32_e64 v18, s[6:7], 0, v15
	v_or_b32_e32 v17, 0x20000000, v24
	v_lshlrev_b64 v[24:25], 10, v[9:10]
	v_cmp_eq_u64_e32 vcc, 0, v[11:12]
	v_subb_co_u32_e64 v19, s[6:7], 0, v16, s[6:7]
	v_cmp_lt_u64_e64 s[6:7], 62, v[18:19]
	v_cndmask_b32_e32 v17, v17, v25, vcc
	v_cndmask_b32_e32 v16, v23, v24, vcc
                                        ; implicit-def: $vgpr23_vgpr24
	s_and_saveexec_b64 s[22:23], s[6:7]
	s_xor_b64 s[6:7], exec, s[22:23]
; %bb.29:
	v_cmp_ne_u64_e32 vcc, 0, v[16:17]
	s_mov_b32 s22, 0
	v_cndmask_b32_e64 v23, 0, 1, vcc
	v_mov_b32_e32 v24, s22
                                        ; implicit-def: $vgpr18
                                        ; implicit-def: $vgpr16_vgpr17
                                        ; implicit-def: $vgpr15
; %bb.30:
	s_andn2_saveexec_b64 s[6:7], s[6:7]
; %bb.31:
	v_lshlrev_b64 v[19:20], v15, v[16:17]
	v_lshrrev_b64 v[23:24], v18, v[16:17]
	v_cmp_ne_u64_e32 vcc, 0, v[19:20]
	v_cndmask_b32_e64 v15, 0, 1, vcc
	v_or_b32_e32 v23, v23, v15
; %bb.32:
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v26, v14
	s_xor_b64 s[22:23], exec, -1
	s_or_b64 s[6:7], s[18:19], exec
	v_mov_b32_e32 v25, v13
.LBB4_33:
	s_or_b64 exec, exec, s[16:17]
	s_andn2_b64 s[18:19], s[18:19], exec
	s_and_b64 s[6:7], s[6:7], exec
	s_and_b64 s[16:17], s[22:23], exec
	s_or_b64 s[18:19], s[18:19], s[6:7]
.LBB4_34:
	s_or_b64 exec, exec, s[20:21]
	s_mov_b64 s[6:7], 0
                                        ; implicit-def: $vgpr17_vgpr18
                                        ; implicit-def: $vgpr19_vgpr20
	s_and_saveexec_b64 s[20:21], s[18:19]
	s_xor_b64 s[18:19], exec, s[20:21]
; %bb.35:
	v_add_co_u32_e32 v13, vcc, v21, v23
	v_addc_co_u32_e32 v14, vcc, v22, v24, vcc
	v_bfrev_b32_e32 v15, 4
	v_add_co_u32_e32 v13, vcc, 0, v13
	v_addc_co_u32_e32 v14, vcc, v14, v15, vcc
	v_cmp_gt_u64_e32 vcc, 2.0, v[13:14]
	s_mov_b64 s[6:7], exec
	v_cndmask_b32_e64 v15, 0, 1, vcc
	v_sub_co_u32_e32 v19, vcc, v25, v15
	v_lshlrev_b64 v[17:18], v15, v[13:14]
	v_subbrev_co_u32_e32 v20, vcc, 0, v26, vcc
; %bb.36:
	s_or_b64 exec, exec, s[18:19]
	s_and_saveexec_b64 s[18:19], s[16:17]
	s_cbranch_execz .LBB4_40
; %bb.37:
	v_cmp_eq_u64_e32 vcc, 0, v[7:8]
	s_mov_b64 s[16:17], -1
	s_and_saveexec_b64 s[20:21], vcc
	s_xor_b64 s[20:21], exec, s[20:21]
; %bb.38:
	v_mov_b32_e32 v5, 0x7ff00000
	v_mov_b32_e32 v6, 0xfff00000
	v_cndmask_b32_e64 v6, v5, v6, s[8:9]
	v_mov_b32_e32 v5, 0
	s_xor_b64 s[16:17], exec, -1
; %bb.39:
	s_or_b64 exec, exec, s[20:21]
	s_andn2_b64 s[14:15], s[14:15], exec
	s_and_b64 s[16:17], s[16:17], exec
	s_or_b64 s[14:15], s[14:15], s[16:17]
.LBB4_40:
	s_or_b64 exec, exec, s[18:19]
	s_and_b64 s[16:17], s[6:7], exec
	s_and_b64 s[14:15], s[14:15], exec
	s_or_saveexec_b64 s[6:7], s[12:13]
	s_mov_b64 s[18:19], 0
	s_xor_b64 exec, exec, s[6:7]
	s_cbranch_execz .LBB4_3
.LBB4_41:
	s_mov_b64 s[12:13], 0x7fe
	v_cmp_lt_i64_e32 vcc, s[12:13], v[11:12]
	s_mov_b64 s[12:13], -1
	s_mov_b64 s[20:21], s[14:15]
	s_and_saveexec_b64 s[18:19], vcc
; %bb.42:
	v_or_b32_e32 v14, v8, v10
	v_or_b32_e32 v13, v7, v9
	v_cmp_ne_u64_e32 vcc, 0, v[13:14]
	s_andn2_b64 s[20:21], s[14:15], exec
	s_and_b64 s[22:23], vcc, exec
	s_xor_b64 s[12:13], exec, -1
	s_or_b64 s[20:21], s[20:21], s[22:23]
; %bb.43:
	s_or_b64 exec, exec, s[18:19]
	s_and_b64 s[18:19], s[12:13], exec
	s_andn2_b64 s[12:13], s[14:15], exec
	s_and_b64 s[14:15], s[20:21], exec
	s_or_b64 s[14:15], s[12:13], s[14:15]
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_cbranch_execz .LBB4_4
.LBB4_44:
	s_mov_b32 s6, 0
	s_mov_b32 s7, 0x7ff00000
	v_and_b32_e32 v6, 0x7ff00000, v1
	v_mov_b32_e32 v5, 0
	v_cmp_ne_u64_e32 vcc, s[6:7], v[5:6]
	v_cmp_eq_u64_e64 s[6:7], 0, v[9:10]
	s_or_b64 vcc, vcc, s[6:7]
	v_cndmask_b32_e32 v6, v1, v3, vcc
	v_cndmask_b32_e32 v5, v0, v2, vcc
	s_or_b64 exec, exec, s[12:13]
	s_mov_b64 s[12:13], 0
	s_and_saveexec_b64 s[6:7], s[18:19]
	s_cbranch_execz .LBB4_5
.LBB4_45:
	v_cmp_ne_u64_e32 vcc, 0, v[11:12]
	s_mov_b64 s[18:19], -1
	s_mov_b64 s[14:15], s[16:17]
	s_and_saveexec_b64 s[12:13], vcc
; %bb.46:
	v_add_co_u32_e32 v2, vcc, v7, v9
	v_addc_co_u32_e32 v3, vcc, v8, v10, vcc
	v_lshlrev_b64 v[17:18], 9, v[2:3]
	v_mov_b32_e32 v20, v12
	v_or_b32_e32 v18, 2.0, v18
	s_xor_b64 s[18:19], exec, -1
	s_or_b64 s[14:15], s[16:17], exec
	v_mov_b32_e32 v19, v11
; %bb.47:
	s_or_b64 exec, exec, s[12:13]
	s_andn2_b64 s[16:17], s[16:17], exec
	s_and_b64 s[14:15], s[14:15], exec
	s_and_b64 s[12:13], s[18:19], exec
	s_or_b64 s[16:17], s[16:17], s[14:15]
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[6:7], s[16:17]
	s_xor_b64 s[14:15], exec, s[6:7]
	s_cbranch_execz .LBB4_6
.LBB4_48:
	v_and_b32_e32 v2, 0xffff, v19
	s_mov_b64 s[6:7], 0x7fc
	v_mov_b32_e32 v3, 0
	v_cmp_lt_u64_e32 vcc, s[6:7], v[2:3]
	s_mov_b64 s[18:19], -1
                                        ; implicit-def: $vgpr5_vgpr6
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB4_56
; %bb.49:
	v_cmp_lt_i64_e32 vcc, -1, v[19:20]
	s_mov_b64 s[20:21], -1
	s_mov_b64 s[6:7], 0
                                        ; implicit-def: $vgpr5_vgpr6
	s_and_saveexec_b64 s[18:19], vcc
	s_xor_b64 s[18:19], exec, s[18:19]
	s_cbranch_execz .LBB4_53
; %bb.50:
	v_add_co_u32_e32 v2, vcc, 0x200, v17
	s_mov_b64 s[22:23], 0x7fd
	v_addc_co_u32_e32 v3, vcc, 0, v18, vcc
	v_cmp_lt_u64_e64 s[6:7], s[22:23], v[19:20]
	v_cmp_gt_i64_e32 vcc, 0, v[2:3]
                                        ; implicit-def: $vgpr5_vgpr6
	s_or_b64 s[6:7], s[6:7], vcc
	s_and_saveexec_b64 s[24:25], s[6:7]
	s_xor_b64 s[6:7], exec, s[24:25]
; %bb.51:
	v_mov_b32_e32 v2, 0x7ff00000
	v_mov_b32_e32 v3, 0xfff00000
	v_cndmask_b32_e64 v6, v2, v3, s[8:9]
	v_mov_b32_e32 v5, 0
	s_xor_b64 s[20:21], exec, -1
; %bb.52:
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v19, s22
	v_mov_b32_e32 v20, s23
	s_and_b64 s[6:7], s[20:21], exec
.LBB4_53:
	s_andn2_saveexec_b64 s[18:19], s[18:19]
; %bb.54:
	v_lshrrev_b64 v[2:3], 1, v[17:18]
	v_and_b32_e32 v9, 1, v17
	v_or_b32_e32 v2, v2, v9
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v18, v3
	v_mov_b32_e32 v20, 0
	s_or_b64 s[6:7], s[6:7], exec
	v_mov_b32_e32 v17, v2
; %bb.55:
	s_or_b64 exec, exec, s[18:19]
	s_orn2_b64 s[18:19], s[6:7], exec
.LBB4_56:
	s_or_b64 exec, exec, s[16:17]
	s_and_saveexec_b64 s[6:7], s[18:19]
	s_cbranch_execz .LBB4_58
; %bb.57:
	v_add_co_u32_e32 v5, vcc, 0x200, v17
	v_and_b32_e32 v2, 0x3ff, v17
	s_mov_b64 s[16:17], 0x200
	v_mov_b32_e32 v3, 0
	v_addc_co_u32_e32 v6, vcc, 0, v18, vcc
	v_cmp_eq_u64_e32 vcc, s[16:17], v[2:3]
	v_lshrrev_b64 v[5:6], 10, v[5:6]
	v_cndmask_b32_e64 v2, 0, 1, vcc
	v_not_b32_e32 v2, v2
	v_and_b32_e32 v5, v5, v2
	v_cmp_ne_u64_e32 vcc, 0, v[5:6]
	v_and_b32_e32 v2, 0x80000000, v1
	v_lshlrev_b32_e32 v3, 20, v19
	v_cndmask_b32_e32 v3, 0, v3, vcc
	v_or_b32_e32 v2, v6, v2
	v_add_co_u32_e32 v5, vcc, 0, v5
	v_addc_co_u32_e32 v6, vcc, v2, v3, vcc
.LBB4_58:
	s_or_b64 exec, exec, s[6:7]
	s_or_b64 exec, exec, s[14:15]
	s_and_saveexec_b64 s[6:7], s[12:13]
	s_cbranch_execnz .LBB4_7
	s_branch .LBB4_8
.LBB4_59:
	v_lshlrev_b64 v[21:22], 10, v[9:10]
	v_lshlrev_b64 v[19:20], 10, v[7:8]
	v_cmp_lt_i64_e32 vcc, -1, v[15:16]
	s_mov_b64 s[14:15], 0
	s_mov_b64 s[18:19], 0
	s_mov_b64 s[12:13], 0
                                        ; implicit-def: $vgpr17_vgpr18
	s_and_saveexec_b64 s[4:5], vcc
	s_xor_b64 s[16:17], exec, s[4:5]
	s_cbranch_execnz .LBB4_63
; %bb.60:
	s_or_saveexec_b64 s[20:21], s[16:17]
	s_mov_b64 s[16:17], s[8:9]
	s_xor_b64 exec, exec, s[20:21]
	s_cbranch_execnz .LBB4_72
.LBB4_61:
	s_or_b64 exec, exec, s[20:21]
	s_and_saveexec_b64 s[4:5], s[18:19]
	s_xor_b64 s[18:19], exec, s[4:5]
	s_cbranch_execnz .LBB4_79
.LBB4_62:
	s_or_b64 exec, exec, s[18:19]
	s_and_saveexec_b64 s[4:5], s[14:15]
	s_cbranch_execnz .LBB4_98
	s_branch .LBB4_101
.LBB4_63:
	s_mov_b64 s[20:21], 0x7ff
	v_cmp_ne_u64_e32 vcc, s[20:21], v[11:12]
	s_mov_b64 s[12:13], 0
	s_mov_b64 s[4:5], 0
                                        ; implicit-def: $vgpr17_vgpr18
	s_and_saveexec_b64 s[18:19], vcc
	s_xor_b64 s[18:19], exec, s[18:19]
	s_cbranch_execz .LBB4_69
; %bb.64:
	v_cmp_eq_u64_e32 vcc, 0, v[13:14]
	v_cndmask_b32_e32 v14, 0, v19, vcc
	v_cndmask_b32_e32 v13, 2.0, v20, vcc
	v_cmp_lt_u64_e32 vcc, 62, v[15:16]
	v_add_co_u32_e64 v16, s[4:5], v14, v19
	v_addc_co_u32_e64 v17, s[4:5], v13, v20, s[4:5]
                                        ; implicit-def: $vgpr13_vgpr14
	s_and_saveexec_b64 s[4:5], vcc
	s_xor_b64 s[4:5], exec, s[4:5]
; %bb.65:
	v_cmp_ne_u64_e32 vcc, 0, v[16:17]
	v_mov_b32_e32 v14, s21
	v_cndmask_b32_e64 v13, 0, 1, vcc
                                        ; implicit-def: $vgpr15
                                        ; implicit-def: $vgpr16_vgpr17
; %bb.66:
	s_andn2_saveexec_b64 s[4:5], s[4:5]
; %bb.67:
	v_sub_u32_e32 v13, 0, v15
	v_lshlrev_b64 v[18:19], v13, v[16:17]
	v_lshrrev_b64 v[13:14], v15, v[16:17]
	v_cmp_ne_u64_e32 vcc, 0, v[18:19]
	v_cndmask_b32_e64 v15, 0, 1, vcc
	v_or_b32_e32 v13, v13, v15
; %bb.68:
	s_or_b64 exec, exec, s[4:5]
	v_sub_co_u32_e32 v17, vcc, v21, v13
	s_mov_b64 s[4:5], exec
	v_subb_co_u32_e32 v18, vcc, v22, v14, vcc
.LBB4_69:
	s_andn2_saveexec_b64 s[18:19], s[18:19]
; %bb.70:
	v_cmp_ne_u64_e32 vcc, 0, v[9:10]
	s_and_b64 s[12:13], vcc, exec
; %bb.71:
	s_or_b64 exec, exec, s[18:19]
	s_and_b64 s[12:13], s[12:13], exec
	s_and_b64 s[18:19], s[4:5], exec
                                        ; implicit-def: $vgpr13_vgpr14
                                        ; implicit-def: $vgpr19_vgpr20
                                        ; implicit-def: $vgpr15
                                        ; implicit-def: $vgpr21_vgpr22
	s_or_saveexec_b64 s[20:21], s[16:17]
	s_mov_b64 s[16:17], s[8:9]
	s_xor_b64 exec, exec, s[20:21]
	s_cbranch_execz .LBB4_61
.LBB4_72:
	s_mov_b64 s[16:17], 0x7ff
	v_cmp_ne_u64_e32 vcc, s[16:17], v[13:14]
	s_mov_b64 s[22:23], -1
	s_mov_b64 s[4:5], s[18:19]
	s_mov_b64 s[24:25], s[8:9]
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB4_78
; %bb.73:
	v_cmp_eq_u64_e32 vcc, 0, v[11:12]
	v_cndmask_b32_e32 v11, 2.0, v22, vcc
	v_cndmask_b32_e32 v12, 0, v21, vcc
	v_sub_co_u32_e32 v17, vcc, 0, v15
	v_subb_co_u32_e32 v18, vcc, 0, v16, vcc
	v_cmp_lt_u64_e32 vcc, 62, v[17:18]
	v_add_co_u32_e64 v21, s[4:5], v12, v21
	v_addc_co_u32_e64 v22, s[4:5], v11, v22, s[4:5]
                                        ; implicit-def: $vgpr11_vgpr12
	s_and_saveexec_b64 s[4:5], vcc
	s_xor_b64 s[4:5], exec, s[4:5]
; %bb.74:
	v_cmp_ne_u64_e32 vcc, 0, v[21:22]
	v_mov_b32_e32 v12, s17
	v_cndmask_b32_e64 v11, 0, 1, vcc
                                        ; implicit-def: $vgpr17
                                        ; implicit-def: $vgpr21_vgpr22
                                        ; implicit-def: $vgpr15
; %bb.75:
	s_andn2_saveexec_b64 s[4:5], s[4:5]
; %bb.76:
	v_lshlrev_b64 v[15:16], v15, v[21:22]
	v_lshrrev_b64 v[11:12], v17, v[21:22]
	v_cmp_ne_u64_e32 vcc, 0, v[15:16]
	v_cndmask_b32_e64 v15, 0, 1, vcc
	v_or_b32_e32 v11, v11, v15
; %bb.77:
	s_or_b64 exec, exec, s[4:5]
	s_xor_b64 s[4:5], s[8:9], -1
	v_sub_co_u32_e32 v17, vcc, v19, v11
	v_subb_co_u32_e32 v18, vcc, v20, v12, vcc
	s_andn2_b64 s[16:17], s[8:9], exec
	s_and_b64 s[4:5], s[4:5], exec
	v_mov_b32_e32 v11, v13
	s_or_b64 s[24:25], s[16:17], s[4:5]
	s_xor_b64 s[22:23], exec, -1
	s_or_b64 s[4:5], s[18:19], exec
	v_mov_b32_e32 v12, v14
.LBB4_78:
	s_or_b64 exec, exec, s[14:15]
	s_andn2_b64 s[14:15], s[8:9], exec
	s_and_b64 s[16:17], s[24:25], exec
	s_andn2_b64 s[18:19], s[18:19], exec
	s_and_b64 s[4:5], s[4:5], exec
	s_or_b64 s[16:17], s[14:15], s[16:17]
	s_and_b64 s[14:15], s[22:23], exec
	s_or_b64 s[18:19], s[18:19], s[4:5]
	s_or_b64 exec, exec, s[20:21]
	s_and_saveexec_b64 s[4:5], s[18:19]
	s_xor_b64 s[18:19], exec, s[4:5]
	s_cbranch_execz .LBB4_62
.LBB4_79:
	v_add_co_u32_e32 v15, vcc, 0, v17
	v_addc_co_u32_e32 v16, vcc, 2.0, v18, vcc
	v_cmp_eq_u32_e32 vcc, 0, v16
	v_cndmask_b32_e64 v5, 0, 1, vcc
	v_cndmask_b32_e32 v6, v16, v17, vcc
	s_mov_b32 s4, 0x10000
	v_lshlrev_b16_e32 v5, 5, v5
	v_lshlrev_b32_e32 v14, 16, v6
	v_cmp_gt_u32_e32 vcc, s4, v6
	v_or_b32_e32 v13, 16, v5
	v_cndmask_b32_e32 v6, v6, v14, vcc
	s_mov_b32 s4, 0x1000000
	v_cndmask_b32_e32 v5, v5, v13, vcc
	v_lshlrev_b32_e32 v14, 8, v6
	v_cmp_gt_u32_e32 vcc, s4, v6
	v_cndmask_b32_e32 v6, v6, v14, vcc
	v_lshrrev_b32_e32 v6, 24, v6
	s_getpc_b64 s[4:5]
	s_add_u32 s4, s4, softfloat_countLeadingZeros8@rel32@lo+4
	s_addc_u32 s5, s5, softfloat_countLeadingZeros8@rel32@hi+12
	global_load_ubyte v6, v6, s[4:5]
	v_or_b32_e32 v13, 8, v5
	v_cndmask_b32_e32 v5, v5, v13, vcc
	v_mov_b32_e32 v13, 10
	s_mov_b64 s[4:5], 0x7fc
	s_waitcnt vmcnt(0)
	v_add_u16_e32 v5, v6, v5
	v_add_u16_e32 v6, -1, v5
	v_sub_u16_e32 v5, 0, v5
	v_bfe_i32 v5, v5, 0, 8
	v_and_b32_e32 v17, 0xff, v6
	v_cmp_lt_i16_sdwa s[20:21], sext(v6), v13 src0_sel:BYTE_0 src1_sel:DWORD
	v_ashrrev_i32_e32 v6, 31, v5
	v_add_co_u32_e32 v13, vcc, v11, v5
	v_addc_co_u32_e32 v14, vcc, v12, v6, vcc
	v_cmp_lt_u32_e32 vcc, s4, v13
	s_or_b64 s[20:21], s[20:21], vcc
                                        ; implicit-def: $vgpr5_vgpr6
	s_and_saveexec_b64 s[22:23], s[20:21]
	s_xor_b64 s[20:21], exec, s[22:23]
	s_cbranch_execz .LBB4_95
; %bb.80:
	v_and_b32_e32 v5, 0xffff, v13
	v_mov_b32_e32 v6, 0
	v_lshlrev_b64 v[11:12], v17, v[15:16]
	v_cmp_lt_u64_e32 vcc, s[4:5], v[5:6]
	s_mov_b64 s[24:25], -1
                                        ; implicit-def: $vgpr5_vgpr6
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB4_92
; %bb.81:
	v_cmp_lt_i64_e32 vcc, -1, v[13:14]
	s_mov_b64 s[26:27], -1
	s_mov_b64 s[4:5], 0
                                        ; implicit-def: $vgpr5_vgpr6
	s_and_saveexec_b64 s[24:25], vcc
	s_xor_b64 s[24:25], exec, s[24:25]
	s_cbranch_execz .LBB4_85
; %bb.82:
	v_add_co_u32_e32 v5, vcc, 0x200, v11
	s_mov_b64 s[28:29], 0x7fd
	v_addc_co_u32_e32 v6, vcc, 0, v12, vcc
	v_cmp_lt_u64_e64 s[4:5], s[28:29], v[13:14]
	v_cmp_gt_i64_e32 vcc, 0, v[5:6]
                                        ; implicit-def: $vgpr5_vgpr6
	s_or_b64 s[4:5], s[4:5], vcc
	s_and_saveexec_b64 vcc, s[4:5]
	s_xor_b64 s[4:5], exec, vcc
; %bb.83:
	v_mov_b32_e32 v5, 0x7ff00000
	v_mov_b32_e32 v6, 0xfff00000
	v_cndmask_b32_e64 v6, v5, v6, s[16:17]
	v_mov_b32_e32 v5, 0
	s_xor_b64 s[26:27], exec, -1
; %bb.84:
	s_or_b64 exec, exec, s[4:5]
	v_mov_b32_e32 v13, s28
	v_mov_b32_e32 v14, s29
	s_and_b64 s[4:5], s[26:27], exec
.LBB4_85:
	s_andn2_saveexec_b64 s[24:25], s[24:25]
	s_cbranch_execz .LBB4_91
; %bb.86:
	v_sub_co_u32_e32 v15, vcc, 0, v13
	v_subb_co_u32_e32 v16, vcc, 0, v14, vcc
	v_cmp_lt_u64_e32 vcc, 62, v[15:16]
	s_and_saveexec_b64 s[26:27], vcc
	s_xor_b64 s[26:27], exec, s[26:27]
; %bb.87:
	v_cmp_ne_u64_e32 vcc, 0, v[11:12]
	s_mov_b32 s28, 0
	v_cndmask_b32_e64 v11, 0, 1, vcc
	v_mov_b32_e32 v12, s28
                                        ; implicit-def: $vgpr15
                                        ; implicit-def: $vgpr13
; %bb.88:
	s_andn2_saveexec_b64 s[26:27], s[26:27]
; %bb.89:
	v_lshlrev_b64 v[13:14], v13, v[11:12]
	v_lshrrev_b64 v[11:12], v15, v[11:12]
	v_cmp_ne_u64_e32 vcc, 0, v[13:14]
	v_cndmask_b32_e64 v13, 0, 1, vcc
	v_or_b32_e32 v11, v11, v13
; %bb.90:
	s_or_b64 exec, exec, s[26:27]
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v14, 0
	s_or_b64 s[4:5], s[4:5], exec
.LBB4_91:
	s_or_b64 exec, exec, s[24:25]
	s_orn2_b64 s[24:25], s[4:5], exec
.LBB4_92:
	s_or_b64 exec, exec, s[22:23]
	s_and_saveexec_b64 s[4:5], s[24:25]
	s_cbranch_execz .LBB4_94
; %bb.93:
	v_and_b32_e32 v5, 0x3ff, v11
	v_add_co_u32_e32 v11, vcc, 0x200, v11
	s_mov_b64 s[22:23], 0x200
	v_mov_b32_e32 v6, 0
	v_addc_co_u32_e32 v12, vcc, 0, v12, vcc
	v_cmp_eq_u64_e32 vcc, s[22:23], v[5:6]
	v_lshrrev_b64 v[11:12], 10, v[11:12]
	v_cndmask_b32_e64 v5, 0, 1, vcc
	v_not_b32_e32 v5, v5
	v_and_b32_e32 v11, v11, v5
	v_bfrev_b32_e32 v5, 1
	v_cmp_ne_u64_e32 vcc, 0, v[11:12]
	v_cndmask_b32_e64 v5, 0, v5, s[16:17]
	v_lshlrev_b32_e32 v6, 20, v13
	v_cndmask_b32_e32 v6, 0, v6, vcc
	v_or_b32_e32 v12, v12, v5
	v_add_co_u32_e32 v5, vcc, 0, v11
	v_addc_co_u32_e32 v6, vcc, v12, v6, vcc
.LBB4_94:
	s_or_b64 exec, exec, s[4:5]
                                        ; implicit-def: $vgpr13
                                        ; implicit-def: $vgpr15_vgpr16
                                        ; implicit-def: $vgpr17
.LBB4_95:
	s_andn2_saveexec_b64 s[4:5], s[20:21]
; %bb.96:
	v_cmp_ne_u64_e32 vcc, 0, v[15:16]
	v_bfrev_b32_e32 v5, 1
	v_lshlrev_b32_e32 v6, 20, v13
	v_cndmask_b32_e64 v5, 0, v5, s[16:17]
	v_cndmask_b32_e32 v6, 0, v6, vcc
	v_add_co_u32_e64 v11, vcc, 0, 0
	v_addc_co_u32_e32 v12, vcc, v6, v5, vcc
	v_add_u32_e32 v5, 54, v17
	v_lshlrev_b64 v[5:6], v5, v[15:16]
	v_add_co_u32_e32 v5, vcc, v11, v5
	v_addc_co_u32_e32 v6, vcc, v12, v6, vcc
; %bb.97:
	s_or_b64 exec, exec, s[4:5]
	s_or_b64 exec, exec, s[18:19]
	s_and_saveexec_b64 s[4:5], s[14:15]
	s_cbranch_execz .LBB4_101
.LBB4_98:
	v_cmp_eq_u64_e32 vcc, 0, v[7:8]
	s_mov_b64 s[14:15], -1
	s_and_saveexec_b64 s[16:17], vcc
	s_xor_b64 s[16:17], exec, s[16:17]
; %bb.99:
	v_mov_b32_e32 v5, 0xfff00000
	v_mov_b32_e32 v6, 0x7ff00000
	v_cndmask_b32_e64 v6, v5, v6, s[8:9]
	v_mov_b32_e32 v5, 0
	s_xor_b64 s[14:15], exec, -1
; %bb.100:
	s_or_b64 exec, exec, s[16:17]
	s_andn2_b64 s[8:9], s[12:13], exec
	s_and_b64 s[12:13], s[14:15], exec
	s_or_b64 s[12:13], s[8:9], s[12:13]
.LBB4_101:
	s_or_b64 exec, exec, s[4:5]
	s_and_b64 s[12:13], s[12:13], exec
                                        ; implicit-def: $vgpr7_vgpr8
                                        ; implicit-def: $vgpr11_vgpr12
	s_andn2_saveexec_b64 s[4:5], s[10:11]
	s_cbranch_execz .LBB4_12
.LBB4_102:
	s_mov_b64 s[8:9], 0x7ff
	v_cmp_ne_u64_e32 vcc, s[8:9], v[11:12]
	s_and_saveexec_b64 s[8:9], vcc
	s_xor_b64 s[8:9], exec, s[8:9]
	s_cbranch_execz .LBB4_106
; %bb.103:
	v_cmp_ne_u64_e32 vcc, v[9:10], v[7:8]
	v_mov_b32_e32 v5, 0
	v_mov_b32_e32 v6, 0
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB4_105
; %bb.104:
	v_sub_co_u32_e32 v5, vcc, v9, v7
	v_subb_co_u32_e32 v6, vcc, v10, v8, vcc
	v_ashrrev_i32_e32 v8, 31, v6
	v_xor_b32_e32 v5, v5, v8
	v_xor_b32_e32 v13, v6, v8
	v_sub_co_u32_e32 v7, vcc, v5, v8
	v_subb_co_u32_e32 v8, vcc, v13, v8, vcc
	v_cmp_eq_u32_e32 vcc, 0, v8
	v_cndmask_b32_e64 v5, 0, 1, vcc
	v_cndmask_b32_e32 v13, v8, v7, vcc
	s_mov_b32 s14, 0x10000
	v_lshlrev_b16_e32 v5, 5, v5
	v_lshlrev_b32_e32 v15, 16, v13
	v_cmp_gt_u32_e32 vcc, s14, v13
	v_or_b32_e32 v14, 16, v5
	v_cndmask_b32_e32 v13, v13, v15, vcc
	s_mov_b32 s14, 0x1000000
	v_cndmask_b32_e32 v5, v5, v14, vcc
	v_lshlrev_b32_e32 v15, 8, v13
	v_cmp_gt_u32_e32 vcc, s14, v13
	v_cndmask_b32_e32 v13, v13, v15, vcc
	v_lshrrev_b32_e32 v13, 24, v13
	s_getpc_b64 s[14:15]
	s_add_u32 s14, s14, softfloat_countLeadingZeros8@rel32@lo+4
	s_addc_u32 s15, s15, softfloat_countLeadingZeros8@rel32@hi+12
	global_load_ubyte v15, v13, s[14:15]
	v_or_b32_e32 v14, 8, v5
	v_cndmask_b32_e32 v5, v5, v14, vcc
	v_add_co_u32_e32 v13, vcc, -1, v11
	v_addc_co_u32_e64 v14, s[14:15], 0, -1, vcc
	v_cmp_gt_u64_e32 vcc, v[13:14], v[11:12]
	v_xor_b32_e32 v6, v6, v1
	v_cndmask_b32_e64 v11, v14, 0, vcc
	v_cndmask_b32_e64 v13, v13, 0, vcc
	v_and_b32_e32 v12, 0x80000000, v6
	s_waitcnt vmcnt(0)
	v_add_u16_e32 v5, v5, v15
	v_add_u16_e32 v14, -11, v5
	v_bfe_i32 v5, v14, 0, 8
	v_ashrrev_i32_e32 v6, 31, v5
	v_sub_co_u32_e32 v5, vcc, v13, v5
	v_subb_co_u32_e32 v6, vcc, v11, v6, vcc
	v_cmp_gt_i64_e32 vcc, 0, v[5:6]
	v_cndmask_b32_e32 v11, v14, v13, vcc
	v_cmp_lt_i64_e32 vcc, 0, v[5:6]
	v_and_b32_e32 v6, 63, v11
	v_cndmask_b32_e32 v5, 0, v5, vcc
	v_lshlrev_b32_e32 v5, 20, v5
	v_add_co_u32_e64 v11, vcc, 0, 0
	v_addc_co_u32_e32 v12, vcc, v5, v12, vcc
	v_and_b32_e32 v5, 0xffff, v6
	v_lshlrev_b64 v[5:6], v5, v[7:8]
	v_add_co_u32_e32 v5, vcc, v11, v5
	v_addc_co_u32_e32 v6, vcc, v12, v6, vcc
.LBB4_105:
	s_or_b64 exec, exec, s[10:11]
                                        ; implicit-def: $vgpr7_vgpr8
.LBB4_106:
	s_or_saveexec_b64 s[8:9], s[8:9]
	s_mov_b64 s[10:11], s[12:13]
	s_xor_b64 exec, exec, s[8:9]
; %bb.107:
	v_or_b32_e32 v6, v8, v10
	v_or_b32_e32 v5, v7, v9
	v_cmp_ne_u64_e32 vcc, 0, v[5:6]
	s_mov_b32 s10, 0
	s_mov_b32 s11, 0xfff80000
	v_mov_b32_e32 v5, s10
	v_mov_b32_e32 v6, s11
	s_andn2_b64 s[10:11], s[12:13], exec
	s_and_b64 s[14:15], vcc, exec
	s_or_b64 s[10:11], s[10:11], s[14:15]
; %bb.108:
	s_or_b64 exec, exec, s[8:9]
	s_andn2_b64 s[8:9], s[12:13], exec
	s_and_b64 s[10:11], s[10:11], exec
	s_or_b64 s[12:13], s[8:9], s[10:11]
	s_or_b64 exec, exec, s[4:5]
	s_and_saveexec_b64 s[8:9], s[12:13]
	s_cbranch_execnz .LBB4_13
	s_branch .LBB4_14
.Lfunc_end4:
	.size	__softfloat_f64_sub, .Lfunc_end4-__softfloat_f64_sub
                                        ; -- End function
	.section	.AMDGPU.csdata
; Function info:
; codeLenInByte = 3324
; NumSgprs: 37
; NumVgprs: 27
; ScratchSize: 8
; MemoryBound: 0
	.type	softfloat_countLeadingZeros8,@object ; @softfloat_countLeadingZeros8
	.section	.rodata,#alloc
softfloat_countLeadingZeros8:
	.ascii	"\b\007\006\006\005\005\005\005\004\004\004\004\004\004\004\004\003\003\003\003\003\003\003\003\003\003\003\003\003\003\003\003\002\002\002\002\002\002\002\002\002\002\002\002\002\002\002\002\002\002\002\002\002\002\002\002\002\002\002\002\002\002\002\002\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001"
	.zero	128
	.size	softfloat_countLeadingZeros8, 256

	.ident	"clang version 15.0.0"
	.section	".note.GNU-stack"
	.addrsig
	.addrsig_sym __softfloat_f64_add
	.addrsig_sym __softfloat_f64_div
	.addrsig_sym __softfloat_f64_mul
	.addrsig_sym __softfloat_f64_sub
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
    .sgpr_count:     9
    .sgpr_spill_count: 0
    .symbol:         _Z6reducePfS_.kd
    .vgpr_count:     6
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   'amdgcn-amd-amdhsa--gfx928:sramecc+'
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata

# __CLANG_OFFLOAD_BUNDLE____END__ hip-amdgcn-amd-amdhsa-gfx928

# __CLANG_OFFLOAD_BUNDLE____START__ host-x86_64-unknown-linux-gnu
	.text
	.file	"reduce_16.cpp"
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
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4                               # -- Begin function main
.LCPI1_0:
	.long	0x3f800000                      # float 1
	.long	0x3f800000                      # float 1
	.long	0x3f800000                      # float 1
	.long	0x3f800000                      # float 1
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
	movl	$512, %edi                      # imm = 0x200
	callq	_Znam
	movq	%rax, %r12
	movl	$512, %edi                      # imm = 0x200
	callq	_Znam
	movq	%rax, %r14
	movl	$1, %edi
	callq	srand
	movaps	.LCPI1_0(%rip), %xmm0           # xmm0 = [1.0E+0,1.0E+0,1.0E+0,1.0E+0]
	movups	%xmm0, (%r12)
	movups	%xmm0, 16(%r12)
	movups	%xmm0, 32(%r12)
	movups	%xmm0, 48(%r12)
	movups	%xmm0, 64(%r12)
	movups	%xmm0, 80(%r12)
	movups	%xmm0, 96(%r12)
	movups	%xmm0, 112(%r12)
	movups	%xmm0, 128(%r12)
	movups	%xmm0, 144(%r12)
	movups	%xmm0, 160(%r12)
	movups	%xmm0, 176(%r12)
	movups	%xmm0, 192(%r12)
	movups	%xmm0, 208(%r12)
	movups	%xmm0, 224(%r12)
	movups	%xmm0, 240(%r12)
	movups	%xmm0, 256(%r12)
	movups	%xmm0, 272(%r12)
	movups	%xmm0, 288(%r12)
	movups	%xmm0, 304(%r12)
	movups	%xmm0, 320(%r12)
	movups	%xmm0, 336(%r12)
	movups	%xmm0, 352(%r12)
	movups	%xmm0, 368(%r12)
	movups	%xmm0, 384(%r12)
	movups	%xmm0, 400(%r12)
	movups	%xmm0, 416(%r12)
	movups	%xmm0, 432(%r12)
	movups	%xmm0, 448(%r12)
	movups	%xmm0, 464(%r12)
	movups	%xmm0, 480(%r12)
	movups	%xmm0, 496(%r12)
	movl	$60, %ebx
	.p2align	4, 0x90
.LBB1_7:                                # =>This Inner Loop Header: Depth=1
	movss	-60(%r12,%rbx), %xmm0           # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movl	$.L.str.1, %edi
	movb	$1, %al
	callq	printf
	movss	-56(%r12,%rbx), %xmm0           # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movl	$.L.str.1, %edi
	movb	$1, %al
	callq	printf
	movss	-52(%r12,%rbx), %xmm0           # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movl	$.L.str.1, %edi
	movb	$1, %al
	callq	printf
	movss	-48(%r12,%rbx), %xmm0           # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movl	$.L.str.1, %edi
	movb	$1, %al
	callq	printf
	movss	-44(%r12,%rbx), %xmm0           # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movl	$.L.str.1, %edi
	movb	$1, %al
	callq	printf
	movss	-40(%r12,%rbx), %xmm0           # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movl	$.L.str.1, %edi
	movb	$1, %al
	callq	printf
	movss	-36(%r12,%rbx), %xmm0           # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movl	$.L.str.1, %edi
	movb	$1, %al
	callq	printf
	movss	-32(%r12,%rbx), %xmm0           # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movl	$.L.str.1, %edi
	movb	$1, %al
	callq	printf
	movss	-28(%r12,%rbx), %xmm0           # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movl	$.L.str.1, %edi
	movb	$1, %al
	callq	printf
	movss	-24(%r12,%rbx), %xmm0           # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movl	$.L.str.1, %edi
	movb	$1, %al
	callq	printf
	movss	-20(%r12,%rbx), %xmm0           # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movl	$.L.str.1, %edi
	movb	$1, %al
	callq	printf
	movss	-16(%r12,%rbx), %xmm0           # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movl	$.L.str.1, %edi
	movb	$1, %al
	callq	printf
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
	addq	$64, %rbx
	cmpq	$572, %rbx                      # imm = 0x23C
	jne	.LBB1_7
# %bb.8:
	movl	$.Lstr.4, %edi
	callq	puts@PLT
	leaq	16(%rsp), %rdi
	movl	$512, %esi                      # imm = 0x200
	callq	hipMalloc
	leaq	8(%rsp), %rdi
	movl	$512, %esi                      # imm = 0x200
	callq	hipMalloc
	movq	16(%rsp), %rdi
	movl	$512, %edx                      # imm = 0x200
	movq	%r12, %rsi
	movl	$1, %ecx
	callq	hipMemcpy
	movabsq	$4294967298, %rdi               # imm = 0x100000002
	leaq	62(%rdi), %rdx
	xorl	%r15d, %r15d
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB1_10
# %bb.9:
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
.LBB1_10:
	movq	8(%rsp), %rsi
	movl	$512, %edx                      # imm = 0x200
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
	jmp	.LBB1_11
.LBB1_1:
	movl	$_ZSt4cerr, %edi
	movl	$.L.str, %esi
	movl	$30, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	_ZSt4cerr(%rip), %rax
	movq	-24(%rax), %rax
	movq	_ZSt4cerr+240(%rax), %rbx
	testq	%rbx, %rbx
	je	.LBB1_12
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
.LBB1_11:
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
.LBB1_12:
	.cfi_def_cfa_offset 160
	callq	_ZSt16__throw_bad_castv
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
	.cfi_endproc
                                        # -- End function
	.section	.text.startup,"ax",@progbits
	.p2align	4, 0x90                         # -- Begin function _GLOBAL__sub_I_reduce_16.cpp
	.type	_GLOBAL__sub_I_reduce_16.cpp,@function
_GLOBAL__sub_I_reduce_16.cpp:           # @_GLOBAL__sub_I_reduce_16.cpp
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
	.size	_GLOBAL__sub_I_reduce_16.cpp, .Lfunc_end2-_GLOBAL__sub_I_reduce_16.cpp
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
	.asciz	"%.1f "
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
	.quad	_GLOBAL__sub_I_reduce_16.cpp
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
	.addrsig_sym _GLOBAL__sub_I_reduce_16.cpp
	.addrsig_sym __hip_module_ctor
	.addrsig_sym __hip_module_dtor
	.addrsig_sym _ZStL8__ioinit
	.addrsig_sym __dso_handle
	.addrsig_sym _Z6reducePfS_
	.addrsig_sym _ZSt4cerr
	.addrsig_sym __hip_fatbin
	.addrsig_sym __hip_fatbin_wrapper

# __CLANG_OFFLOAD_BUNDLE____END__ host-x86_64-unknown-linux-gnu
