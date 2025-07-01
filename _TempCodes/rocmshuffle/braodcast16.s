	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx906"
	.amdhsa_code_object_version 4
	.text
	.globl	broadcast
	.p2align	8
	.type	broadcast,@function
broadcast:
	s_load_dwordx2 s[0:1], s[8:9], 0x0
	s_lshl_b32 s2, s12, 6
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 2
	v_mbcnt_lo_u32_b32 v0, -1, 0
	s_waitcnt lgkmcnt(0)
	s_add_u32 s0, s0, s2
	s_addc_u32 s1, s1, s3
	s_load_dword s2, s[0:1], 0x0
	v_mbcnt_hi_u32_b32 v0, -1, v0
	v_and_b32_e32 v1, 0x70, v0
	v_add_u32_e32 v1, 16, v1
	v_lshlrev_b32_e32 v0, 2, v0
	v_cmp_gt_i32_e32 vcc, 1, v1
	v_cndmask_b32_e32 v0, 0, v0, vcc
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v1, s2
	ds_bpermute_b32 v0, v0, v1
	v_mov_b32_e32 v1, 0
	s_waitcnt lgkmcnt(0)
	global_store_dword v1, v0, s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel broadcast
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 64
		.amdhsa_user_sgpr_count 12
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 1
		.amdhsa_user_sgpr_queue_ptr 1
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 1
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 13
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
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
	.size	broadcast, .Lfunc_end0-broadcast

	.set broadcast.num_vgpr, 2
	.set broadcast.num_agpr, 0
	.set broadcast.numbered_sgpr, 13
	.set broadcast.private_seg_size, 0
	.set broadcast.uses_vcc, 1
	.set broadcast.uses_flat_scratch, 0
	.set broadcast.has_dyn_sized_stack, 0
	.set broadcast.has_recursion, 0
	.set broadcast.has_indirect_call, 0
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         16
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         24
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         32
        .size:           8
        .value_kind:     hidden_hostcall_buffer
      - .offset:         40
        .size:           8
        .value_kind:     hidden_default_queue
      - .offset:         48
        .size:           8
        .value_kind:     hidden_completion_action
      - .offset:         56
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 64
    .max_flat_workgroup_size: 1
    .name:           broadcast
    .private_segment_fixed_size: 0
    .sgpr_count:     17
    .sgpr_spill_count: 0
    .symbol:         broadcast.kd
    .vgpr_count:     2
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx906
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
