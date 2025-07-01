	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx906"
	.amdhsa_code_object_version 4
	.text
	.globl	reduce
	.p2align	8
	.type	reduce,@function
reduce:
	s_load_dwordx4 s[0:3], s[8:9], 0x0
	v_mov_b32_e32 v0, 0
	s_waitcnt lgkmcnt(0)
	s_load_dword s0, s[0:1], 0x0
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v1, s0
	global_store_dword v0, v1, s[2:3]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel reduce
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 72
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
		.amdhsa_next_free_sgpr 10
		.amdhsa_reserve_vcc 0
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
	.size	reduce, .Lfunc_end0-reduce

	.set reduce.num_vgpr, 2
	.set reduce.num_agpr, 0
	.set reduce.numbered_sgpr, 10
	.set reduce.private_seg_size, 0
	.set reduce.uses_vcc, 0
	.set reduce.uses_flat_scratch, 0
	.set reduce.has_dyn_sized_stack, 0
	.set reduce.has_recursion, 0
	.set reduce.has_indirect_call, 0
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
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         24
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         32
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         40
        .size:           8
        .value_kind:     hidden_hostcall_buffer
      - .offset:         48
        .size:           8
        .value_kind:     hidden_default_queue
      - .offset:         56
        .size:           8
        .value_kind:     hidden_completion_action
      - .offset:         64
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 72
    .max_flat_workgroup_size: 1
    .name:           reduce
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         reduce.kd
    .vgpr_count:     2
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx906
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
