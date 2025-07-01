	s_load_dwordx4 s[0:3], s[4:5], 0x0
	v_lshlrev_b32_e32 v0, 2, v0
	v_mbcnt_lo_u32_b32 v2, -1, 0
	v_mbcnt_hi_u32_b32 v2, -1, v2
	v_and_b32_e32 v3, 3, v2
	s_waitcnt lgkmcnt(0)
	global_load_dword v1, v0, s[0:1]
	v_cmp_ne_u32_e32 vcc, 3, v3
	v_addc_co_u32_e32 v4, vcc, 0, v2, vcc
	v_lshlrev_b32_e32 v4, 2, v4
	v_cmp_gt_u32_e32 vcc, 2, v3
	v_cndmask_b32_e64 v3, 0, 1, vcc
	v_lshlrev_b32_e32 v3, 1, v3
	v_add_lshl_u32 v2, v3, v2, 2
	s_waitcnt vmcnt(0)
	ds_bpermute_b32 v4, v4, v1
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v1, v1, v4
	ds_bpermute_b32 v2, v2, v1
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v1, v1, v2
	global_store_dword v0, v1, s[2:3]
	s_endpgm