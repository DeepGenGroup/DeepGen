	s_load_dwordx4 s[0:3], s[8:9], 0x0
	s_lshl_b32 s4, s12, 6
	s_ashr_i32 s5, s4, 31
	s_lshl_b64 s[4:5], s[4:5], 2
	v_mbcnt_lo_u32_b32 v0, -1, 0
	s_waitcnt lgkmcnt(0)
	s_add_u32 s0, s0, s4
	s_addc_u32 s1, s1, s5
	s_load_dword s0, s[0:1], 0x0
	v_mbcnt_hi_u32_b32 v0, -1, v0
	v_and_b32_e32 v1, 0x70, v0
	v_add_u32_e32 v1, 16, v1
	v_add_u32_e32 v2, 1, v0
	v_cmp_lt_i32_e32 vcc, v2, v1
	v_cndmask_b32_e32 v2, v0, v2, vcc
	v_lshlrev_b32_e32 v2, 2, v2
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v3, s0
	ds_bpermute_b32 v2, v2, v3
	v_add_u32_e32 v3, 2, v0
	v_cmp_lt_i32_e32 vcc, v3, v1
	v_cndmask_b32_e32 v0, v0, v3, vcc
	v_lshlrev_b32_e32 v0, 2, v0
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v2, s0, v2
	ds_bpermute_b32 v0, v0, v2
	s_add_u32 s0, s2, s4
	s_addc_u32 s1, s3, s5
	v_mov_b32_e32 v1, 0
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v0, v2, v0
	global_store_dword v1, v0, s[0:1]
	s_endpgm