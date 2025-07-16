import triton
import triton.language as tl
import torch
import numpy as np
import kcg.Utils as utils

# @triton.jit
# def bmm_kernel(
#   a_ptr, b_ptr, c_ptr, M, N, K,
#   stride_ab, stride_am, stride_ak, stride_bb, stride_bk, stride_bn, stride_cb, stride_cm, stride_cn,
#   BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr
# ):
#   pid_batch = tl.program_id(axis=1)
#   pid = tl.program_id(axis=0)
#   num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
#   num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
#   num_pid_in_group = GROUP_SIZE_M * num_pid_n
#   group_id = pid // num_pid_in_group
#   first_pid_m = group_id * GROUP_SIZE_M
#   group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
#   pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
#   pid_n = (pid % num_pid_in_group) // group_size_m
#   offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
#   offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
#   offs_k = tl.arange(0, BLOCK_SIZE_K)
#   a_ptrs = a_ptr + pid_batch * stride_ab + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
#   b_ptrs = b_ptr + pid_batch * stride_bb + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
#   accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
#   for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
#     a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
#     b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
#     accumulator += tl.dot(a, b)
#     a_ptrs += BLOCK_SIZE_K * stride_ak
#     b_ptrs += BLOCK_SIZE_K * stride_bk
#   c = accumulator
#   offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#   offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#   c_ptrs = c_ptr + pid_batch * stride_cb + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
#   c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
#   tl.store(c_ptrs, c, mask=c_mask)

m,n,k = 512,2048,8192
def get_gemm_configs(m,n,k) :
  bm = [32,64,128]
  bn = [32,64,128]
  bk = [8,16,32,64]
  tm = [1,2,4]
  tn = [1,2,4]
  ret = []
  for _bm in bm :
    for _bn in bn :
      for _bk in bk :
        for _tm in tm :
          for _tn in tn :
            thm = _bm // _tm
            thn = _bn // _tn
            if m / _bm < 4 or n / _bn < 4 :
              continue
            th = thm * thn
            if _bm % _tm == 0 and _bn % _tn == 0 and th >= 64 and th % 64 == 0:
              ret.append(triton.Config({'BLOCK_SIZE_M': _bm, 'BLOCK_SIZE_N': _bn, 'BLOCK_SIZE_K': _bk, 'GROUP_SIZE_M': 4, }, num_stages=0,num_warps=th // 64))  
  return ret

@triton.autotune(
  configs=[
      # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 4, }, num_stages=0,num_warps=4),  # best
      # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, }, num_stages=0,num_warps=2),  # best
      # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, }, num_stages=0,num_warps=4),  # best
      # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, }, num_stages=0,num_warps=4),  # best
      triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 4, }, num_stages=0,num_warps=4),  # not best
  ]
  ,key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
  a_ptr, b_ptr, c_ptr, M, N, K,
  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr
):
  pid = tl.program_id(axis=0)
  num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
  num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
  num_pid_in_group = GROUP_SIZE_M * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * GROUP_SIZE_M
  group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
  pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
  offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
  offs_k = tl.arange(0, BLOCK_SIZE_K)
  a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
  b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
  accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
  for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
    accumulator += tl.dot(a, b)
    a_ptrs += BLOCK_SIZE_K * stride_ak
    b_ptrs += BLOCK_SIZE_K * stride_bk
  c = accumulator
  offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
  offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
  c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
  tl.store(c_ptrs, c, mask=c_mask)

# def bmm(a : torch.Tensor, b:torch.Tensor):
#   # Check constraints
#   print('[bmm]',a.shape,b.shape,flush=True)
#   assert a.dim() <= 4 and b.dim() <= 4
#   if a.dim() >= 4:
#     for d in a.shape[0:-3] :
#       assert d == 1
#   _ba,_ma,_ka = a.shape[-3:]
#   _bb,_kb,_nb = b.shape[-3:]
#   # assert a.dim() == 3 and b.dim() == 3, f"Inputs must be 3D tensors : {a.shape}, {b.shape}"
#   assert _ba == _bb, "Batch sizes must match"
#   assert _ka == _kb, "Incompatible dimensions"
#   if not a.is_contiguous() :
#     a = a.contiguous()
#   if not b.is_contiguous():
#     b = b.contiguous()
#   B, M, K = a.shape[-3:]
#   B, K, N = b.shape[-3:]
#   c = torch.empty((B, M, N), device=a.device, dtype=torch.float32).contiguous()
#   grid = lambda META: (
#     triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
#     B  # batch dimension
#   )
#   bmm_kernel[grid](
#     a, b, c, M, N, K,
#     a.stride(0), a.stride(1), a.stride(2),
#     b.stride(0), b.stride(1), b.stride(2),
#     c.stride(0), c.stride(1), c.stride(2), 
#     128, 64, 16, 4
#   )
#   return c

def matmul(a, b):
  # Check constraints.
  assert a.shape[1] == b.shape[0], f"Incompatible dimensions, {a.shape, b.shape}"
  assert a.is_contiguous(), "Matrix A must be contiguous"
  M, K = a.shape
  K, N = b.shape
  # Allocates output.
  c = torch.empty((M, N), device=a.device, dtype=torch.float32)
  # 1D launch kernel where each block gets its own program.
  grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
  matmul_kernel[grid](
    a, b, c, M, N, K,
    a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
    # 128, 64, 16, 4
  )
  return c

@triton.jit
def bmm_kernel(
    a_ptr, b_ptr, c_ptr, 
    M, N, K,
    stride_ab, stride_am, stride_ak, 
    stride_bb, stride_bk, stride_bn, 
    stride_cb, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr, 
    GROUP_SIZE_M: tl.constexpr
):
    pid_batch = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)
    
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + pid_batch * stride_ab + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # 关键修正：b_ptr的索引顺序
    b_ptrs = b_ptr + pid_batch * stride_bb + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # 计算K维度的迭代次数
    num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
    
    for k in range(0, num_k_blocks):
        k_base = k * BLOCK_SIZE_K
        # 计算当前块的实际K维度边界
        k_remaining = K - k_base
        
        # 使用全局坐标进行边界检查
        a_mask = (offs_am[:, None] < M) & ((k_base + offs_k[None, :]) < K)
        b_mask = ((k_base + offs_k[:, None]) < K) & (offs_bn[None, :] < N)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # 关键修正：确保使用正确的数据类型
        a = a.to(tl.float32)
        b = b.to(tl.float32)
        
        # 累加矩阵乘法结果
        accumulator += tl.dot(a, b, allow_tf32=False)
        
        # 移动到下一个K块
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # 准备存储结果
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + pid_batch * stride_cb + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    
    # 结果边界检查
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    # 将结果转换为与输入相同的数据类型
    c = accumulator.to(tl.float32 if c_ptr.dtype.element_ty != tl.float16 else tl.float16)
    tl.store(c_ptrs, c, mask=c_mask)

def bmm(a: torch.Tensor, b: torch.Tensor):
    print('[bmm]', a.shape, b.shape, flush=True)
    # 验证输入维度
    assert a.dim() >= 2 and b.dim() >= 2, "Inputs must be at least 2D"
    assert a.shape[-1] == b.shape[-2], "Incompatible dimensions for matrix multiplication"
    # 保存原始形状
    orig_shape_a = a.shape
    orig_shape_b = b.shape
    outshape = a.shape
    
    # 展平批次维度
    a = a.reshape(-1, a.shape[-2], a.shape[-1])
    b = b.reshape(-1, b.shape[-2], b.shape[-1])
    
    B, M, K = a.shape
    B2, K2, N = b.shape
    assert K == K2, f"Incompatible dimensions: a K={K}, b K={K2}"
    batch = max(B,B2)
    if B < batch :
      a = a.expand(batch,-1,-1)
      outshape = b.shape
    if B2 < batch :
      b = b.expand(batch,-1,-1)
      outshape = a.shape
    # assert B == B2, f"Batch size mismatch: a {B} vs b {B2}"
    
    # 创建输出张量（保持输入数据类型）
    c = torch.empty((batch, M, N), device=a.device, dtype=a.dtype)
    
    # 计算网格大小
    def grid(META):
        return (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
            batch  # 批次维度
        )
    
    # 根据问题规模调整块大小
    if M >= 1024 and N >= 1024:
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
    else:
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        
    BLOCK_SIZE_K = 32 if K >= 512 else 16
    
    # 调用核函数
    bmm_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
        BLOCK_SIZE_M=BLOCK_SIZE_M, 
        BLOCK_SIZE_N=BLOCK_SIZE_N, 
        BLOCK_SIZE_K=BLOCK_SIZE_K, 
        GROUP_SIZE_M=8
    )
    
    # 恢复原始形状
    final_shape = outshape[:-2] + (M, N)
    return c.view(final_shape)

def clear_l2_cache(device=None):
    """清除 GPU L2 缓存的专业方法"""
    if device is None:
        device = torch.cuda.current_device()
    
    # 获取 GPU L2 缓存大小
    total_mem = torch.cuda.get_device_properties(device).total_memory
    # print(f'totalMEm = {total_mem}')
    # l2_size = 4 * 1024 * 1024  # 典型 L2 缓存大小 (根据 GPU 调整)
    l2_size = total_mem  # 典型 L2 缓存大小 (根据 GPU 调整)
    
    # 创建足够覆盖 L2 缓存的随机数据
    buffer_size = int( 0.8 * l2_size)  # 2 倍 L2 大小确保覆盖
    fill_buffer = torch.randn(buffer_size // 4, dtype=torch.float32, device=device)  # 每个 float32 占 4 字节
    
    # 执行覆盖操作
    fill_buffer.zero_()  # 确保实际写入操作
    torch.cuda.synchronize(device)


def test_perf(mm_base, mm_benchmark, testCount) :
  st = torch.cuda.Event(enable_timing=True)
  et = torch.cuda.Event(enable_timing=True)
  st_bench = torch.cuda.Event(enable_timing=True)
  et_bench = torch.cuda.Event(enable_timing=True)
  times_base = []
  times_bench = []
  for i in range(testCount) :
    st.record()
    r0 = mm_base()
    et.record()
    torch.cuda.synchronize()
    eps = st.elapsed_time(et)
    
    st_bench.record()
    r = mm_benchmark()
    et_bench.record()
    torch.cuda.synchronize()
    eps_bench = st_bench.elapsed_time(et_bench)
    
    if torch.allclose(r,r0,1e-3,1e-3) :
      times_base.append(eps)
      times_bench.append(eps_bench)
  t0 = np.median(times_base)
  t = np.median(times_bench)
  return (t0,t)

def testGEMMTriton(m,n,k, filepath = None) :
  # m,n,k = 512,2048,8192
  utils.DeviceInfo.init_cuda([7])
  a = torch.randn((m,k), device='cuda:7', dtype=torch.float32)
  b = torch.randn((k,n), device='cuda:7', dtype=torch.float32)
  c = matmul(a, b)
  
  # a = torch.randn((1,12 , 1024, 64), device='cuda', dtype=torch.float32)
  # b = torch.randn((1,12 , 64, 1024), device='cuda', dtype=torch.float32)
  # c = bmm(a, b)
  d = torch.matmul(a,b)
  
  def f_mm_base() :
    return torch.matmul(a,b)
  def f_mm_bench() :
    return matmul(a,b)
  print(f'-------- perf test: m={m} n={n} k = {k} ')
  (t_torch, t_triton) = test_perf(f_mm_base, f_mm_bench, 10)
  if filepath is not None:
    import json
    with open(filepath) as f:
      perfs_ours_vs_torch = json.load(f)
    t_ours = perfs_ours_vs_torch['testResult'][0]['time']
    print(f'---- GEMM test : m,n,k = {m,n,k}')
    print(f'  time(ms) : torch = {t_torch}, triton = {t_triton}, ours = {t_ours}')
    print(f'  ours speedup : 和torch比 = {t_torch / t_ours}, 和triton比 = {t_triton / t_ours}')
    
if __name__ == '__main__' :
  cases = [
    [512,2048,8192, '/home/xushilong/DeepGen/项目交付/gemm_result/res-mm-512-2048-8192.json'],
    [1024,2048,4096, '/home/xushilong/DeepGen/项目交付/gemm_result/res-mm-1024-2048-4096.json'],
    [1024,4096,2048, '/home/xushilong/DeepGen/项目交付/gemm_result/res-mm-1024-4096-2048.json'],
    [2048,2048,2048, '/home/xushilong/DeepGen/项目交付/gemm_result/res-mm-2048-2048-2048.json'],
  ]
  for e in cases :
    testGEMMTriton(*e) 
  
# -------- perf test: m=512 n=2048 k = 8192 
# torch = 2.8954415321350098, triton = 3.507841944694519, speedup = 0.8254196106281977
    # {
    #   "name": "kcg_MM_bM512N2048K8192isAT1W64_BM64BN64BK16TM4TN4BLY2BLX2WLY8WLX8GLWA4GLWB4BSWM4BSWN4WSWM4WSWN4LSU1Map4GSW0UN16RP0SP1LC1RC0",
    #   "speedup": 1.1667429692735567,
    #   "time": 2.4747190475463867,
    #   "time_base": 2.8873610496520996
    # },
    ## 加速比 (和triton比 : 1.4174， 和torch比：)
    
# -------- perf test: m=1024 n=2048 k = 4096 
# torch = 2.7201614379882812, triton = 3.0925610065460205, speedup = 0.879582143159187
# -------- perf test: m=1024 n=4096 k = 2048 
# torch = 2.212561011314392, triton = 2.955441474914551, speedup = 0.7486397650213537
# -------- perf test: m=2048 n=2048 k = 2048 
# torch = 2.2156810760498047, triton = 2.961761951446533, speedup = 0.7480955972736633