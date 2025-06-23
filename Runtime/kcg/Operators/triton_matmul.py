import triton
import triton.language as tl
import torch


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
    128, 64, 16, 4
  )
  return c


import torch
import triton
import triton.language as tl

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
    aa = a
    bb = b
    # 保存原始形状
    orig_shape_a = a.shape
    orig_shape_b = b.shape
    
    # 展平批次维度
    a = a.view(-1, a.shape[-2], a.shape[-1])
    b = b.view(-1, b.shape[-2], b.shape[-1])
    
    B, M, K = a.shape
    B2, K2, N = b.shape
    assert K == K2, f"Incompatible dimensions: a K={K}, b K={K2}"
    assert B == B2, f"Batch size mismatch: a {B} vs b {B2}"
    
    # 创建输出张量（保持输入数据类型）
    c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
    
    # 计算网格大小
    def grid(META):
        return (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
            B  # 批次维度
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
    final_shape = orig_shape_a[:-2] + (M, N)
    return c.view(final_shape)
  
if __name__ == "__main__":
  # ...
  a = torch.randn((1, 12, 1024, 64), device='cuda', dtype=torch.float32)
  b = torch.randn((1, 12, 64, 1024), device='cuda', dtype=torch.float32)
  # c = matmul(a, b)
  
  c = bmm(a, b)
  d = torch.matmul(a,b)
  print(c.shape, d.shape)
  if torch.allclose(c,d, atol=1e-5,rtol=1e-5):
    print("test OK!")
  else:
    print("test error!")
    
  print(c)