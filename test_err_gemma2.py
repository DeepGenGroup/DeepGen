import torch

torch.set_printoptions(precision=6, sci_mode=False)

def causal_upper_mask(S: int, device, dtype):
    # strict upper triangle = -inf, else 0
    return torch.where(
        torch.triu(torch.ones((S, S), device=device, dtype=torch.bool), diagonal=1),
        torch.full((S, S), float("-inf"), device=device, dtype=dtype),
        torch.zeros((S, S), device=device, dtype=dtype),
    )

def gemma2_before_preprocess(q, k, v, scale8=8.0, tanh_scale=50.0):
    """
    before preprocess (你的原始 exp 版):
      y = tanh((QK/8)/50)*50 + mask
      p = exp(y-m) / sum(exp(y-m))
      out = p @ V
    """
    B, S, H, D = q.shape
    device, dtype = q.device, q.dtype

    qh = q.permute(0, 2, 1, 3)   # [B,H,S,D]
    kh = k.permute(0, 2, 3, 1)   # [B,H,D,S]
    vh = v.permute(0, 2, 1, 3)   # [B,H,S,D]

    mask = causal_upper_mask(S, device, dtype).unsqueeze(0).unsqueeze(0)

    scores = torch.matmul(qh, kh) / scale8
    y = torch.tanh(scores / tanh_scale) * tanh_scale
    y = y + mask

    m = y.max(dim=-1, keepdim=True).values
    ex = torch.exp(y - m)
    denom = ex.sum(dim=-1, keepdim=True)
    p = ex / denom

    out = torch.matmul(p, vh)
    return out.permute(0, 2, 1, 3)

def gemma2_transformed_noexp2(q, k, v, scale8=8.0, tanh_scale=50.0):
    """
    去掉 mul+exp2 后的“变换版”（结构上对应 exp2 那套）：
      y = ...
      m = max(y)
      em = exp(m)
      denom = sum(exp(y)) / em          # 先 sum 再除（等价 sum(exp(y)/em)）
      p = exp(y) / (em * denom)
      out = p @ V
    """
    B, S, H, D = q.shape
    device, dtype = q.device, q.dtype

    qh = q.permute(0, 2, 1, 3)
    kh = k.permute(0, 2, 3, 1)
    vh = v.permute(0, 2, 1, 3)

    mask = causal_upper_mask(S, device, dtype).unsqueeze(0).unsqueeze(0)

    scores = torch.matmul(qh, kh) / scale8
    y = torch.tanh(scores / tanh_scale) * tanh_scale
    y = y + mask

    m = y.max(dim=-1, keepdim=True).values
    em = torch.exp(m)

    sum_ex = torch.exp(y).sum(dim=-1, keepdim=True)  # sum(exp(y))
    denom = sum_ex / em                               # 先 sum 再除
    p = torch.exp(y) / (em * denom)

    out = torch.matmul(p, vh)
    return out.permute(0, 2, 1, 3)

# ===== split kernels (对应 p77/p69 的切分点，但已去掉 exp2) =====

def gemma2_p77_noexp2(q, k, scale8=8.0, tanh_scale=50.0):
    """
    类比 Gemma2_p77：
      y = ...
      m = max(y)
      em = exp(m)
      scores = (Q / 8) @ K
      denom = sum(exp(y)) / exp(m)      # 先 sum 再除
    return (em, denom)
    """
    B, S, H, D = q.shape
    device, dtype = q.device, q.dtype

    qh = q.permute(0, 2, 1, 3)
    kh = k.permute(0, 2, 3, 1)

    mask = causal_upper_mask(S, device, dtype).unsqueeze(0).unsqueeze(0)

    scores = torch.matmul(qh / scale8, kh)
    y = torch.tanh(scores / tanh_scale) * tanh_scale
    y = y + mask

    m = y.max(dim=-1, keepdim=True).values
    em = torch.exp(m)
    sum_ex = torch.exp(y).sum(dim=-1, keepdim=True)
    denom = sum_ex / em
    return em, denom

def gemma2_p69_noexp2(q, v, k, em, denom, scale8=8.0, tanh_scale=50.0):
    """
    类比 Gemma2_p69（recompute y）：
      scores = (Q / 8) @ K
      out = (exp(y) @ V) / (em * denom)
    return (em, out)
    """
    B, S, H, D = q.shape
    device, dtype = q.device, q.dtype

    qh = q.permute(0, 2, 1, 3)
    kh = k.permute(0, 2, 3, 1)
    vh = v.permute(0, 2, 1, 3)

    mask = causal_upper_mask(S, device, dtype).unsqueeze(0).unsqueeze(0)

    scores = torch.matmul(qh / scale8, kh)
    y = torch.tanh(scores / tanh_scale) * tanh_scale
    y = y + mask

    out = (torch.matmul(torch.exp(y), vh) / (em * denom)).permute(0, 2, 1, 3)
    return em, out

def report(tag, a, b):
    diff = (a - b).abs()
    max_abs = diff.max().item()
    max_rel = (diff / (b.abs() + 1e-12)).max().item()
    print(f"{tag}: max_abs={max_abs:.6e}, max_rel={max_rel:.6e}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    B, S, H, D = 1, 4096, 32, 64
    torch.manual_seed(0)

    q = torch.randn(B, S, H, D, device=device, dtype=dtype)
    k = torch.randn(B, S, H, D, device=device, dtype=dtype)
    v = torch.randn(B, S, H, D, device=device, dtype=dtype)

    with torch.no_grad():
        y_before = gemma2_before_preprocess(q, k, v)
        y_trans  = gemma2_transformed_noexp2(q, k, v)

        em, denom = gemma2_p77_noexp2(q, k)
        em2, y_split = gemma2_p69_noexp2(q, v, k, em, denom)

    report("transformed_vs_before", y_trans, y_before)
    report("split_vs_transformed ", y_split, y_trans)
    report("split_vs_before      ", y_split, y_before)

    print("done.",
          "device", device,
          "q", tuple(q.shape),
          "scores", (B, H, S, S),
          "out", tuple(y_before.shape))

if __name__ == "__main__":
    main()