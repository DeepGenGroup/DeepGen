import torch

torch.set_printoptions(precision=6, sci_mode=False)

def attn_stablehlo_original(q, k, v, scale=8.0):
    qh = q.permute(0, 2, 1, 3)
    kh = k.permute(0, 2, 3, 1)
    vh = v.permute(0, 2, 1, 3)

    scores = torch.matmul(qh, kh) / scale
    m = scores.max(dim=-1, keepdim=True).values
    ex = torch.exp(scores - m)
    denom = ex.sum(dim=-1, keepdim=True)
    p = ex / denom

    out = torch.matmul(p, vh)
    return out.permute(0, 2, 1, 3)


def attn_stablehlo_transformed_noexp2(q, k, v, scale=8.0):
    qh = q.permute(0, 2, 1, 3)
    kh = k.permute(0, 2, 3, 1)
    vh = v.permute(0, 2, 1, 3)

    scores = torch.matmul(qh, kh) / scale
    m = scores.max(dim=-1, keepdim=True).values

    ex = torch.exp(scores)
    em = torch.exp(m)
    tmp = ex / em
    denom = tmp.sum(dim=-1, keepdim=True)
    p = tmp / denom

    out = torch.matmul(p, vh)
    return out.permute(0, 2, 1, 3)


def kernel1_em_denom_sum_then_div(q, k, scale=8.0):
    """
    kernel1:
      em    = exp(max(scores))
      denom = sum(exp(scores)) / em      # 先 sum 再除
    """
    qh = q.permute(0, 2, 1, 3)
    kh = k.permute(0, 2, 3, 1)
    scores = torch.matmul(qh, kh) / scale

    m = scores.max(dim=-1, keepdim=True).values
    em = torch.exp(m)
    sum_ex = torch.exp(scores).sum(dim=-1, keepdim=True)
    denom = sum_ex / em
    return em, denom


def kernel2_use_em_denom_sum_then_div(q, k, v, em, denom, scale=8.0):
    """
    kernel2:
      p = (exp(scores)/em) / denom
        = exp(scores) / (em * denom)
    """
    qh = q.permute(0, 2, 1, 3)
    kh = k.permute(0, 2, 3, 1)
    vh = v.permute(0, 2, 1, 3)

    scores = torch.matmul(qh, kh) / scale
    p = torch.exp(scores) / (em * denom)
    out = torch.matmul(p, vh)
    return out.permute(0, 2, 1, 3)


def report(tag, a, b):
    diff = (a - b).abs()
    max_abs = diff.max().item()
    max_rel = (diff / (b.abs() + 1e-12)).max().item()
    print(f"{tag}: max_abs={max_abs:.6e}, max_rel={max_rel:.6e}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    B, S, H, D = 1, 2048, 32, 64
    scale = 8.0

    torch.manual_seed(0)
    q = torch.randn(B, S, H, D, device=device, dtype=dtype)
    k = torch.randn(B, S, H, D, device=device, dtype=dtype)
    v = torch.randn(B, S, H, D, device=device, dtype=dtype)

    with torch.no_grad():
        y_orig = attn_stablehlo_original(q, k, v, scale=scale)
        y_trans = attn_stablehlo_transformed_noexp2(q, k, v, scale=scale)

        em, denom = kernel1_em_denom_sum_then_div(q, k, scale=scale)
        y_split = kernel2_use_em_denom_sum_then_div(q, k, v, em, denom, scale=scale)

    report("transformed_vs_original", y_trans, y_orig)
    report("split_vs_transformed    ", y_split, y_trans)
    report("split_vs_original       ", y_split, y_orig)

    print("done.",
          "q", tuple(q.shape),
          "scores", (B, H, S, S),
          "out", tuple(y_orig.shape),
          "device", device)

if __name__ == "__main__":
    main()