import torch

torch.set_printoptions(precision=6, sci_mode=False)

# ============================================================
# 1) 初始 IR（before preprocess, 稳定 exp(scores-max)）
# ============================================================
def h2o_ir_initial(q, k, v, scale8=8.0):
    qh = q.permute(0, 2, 1, 3)            # [B,H,S,D]
    kh = k.permute(0, 2, 3, 1)            # [B,H,D,S]
    vh = v.permute(0, 2, 1, 3)            # [B,H,S,D]

    scores = torch.matmul(qh, kh) / scale8             # [B,H,S,S]
    m = scores.max(dim=-1, keepdim=True).values        # [B,H,S,1]
    ex = torch.exp(scores - m)                         # [B,H,S,S]
    den = ex.sum(dim=-1, keepdim=True)                 # [B,H,S,1]
    p = ex / den                                       # [B,H,S,S]

    out = torch.matmul(p, vh).permute(0, 2, 1, 3)      # [B,S,H,D]
    row_sum = p.sum(dim=2, keepdim=False)              # [B,H,S]  (dim=2)
    return out, row_sum


# ============================================================
# 2) IR reorder 后（整体函数，但按 k1/k2/k3 的数据流重排）
#    k1: em=exp(max), w=exp(scores), den=sum(w)
#    k2: expr1=den/em, expr2=w/em, p=expr2/expr1, row_sum=sum(p, dim=2)
#    k3: out = p @ V
# ============================================================
def h2o_ir_reordered(q, k, v, scale8=8.0):
    qh = q.permute(0, 2, 1, 3)
    kh = k.permute(0, 2, 3, 1)
    vh = v.permute(0, 2, 1, 3)

    scores = torch.matmul(qh, kh) / scale8

    # === k1 equivalent ===
    m = scores.max(dim=-1, keepdim=True).values        # [B,H,S,1]
    em = torch.exp(m)                                  # exp(max)
    w = torch.exp(scores)
    den = w.sum(dim=-1, keepdim=True)                  # reduce_sum (softmax 分母)

    # === k2 equivalent: 按三式写法 ===
    expr1 = den / em
    expr2 = w / em
    p = expr2 / expr1
    row_sum = p.sum(dim=2, keepdim=False)              # [B,H,S]

    # === k3 equivalent: 第二个 dot ===
    out = torch.matmul(p, vh).permute(0, 2, 1, 3)      # [B,S,H,D]

    return out, row_sum


# ============================================================
# 3) Split kernels（按 reordered 的三式拆分）
# ============================================================
def h2o_k1(q, k, scale8=8.0):
    """
    Kernel1 返回：
      em    = exp(max(scores))            [B,H,S,1]
      expr1 = den / em                    [B,H,S,1]
      其中 den = reduce_sum(exp(scores), dim=-1)
    """
    qh = q.permute(0, 2, 1, 3)
    kh = k.permute(0, 2, 3, 1)
    scores = torch.matmul(qh / scale8, kh)

    m = scores.max(dim=-1, keepdim=True).values
    em = torch.exp(m)
    w = torch.exp(scores)
    den = w.sum(dim=-1, keepdim=True)
    expr1 = den / em
    return em, expr1


def h2o_k2(q, k, em, expr1, scale8=8.0):
    """
    Kernel2：使用 em 和 expr1，从头算到 row_sum
      expr2 = w / em
      p = expr2 / expr1
      row_sum = reduce_sum(p, dim=2, keep_dim=false)
    """
    qh = q.permute(0, 2, 1, 3)
    kh = k.permute(0, 2, 3, 1)
    scores = torch.matmul(qh / scale8, kh)

    # 这里 w = exp(scores)，所以 expr2 = w / em = exp(scores) / em
    expr2 = torch.exp(scores) / em
    p = expr2 / expr1
    row_sum = p.sum(dim=2, keepdim=False)   # [B,H,S]
    return row_sum


def h2o_k3(q, k, v, em, expr1, scale8=8.0):
    """
    Kernel3：使用 em 和 expr1，从头算到 out
      out = (w @ V) / (em * expr1)
    """
    qh = q.permute(0, 2, 1, 3)
    kh = k.permute(0, 2, 3, 1)
    vh = v.permute(0, 2, 1, 3)              # [B,H,S,D]
    scores = torch.matmul(qh / scale8, kh)

    out = (torch.matmul(torch.exp(scores), vh) / (em * expr1)).permute(0, 2, 1, 3)  # [B,S,H,D]
    return out


def h2o_split(q, k, v, scale8=8.0):
    em, expr1 = h2o_k1(q, k, scale8=scale8)
    row_sum = h2o_k2(q, k, em, expr1, scale8=scale8)
    out = h2o_k3(q, k, v, em, expr1, scale8=scale8)
    return out, row_sum


# ============================================================
# Validation
# ============================================================
def report(tag, a, b):
    diff = (a - b).abs()
    max_abs = diff.max().item()
    max_rel = (diff / (b.abs() + 1e-12)).max().item()
    print(f"{tag}: max_abs={max_abs:.6e}, max_rel={max_rel:.6e}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    B, S, H, D = 1, 4096, 32, 64
    scale8 = 8.0

    torch.manual_seed(0)
    q = torch.randn(B, S, H, D, device=device, dtype=dtype)
    k = torch.randn(B, S, H, D, device=device, dtype=dtype)
    v = torch.randn(B, S, H, D, device=device, dtype=dtype)

    with torch.no_grad():
        out_init, rs_init = h2o_ir_initial(q, k, v, scale8)
        out_reo,  rs_reo  = h2o_ir_reordered(q, k, v, scale8)
        out_sp,   rs_sp   = h2o_split(q, k, v, scale8)

    # out 对比
    report("reordered_out_vs_initial", out_reo, out_init)
    report("split_out_vs_initial   ", out_sp, out_init)
    report("split_out_vs_reordered ", out_sp, out_reo)

    # row_sum 对比
    report("reordered_rs_vs_initial", rs_reo, rs_init)
    report("split_rs_vs_initial    ", rs_sp, rs_init)
    report("split_rs_vs_reordered  ", rs_sp, rs_reo)

    print("done.",
          "device", device,
          "q", tuple(q.shape),
          "scores", (B, H, S, S),
          "out", tuple(out_init.shape),
          "row_sum", tuple(rs_init.shape))

if __name__ == "__main__":
    main()