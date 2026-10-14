"""Lab 32: why FlashAttention exists - online (streaming) softmax, tiled, never materialising N x N."""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
N, D = 512, 64
q = torch.randn(N, D)
k = torch.randn(N, D)
v = torch.randn(N, D)
scale = D**-0.5

# 1. standard attention: materialise the full N x N score matrix
scores = (q @ k.T) * scale
ref = scores.softmax(-1) @ v
print(f"standard attention materialises a {N}x{N} score matrix = "
      f"{scores.numel() * scores.element_size() / 1024:.0f} KB for one head at N={N}")
print(f"at N=32,768 that same matrix would be {32768 ** 2 * 2 / 1024 ** 3:.1f} GB per head in bf16 "
      "-> this is what runs you out of memory, not the weights")

# 2. online softmax: one pass, constant extra memory, exact result
def flash_like(q, k, v, block=64):
    out = torch.zeros_like(q)
    m = torch.full((q.shape[0], 1), float("-inf"))   # running max
    l = torch.zeros(q.shape[0], 1)                   # running sum of exp
    for start in range(0, k.shape[0], block):        # stream over K/V blocks
        kb, vb = k[start:start + block], v[start:start + block]
        s = (q @ kb.T) * scale
        m_new = torch.maximum(m, s.max(-1, keepdim=True).values)
        correction = torch.exp(m - m_new)
        p = torch.exp(s - m_new)
        l = l * correction + p.sum(-1, keepdim=True)
        out = out * correction + p @ vb              # rescale what we already have, then add
        m = m_new
    return out / l


flash = flash_like(q, k, v)
print(f"\nonline-softmax result matches standard attention: {torch.allclose(flash, ref, atol=1e-5)} "
      f"(max diff {(flash - ref).abs().max():.2e})")
print("peak score memory in the streaming version: one block only ->",
      f"{64 * N * 4 / 1024:.0f} KB instead of {scores.numel() * 4 / 1024:.0f} KB (and it is O(1) in N per block)")

print("\nThe maths that makes it exact: softmax can be rescaled after the fact.")
print("  keep running max m and sum l; when a block raises the max, multiply the old partial")
print("  output and l by exp(m_old - m_new). Nothing is approximated.")
print("\nWhy it is faster on a GPU: attention is memory-bandwidth bound, not FLOP bound. Writing and")
print("re-reading the N x N matrix to HBM is the bottleneck; keeping tiles in SRAM removes it.")
print("FlashAttention-2/3 add better work partitioning and (v3) Hopper-specific async copies.")
print("You rarely write this yourself: torch SDPA and the model's attn_implementation call it for you.")
