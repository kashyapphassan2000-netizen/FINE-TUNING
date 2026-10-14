"""Lab 33: torch scaled_dot_product_attention - one API, several kernels, exact semantics."""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

torch.manual_seed(0)
B, H, KV, T, D = 1, 8, 2, 16, 32
q = torch.randn(B, H, T, D)
k = torch.randn(B, KV, T, D)
v = torch.randn(B, KV, T, D)


def manual(q, k, v, causal=True):
    kk = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    vv = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    s = (q @ kk.transpose(-1, -2)) / q.shape[-1] ** 0.5
    if causal:
        s = s + torch.triu(torch.full((q.shape[-2], kk.shape[-2]), float("-inf")), diagonal=1)
    return s.softmax(-1) @ vv


sdpa_gqa = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
print("1. enable_gqa=True does the KV-head repeat for you:", torch.allclose(sdpa_gqa, manual(q, k, v), atol=1e-5))
print("   signature: scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,")
print("                                           is_causal=False, scale=None, enable_gqa=False)")

kk, vv = k.repeat_interleave(4, 1), v.repeat_interleave(4, 1)
mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
print("2. is_causal=True equals passing that mask explicitly:",
      torch.allclose(F.scaled_dot_product_attention(q, kk, vv, is_causal=True),
                     F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask), atol=1e-6))
print("   (is_causal is cheaper: the kernel skips the masked blocks instead of computing them)")

print("3. custom scale:", torch.allclose(
    F.scaled_dot_product_attention(q, kk, vv, scale=0.1),
    ((q @ kk.transpose(-1, -2)) * 0.1).softmax(-1) @ vv, atol=1e-5))

print("4. backends in this build:", [b for b in dir(SDPBackend) if b.isupper() and b != "ERROR"])
with sdpa_kernel(SDPBackend.MATH):
    math_out = F.scaled_dot_product_attention(q, kk, vv, is_causal=True)
print("   forcing the MATH backend gives the same numbers:", torch.allclose(math_out, sdpa_gqa, atol=1e-5))
print("   FLASH_ATTENTION / EFFICIENT_ATTENTION / CUDNN_ATTENTION are CUDA-only and are picked")
print("   automatically when dtype, head_dim, masking and alignment allow it.")

print("\n5. a decode step: one query token against the whole cached K/V")
q1 = torch.randn(B, H, 1, D)
kc, vc = torch.randn(B, KV, 513, D), torch.randn(B, KV, 513, D)
step = F.scaled_dot_product_attention(q1, kc, vc, enable_gqa=True)   # no mask needed: q attends to all past
print("   shapes:", tuple(q1.shape), "x", tuple(kc.shape), "->", tuple(step.shape),
      "| is_causal must be False here (a single query may see every cached key)")
print("\nIn transformers: model.config._attn_implementation in", ["eager", "sdpa", "flash_attention_2/3/4", "flex_attention"])
print("eager materialises the matrix (debuggable, memory-hungry); sdpa is the safe default.")
