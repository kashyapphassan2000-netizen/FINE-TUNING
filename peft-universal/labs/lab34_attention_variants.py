"""Lab 34: MHA -> MQA -> GQA -> MLA. Same interface, very different KV cost. All four implemented and run."""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, T, H, D = 1, 32, 16, 64          # 16 query heads, head_dim 64 -> model width 1024
HID = H * D
x = torch.randn(B, T, HID)


def attend(q, k, v):
    kk = k.repeat_interleave(q.shape[1] // k.shape[1], 1)
    vv = v.repeat_interleave(q.shape[1] // v.shape[1], 1)
    return F.scaled_dot_product_attention(q, kk, vv, is_causal=True)


class Variant(torch.nn.Module):
    """kv_heads = H -> MHA, 1 -> MQA, anything between -> GQA."""

    def __init__(self, kv_heads):
        super().__init__()
        self.kv = kv_heads
        self.q = torch.nn.Linear(HID, H * D, bias=False)
        self.k = torch.nn.Linear(HID, kv_heads * D, bias=False)
        self.v = torch.nn.Linear(HID, kv_heads * D, bias=False)
        self.o = torch.nn.Linear(H * D, HID, bias=False)

    def forward(self, x):
        b, t, _ = x.shape
        q = self.q(x).view(b, t, H, D).transpose(1, 2)
        k = self.k(x).view(b, t, self.kv, D).transpose(1, 2)
        v = self.v(x).view(b, t, self.kv, D).transpose(1, 2)
        self.cache_bytes = (k.numel() + v.numel()) * 2 // t      # per token, bf16
        return self.o(attend(q, k, v).transpose(1, 2).reshape(b, t, H * D))


class MLA(torch.nn.Module):
    """DeepSeek-style: cache ONE low-rank latent per token instead of K and V per head."""

    def __init__(self, rank=128, rope_dim=16):
        super().__init__()
        self.rank, self.rope_dim = rank, rope_dim
        self.kv_a = torch.nn.Linear(HID, rank + rope_dim, bias=False)   # down-project: this is what is cached
        self.kv_b = torch.nn.Linear(rank, H * 2 * D, bias=False)        # up-project to per-head K and V
        self.q = torch.nn.Linear(HID, H * D, bias=False)
        self.o = torch.nn.Linear(H * D, HID, bias=False)

    def forward(self, x):
        b, t, _ = x.shape
        latent = self.kv_a(x)                                   # (b, t, rank + rope_dim)  <-- the cache
        self.cache_bytes = latent.numel() * 2 // t
        c, _rope = latent[..., :self.rank], latent[..., self.rank:]
        kv = self.kv_b(c).view(b, t, H, 2 * D).transpose(1, 2)
        k, v = kv[..., :D], kv[..., D:]
        q = self.q(x).view(b, t, H, D).transpose(1, 2)
        return self.o(attend(q, k, v).transpose(1, 2).reshape(b, t, H * D))


print(f"model width {HID}, {H} query heads, head_dim {D}, bf16 KV cache\n")
print(f"{'variant':<28}{'kv heads':>9}{'attn params':>13}{'KV bytes/token':>16}{'vs MHA':>9}  output")
base = None
for name, m in [("MHA  (kv_heads = 16)", Variant(16)), ("GQA  (kv_heads = 4)", Variant(4)),
                ("MQA  (kv_heads = 1)", Variant(1)), ("MLA  (latent rank 128)", MLA())]:
    y = m(x)
    p = sum(w.numel() for w in m.parameters())
    base = base or m.cache_bytes
    kvh = getattr(m, "kv", "-")
    print(f"{name:<28}{str(kvh):>9}{p:>13,}{m.cache_bytes:>16,}{base / m.cache_bytes:>8.1f}x  {tuple(y.shape)}")

print("\nWhat you trade:")
print("  MHA  every head has its own K and V. Best expressiveness, biggest cache. Used by GPT-2 era models.")
print("  MQA  one shared K/V for all heads. Smallest cache, some quality loss reported in the literature.")
print("  GQA  groups of query heads share a K/V. The industry default (Llama 3, Qwen, Mistral): near-MHA")
print("       quality at MQA-like cache cost. kv_heads is usually 4-8.")
print("  MLA  cache a compressed latent, rebuild per-head K/V on the fly. Smallest cache per layer at high")
print("       head counts (DeepSeek-V3: 128 heads but only 576 numbers cached per token per layer).")
print("       Cost: extra matmuls at decode, and a more complex kernel. RoPE needs a separate small")
print("       'decoupled' part because rotation does not commute with the low-rank absorption.")
print("\nPractical rule: you cannot change this after pretraining. It is a model-architecture decision that")
print("fixes your serving economics. If you are choosing a base model for long context, read kv_heads first.")
