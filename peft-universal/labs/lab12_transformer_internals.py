"""Lab 12: transformer internals from scratch, each piece checked against the real transformers code."""
import torch
import torch.nn.functional as F
from transformers import LlamaConfig
from transformers.models.llama import modeling_llama as ml

torch.manual_seed(0)
B, T, H, KV, D = 1, 6, 4, 2, 8          # batch, tokens, query heads, kv heads, head_dim
hidden = H * D
x = torch.randn(B, T, hidden)

# 1. RMSNorm
def my_rmsnorm(v, weight, eps=1e-6):
    return weight * v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)

ref_norm = ml.LlamaRMSNorm(hidden)
with torch.no_grad():
    ref_norm.weight.normal_()
print("1. RMSNorm matches transformers:", torch.allclose(my_rmsnorm(x, ref_norm.weight), ref_norm(x), atol=1e-6))
print("   RMSNorm divides by the root-mean-square only: no mean subtraction, so it is cheaper than LayerNorm.")

# 2. RoPE: rotate q and k by an angle that depends on the position
cfg = LlamaConfig(hidden_size=hidden, num_attention_heads=H, num_key_value_heads=KV, head_dim=D,
                  num_hidden_layers=1, intermediate_size=16, vocab_size=32)
rope = ml.LlamaRotaryEmbedding(config=cfg)
pos = torch.arange(T).unsqueeze(0)
q = torch.randn(B, H, T, D)
k = torch.randn(B, KV, T, D)
cos, sin = rope(q, pos)
q_rot, k_rot = ml.apply_rotary_pos_emb(q, k, cos, sin)


def my_rope(t, cos, sin):
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    half = t.shape[-1] // 2
    rotated = torch.cat((-t[..., half:], t[..., :half]), dim=-1)
    return t * c + rotated * s


print("2. RoPE matches transformers:", torch.allclose(my_rope(q, cos, sin), q_rot, atol=1e-6))
d0 = (q_rot[0, 0, 0] @ k_rot[0, 0, 0]) / (q[0, 0, 0] @ k[0, 0, 0])
print(f"   RoPE changes the q.k score with distance: ratio at distance 0 = {d0:.3f} (unchanged at the same position)")

# 3. GQA: KV heads are shared by several query heads
def repeat_kv(t, n_rep):
    return t.repeat_interleave(n_rep, dim=1)


scores = (q_rot @ repeat_kv(k_rot, H // KV).transpose(-1, -2)) / D**0.5
causal = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
attn = (scores + causal).softmax(-1)
v = torch.randn(B, KV, T, D)
mine = attn @ repeat_kv(v, H // KV)
ref = F.scaled_dot_product_attention(q_rot, repeat_kv(k_rot, H // KV), repeat_kv(v, H // KV), is_causal=True)
print("3. GQA attention matches torch SDPA:", torch.allclose(mine, ref, atol=1e-5))
print(f"   {H} query heads share {KV} KV heads -> KV cache is {H / KV:.0f}x smaller than full MHA")
print("   causal mask: row 0 attends to 1 token, row 5 to 6:",
      [int((attn[0, 0, i] > 0).sum()) for i in range(T)])

# 4. SwiGLU MLP
mlp = ml.LlamaMLP(cfg)
my_mlp = mlp.down_proj(F.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
print("4. SwiGLU = down(silu(gate(x)) * up(x)) matches LlamaMLP:", torch.allclose(my_mlp, mlp(x), atol=1e-6))

# 5. KV cache: generating with a cache equals recomputing the whole sequence
model = ml.LlamaForCausalLM(cfg).eval()
ids = torch.randint(0, 32, (1, 5))
with torch.no_grad():
    full = model(ids).logits[:, -1]
    cached = model(ids[:, :-1], use_cache=True)
    step = model(ids[:, -1:], past_key_values=cached.past_key_values).logits[:, -1]
print("5. KV cache gives the same next-token logits:", torch.allclose(full, step, atol=1e-4))
print("   without a cache, generating token n redoes the work for all n-1 previous tokens.")
print("\nblock order: x -> RMSNorm -> attention(Q,K,V + RoPE) -> +residual -> RMSNorm -> SwiGLU MLP -> +residual")
