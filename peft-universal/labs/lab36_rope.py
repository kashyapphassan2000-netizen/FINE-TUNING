"""Lab 36: RoPE - the relative-distance property, and what long-context scaling actually changes."""
import math
import torch
from transformers import LlamaConfig
from transformers.models.llama import modeling_llama as ml

torch.manual_seed(0)
D, T = 64, 16
cfg = LlamaConfig(hidden_size=D * 2, num_attention_heads=2, num_key_value_heads=2, head_dim=D,
                  num_hidden_layers=1, intermediate_size=8, vocab_size=8)
rope = ml.LlamaRotaryEmbedding(config=cfg)
pos = torch.arange(T).unsqueeze(0)
q = torch.randn(1, 1, T, D)
k = torch.randn(1, 1, T, D)
cos, sin = rope(q, pos)
qr, kr = ml.apply_rotary_pos_emb(q, k, cos, sin)

print("1. THE property: after rotation, the score between position i and j depends only on (i - j).")
same_q = q[0, 0, 0].repeat(T, 1).unsqueeze(0).unsqueeze(0)
same_k = k[0, 0, 0].repeat(T, 1).unsqueeze(0).unsqueeze(0)
qq, kk = ml.apply_rotary_pos_emb(same_q, same_k, cos, sin)
pairs = [(2, 0), (5, 3), (9, 7), (15, 13)]      # all distance 2
scores = [float(qq[0, 0, i] @ kk[0, 0, j]) for i, j in pairs]
print(f"   same q,k vectors at distance 2, positions {pairs}")
print(f"   scores: {[round(s, 4) for s in scores]}  -> all equal: "
      f"{max(scores) - min(scores) < 1e-3}")

print("\n2. How it works: split the head dim into pairs and rotate pair m by angle pos * theta_m")
base = 10000.0
freqs = [1.0 / (base ** (2 * i / D)) for i in range(D // 2)]
print(f"   theta_m = base^(-2m/d), base={base:g}, d={D}")
print(f"   fastest pair rotates {freqs[0]:.4f} rad/token (period {2 * math.pi / freqs[0]:.0f} tokens)")
print(f"   slowest pair rotates {freqs[-1]:.2e} rad/token (period {2 * math.pi / freqs[-1]:,.0f} tokens)")
print("   low-frequency pairs carry long-range position, high-frequency pairs local order.")

print("\n3. Why context extension needs care: at positions beyond training length, the slow pairs enter")
print("   angles the model never saw. Three standard fixes (all present in transformers 5.17):")
for name, formula in [("linear (position interpolation)", "pos -> pos / factor : squeezes all frequencies"),
                      ("dynamic NTK", "raise base with the sequence length"),
                      ("yarn", "interpolate low frequencies, keep high ones, plus an attention temperature"),
                      ("llama3", "piecewise: keep high freqs, interpolate low freqs, smooth in between"),
                      ("longrope", "per-dimension learned scaling factors")]:
    print(f"     {name:<32} {formula}")
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
print(f"   available rope_type values: {sorted(ROPE_INIT_FUNCTIONS.keys())}")

print("\n4. Measured effect of linear interpolation (factor 4) on the rotation angle at position 1000:")
for factor in (1, 4):
    ang = 1000 / factor * freqs[len(freqs) // 2]
    print(f"   factor={factor}: middle-frequency angle = {ang:8.2f} rad "
          f"({'inside' if factor > 1 else 'far outside'} the range seen during training)")
print("\n5. Config in transformers 5.17 lives under `rope_parameters`, e.g. Llama default:")
print(f"   {LlamaConfig().to_dict()['rope_parameters']}")
print("   gpt-oss ships yarn with factor 32.0 and original_max_position_embeddings 4096.")
print("   Gemma-3 uses TWO rope settings: theta 10,000 on sliding layers and 1,000,000 on full layers.")
print("\nPractical: extending context is a config + fine-tune job, not a flag you flip at serving time.")
print("Always re-evaluate long-context quality after changing rope scaling (needle tests, not perplexity alone).")
