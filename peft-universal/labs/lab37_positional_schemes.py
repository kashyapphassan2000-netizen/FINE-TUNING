"""Lab 37: positional encoding schemes side by side, including what each does beyond training length."""
import math
import torch

torch.manual_seed(0)
T, D = 12, 32


def sinusoidal(T, D, base=10000.0):
    pos = torch.arange(T).unsqueeze(1)
    i = torch.arange(0, D, 2)
    ang = pos / base ** (i / D)
    pe = torch.zeros(T, D)
    pe[:, 0::2], pe[:, 1::2] = ang.sin(), ang.cos()
    return pe


def alibi_slopes(H):
    start = 2 ** (-8 / H)
    return torch.tensor([start ** (i + 1) for i in range(H)])


print("A. ABSOLUTE schemes add a vector to the token embedding at layer 0")
pe = sinusoidal(T, D)
print(f"   sinusoidal: fixed formula, no parameters, defined for ANY position (shape {tuple(pe.shape)})")
print(f"   learned:    nn.Embedding(max_positions, d_model) -> {T * D:,} trained numbers here;")
print("               undefined beyond max_positions, which is why GPT-2 style models hard-stop at 1024")
sim = torch.nn.functional.cosine_similarity(pe[0].unsqueeze(0), pe, dim=-1)
print(f"   similarity of position 0 to positions 0..5: {[round(float(s), 3) for s in sim[:6]]} (decays smoothly)")

print("\nB. RELATIVE / BIAS schemes act inside attention, at every layer")
H = 8
slopes = alibi_slopes(H)
dist = torch.arange(T).unsqueeze(0) - torch.arange(T).unsqueeze(1)
bias = slopes.view(H, 1, 1) * dist.clamp(max=0)
print(f"   ALiBi: score += -slope * distance. slopes for {H} heads: "
      f"{[round(float(s), 4) for s in slopes[:4]]} ...")
print(f"   head 0 bias at distances 0,1,4,16: "
      f"{[round(float(slopes[0] * -d), 3) for d in (0, 1, 4, 16)]} -> a soft recency prior, extrapolates by design")
print("   RoPE:  rotate Q and K by position; the score becomes a function of (i - j). See lab 36.")
print("   NoPE:  no positional signal at all; causal masking alone gives the model an order signal.")
print("          Works in some recent hybrid designs, usually mixed with other layers.")

print("\nC. Comparison")
rows = [
    ("sinusoidal absolute", "added at input", "no params", "poor beyond training length", "original Transformer"),
    ("learned absolute", "added at input", "max_pos x d params", "impossible beyond max_pos", "GPT-2, BERT"),
    ("ALiBi", "bias in attention", "no params", "good, by construction", "BLOOM, MPT"),
    ("RoPE", "rotation of Q,K", "no params", "needs scaling (yarn/llama3)", "Llama, Qwen, Mistral, DeepSeek"),
    ("RoPE + sliding window", "rotation + local mask", "no params", "local layers bounded", "Gemma-3, gpt-oss"),
    ("NoPE", "nothing", "no params", "architecture dependent", "some hybrid models"),
]
print(f"{'scheme':<24}{'where it acts':<20}{'params':<20}{'beyond train length':<30}seen in")
for r in rows:
    print(f"{r[0]:<24}{r[1]:<20}{r[2]:<20}{r[3]:<30}{r[4]}")

print("\nD. Why this matters for the KV cache")
print("   RoPE is applied to K BEFORE caching, so a cached key already carries its absolute position.")
print("   Changing rope settings after the cache was filled corrupts it - this is why prefix caches are")
print("   keyed by model + rope config, and why you cannot mix two rope scalings in one server.")
print("   ALiBi caches raw K (the bias is added at score time), and MLA must keep a separate small")
print("   'decoupled' RoPE part because rotation does not commute with the low-rank absorption (lab 34).")
