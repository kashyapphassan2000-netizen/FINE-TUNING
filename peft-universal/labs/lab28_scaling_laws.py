"""Lab 28: the compute budget problem - scaling laws, and what they mean for CPT and fine-tuning.

Formulas (sources: Kaplan et al. 2020 for C = 6ND; Hoffmann et al. 2022 "Chinchilla" for the loss fit):
  training compute     C  ~= 6 x N x D            N = parameters, D = training tokens (forward 2ND + backward 4ND)
  Chinchilla loss fit  L(N, D) = E + A / N^alpha + B / D^beta
                       E = 1.69, A = 406.4, B = 410.7, alpha = 0.34, beta = 0.28  (their fitted constants)
  compute-optimal      D* ~= 20 x N tokens (rule of thumb from the same paper's approaches 1-2; see section 2)
  PEFT on a frozen base  C ~= 4ND (no weight gradients) or 6ND with gradient checkpointing (recompute)
Everything printed is arithmetic you can redo by hand.
"""
import math

E, A, B, ALPHA, BETA = 1.69, 406.4, 410.7, 0.34, 0.28


def loss(n, d):
    return E + A / n**ALPHA + B / d**BETA


def gpu_hours(flops, peak_flops=989e12, mfu=0.40):
    """H100 SXM dense bf16 peak ~989 TFLOP/s; 35-45% MFU is a realistic training utilisation."""
    return flops / (peak_flops * mfu) / 3600


print("1) PRE-TRAINING: why nobody pre-trains for fun")
print(f"{'model':>8}{'tokens':>10}{'FLOPs (6ND)':>14}{'H100-hours @40% MFU':>22}{'Chinchilla loss':>17}")
for n, d in [(1e9, 20e9), (8e9, 15e12), (70e9, 15e12), (405e9, 15.6e12)]:
    c = 6 * n * d
    print(f"{n / 1e9:>7.0f}B{d / 1e12:>9.2f}T{c:>14.2e}{gpu_hours(c):>22,.0f}{loss(n, d):>17.3f}")
c8 = gpu_hours(6 * 8e9 * 15e12)
print(f"  -> an 8B model on 15T tokens is ~{c8 / 1e3:,.0f}k H100-hours: at $2-3/h that is "
      f"${c8 * 2 / 1e6:.1f}-{c8 * 3 / 1e6:.1f}M of GPU time, before failed runs and ablations.")

print("\n2) COMPUTE-OPTIMAL split for a fixed budget (minimise L(N,D) subject to 6ND = C)")
for c in (1e21, 1e22, 1e23, 1e24):
    best = min(((loss(n, c / (6 * n)), n) for n in [10 ** (x / 20) for x in range(160, 240)]))
    n = best[1]; d = c / (6 * n)
    print(f"  C={c:.0e}: N*={n / 1e9:6.2f}B  D*={d / 1e9:8.1f}B tokens  D/N={d / n:5.1f}  loss={best[0]:.3f}")
print("  -> HONEST NOTE: this parametric fit (Chinchilla 'approach 3') gives D/N of ~50-100. The famous")
print("     'D ~ 20 x N' comes from the paper's approaches 1 and 2; a 2024 replication (Besiroglu et al.) found")
print("     the published approach-3 constants inconsistent and, refitted, close to ~20. Use 20x as the rule of thumb.")
print("  -> Modern models deliberately OVER-train small models (Llama-3 8B: D/N ~ 1,900) because serving cost")
print("     depends on N, not on D: a smaller model trained longer is cheaper to SERVE to millions of users.")

print("\n3) CONTINUED PRE-TRAINING (CPT) budget: the same formula, much smaller D")
for n, d in [(8e9, 1e9), (8e9, 10e9), (70e9, 10e9)]:
    full = 6 * n * d
    print(f"  CPT {n / 1e9:.0f}B on {d / 1e9:.0f}B domain tokens: full-FT {full:.2e} FLOPs = "
          f"{gpu_hours(full):,.0f} H100-h ; LoRA (frozen base, ckpt) ~same FLOPs, far less MEMORY")
print("  -> CPT FLOPs barely drop with LoRA (the forward/backward through the frozen base still runs);")
print("     LoRA saves memory (optimizer state, gradients), not compute. Plan CPT by tokens x params.")

print("\n4) SFT budget: why SFT is cheap")
n, rows, toks = 8e9, 20_000, 600
d = rows * toks * 3          # 3 epochs
c = 6 * n * d
print(f"  {rows:,} examples x {toks} tokens x 3 epochs = {d / 1e6:.0f}M tokens -> {c:.2e} FLOPs "
      f"= {gpu_hours(c):.1f} H100-hours at 40% MFU (real runs: 1.5-3x that with padding/eval/IO)")

print("\n5) DATA-LIMITED regime: what more data buys you (fixed 8B model)")
for d in (1e9, 1e10, 1e11, 1e12, 1e13):
    print(f"  D={d:.0e}: Chinchilla loss {loss(8e9, d):.3f}")
print("""
READING
- Pre-training a useful base model costs 10^5-10^7 GPU-hours. You will almost always START from one.
- CPT is pre-training at small D: worth it only when you have 100M-10B+ tokens of domain text the base
  has not seen (new language, proprietary code, specialised notation). Otherwise go straight to SFT.
- SFT is 10^0-10^2 GPU-hours: data quality, not compute, is the binding constraint.
- The diminishing return in (5) is why '10x more synthetic data' rarely helps: the B/D^beta term shrinks
  slowly, and synthetic data that does not add NEW information does not even buy that.""")
