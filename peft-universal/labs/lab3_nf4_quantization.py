"""Lab 3: NF4 (4-bit NormalFloat) from the QLoRA paper, checked against bitsandbytes."""
import torch

N = torch.distributions.Normal(0.0, 1.0)


def nf4_codebook(offset: float = 0.9677083) -> torch.Tensor:
    pos = N.icdf(torch.linspace(offset, 0.5, 9)[:-1])      # 8 positive levels
    neg = -N.icdf(torch.linspace(offset, 0.5, 8)[:-1])     # 7 negative levels
    v = torch.cat([pos, torch.zeros(1), neg]).sort().values
    return v / v.abs().max()


ours = nf4_codebook()
print("our NF4 codebook :", [round(x, 4) for x in ours.tolist()])
try:
    import bitsandbytes.functional as F

    ref = F.create_normal_map()
    ref_vals = torch.tensor(sorted({round(float(v), 6) for v in ref}))
    print("bitsandbytes NF4 :", [round(x, 4) for x in ref_vals.tolist()])
    print("match (atol 1e-4):", torch.allclose(ours, ref_vals.float(), atol=1e-4))
except Exception as e:  # bitsandbytes not installed
    print("bitsandbytes not available:", e)


def quantize(w: torch.Tensor, codebook: torch.Tensor, block: int = 64):
    w = w.flatten()
    pad = (-len(w)) % block
    wb = torch.nn.functional.pad(w, (0, pad)).view(-1, block)
    absmax = wb.abs().max(1, keepdim=True).values.clamp_min(1e-12)   # one scale per block
    idx = ((wb / absmax).unsqueeze(-1) - codebook).abs().argmin(-1)     # 4-bit index per weight
    deq = codebook[idx] * absmax
    return deq.flatten()[: len(w)], absmax


torch.manual_seed(0)
w = torch.randn(4096 * 64) * 0.02            # pretrained weights are roughly normal
uniform4 = torch.linspace(-1, 1, 16)          # naive 4-bit grid
for name, cb in (("NF4", ours), ("uniform int4", uniform4)):
    deq, _ = quantize(w, cb)
    print(f"{name:<13} mean abs error = {(deq - w).abs().mean():.3e}")

bits_const = 32 / 64
bits_dq = 8 / 64 + 32 / (64 * 256)
print(f"\nquantization constants: {bits_const:.3f} bit/param without double quant, {bits_dq:.3f} with it")
print(f"=> saving {bits_const - bits_dq:.3f} bit/param; for 65B params = {(bits_const - bits_dq) * 65e9 / 8 / 1024**3:.2f} GiB")
