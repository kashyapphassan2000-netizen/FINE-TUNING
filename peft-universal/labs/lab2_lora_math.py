"""Lab 2: LoRA from scratch.  h = W0 x + (alpha/r) * B A x"""
import torch

torch.manual_seed(0)
d_in, d_out, r, alpha = 512, 512, 8, 16


class LoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, r: int, alpha: float, rslora: bool = False):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = torch.nn.Parameter(torch.empty(r, base.in_features))
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, r))  # zero => starts identical to base
        torch.nn.init.kaiming_uniform_(self.A, a=5**0.5)  # same init family PEFT uses
        self.scale = alpha / (r**0.5 if rslora else r)

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)

    def merged_weight(self):
        return self.base.weight + self.scale * self.B @ self.A


base = torch.nn.Linear(d_in, d_out, bias=False)
lora = LoRALinear(base, r, alpha)
x = torch.randn(4, d_in)
print("1) output identical at init:", torch.allclose(base(x), lora(x)))
print(f"2) trainable params: full={d_in * d_out:,}  lora={r * (d_in + d_out):,}  ({100 * r * (d_in + d_out) / (d_in * d_out):.2f}%)")

# a target change that is truly rank 4
U, V = torch.randn(d_out, 4), torch.randn(4, d_in)
W_target = base.weight.detach() + 0.05 * U @ V
opt = torch.optim.Adam([lora.A, lora.B], lr=1e-3)
for step in range(1500):
    xb = torch.randn(64, d_in)
    loss = ((lora(xb) - xb @ W_target.T) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
print(f"3) final MSE after 1500 steps: {loss.item():.2e}")
delta = (lora.merged_weight() - base.weight).detach()
s = torch.linalg.svdvals(delta)
print("4) singular values of learned delta (top 10):", [round(v, 3) for v in s[:10].tolist()])
print("   rank(delta) <= r by construction:", int((s > 1e-4 * s[0]).sum()), "<=", r)
xb = torch.randn(3, d_in)
print("5) merge is exact:", torch.allclose(lora(xb), xb @ lora.merged_weight().T, atol=1e-5))
for rr in (4, 16, 64, 256):
    print(f"6) r={rr:>3}: classic scale alpha/r={alpha / rr:.3f}   rsLoRA scale alpha/sqrt(r)={alpha / rr**0.5:.3f}")
