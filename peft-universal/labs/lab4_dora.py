"""Lab 4: DoRA.  W' = m * (W0 + s*B A) / ||W0 + s*B A||_col     (m is a trainable per-column magnitude)"""
import torch

torch.manual_seed(0)
d_in, d_out, r, s = 64, 32, 4, 2.0
W0 = torch.randn(d_out, d_in)
m = torch.nn.Parameter(W0.norm(dim=0, keepdim=True).clone())   # column norms, shape (1, d_in)
A = torch.nn.Parameter(torch.randn(r, d_in) * 0.01)
B = torch.nn.Parameter(torch.zeros(d_out, r))


def dora_weight():
    V = W0 + s * B @ A
    return m * V / V.norm(dim=0, keepdim=True)   # PEFT detaches this norm in backward to save memory


print("1) identical to W0 at init:", torch.allclose(dora_weight(), W0, atol=1e-5))
print(f"2) trainable: lora={A.numel() + B.numel()}  +magnitude={m.numel()}  (DoRA adds d_in numbers per layer here)")

# target = rescale some columns only (pure magnitude change) -> DoRA can do this with m alone
scale = torch.ones(1, d_in)
scale[0, :8] = 1.5
target = W0 * scale


def run(use_dora: bool, steps: int = 400):
    torch.manual_seed(1)
    mm = torch.nn.Parameter(W0.norm(dim=0, keepdim=True).clone())
    AA = torch.nn.Parameter(torch.randn(r, d_in) * 0.01)
    BB = torch.nn.Parameter(torch.zeros(d_out, r))
    params = [AA, BB, mm] if use_dora else [AA, BB]
    opt = torch.optim.Adam(params, lr=1e-2)
    for _ in range(steps):
        V = W0 + s * BB @ AA
        W = mm * V / V.norm(dim=0, keepdim=True) if use_dora else V
        loss = ((W - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()


print(f"3) column-rescale task, rank {r}:  LoRA loss={run(False):.4f}   DoRA loss={run(True):.6f}")
print("   (8 rescaled columns need rank 8 for LoRA; DoRA's magnitude vector handles it directly)")
