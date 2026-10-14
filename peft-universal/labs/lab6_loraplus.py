"""Lab 6: LoRA+. Same optimizer, but lr_B = ratio * lr_A. Compare loss after a fixed number of steps."""
import torch

d, r, steps = 256, 8, 300
torch.manual_seed(0)
W0 = torch.randn(d, d) / d**0.5
target = W0 + (torch.randn(d, r) @ torch.randn(r, d)) * 0.02


def run(ratio: float, lr: float = 2e-3) -> float:
    torch.manual_seed(1)
    A = torch.nn.Parameter(torch.randn(r, d) / d**0.5)
    B = torch.nn.Parameter(torch.zeros(d, r))
    opt = torch.optim.AdamW([{"params": [A], "lr": lr}, {"params": [B], "lr": lr * ratio}], weight_decay=0.0)
    for _ in range(steps):
        x = torch.randn(128, d)
        loss = ((x @ (W0 + B @ A).T - x @ target.T) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()


for ratio in (1, 4, 16, 64):
    print(f"lr_B/lr_A = {ratio:>3}: loss after {steps} steps = {run(ratio):.3e}")
print("Too large a ratio can destabilise training; the LoRA+ paper suggests ~16 as a starting point.")
