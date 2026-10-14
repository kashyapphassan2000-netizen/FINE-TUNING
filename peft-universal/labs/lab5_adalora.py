"""Lab 5: AdaLoRA ideas. dW = P diag(E) Q, importance-based pruning of E under a shrinking budget."""
import torch

torch.manual_seed(0)
layers, d, init_r, target_r = 4, 48, 8, 3
true_ranks = [6, 1, 3, 2]                                    # layers need different ranks (avg = 3)
targets = [torch.randn(d, k) @ torch.randn(k, d) * 0.3 for k in true_ranks]
P = [torch.nn.Parameter(torch.randn(d, init_r) * 0.1) for _ in range(layers)]
Q = [torch.nn.Parameter(torch.randn(init_r, d) * 0.1) for _ in range(layers)]
E = [torch.nn.Parameter(torch.zeros(init_r)) for _ in range(layers)]
mask = [torch.ones(init_r) for _ in range(layers)]
ema_s = [torch.zeros(init_r) for _ in range(layers)]
ema_u = [torch.zeros(init_r) for _ in range(layers)]
b1 = b2 = 0.85
T, tinit, tfinal = 1500, 150, 300
opt = torch.optim.Adam(P + Q + E, lr=5e-3)


def budget(t):  # cubic schedule from init to target, as in the paper
    total0, total1 = layers * init_r, layers * target_r
    if t < tinit:
        return total0
    if t > T - tfinal:
        return total1
    frac = 1 - (t - tinit) / (T - tinit - tfinal)
    return int(total1 + (total0 - total1) * frac**3)


for t in range(T):
    loss = sum(((P[i] @ torch.diag(E[i] * mask[i]) @ Q[i] - targets[i]) ** 2).mean() for i in range(layers))
    orth = sum((P[i].T @ P[i] - torch.eye(init_r)).norm() + (Q[i] @ Q[i].T - torch.eye(init_r)).norm() for i in range(layers))
    opt.zero_grad()
    (loss + 0.1 * orth).backward()
    opt.step()
    with torch.no_grad():
        scores = []
        for i in range(layers):
            s = (E[i] * E[i].grad).abs()                       # sensitivity |w * dL/dw|
            u = (s - ema_s[i]).abs()                           # uncertainty
            ema_s[i] = b1 * ema_s[i] + (1 - b1) * s
            ema_u[i] = b2 * ema_u[i] + (1 - b2) * u
            scores.append(ema_s[i] * ema_u[i])
        k = budget(t)
        if tinit <= t:
            flat = torch.cat(scores)
            keep = torch.zeros_like(flat)
            keep[flat.topk(k).indices] = 1
            for i in range(layers):
                mask[i] = keep[i * init_r:(i + 1) * init_r]
    if t % 300 == 0 or t == T - 1:
        print(f"step {t:>4} budget={budget(t):>2} loss={loss.item():.4f} ranks={[int(m.sum()) for m in mask]}")
print("true ranks :", true_ranks)
print("final ranks:", [int(m.sum()) for m in mask], "(same total budget, redistributed by importance)")
