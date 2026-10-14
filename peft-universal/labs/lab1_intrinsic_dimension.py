"""Lab 1: a model can be trained inside a tiny random subspace.  theta = theta0 + P z,  only z (d numbers) trains."""
import torch

torch.manual_seed(0)
X = torch.randn(2000, 20)
y = ((X[:, :5].sum(1) > 0) ^ (X[:, 5] > 0.5)).long()


def make():
    return torch.nn.Sequential(torch.nn.Linear(20, 64), torch.nn.ReLU(), torch.nn.Linear(64, 2))


def flat(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def train_subspace(d: int | None, steps: int = 600) -> float:
    torch.manual_seed(1)
    model = make()
    theta0 = flat(model)
    D = theta0.numel()
    shapes = [p.shape for p in model.parameters()]
    if d is None:  # full fine-tuning
        params = [torch.nn.Parameter(theta0.clone())]
        P = None
    else:
        P = torch.randn(D, d) / d**0.5  # random projection, fixed
        params = [torch.nn.Parameter(torch.zeros(d))]
    opt = torch.optim.Adam(params, lr=1e-2)
    for _ in range(steps):
        theta = params[0] if P is None else theta0 + P @ params[0]
        out, i = X, 0
        ws = []
        for s in shapes:
            n = torch.Size(s).numel()
            ws.append(theta[i:i + n].view(s))
            i += n
        h = torch.relu(out @ ws[0].T + ws[1])
        logits = h @ ws[2].T + ws[3]
        loss = torch.nn.functional.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return (logits.argmax(1) == y).float().mean().item(), D


full, D = train_subspace(None)
print(f"total parameters D = {D}")
print(f"full training accuracy       = {full:.3f}")
for d in (2, 5, 10, 25, 50, 100, 200):
    acc, _ = train_subspace(d)
    flag = "  <- reaches 90% of full" if acc >= 0.9 * full else ""
    print(f"d = {d:>4}  ({100 * d / D:5.2f}% of D)  accuracy = {acc:.3f}{flag}")
