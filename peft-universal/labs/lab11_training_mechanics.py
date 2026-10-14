"""Lab 11: what actually changes during training. Forward -> loss -> backward -> gradient -> optimizer -> update.
Everything here is checked against a hand-computed value."""
import torch

torch.manual_seed(0)
x = torch.tensor([[2.0]])
y_true = torch.tensor([[10.0]])
w = torch.nn.Parameter(torch.tensor([[3.0]]))
b = torch.nn.Parameter(torch.tensor([1.0]))
lr = 0.1

print("start: w=3.0 b=1.0 x=2.0 target=10.0   (mean squared error)")
y = x @ w.T + b                                   # 1. FORWARD
loss = ((y - y_true) ** 2).mean()                 # 2. LOSS
print(f"1. forward : y = w*x + b = 3*2 + 1 = {y.item():.1f}")
print(f"2. loss    : (y - target)^2 = ({y.item():.1f} - 10)^2 = {loss.item():.1f}")
loss.backward()                                   # 3. BACKPROP
print(f"3. gradient: dL/dw = 2*(y-target)*x = 2*({y.item():.1f}-10)*2 = {w.grad.item():.1f}"
      f"   (autograd says {w.grad.item():.1f})")
print(f"            dL/db = 2*(y-target)   = {b.grad.item():.1f}")
manual_w = w.item() - lr * w.grad.item()          # 4. UPDATE (plain SGD)
with torch.no_grad():
    w -= lr * w.grad
    b -= lr * b.grad
print(f"4. update  : w_new = w - lr*grad = 3.0 - 0.1*(-12.0) = {manual_w:.1f}  (tensor now {w.item():.1f})")
y2 = x @ w.T + b
print(f"5. check   : new prediction {y2.item():.2f}, new loss {((y2 - y_true) ** 2).mean().item():.3f} (was {loss.item():.1f})")

print("\nSame gradient, three optimizers, first step from w=3.0:")
for name, make in [("SGD lr=0.1", lambda p: torch.optim.SGD(p, lr=0.1)),
                   ("SGD+momentum", lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9)),
                   ("AdamW lr=0.1", lambda p: torch.optim.AdamW(p, lr=0.1))]:
    ww = torch.nn.Parameter(torch.tensor([[3.0]]))
    bb = torch.nn.Parameter(torch.tensor([1.0]))
    opt = make([ww, bb])
    for step in range(3):
        opt.zero_grad()
        l = ((x @ ww.T + bb - y_true) ** 2).mean()
        l.backward()
        opt.step()
    print(f"  {name:<14} after 3 steps: w={ww.item():6.3f} loss={l.item():8.3f}")
print("AdamW normalises by the gradient's own size, so its first steps are ~lr regardless of gradient scale.")

print("\nWhat PEFT changes: only SOME parameters get step 4.")
lin = torch.nn.Linear(4, 4, bias=False)
A = torch.nn.Parameter(torch.randn(2, 4) * 0.02)
B = torch.nn.Parameter(torch.zeros(4, 2))
lin.weight.requires_grad_(False)
out = lin(torch.randn(1, 4)) + (torch.randn(1, 4) @ A.T @ B.T)
out.sum().backward()
print(f"  frozen base weight grad : {lin.weight.grad}")
print(f"  adapter A grad is not None: {A.grad is not None}, B grad is not None: {B.grad is not None}")
print("  -> no gradient is stored for the base weights, which is where the memory saving comes from.")
