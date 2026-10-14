"""Fine-tune YOUR OWN PyTorch model (not a transformers class) with LoRA, end to end, on CPU in seconds.

Pattern for any custom nn.Module: wrap it in uadapt.UniversalModelAdapter -> inspect -> discover targets by
TRACING (no layer names needed) -> inject LoRA -> validate gradients on a real batch -> train -> export
(adapter.pt + exactly merged weights). Replace MyModel, make_batch and loss_fn with yours.

python examples/custom_model/finetune_my_model.py
"""
import torch
from torch import nn

from uadapt.adapter import UniversalModelAdapter

torch.manual_seed(0)


class MyModel(nn.Module):
    """Stand-in for 'my own model': a small sensor-sequence classifier (conv front-end + MLP head)."""

    def __init__(self, channels=4, hidden=64, classes=3):
        super().__init__()
        self.conv = nn.Conv1d(channels, hidden, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.head = nn.Linear(hidden, classes)

    def forward(self, x):                                   # x: [batch, channels, time]
        h = self.conv(x).mean(-1)
        return self.head(self.mlp(self.norm(h)) + h)


def make_batch(n=64):
    x = torch.randn(n, 4, 32)
    y = (x[:, 0].mean(-1) > 0).long() + (x[:, 1].mean(-1) > 0.3).long()   # a rule the model must learn
    return {"x": x}, y


model = MyModel()
# "pre-training": the model you already have. Here: a few steps on a DIFFERENT rule, then frozen.
opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
for _ in range(50):
    inputs, _ = make_batch()
    loss = nn.functional.cross_entropy(model(**inputs), (inputs["x"][:, 2].mean(-1) > 0).long())
    opt.zero_grad(); loss.backward(); opt.step()

sample, _ = make_batch(4)
ua = UniversalModelAdapter(model, sample)
print("inspect   :", ua.inspect()["params"])
print("classify  :", ua.classify()["labels"])
found = ua.discover_trainable_modules()
print("LoRA targets found by tracing:", found["lora_targets"])
print("inject    :", {k: v for k, v in ua.build_training_pipeline(r=4, alpha=8, train_also=["head"]).items() if k != "wrapped"})

data = [make_batch() for _ in range(30)]


def loss_fn(m, batch):
    inputs, y = batch
    return nn.functional.cross_entropy(m(**inputs), y)


def accuracy(m):
    inputs, y = make_batch(512)
    return {"acc": float((m(**inputs).argmax(-1) == y).float().mean())}


print("before    :", ua.evaluate(accuracy))
print("gradients :", {k: v for k, v in ua.validate_training(loss_fn, data[0]).items() if k in ("ok", "problems")})
ua.model.train()
res = ua.train(data, loss_fn, lr=5e-3, max_steps=150, log_every=50)
print("after     :", ua.evaluate(accuracy))
with torch.no_grad():
    inputs, _ = make_batch(8)
    before_merge = ua.model(**inputs)
print("export    :", ua.export("outputs/my_model_lora", merge=True))
with torch.no_grad():
    print("merge exact:", torch.allclose(before_merge, ua.model(**inputs), atol=1e-5))
print("serve     :", ua.serve())
