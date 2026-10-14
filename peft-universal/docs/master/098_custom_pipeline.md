# 98. Custom architectures: the whole training pipeline yourself

:::warn CUSTOM ENGINEERING REQUIRED
No Hugging Face class, no Trainer, no PEFT support, no tutorial. Everything below works on a plain `nn.Module` and is exercised by lab 60's custom rows and the tests.
:::

## 98.1 From `class MyModel(nn.Module)` to a trained, merged model

```py 1. model, 2. dataset, 3. loader + collator, 4. loss, 5. adapt, 6. check, 7. train, 8. eval, 9. export
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from uadapt import lora as L
from uadapt.adapter import UniversalModelAdapter
from uadapt.loop import train

class MyModel(nn.Module):                                   # 1
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(16, 64), nn.GELU(), nn.Linear(64, 64))
        self.head = nn.Linear(64, 3)
    def forward(self, x):
        return self.head(self.enc(x))

class MyData(Dataset):                                      # 2
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return {"x": self.X[i], "y": self.y[i]}

def collate(rows):                                          # 3  (pad / stack / build masks here)
    return {"x": torch.stack([r["x"] for r in rows]), "y": torch.tensor([r["y"] for r in rows])}

loader = DataLoader(MyData(torch.randn(256, 16), torch.randint(0, 3, (256,)).tolist()),
                    batch_size=32, shuffle=True, collate_fn=collate)
loss_fn = lambda m, b: nn.functional.cross_entropy(m(b["x"]), b["y"])   # 4

model = MyModel()
sample = next(iter(loader))
ad = UniversalModelAdapter(model, (sample["x"],))
d = ad.discover_trainable_modules()                         # 5
L.inject(model, d["lora_targets"], r=8, train_also=d["train_fully_or_freeze"])
assert ad.validate_training(loss_fn, sample)["ok"]         # 6  gradients reach the adapters
res = train(model, loader, loss_fn, lr=2e-3, max_steps=200, accum=2, clip=1.0,   # 7
            amp_dtype="bf16", eval_fn=lambda m: {"acc": 0.0}, eval_every=50)
torch.save(L.adapter_state_dict(model, also=["head"]), "adapter.pt")           # 9
```

## 98.2 The loop, and what each feature is for

@file src/uadapt/loop.py

|Feature|Argument|Why|
|---|---|---|
|mixed precision|`amp_dtype="bf16"`|half the activation memory, faster matmuls; bf16 avoids fp16 overflow|
|gradient accumulation|`accum=N`|effective batch N× larger without the memory|
|gradient clipping|`clip=1.0`|stops one bad batch from destroying the weights|
|warmup + cosine|`warmup=0.03`|stable start, smooth decay|
|NaN guard|built in|stops at the first non-finite loss instead of training on garbage|
|gradient check on step 0|built in|refuses to continue if nothing (or the wrong thing) receives gradients|
|checkpoints|`ckpt_dir`, `ckpt_every`|adapter weights only, so checkpoints are small|

## 98.3 Scaling it out (syntax; not executed here: no GPUs)

```py gradient checkpointing on your own blocks: recompute activations in backward
from torch.utils.checkpoint import checkpoint
class Block(nn.Module):
    def forward(self, x):
        return checkpoint(self._inner, x, use_reentrant=False)
```

```py DDP: one process per GPU, launched with torchrun
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
dist.init_process_group("nccl"); rank = dist.get_rank(); torch.cuda.set_device(rank)
model = DDP(model.cuda(rank), device_ids=[rank])
sampler = torch.utils.data.distributed.DistributedSampler(dataset)   # each rank sees a different shard
# torchrun --nproc_per_node=4 train_my_model.py
```

```py FSDP2: shard parameters across GPUs when one GPU cannot hold the model
from torch.distributed.fsdp import fully_shard
for block in model.blocks:
    fully_shard(block)
fully_shard(model)
```

:::warn Distributed training with custom adapters
Inject adapters BEFORE wrapping with DDP/FSDP, freeze the base first, and save adapter weights from rank 0 only (with FSDP, gather the full state dict first). This was not executed in this guide.
:::

## 98.4 Quantization of a custom model

bitsandbytes 4-bit replaces `nn.Linear` layers; convolutions and custom parameters stay in their dtype. For a custom model: quantize only the Linear-heavy parts, keep adapters in bf16/fp32, and measure the memory you actually saved.
