"""Lab 29: Multi-Token Prediction (MTP) - predict t+1, t+2, t+3 from the same hidden state.

Two published designs:
  - parallel heads on a shared trunk (Gloeckle et al., Meta 2024, "Better & Faster LLMs via Multi-token Prediction")
  - sequential MTP modules that chain the previous depth's representation (DeepSeek-V3, 2024)
Both add extra prediction targets during TRAINING (denser signal, better planning) and can be reused at
INFERENCE as a built-in draft model for speculative decoding (predict 2-3 tokens, verify with the main head).

This lab trains a tiny causal transformer twice on the same text:
  A) next-token only      B) next-token + heads for t+2 and t+3 (parallel-head MTP, loss weight lam)
and reports per-head accuracy plus the self-speculative ACCEPTANCE rate of the t+2 head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
import random

rng = random.Random(0)
TEMPLATES = ["the {dev} {qty} must stay below {num} {unit} during {phase} .",
             "reset the {dev} by holding the {col} button for {num} seconds .",
             "if the {dev} {qty} exceeds {num} {unit} stop the {phase} and call support ."]
SLOTS = {"dev": ["pump", "valve", "compressor", "motor"], "qty": ["pressure", "temperature", "speed"],
         "num": ["two", "four", "six", "eight", "ten"], "unit": ["bar", "degrees", "rpm"],
         "phase": ["calibration", "startup", "cleaning"], "col": ["red", "green", "blue"]}
# random slot fills: some next tokens are predictable (grammar), some are not (the slot values)
words = " ".join(rng.choice(TEMPLATES).format(**{k: rng.choice(v) for k, v in SLOTS.items()})
                 for _ in range(400)).split()
vocab = sorted(set(words))
stoi = {w: i for i, w in enumerate(vocab)}
data = torch.tensor([stoi[w] for w in words])
V, D, T, K = len(vocab), 64, 24, 3          # K = how many future tokens are predicted (1 = plain LM)


class Tiny(nn.Module):
    def __init__(self, k_heads: int):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.pos = nn.Embedding(T, D)
        layer = nn.TransformerEncoderLayer(D, 4, 128, batch_first=True, dropout=0.0)
        self.trunk = nn.TransformerEncoder(layer, 2)
        # head 0 = the normal next-token head; heads 1.. = t+2, t+3 (each a small MLP + shared unembedding)
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(D, D), nn.GELU()) for _ in range(k_heads)])
        self.unembed = nn.Linear(D, V)

    def forward(self, x):
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool), 1)   # causal
        h = self.trunk(self.emb(x) + self.pos(torch.arange(x.shape[1])), mask=mask)
        return [self.unembed(hd(h)) for hd in self.heads]          # list of [B, T, V]


def batch(bs=32):
    ix = torch.randint(0, len(data) - T - K, (bs,))
    x = torch.stack([data[i:i + T] for i in ix])
    ys = [torch.stack([data[i + 1 + j:i + 1 + j + T] for i in ix]) for j in range(K)]   # t+1, t+2, t+3
    return x, ys


def train(k_heads: int, steps=400, lam=0.5):
    m = Tiny(k_heads)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(steps):
        x, ys = batch()
        outs = m(x)
        # main loss + lam * mean of the extra-depth losses (DeepSeek-V3 uses a small lam such as 0.1-0.3)
        main = F.cross_entropy(outs[0].reshape(-1, V), ys[0].reshape(-1))
        extra = [F.cross_entropy(o.reshape(-1, V), y.reshape(-1)) for o, y in zip(outs[1:], ys[1:])]
        loss = main + (lam * sum(extra) / len(extra) if extra else 0.0)
        opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def report(m, name):
    torch.manual_seed(1)
    x, ys = batch(256)
    outs = m(x)
    accs = [(o.argmax(-1) == y).float().mean().item() for o, y in zip(outs, ys)]
    line = "  ".join(f"t+{i + 1} acc {a:.3f}" for i, a in enumerate(accs))
    print(f"{name:<28}{line}")
    if len(outs) > 1:
        # self-speculative decoding: at position p, head0 gives t+1, head1 DRAFTS t+2.
        # Accept the draft if, after appending t+1, the main head at p+1 predicts the same token.
        draft = outs[1].argmax(-1)[:, :-1]
        verify = outs[0].argmax(-1)[:, 1:]
        acc_rate = (draft == verify).float().mean().item()
        print(f"{'':<28}self-speculative acceptance of the t+2 draft: {acc_rate:.3f}  "
              f"(expected tokens per main-head step ~ {1 + acc_rate:.2f})")


print(f"vocab {V} words, context {T}, corpus {len(words)} tokens\n")
report(train(1), "A) next-token only")
report(train(K), f"B) MTP with {K} heads")
print("""
READING
- The t+1 accuracy is the model you SHIP; the extra heads are training signal (and a free draft model).
- MTP helps most at scale and on structured outputs (code, maths): Meta reported gains growing with model
  size; DeepSeek-V3 used one extra depth (D=1) and reported ~85-90% acceptance of the second token,
  ~1.8x decoding speed. On a toy model the effect on t+1 is small; the acceptance rate is the useful number.
- Fine-tuning note: if a base model was trained WITH MTP modules (DeepSeek-V3/R1, some Qwen/GLM variants),
  most fine-tuning recipes DROP or freeze them; serving engines (vLLM/SGLang) can still use them as the
  speculative draft ('--speculative-config' with an MTP method) only if you KEEP them in the checkpoint.""")
