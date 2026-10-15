# Transformers 101 - Part 7 ELITE
# How modern LLMs (Llama / Qwen / Mistral / DeepSeek family) change the 2017 design,
# plus a mechanistic-interpretability experiment (induction heads).
# Each "# %% ID" block is ONE Colab cell. Run them in order in a NEW notebook (T4 GPU).

# %% 7.0-A
import math
import time
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| device:", device)

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = urllib.request.urlopen(url).read().decode("utf-8")
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

BLOCK_SIZE = 128
BATCH_SIZE = 64

def get_batch(split):
    src = train_data if split == "train" else val_data
    ix = torch.randint(len(src) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([src[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([src[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(device), y.to(device)

print("data ready:", f"{len(train_data):,}", "train tokens, vocab", vocab_size)

# %% 7.1-A
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_tables(max_len, d_head, base=10000.0):
    freq = base ** (-torch.arange(0, d_head, 2).float() / d_head)
    angles = torch.outer(torch.arange(max_len).float(), freq)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)


class GQAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.h, self.kv = n_heads, n_kv_heads
        self.dh = d_model // n_heads
        self.wq = nn.Linear(d_model, n_heads * self.dh, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.dh, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.dh, bias=False)
        self.wo = nn.Linear(n_heads * self.dh, d_model, bias=False)

    def forward(self, x, cos, sin, cache=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.wk(x).view(B, T, self.kv, self.dh).transpose(1, 2)
        v = self.wv(x).view(B, T, self.kv, self.dh).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            if cache["k"] is not None:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        k = k.repeat_interleave(self.h // self.kv, dim=1)
        v = v.repeat_interleave(self.h // self.kv, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))
        return self.wo(out.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        hidden = 64 * math.ceil((8 * d_model / 3) / 64)
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class ModernBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = GQAttention(d_model, n_heads, n_kv_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.norm1(x), cos, sin, cache)
        x = x + self.ffn(self.norm2(x))
        return x

print("modern building blocks ready")

# %% 7.1-B
class ModernGPT(nn.Module):
    def __init__(self, vocab_size, block_size, d_model=256, n_heads=8, n_kv_heads=2,
                 n_layers=4, max_len=1024):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [ModernBlock(d_model, n_heads, n_kv_heads) for _ in range(n_layers)])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        cos, sin = rope_tables(max_len, d_model // n_heads)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, caches=None, start_pos=0):
        B, T = idx.shape
        cos = self.cos[start_pos:start_pos + T]
        sin = self.sin[start_pos:start_pos + T]
        x = self.tok_emb(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, None if caches is None else caches[i])
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, use_cache=True):
        if not use_cache:
            for _ in range(max_new_tokens):
                logits, _ = self(idx[:, -self.block_size:])
                idx = torch.cat([idx, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
            return idx
        caches = [{"k": None, "v": None} for _ in self.blocks]
        logits, _ = self(idx, caches=caches, start_pos=0)
        for _ in range(max_new_tokens):
            next_id = logits[:, -1].argmax(-1, keepdim=True)
            idx = torch.cat([idx, next_id], dim=1)
            logits, _ = self(next_id, caches=caches, start_pos=idx.size(1) - 1)
        return idx

torch.manual_seed(42)
modern = ModernGPT(vocab_size, BLOCK_SIZE).to(device)
print("ModernGPT parameters:", f"{sum(p.numel() for p in modern.parameters()):,}")
xb, yb = get_batch("train")
_, loss = modern(xb, yb)
print("loss before training:", round(loss.item(), 3))

# %% 7.1-C
MAX_ITERS = 3000
EVAL_EVERY = 300
EVAL_ITERS = 50
use_amp = device == "cuda"

@torch.no_grad()
def estimate_loss(m):
    m.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(split)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                _, l = m(x, y)
            losses[k] = l.item()
        out[split] = losses.mean().item()
    m.train()
    return out

optimizer = torch.optim.AdamW(modern.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=MAX_ITERS,
                                                pct_start=0.1)
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
t0 = time.time()
for it in range(MAX_ITERS + 1):
    if it % EVAL_EVERY == 0:
        l = estimate_loss(modern)
        print(f"step {it:5d} | train {l['train']:.3f} | val {l['val']:.3f} | {time.time() - t0:6.1f}s")
    if it == MAX_ITERS:
        break
    x, y = get_batch("train")
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
        _, loss = modern(x, y)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(modern.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

# %% 7.2-A
modern.eval()
prompt = torch.tensor([encode("ROMEO:")], device=device)
N_NEW = 110

def timed(use_cache):
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    out = modern.generate(prompt, N_NEW, use_cache=use_cache)
    if device == "cuda":
        torch.cuda.synchronize()
    return out, time.time() - t

modern.generate(prompt, 5, use_cache=True)
slow, t_slow = timed(use_cache=False)
fast, t_fast = timed(use_cache=True)
print("identical text with and without cache:", torch.equal(slow, fast))
print(f"no cache : {N_NEW / t_slow:7.1f} tokens/s")
print(f"KV cache : {N_NEW / t_fast:7.1f} tokens/s  ({t_slow / t_fast:.1f}x faster)")
print()
print(decode(fast[0].tolist()))

# %% 7.2-B
def kv_cache_gb(layers, kv_heads, d_head, seq_len, batch=1, bytes_per_value=2):
    return 2 * layers * kv_heads * d_head * seq_len * batch * bytes_per_value / 1e9

print("KV cache for ONE 32k-token conversation, 16-bit values:")
print(f"  8B model, full multi-head (32 KV heads): {kv_cache_gb(32, 32, 128, 32768):5.1f} GB")
print(f"  8B model, GQA (8 KV heads)             : {kv_cache_gb(32, 8, 128, 32768):5.1f} GB")
print(f"  8B model, MQA (1 KV head)              : {kv_cache_gb(32, 1, 128, 32768):5.1f} GB")
print(f"  Qwen3-0.6B (28 layers, 8 KV heads)     : {kv_cache_gb(28, 8, 128, 32768):5.1f} GB")

# %% 7.3-A
from transformers import AutoTokenizer, AutoModelForCausalLM

g_tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
g_model = AutoModelForCausalLM.from_pretrained(
    "openai-community/gpt2", attn_implementation="eager").to(device).eval()

torch.manual_seed(0)
L = 50
rand = torch.randint(1000, 10000, (1, L))
seq = torch.cat([torch.tensor([[g_tok.bos_token_id]]), rand, rand], dim=1).to(device)
with torch.no_grad():
    out = g_model(seq, output_attentions=True)
logp = F.log_softmax(out.logits[0, :-1].float(), dim=-1)
tok_loss = -logp.gather(1, seq[0, 1:].unsqueeze(1)).squeeze(1)
print(f"average loss on 1st copy of the random tokens: {tok_loss[:L].mean().item():.2f}")
print(f"average loss on 2nd copy of the random tokens: {tok_loss[L:].mean().item():.2f}")

# %% 7.3-B
att = torch.stack(out.attentions)[:, 0]
q_pos = torch.arange(L + 1, 2 * L + 1, device=device)
k_pos = q_pos - L + 1
induction = att[:, :, q_pos, k_pos].mean(-1).cpu()
plt.figure(figsize=(6, 5))
plt.imshow(induction, cmap="magma")
plt.xlabel("head")
plt.ylabel("layer")
plt.title("Induction score of every GPT-2 head")
plt.colorbar()
plt.show()
top = torch.topk(induction.flatten(), 5)
for score, flat in zip(top.values.tolist(), top.indices.tolist()):
    print(f"layer {flat // 12:2d} head {flat % 12:2d}  induction score {score:.2f}")
