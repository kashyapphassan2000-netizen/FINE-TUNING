# Transformers 101 - Part 6 ADVANCED
# Topics 7 (Encoder-Decoder), 8 (Encoder-Only), 9 (Decoder-Only), 10 (Cross-Attention)
# Each "# %% ID" block is ONE Colab cell. Run them in order in a NEW notebook (T4 GPU).

# %% 6.0-A
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| device:", device)

# %% 6.0-B
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, x_q, x_kv=None, mask=None):
        if x_kv is None:
            x_kv = x_q
        Q = self.split_heads(self.W_q(x_q))
        K = self.split_heads(self.W_k(x_kv))
        V = self.split_heads(self.W_v(x_kv))
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        out = weights @ V
        B, H, T, Dh = out.shape
        out = out.transpose(1, 2).contiguous().view(B, T, H * Dh)
        return self.W_o(out), weights


def causal_mask(T):
    return torch.tril(torch.ones(T, T)).view(1, 1, T, T)


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x, mask=None):
        a, _ = self.attn(self.ln1(x), mask=mask)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x

print("building blocks ready")

# %% 6.1-A
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

t5_tok = AutoTokenizer.from_pretrained("google-t5/t5-small")
t5 = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small").to(device)

def t5_run(prompt, max_new_tokens=40):
    inputs = t5_tok(prompt, return_tensors="pt").to(device)
    out = t5.generate(**inputs, max_new_tokens=max_new_tokens)
    return t5_tok.decode(out[0], skip_special_tokens=True)

print(t5_run("translate English to German: The house is wonderful."))
print(t5_run("summarize: The Transformer is a neural network that reads a whole sentence at "
             "once and uses attention to decide which words matter for each other word. It "
             "replaced recurrent networks in translation and now powers chatbots."))

# %% 6.1-B
cfg = t5.config
print("encoder layers:", cfg.num_layers, "| decoder layers:", cfg.num_decoder_layers,
      "| d_model:", cfg.d_model, "| heads:", cfg.num_heads)
print("\nInside ONE decoder layer:")
for name, module in t5.decoder.block[0].named_children():
    print(" ", name)
    for sub_name, sub in module.named_children():
        print("     ", sub_name, "->", type(sub).__name__)
print("\nTotal parameters:", f"{sum(p.numel() for p in t5.parameters()):,}")

# %% 6.2-A
from transformers import pipeline

fill = pipeline("fill-mask", model="google-bert/bert-base-uncased",
                device=0 if device == "cuda" else -1)
for guess in fill("The [MASK] barked at the mailman all morning.")[:5]:
    print(f"{guess['token_str']:>10}  {guess['score']:.3f}")

# %% 6.2-B
from transformers import AutoModel

bert_tok = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
bert = AutoModel.from_pretrained("google-bert/bert-base-uncased").to(device).eval()

def word_vector(sentence, word):
    enc = bert_tok(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        hidden = bert(**enc).last_hidden_state[0]
    toks = bert_tok.convert_ids_to_tokens(enc["input_ids"][0])
    return hidden[toks.index(word)]

river1 = word_vector("We sat on the bank of the river and fished.", "bank")
river2 = word_vector("The river bank was muddy after the rain.", "bank")
money = word_vector("I deposited my salary at the bank today.", "bank")
print("river bank vs river bank:", round(F.cosine_similarity(river1, river2, dim=0).item(), 3))
print("river bank vs money bank:", round(F.cosine_similarity(river1, money, dim=0).item(), 3))

# %% 6.2-C
st_name = "sentence-transformers/all-MiniLM-L6-v2"
st_tok = AutoTokenizer.from_pretrained(st_name)
st_model = AutoModel.from_pretrained(st_name).to(device).eval()

def embed(sentences):
    enc = st_tok(sentences, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        hidden = st_model(**enc).last_hidden_state
    m = enc["attention_mask"].unsqueeze(-1).float()
    vecs = (hidden * m).sum(1) / m.sum(1)
    return F.normalize(vecs, dim=-1)

docs = ["How do I reset my password?",
        "Our office is closed on public holidays.",
        "Refunds are processed within 5 business days.",
        "The GPU ran out of memory during training."]
query = "I forgot my login credentials"
scores = embed([query]) @ embed(docs).T
for d, s in sorted(zip(docs, scores[0].tolist()), key=lambda p: -p[1]):
    print(f"{s:.3f}  {d}")

# %% 6.3-A
import urllib.request

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
print("characters in file:", f"{len(text):,}")
print("vocabulary size   :", vocab_size)
print("train / val tokens:", f"{len(train_data):,}", "/", f"{len(val_data):,}")
print(text[:120])

# %% 6.3-B
BLOCK_SIZE = 128
BATCH_SIZE = 64

def get_batch(split):
    src = train_data if split == "train" else val_data
    ix = torch.randint(len(src) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([src[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([src[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(device), y.to(device)

xb, yb = get_batch("train")
print("x batch:", tuple(xb.shape), " y batch:", tuple(yb.shape))
print("x:", repr(decode(xb[0, :20].tolist())))
print("y:", repr(decode(yb[0, :20].tolist())), "<- same text shifted left by one")

# %% 6.3-C
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size, d_model=256, n_heads=4, n_layers=4, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.register_buffer("mask", causal_mask(block_size))

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        mask = self.mask[:, :, :T, :T]
        for block in self.blocks:
            x = block(x, mask)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

torch.manual_seed(42)
model = MiniGPT(vocab_size, BLOCK_SIZE).to(device)
print("MiniGPT parameters:", f"{sum(p.numel() for p in model.parameters()):,}")
_, loss = model(xb, yb)
print("loss before training:", round(loss.item(), 3), "| random-guess loss = ln(65) =", round(math.log(vocab_size), 3))

# %% 6.3-D
MAX_ITERS = 3000
EVAL_EVERY = 300
EVAL_ITERS = 50
LR = 1e-3

@torch.no_grad()
def estimate_loss(m):
    m.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(split)
            _, l = m(x, y)
            losses[k] = l.item()
        out[split] = losses.mean().item()
    m.train()
    return out

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
history = []
t0 = time.time()
for it in range(MAX_ITERS + 1):
    if it % EVAL_EVERY == 0:
        l = estimate_loss(model)
        history.append((it, l["train"], l["val"]))
        print(f"step {it:5d} | train {l['train']:.3f} | val {l['val']:.3f} | {time.time() - t0:6.1f}s")
    if it == MAX_ITERS:
        break
    x, y = get_batch("train")
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

steps, tr, va = zip(*history)
plt.figure(figsize=(6, 3.5))
plt.plot(steps, tr, label="train")
plt.plot(steps, va, label="val")
plt.xlabel("step")
plt.ylabel("loss")
plt.legend()
plt.title("MiniGPT learning Shakespeare")
plt.show()

# %% 6.3-E
model.eval()
start = torch.tensor([encode("ROMEO:")], device=device)
torch.manual_seed(7)
sample = model.generate(start, max_new_tokens=300, temperature=0.8, top_k=20)
print(decode(sample[0].tolist()))

# %% 6.3-F
from transformers import AutoModelForCausalLM

qwen_name = "Qwen/Qwen3-0.6B"
q_tok = AutoTokenizer.from_pretrained(qwen_name)
qwen = AutoModelForCausalLM.from_pretrained(qwen_name, dtype="auto").to(device)
c = qwen.config
print(f"layers={c.num_hidden_layers}  d_model={c.hidden_size}  query heads={c.num_attention_heads}  "
      f"key/value heads={c.num_key_value_heads}  context={c.max_position_embeddings}")
messages = [{"role": "user", "content": "In one sentence, what does a decoder-only transformer do?"}]
prompt = q_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
inputs = q_tok(prompt, return_tensors="pt").to(device)
out = qwen.generate(**inputs, max_new_tokens=60, do_sample=False)
print(q_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))

# %% 6.4-A
SRC_LEN = 8
BOS = 10
S2S_VOCAB = 11

def make_batch(n):
    src = torch.randint(0, 10, (n, SRC_LEN))
    tgt = src.flip(1)
    dec_in = torch.cat([torch.full((n, 1), BOS), tgt[:, :-1]], dim=1)
    return src.to(device), dec_in.to(device), tgt.to(device)

s, d_in, t = make_batch(1)
print("source (encoder input)  :", s[0].tolist())
print("decoder input (BOS=10)  :", d_in[0].tolist())
print("target (what to predict):", t[0].tolist())

# %% 6.4-B
class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x, memory, tgt_mask, src_mask=None):
        a, _ = self.self_attn(self.ln1(x), mask=tgt_mask)
        x = x + a
        c, cross_w = self.cross_attn(self.ln2(x), x_kv=memory, mask=src_mask)
        x = x + c
        x = x + self.ffn(self.ln3(x))
        return x, cross_w


class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab, max_len, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.src_emb = nn.Embedding(vocab, d_model)
        self.tgt_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.encoder = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.enc_ln = nn.LayerNorm(d_model)
        self.decoder = nn.ModuleList([DecoderBlock(d_model, n_heads) for _ in range(n_layers)])
        self.dec_ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def encode(self, src):
        pos = torch.arange(src.size(1), device=src.device)
        x = self.src_emb(src) + self.pos_emb(pos)
        for block in self.encoder:
            x = block(x)
        return self.enc_ln(x)

    def decode(self, dec_in, memory):
        T = dec_in.size(1)
        pos = torch.arange(T, device=dec_in.device)
        x = self.tgt_emb(dec_in) + self.pos_emb(pos)
        mask = causal_mask(T).to(dec_in.device)
        for block in self.decoder:
            x, cross_w = block(x, memory, mask)
        return self.head(self.dec_ln(x)), cross_w

    def forward(self, src, dec_in):
        return self.decode(dec_in, self.encode(src))

torch.manual_seed(0)
s2s = Seq2SeqTransformer(S2S_VOCAB, SRC_LEN).to(device)
print("Seq2Seq parameters:", f"{sum(p.numel() for p in s2s.parameters()):,}")

# %% 6.4-C
opt = torch.optim.AdamW(s2s.parameters(), lr=3e-3)
for step in range(1501):
    src, dec_in, tgt = make_batch(128)
    logits, _ = s2s(src, dec_in)
    loss = F.cross_entropy(logits.reshape(-1, S2S_VOCAB), tgt.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if step % 300 == 0:
        print(f"step {step:4d} | loss {loss.item():.4f}")

# %% 6.4-D
@torch.no_grad()
def greedy_decode(model, src):
    model.eval()
    memory = model.encode(src)
    dec = torch.full((src.size(0), 1), BOS, device=src.device)
    for _ in range(SRC_LEN):
        logits, w = model.decode(dec, memory)
        next_id = logits[:, -1].argmax(-1, keepdim=True)
        dec = torch.cat([dec, next_id], dim=1)
    return dec[:, 1:], w

src, _, tgt = make_batch(1000)
pred, _ = greedy_decode(s2s, src)
acc = (pred == tgt).all(dim=1).float().mean().item()
print("whole-sequence accuracy on 1000 new examples:", f"{acc:.1%}")
print("example source :", src[0].tolist())
print("model output   :", pred[0].tolist())

# %% 6.4-E
example = torch.tensor([[3, 1, 4, 1, 5, 9, 2, 6]], device=device)
pred, cross_w = greedy_decode(s2s, example)
A = cross_w[0].mean(0).cpu()
plt.figure(figsize=(5, 4.5))
plt.imshow(A, cmap="viridis")
plt.xticks(range(SRC_LEN), example[0].tolist())
plt.yticks(range(SRC_LEN), pred[0].tolist())
plt.xlabel("encoder position (source digit)")
plt.ylabel("decoder step (digit being written)")
plt.title("Cross-attention: decoder looks at the source")
plt.colorbar()
plt.show()
print("output:", pred[0].tolist())
