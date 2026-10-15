# Transformers 101 - Part 5 INTERMEDIATE
# Topics 4 (Multi-Head Attention), 5 (Masked Multi-Head Attention), 6 (Positional Encoding)
# + the full Transformer block that Part 6 builds on.
# Each "# %% ID" block is ONE Colab cell. Run them in order in a NEW notebook.

# %% 5.0-A
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| device:", device)

# %% 5.1-A
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
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

mha = MultiHeadAttention(d_model=32, n_heads=4)
x = torch.randn(2, 6, 32)
y, w = mha(x)
print("input  x      :", tuple(x.shape))
print("output y      :", tuple(y.shape))
print("weights       :", tuple(w.shape), "<- (batch, heads, query, key)")
print("parameters    :", sum(p.numel() for p in mha.parameters()))

# %% 5.1-B
B, T, D, H = 2, 6, 32, 4
q = mha.W_q(x)
print("1. after W_q              :", tuple(q.shape))
q = q.view(B, T, H, D // H)
print("2. view into heads        :", tuple(q.shape))
q = q.transpose(1, 2)
print("3. heads to the front     :", tuple(q.shape))
k = mha.W_k(x).view(B, T, H, D // H).transpose(1, 2)
s = q @ k.transpose(-2, -1)
print("4. scores per head        :", tuple(s.shape))
o = torch.softmax(s / math.sqrt(D // H), -1) @ mha.W_v(x).view(B, T, H, D // H).transpose(1, 2)
print("5. weighted values        :", tuple(o.shape))
o = o.transpose(1, 2).reshape(B, T, D)
print("6. heads glued back       :", tuple(o.shape))

# %% 5.1-C
ref = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
with torch.no_grad():
    ref.in_proj_weight.copy_(torch.cat([mha.W_q.weight, mha.W_k.weight, mha.W_v.weight]))
    ref.in_proj_bias.copy_(torch.cat([mha.W_q.bias, mha.W_k.bias, mha.W_v.bias]))
    ref.out_proj.weight.copy_(mha.W_o.weight)
    ref.out_proj.bias.copy_(mha.W_o.bias)
y_ref, _ = ref(x, x, x)
print("my MHA == torch.nn.MultiheadAttention:", torch.allclose(y, y_ref, atol=1e-6))

# %% 5.2-A
def causal_mask(T):
    return torch.tril(torch.ones(T, T)).view(1, 1, T, T)

print(causal_mask(5)[0, 0])

# %% 5.2-B
torch.manual_seed(0)
x1 = torch.randn(1, 6, 32)
x2 = x1.clone()
x2[0, -1] = torch.randn(32)
mask = causal_mask(6)
y1, w1 = mha(x1, mask=mask)
y2, _ = mha(x2, mask=mask)
print("WITH mask   : first 5 outputs unchanged? ", torch.allclose(y1[0, :5], y2[0, :5], atol=1e-6))
print("WITH mask   : last output changed?       ", not torch.allclose(y1[0, 5], y2[0, 5]))
u1, _ = mha(x1)
u2, _ = mha(x2)
print("WITHOUT mask: first 5 outputs unchanged? ", torch.allclose(u1[0, :5], u2[0, :5], atol=1e-6))
print("head 0 weights with mask:")
print(w1[0, 0].detach().round(decimals=2))

# %% 5.2-C
tokens = torch.tensor([[5, 7, 9, 0, 0],
                       [3, 4, 0, 0, 0]])
PAD = 0
pad_mask = (tokens != PAD).view(2, 1, 1, 5).int()
print("padding mask, sentence 1:", pad_mask[0, 0, 0].tolist())
print("padding mask, sentence 2:", pad_mask[1, 0, 0].tolist())
combined = pad_mask * causal_mask(5).int()
print("causal + padding, sentence 2:")
print(combined[1, 0])

# %% 5.3-A
torch.manual_seed(0)
x = torch.randn(1, 5, 32)
perm = torch.tensor([4, 2, 0, 3, 1])
y, _ = mha(x)
y_shuffled_input, _ = mha(x[:, perm])
print("Shuffle the words, outputs are just shuffled too:",
      torch.allclose(y[:, perm], y_shuffled_input, atol=1e-6))
print("=> without position info, 'dog bites man' == 'man bites dog' to attention")

# %% 5.3-B
def sinusoidal_pe(max_len, d_model):
    pos = torch.arange(max_len).unsqueeze(1)
    i = torch.arange(0, d_model, 2)
    freq = torch.exp(-math.log(10000.0) * i / d_model)
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)
    return pe

pe = sinusoidal_pe(100, 64)
print("PE table:", tuple(pe.shape))
print("position 0, first 6 numbers:", [round(v, 3) for v in pe[0, :6].tolist()])
print("position 1, first 6 numbers:", [round(v, 3) for v in pe[1, :6].tolist()])
plt.figure(figsize=(8, 4))
plt.imshow(pe.T, aspect="auto", cmap="RdBu")
plt.xlabel("position in sentence")
plt.ylabel("embedding dimension")
plt.title("Sinusoidal positional encoding (each column = one position's 'barcode')")
plt.colorbar()
plt.show()

# %% 5.3-C
sims = pe @ pe[20]
plt.figure(figsize=(7, 3))
plt.plot(sims)
plt.axvline(20, color="red", linestyle="--")
plt.xlabel("position")
plt.ylabel("dot product with position 20")
plt.title("Nearby positions look similar, far ones look different")
plt.show()
print("similarity 20 vs 20:", round(sims[20].item(), 2))
print("similarity 20 vs 21:", round(sims[21].item(), 2))
print("similarity 20 vs 60:", round(sims[60].item(), 2))

# %% 5.3-D
def rope(x, base=10000.0):
    T, D = x.shape[-2], x.shape[-1]
    pos = torch.arange(T, dtype=torch.float32).unsqueeze(1)
    freq = base ** (-torch.arange(0, D, 2, dtype=torch.float32) / D)
    angle = pos * freq
    cos, sin = angle.cos(), angle.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out

torch.manual_seed(0)
q_vec, k_vec = torch.randn(64), torch.randn(64)
Qs = rope(q_vec.repeat(50, 1))
Ks = rope(k_vec.repeat(50, 1))
print("query at 3, key at 1  (distance 2):", round((Qs[3] @ Ks[1]).item(), 4))
print("query at 30, key at 28 (distance 2):", round((Qs[30] @ Ks[28]).item(), 4))
print("query at 30, key at 10 (distance 20):", round((Qs[30] @ Ks[10]).item(), 4))
print("RoPE keeps vector length:", torch.allclose(Qs.norm(dim=-1), q_vec.norm().repeat(50)))

# %% 5.4-A
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

block = Block(d_model=64, n_heads=4)
x = torch.randn(2, 10, 64)
y = block(x, mask=causal_mask(10))
print("block in :", tuple(x.shape))
print("block out:", tuple(y.shape))
print("block parameters:", sum(p.numel() for p in block.parameters()))
