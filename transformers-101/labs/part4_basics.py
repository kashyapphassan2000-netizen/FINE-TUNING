# Transformers 101 - Part 4 BASICS
# Topics 1 (Embeddings), 2 (The Attention Mechanism), 3 (Self-Attention)
# Each "# %% ID" block is ONE Colab cell. Run them in order.

# %% 4.0-A
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| device:", device)

# %% 4.1-A
sentence = "the cat sat on the mat"
words = sentence.split()
vocab = sorted(set(words))
stoi = {w: i for i, w in enumerate(vocab)}
itos = {i: w for w, i in stoi.items()}
ids = [stoi[w] for w in words]
print("words:", words)
print("vocab:", vocab)
print("ids  :", ids)

# %% 4.1-B
one_hot = F.one_hot(torch.tensor(ids), num_classes=len(vocab)).float()
print(one_hot)
cat_vec, mat_vec = one_hot[1], one_hot[5]
print("cat . mat =", (cat_vec @ mat_vec).item())
print("cat . cat =", (cat_vec @ cat_vec).item())

# %% 4.1-C
d_model = 4
emb = nn.Embedding(num_embeddings=len(vocab), embedding_dim=d_model)
X = emb(torch.tensor(ids))
print("X shape:", tuple(X.shape))
print(X)
same = one_hot @ emb.weight
print("lookup == one_hot @ W ?", torch.allclose(X, same))

# %% 4.1-D
from transformers import AutoTokenizer, AutoModel

tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
gpt2 = AutoModel.from_pretrained("openai-community/gpt2", attn_implementation="eager")
E = gpt2.get_input_embeddings().weight.detach()
print("GPT-2 embedding table:", tuple(E.shape))

def vec(word):
    token_ids = tok.encode(" " + word)
    assert len(token_ids) == 1, f"'{word}' splits into {len(token_ids)} tokens"
    return E[token_ids[0]]

def cos(a, b):
    return F.cosine_similarity(vec(a), vec(b), dim=0).item()

pairs = [("king", "queen"), ("king", "man"), ("king", "banana"),
         ("Paris", "London"), ("Paris", "pizza"), ("happy", "glad"), ("happy", "truck")]
for a, b in pairs:
    print(f"{a:>6} vs {b:<7} cosine = {cos(a, b):.3f}")

# %% 4.1-E
plot_words = ["king", "queen", "man", "woman", "prince", "apple", "banana", "orange",
              "car", "bus", "train", "truck", "Monday", "Friday", "Sunday"]
V = torch.stack([vec(w) for w in plot_words])
V = V - V.mean(dim=0)
U, S, Vt = torch.linalg.svd(V, full_matrices=False)
P = V @ Vt[:2].T
plt.figure(figsize=(6.5, 5))
plt.scatter(P[:, 0], P[:, 1])
for w, (x, y) in zip(plot_words, P.tolist()):
    plt.annotate(w, (x, y), fontsize=11)
plt.title("GPT-2 token embeddings squashed to 2-D (PCA)")
plt.show()

# %% 4.2-A
keys = torch.tensor([[1.0, 0.0],    # "cat"
                     [0.0, 1.0],    # "sat"
                     [1.0, 1.0]])   # "mat"
values = torch.tensor([[10.0, 0.0],
                       [0.0, 10.0],
                       [5.0, 5.0]])
query = torch.tensor([1.0, 0.0])
scores = keys @ query
weights = torch.softmax(scores, dim=0)
output = weights @ values
print("scores :", scores)
print("weights:", weights.round(decimals=4))
print("output :", output.round(decimals=3))

# %% 4.2-B
torch.manual_seed(0)
for d in [2, 64, 512]:
    q = torch.randn(d)
    K = torch.randn(5, d)
    raw = K @ q
    w_raw = torch.softmax(raw, dim=0).max().item()
    w_scaled = torch.softmax(raw / d ** 0.5, dim=0).max().item()
    print(f"d={d:4d}  score std={raw.std().item():6.2f}  "
          f"biggest weight: unscaled={w_raw:.3f}  scaled={w_scaled:.3f}")

# %% 4.2-C
def attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / d_k ** 0.5
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ V, weights

torch.manual_seed(1)
Q = torch.randn(1, 6, 8)
K = torch.randn(1, 6, 8)
V = torch.randn(1, 6, 8)
mine, w = attention(Q, K, V)
official = F.scaled_dot_product_attention(Q, K, V)
print("output shape :", tuple(mine.shape))
print("weights shape:", tuple(w.shape))
print("each row of weights sums to 1:", torch.allclose(w.sum(-1), torch.ones(1, 6)))
print("matches PyTorch SDPA:", torch.allclose(mine, official, atol=1e-6))

# %% 4.3-A
class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_k, bias=False)

    def forward(self, x, mask=None):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        return attention(Q, K, V, mask)

torch.manual_seed(42)
sa = SelfAttention(d_model=4, d_k=4)
x = X.detach().unsqueeze(0)
out, w = sa(x)
print("input  :", tuple(x.shape))
print("output :", tuple(out.shape))
print("weights:", tuple(w.shape))
print(w[0].detach().round(decimals=2))

# %% 4.3-B
text = "The animal didn't cross the street because it was too tired"
enc = tok(text, return_tensors="pt")
tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
with torch.no_grad():
    res = gpt2(**enc, output_attentions=True)
att = torch.stack(res.attentions)[:, 0]
it_pos = tokens.index("Ġit")
animal_pos = tokens.index("Ġanimal")
street_pos = tokens.index("Ġstreet")
score = att[:, :, it_pos, animal_pos]
layer, head = divmod(score.argmax().item(), score.shape[1])
print("number of layers x heads:", tuple(score.shape))
print(f"best 'it'->'animal' head: layer {layer}, head {head}")
print(f"  weight on 'animal' = {att[layer, head, it_pos, animal_pos].item():.2f}")
print(f"  weight on 'street' = {att[layer, head, it_pos, street_pos].item():.2f}")

# %% 4.3-C
clean = [t.replace("Ġ", "") for t in tokens]
A = att[layer, head].numpy()
plt.figure(figsize=(7, 6))
plt.imshow(A, cmap="viridis")
plt.xticks(range(len(clean)), clean, rotation=90)
plt.yticks(range(len(clean)), clean)
plt.xlabel("looked at (key)")
plt.ylabel("looking (query)")
plt.title(f"GPT-2 self-attention, layer {layer} head {head}")
plt.colorbar()
plt.tight_layout()
plt.show()
