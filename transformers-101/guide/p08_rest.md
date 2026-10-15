# PART 8 — Tools and Resources Map

Prices and versions checked 15 Oct 2026. Star counts are approximate and change daily — marked (verify).

| Tool | What it does | Free / Open-source / Paid | Exact cost + free-tier limits | Who uses it | Platform | Official link (+ stars) | Best for | My verdict |
|---|---|---|---|---|---|---|---|---|
| PyTorch 2.14.1 | Tensors, autograd, `nn`, SDPA/FlashAttention | Open-source (BSD) | Free | Meta, OpenAI, Anthropic, most labs and companies | Win / Linux / macOS | pytorch.org [1] | Building and training models | **Must learn.** The industry default |
| Hugging Face Transformers 5.20 | Load/run/fine-tune 1000s of pretrained models; v5 is PyTorch-only | Open-source (Apache-2.0) | Free | Everyone from students to Google, Meta, startups | All | github.com/huggingface/transformers (~160k ★, verify) [2] | Using real models fast; reading reference `modeling_*.py` code | **Must learn.** Read its model code as a textbook |
| Google Colab | Hosted Jupyter notebooks with GPU | Freemium | Free: T4 when available, ≤12 h sessions, ~90 min idle timeout. Pro $9.99/mo = 100 compute units; Pro+ ~$49.99/mo (verify) [3][4] | Students, Kaggle-style practitioners, researchers prototyping | Browser | colab.research.google.com | This whole guide | **Best free start.** Don't pay until free T4 limits block you weekly |
| Kaggle Notebooks | Hosted notebooks with T4×2 / P100 GPUs | Free | ~30 GPU-h/week (floating; check your settings), 12 h sessions [5] | Students, competition players | Browser | kaggle.com | Backup when Colab has no GPU | **Great backup.** Requires phone verification for GPU |
| Hugging Face Hub | Hosts models, datasets, demo apps (Spaces) | Freemium | Free public repos; PRO $9/mo (more ZeroGPU quota, private storage) [6] | Whole open-model ecosystem | Browser / API | huggingface.co | Downloading models, publishing your own | **Must have an account** for portfolio |
| VS Code + Jupyter ext. | Local editor and notebooks | Free | Free | Most engineers | Win / Linux / macOS | code.visualstudio.com [8] | Local CPU work, real projects | **Use for projects** after the labs |
| bertviz | Interactive attention visualisation in notebooks | Open-source | Free | Educators, interpretability beginners | Python | github.com/jessevig/bertviz (~7k ★, verify) [27] | Seeing heads interactively | Nice to have; your own heatmaps (4.3-C) teach more |
| TransformerLens 3.x | Hook into every activation of a Transformer, ablate/patch | Open-source (MIT) | Free | Interpretability researchers (Anthropic alumni, DeepMind, MATS scholars) | Python | github.com/TransformerLensOrg/TransformerLens [26] | Mechanistic interpretability (Part 7.4 extension) | **The** interpretability tool |
| nanoGPT / nanochat | Minimal, readable GPT training code / full ChatGPT pipeline (~$100 run) | Open-source (MIT) | Free (GPU rental for nanochat full run ≈ $100) | Students, researchers, interview prep | Python | github.com/karpathy/nanoGPT (~56k ★), github.com/karpathy/nanochat (~56k ★, verify) [24] | Seeing a complete real training stack | **Best code to study next** |
| LLMs-from-scratch | Book code: build GPT, pre-train, fine-tune | Open-source | Free code; book is paid (~$50, verify) | Learners, engineers switching to AI | Python / notebooks | github.com/rasbt/LLMs-from-scratch (~95k ★, verify) [25] | Structured self-study | **Best book-style follow-up** |
| FlashAttention (flash-attn) | Fast fused attention kernels (FA2/FA3, FA4 for Blackwell) | Open-source (BSD) | Free | Every big training/inference stack | Linux + NVIDIA | github.com/Dao-AILab/flash-attention [19] | Training/inference speed | Use via PyTorch SDPA first; install directly only on Linux + recent NVIDIA GPUs |
| vLLM | High-throughput LLM serving (paged KV cache, batching) | Open-source (Apache-2.0) | Free | Production inference at many companies | Linux + GPU | github.com/vllm-project/vllm [40] | Serving models as an API | **Next step after this guide** for engineering jobs |
| RunPod / Vast.ai / Lambda | Rent GPUs by the hour | Paid | Roughly $0.2–0.5/h for 16–24 GB GPUs, ~$2–3/h for H100-class (verify — changes weekly) | Indie researchers, startups | Browser + SSH | runpod.io, vast.ai, lambda.ai | nanochat-scale runs | Only when you outgrow free tiers |

**Best free setup:** Google Colab (T4) + Kaggle as backup + Hugging Face free account + GitHub for saving work. This covers 100% of Parts 2–7.

**Best paid setup:** a rented single H100/A100 by the hour (RunPod/Lambda) + local VS Code over SSH, with `torch.compile`, bf16 and FlashAttention. Use it for nanochat-scale experiments.

**Best setup within your budget (free):** the free setup above. Spend ₹0 until you have finished Part 7 and Project 2 (Part 11); then, if you need it, Colab Pro ($9.99/mo) is the cheapest upgrade.

**Open-source repositories worth studying**

1. **karpathy/nanoGPT** — ~300 lines of clean GPT-2 training; your MiniGPT is a simplified version of it.
2. **karpathy/nanochat** — tokenizer → pre-training → SFT → RL → chat UI in one repo; shows how a real ChatGPT-style product is assembled.
3. **huggingface/transformers** `src/transformers/models/llama/modeling_llama.py` and `.../qwen3/modeling_qwen3.py` — production versions of every Part 7 component (RMSNorm, RoPE, GQA, SwiGLU, KV cache).
4. **KellerJordan/modded-nanogpt** — how elite engineers squeeze speed out of training (speedrun records with exact diffs).
5. **TransformerLensOrg/TransformerLens** — the standard toolkit for interpretability research.

# PART 9 — Keep It Working

| Task | How |
|---|---|
| **Save** | In Colab press **Ctrl+S** (or **File > Save**). Notebooks are in Google Drive → **My Drive > Colab Notebooks**. |
| **Back up to GitHub** | In Colab: **File > Save a copy in GitHub** → authorise → choose your repo → **OK**. Do this after every Part. |
| **Save a trained model** | Add a cell: `torch.save(model.state_dict(), "minigpt.pt")`, then mount Drive (`from google.colab import drive; drive.mount("/content/drive")`) and save to `/content/drive/MyDrive/minigpt.pt`. Colab's local disk is **wiped** when the session ends. |
| **Load a saved model** | Re-create the class with the same settings, then `model.load_state_dict(torch.load(PATH, map_location=device))`. |
| **Restart cleanly** | **Runtime > Restart session**, then **Runtime > Run all** — proves your notebook works top-to-bottom. |
| **Undo a bad cell edit** | Click inside the cell and press **Ctrl+Z**; for deleted cells use **Edit > Undo delete cell**. Older versions: **File > Revision history**. |
| **Free GPU memory** | `del model; import gc; gc.collect(); torch.cuda.empty_cache()` — or restart the session. |
| **Update libraries** | `!pip install -q -U transformers` then **Runtime > Restart session**. Re-run Section 2.4 to confirm versions. Pin versions in real projects: `transformers==5.20.0`. |
| **Local Windows env** | Activate with `.\.venv\Scripts\Activate.ps1`; update with `pip install -U transformers`; freeze with `pip freeze > requirements.txt`. |
| **Keep the knowledge** | Every 2 weeks redo the Part 13 self-test without notes; re-derive attention on paper once a month (interview muscle memory). |

# PART 10 — Troubleshooting

| # | Symptom | Likely cause | Exact fix (steps) |
|---|---|---|---|
| 1 | "Cannot connect to GPU backend" | Free GPUs busy or your quota used up | 1. Click **Connect without GPU** and do Parts 4–5 on CPU. 2. Retry later. 3. Or open kaggle.com → **Code > New Notebook** → right panel **Session options > Accelerator > GPU T4 x2** (phone verification needed). |
| 2 | `NameError: name 'X' is not defined` | A cell earlier in the notebook was not run (or the session restarted) | Click the failing cell → **Runtime > Run before** → run the failing cell again. |
| 3 | `CUDA out of memory` | Batch or model too large, or old models still in memory | 1. `del` old models + `torch.cuda.empty_cache()`. 2. Halve `BATCH_SIZE`. 3. **Runtime > Restart session** and run only the needed cells. |
| 4 | `RuntimeError: Expected all tensors to be on the same device` | Model on GPU, data on CPU (or the reverse) | Add `.to(device)` to the model **and** every input tensor (see `get_batch`). |
| 5 | `mat1 and mat2 shapes cannot be multiplied` / `shape '[2, 6, 4, 8]' is invalid` | Wrong `d_model`, `n_heads` or a `view` that does not match the data | Print `.shape` before the failing line; check `d_model % n_heads == 0`; re-read the shape table in 5.1.2. |
| 6 | Loss becomes `nan` | float16 overflow, learning rate too high, or a fully-masked attention row | 1. Use `GradScaler` with float16 (Part 7). 2. Lower LR ×0.3. 3. Make sure every query can see at least itself (Section 5.2.3 warning). |
| 7 | Loss drops to ~0 instantly but generated text is garbage | Causal mask missing → model copies the visible next token | Re-run the leakage test (5.2-B) on your model; pass `mask` into every block. |
| 8 | Generated text is one repeated character/word | Greedy decoding or very low temperature on a small model | Use `temperature=0.8`, `top_k=20` (6.3-E); train longer. |
| 9 | `OSError: ... is not a valid model identifier` or HTTP 401/429 | Typo in model name, gated model, or rate limit | Copy the exact name from huggingface.co; for gated models accept the licence on the model page and log in (`from huggingface_hub import login; login()`); for 429 wait a minute. |
| 10 | `TypeError: ... got an unexpected keyword argument 'dtype'` | Old Transformers (v4.x) in Colab | `!pip install -q -U transformers` → **Runtime > Restart session** (Section 2.4). |

# PART 11 — Practice Projects

## Project 1 (easy) — "Attention Explainer" notebook

**Goal:** a notebook that takes any sentence and shows how a real model's heads attend.

**Requirements:**

- Input box (a Python variable is fine) for a sentence; works with `openai-community/gpt2` and `google-bert/bert-base-uncased`.
- Plot the attention map for any chosen layer/head, plus a 12×12 grid of "how much does token A attend to token B" for a chosen pair.
- A written paragraph explaining one head's behaviour, with numbers.

**Success criteria:** runs top-to-bottom after **Runtime > Restart session > Run all**; handles words that split into several tokens without crashing; explanation cites measured weights.

## Project 2 (medium) — Train and publish a tiny decoder-only model on your own text

**Goal:** train your Part 7 ModernGPT on a text corpus you choose (e.g. Kannada or English public-domain books, your notes, song lyrics with permission) and publish it.

**Requirements:**

- A tokenizer that is not character-level (train a BPE tokenizer with the Hugging Face `tokenizers` library, 2k–8k vocab).
- Train/validation split, a logged loss curve, KV-cache generation.
- An ablation: compare 2 settings (e.g. GQA 2 vs 8 KV heads, or RoPE vs learned positions) at a fixed token budget.
- Upload weights + a model card to the Hugging Face Hub; code on GitHub with a README.

**Success criteria:** validation loss clearly below the unigram baseline; ablation table with ≥2 seeds; someone else can reproduce your numbers from the README.

## Project 3 (hard) — Encoder–decoder for a real task, with an attention report

**Goal:** build a from-scratch encoder–decoder Transformer (your Part 6.4 code, scaled up) that solves a real sequence-to-sequence task — e.g. English → Kannada transliteration, date-format normalisation, or SQL generation from short questions.

**Requirements:**

- Proper padding masks for variable-length inputs and outputs; beam search **or** sampling in addition to greedy decoding.
- Compare against fine-tuning `google-t5/t5-small` on the same data.
- Cross-attention visualisations for correct and incorrect predictions, with an error analysis.
- A short technical report (2–4 pages) with results table and lessons learned.

**Success criteria:** reported exact-match or BLEU/chrF on a held-out test set; honest comparison vs T5; error analysis that uses the cross-attention maps; project in your GitHub portfolio with a clean README.

# PART 12 — Cheat Sheet

## Part 1–2 — Setup and basics

```text
Colab URL ............ colab.research.google.com
GPU on ............... Runtime > Change runtime type > T4 GPU > Save
Run cell ............. Shift+Enter        New cell: + Code      Save: Ctrl+S
Restart .............. Runtime > Restart session     Run all: Runtime > Run all
Run earlier cells .... Runtime > Run before
Save to GitHub ....... File > Save a copy in GitHub
```

```python
!pip install -q -U transformers           # then Runtime > Restart session
import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"
```

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers matplotlib numpy jupyter
```

## Part 3 — Key facts

- `Attention(Q,K,V) = softmax(Q Kᵀ / √d_k) V`
- Self-attention: Q, K, V from the same sequence. Cross-attention: Q from decoder, K/V from encoder.
- Multi-head: `d_head = d_model / n_heads`; split → attend → concat → `W_O`.
- Mask: blocked score = −∞ → weight 0. Causal = `torch.tril(ones(T,T))`.
- Attention without positions is order-blind (permutation-equivariant).
- Cost of attention ∝ T² (sequence length squared).
- Encoder-only = understand (BERT). Decoder-only = generate (GPT). Encoder–decoder = transform (T5).

## Part 4 — Embeddings and attention

```python
emb = nn.Embedding(vocab_size, d_model)            # ID -> vector (a learned lookup table)
X = emb(torch.tensor(ids))                         # == F.one_hot(ids, V).float() @ emb.weight
F.cosine_similarity(a, b, dim=0)                   # similarity in [-1, 1]

def attention(Q, K, V, mask=None):
    scores = Q @ K.transpose(-2, -1) / Q.size(-1) ** 0.5
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    w = torch.softmax(scores, dim=-1)
    return w @ V, w

F.scaled_dot_product_attention(Q, K, V, is_causal=True)   # official fused version
model = AutoModel.from_pretrained(name, attn_implementation="eager")  # to get attention weights
out = model(**enc, output_attentions=True); out.attentions  # tuple: one (B, H, T, T) per layer
```

Worked example to remember: keys `[1,0],[0,1],[1,1]`, query `[1,0]` → scores `[1,0,1]` → weights `[0.42,0.16,0.42]` → output `[6.33,3.67]`.

## Part 5 — Multi-head, masks, positions, block

```python
# split heads / merge heads
x.view(B, T, H, D // H).transpose(1, 2)            # (B,T,D) -> (B,H,T,Dh)
out.transpose(1, 2).contiguous().view(B, T, D)     # (B,H,T,Dh) -> (B,T,D)

causal = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
padding = (tokens != PAD).view(B, 1, 1, T)
combined = causal * padding

# sinusoidal PE
freq = torch.exp(-math.log(10000.0) * torch.arange(0, d, 2) / d)
pe[:, 0::2] = torch.sin(pos * freq); pe[:, 1::2] = torch.cos(pos * freq)

# RoPE (rotate pairs)
x1, x2 = x[..., 0::2], x[..., 1::2]
out[..., 0::2] = x1 * cos - x2 * sin;  out[..., 1::2] = x1 * sin + x2 * cos

# pre-norm block
x = x + attn(ln1(x), mask=mask)
x = x + ffn(ln2(x))                                # ffn: d -> 4d -> GELU -> d
```

## Part 6 — Families and training

```python
AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")       # encoder-decoder
pipeline("fill-mask", model="google-bert/bert-base-uncased")       # encoder-only
AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype="auto")  # decoder-only
q_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# mean pooling for sentence embeddings
vecs = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)

# next-token batches
x = data[i:i+T];  y = data[i+1:i+T+1]

# training loop
logits, loss = model(x, y)
optimizer.zero_grad(set_to_none=True); loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()

# cross-attention = MHA with different K/V source
c, w = cross_attn(ln2(x), x_kv=memory, mask=src_mask)
```

Key numbers: random-guess loss = ln(vocab) (ln 65 = 4.17); MiniGPT 3.23 M params → val ≈ 1.5–1.6 after 3,000 steps on T4.

## Part 7 — Elite

```python
rms = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight   # RMSNorm
ffn = w_down(F.silu(w_gate(x)) * w_up(x))                               # SwiGLU
k = k.repeat_interleave(n_heads // n_kv_heads, dim=1)                   # GQA
k = torch.cat([cache_k, k_new], dim=2)                                  # KV cache append
with torch.autocast(device_type="cuda", dtype=torch.float16): ...       # mixed precision
scaler = torch.amp.GradScaler("cuda")                                   # with float16
torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
```

- KV cache bytes = `2 × layers × kv_heads × head_dim × tokens × batch × bytes_per_value`
- 8B-class model, 32k tokens, fp16: MHA ≈ 17.2 GB, GQA(8) ≈ 4.3 GB, MQA ≈ 0.5 GB.
- Induction head test: `[BOS] + R + R`, score attention from position i to i − L + 1.
- Modern recipe: pre-RMSNorm (+QK-norm), RoPE, SwiGLU, GQA/MLA, FlashAttention, MoE, KV cache, hybrid/sparse attention for long context.

# PART 13 — Self-Test

1. **What happens if** you remove the `/ √d_k` scaling in a model with `d_head = 512`? Describe the effect on the attention weights and on training.
2. **Do this:** by hand, compute attention for keys `[[2,0],[0,2]]`, values `[[1,1],[3,5]]`, query `[1,1]` (no scaling). Give the weights and the output.
3. **What happens if** you train MiniGPT **without** the causal mask? What will the training loss and the generated text look like, and why?
4. **Do this:** write the one line of PyTorch that turns `(B, T, D)` into `(B, H, T, D/H)` correctly.
5. **What happens if** you shuffle the order of the words in a sentence fed to a self-attention layer that has **no** positional encoding?
6. You need a model to power **semantic search over company documents**. Which family do you choose and why? What pooling do you apply?
7. **Do this:** compute the KV-cache size in GB for a model with 32 layers, 8 KV heads, head_dim 128, at 8,192 tokens, batch 4, in 16-bit precision.
8. In cross-attention, which sequence provides Q and which provides K and V? **What happens if** you swap them?
9. **What happens if** you apply RoPE with `start_pos = 0` for every generated token while using a KV cache?
10. **Do this:** name the five changes that turn a GPT-2-style block into a Llama-3/Qwen3-style block, and give one reason for each.

# PART 14 — Sources

1. PyTorch 2.14.1 General Availability (30 Sep 2026) — https://dev-discuss.pytorch.org/t/pytorch-2-14-1-general-availability/3455
2. Hugging Face Transformers releases — https://github.com/huggingface/transformers/releases · PyPI: https://pypi.org/project/transformers/ (5.20.0 installed and tested for this guide)
3. Google Colab — https://colab.research.google.com · plans: https://colab.research.google.com/signup
4. Colab FAQ (limits) — https://research.google.com/colaboratory/faq.html
5. Kaggle Notebooks docs — https://www.kaggle.com/docs/notebooks
6. Hugging Face pricing — https://huggingface.co/pricing
7. Python for Windows — https://www.python.org/downloads/windows/
8. Visual Studio Code — https://code.visualstudio.com/
9. Vaswani et al., *Attention Is All You Need* (2017) — https://arxiv.org/abs/1706.03762
10. Bahdanau, Cho, Bengio, *Neural Machine Translation by Jointly Learning to Align and Translate* (2014) — https://arxiv.org/abs/1409.0473
11. Devlin et al., *BERT* (2018) — https://arxiv.org/abs/1810.04805
12. Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2, 2019) — https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
13. Raffel et al., *T5* (2019) — https://arxiv.org/abs/1910.10683
14. Su et al., *RoFormer: Rotary Position Embedding* (2021) — https://arxiv.org/abs/2104.09864
15. Ainslie et al., *GQA* (2023) — https://arxiv.org/abs/2305.13245
16. Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need* (MQA, 2019) — https://arxiv.org/abs/1911.02150
17. Shazeer, *GLU Variants Improve Transformer* (SwiGLU, 2020) — https://arxiv.org/abs/2002.05202
18. Zhang & Sennrich, *Root Mean Square Layer Normalization* (2019) — https://arxiv.org/abs/1910.07467
19. Dao et al., *FlashAttention* (2022) — https://arxiv.org/abs/2205.14135 · repo: https://github.com/Dao-AILab/flash-attention · FlashAttention-4 (Mar 2026): https://arxiv.org/abs/2603.05451 · https://together.ai/blog/flashattention-4
20. DeepSeek-AI, *DeepSeek-V3 Technical Report* (MLA, MoE) — https://arxiv.org/abs/2412.19437
21. Olsson et al., *In-context Learning and Induction Heads* (Anthropic, 2022) — https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
22. Elhage et al., *A Mathematical Framework for Transformer Circuits* (2021) — https://transformer-circuits.pub/2021/framework/index.html
23. Jay Alammar, *The Illustrated Transformer* — https://jalammar.github.io/illustrated-transformer/
24. Karpathy: nanoGPT — https://github.com/karpathy/nanoGPT · nanochat — https://github.com/karpathy/nanochat · *Let's build GPT* video — https://www.youtube.com/watch?v=kCc8FmEb1nY
25. Sebastian Raschka: LLMs-from-scratch — https://github.com/rasbt/LLMs-from-scratch · *The Big LLM Architecture Comparison* — https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison · LLM Architecture Gallery (hybrid attention) — https://sebastianraschka.com/llm-architecture-gallery/hybrid-attention/ · *10 Architectures from Jan–Feb 2026* — https://magazine.sebastianraschka.com/p/a-dream-of-spring-for-open-weight
26. TransformerLens — https://github.com/TransformerLensOrg/TransformerLens
27. bertviz — https://github.com/jessevig/bertviz
28. Tiny Shakespeare dataset — https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
29. Models used: https://huggingface.co/openai-community/gpt2 · https://huggingface.co/google-bert/bert-base-uncased · https://huggingface.co/google-t5/t5-small · https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 · https://huggingface.co/Qwen/Qwen3-0.6B
30. Google Docs — Use Markdown (Paste from Markdown) — https://support.google.com/docs/answer/12014036
31. Qwen3-Next / hybrid attention overview — https://sebastianraschka.com/llm-architecture-gallery/hybrid-attention/ · Qiu et al., *Gated Attention for Large Language Models* — https://arxiv.org/abs/2505.06708 (verify)
32. Gu & Dao, *Mamba* (2023) — https://arxiv.org/abs/2312.00752
33. PyTorch FlexAttention + FlashAttention-4 backend (PyTorch blog, Mar 2026) — https://pytorch.org/blog/ (verify exact post)
34. PyTorch `scaled_dot_product_attention` docs — https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
35. Harvard NLP, *The Annotated Transformer* — https://nlp.seas.harvard.edu/annotated-transformer/
36. 3Blue1Brown, Neural networks / Transformers series — https://www.3blue1brown.com/topics/neural-networks
37. Free cloud GPU overview 2026 (Kaggle quota notes, third-party) — https://gpuperhour.com/blog/free-cloud-gpus-and-credits
38. Fedus et al., *Switch Transformers* (MoE, 2021) — https://arxiv.org/abs/2101.03961
39. Colab pricing summaries (third-party, 2026) — https://www.hivenet.com/post/google-colaboratory-gpu-complete-guide-to-free-cloud-gpu-access-and-limitations · https://gpuperhour.com/blog/google-colab-alternatives
40. vLLM — https://github.com/vllm-project/vllm
41. Hugging Face LLM Course — https://huggingface.co/learn/llm-course
42. Stanford CS336 *Language Modeling from Scratch* — https://stanford-cs336.github.io/
43. modded-nanogpt — https://github.com/KellerJordan/modded-nanogpt
44. Warner et al., *ModernBERT* (2024) — https://arxiv.org/abs/2412.13663
45. Radford et al., *Whisper* (2022) — https://arxiv.org/abs/2212.04356
46. Xiong et al., *On Layer Normalization in the Transformer Architecture* (2020) — https://arxiv.org/abs/2002.04745

# ANSWER KEY

1. Scores have a standard deviation of about √512 ≈ 22, so softmax becomes nearly one-hot (one key ≈100%, the rest ≈0%, as in 4.2-B). Gradients through the near-zero weights vanish, attention cannot learn to spread or shift, and training is slow or unstable.
2. Scores: `[1,1]·[2,0] = 2`, `[1,1]·[0,2] = 2` → softmax `[0.5, 0.5]` → output `0.5·[1,1] + 0.5·[3,5] = [2, 3]`.
3. Each position can see the next character, so the model learns to copy it: training loss falls towards ~0 very quickly, but at generation time the future does not exist yet, so the output is garbage. (Symptom row 7, Part 10.)
4. `x.view(B, T, H, D // H).transpose(1, 2)`
5. The outputs are the same vectors in the shuffled order (permutation-equivariance, 5.3-A). The model cannot tell "dog bites man" from "man bites dog".
6. Encoder-only (bidirectional, trained for embeddings — e.g. a sentence-transformers or BGE/E5/Qwen3-Embedding model). Use the model's documented pooling, usually mean pooling over non-padding tokens (with the attention mask), then L2-normalise and rank by cosine similarity.
7. `2 × 32 × 8 × 128 × 8192 × 4 × 2 bytes = 4,294,967,296 bytes ≈ 4.29 GB`.
8. Q from the decoder (the side writing); K and V from the encoder output (the side being read). If swapped, the output has the *source* length and is about the wrong thing: the residual add fails with a shape error (or, if lengths happen to match, the model cannot learn the task).
9. Every new token is rotated as if it were at position 0, so relative distances between the new token and cached keys are wrong; quality degrades rapidly and outputs no longer match no-cache generation (`identical text: False`).
10. (1) LayerNorm → **RMSNorm** (cheaper, as stable); (2) learned/sinusoidal positions → **RoPE** (relative distance, longer context); (3) GELU MLP → **SwiGLU** (better quality per parameter); (4) MHA → **GQA** (smaller KV cache, faster inference); (5) naive attention → **fused SDPA/FlashAttention** with a **KV cache** at inference (speed and memory). Also acceptable: QK-norm, no biases, MoE, weight-tying choice.
