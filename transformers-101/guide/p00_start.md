<div class="doctitle">Transformers 101 — Zero to Elite Lab Manual</div>

<div class="subtitle">Embeddings · Attention · Self-Attention · Multi-Head · Masked · Positional Encoding · Encoder–Decoder · Encoder-Only · Decoder-Only · Cross-Attention</div>

<div class="assume"><b>Assumptions:</b> you are new to Transformers but can write basic Python and know what a neural network layer is (you covered ANN/CNN/NLP earlier in your course). You use a Windows 11 laptop with Chrome. Your goal is to be job-ready as an AI engineer and to build real projects. Budget: free (Google Colab free tier). Versions checked on 15 Oct 2026: PyTorch 2.14.1 [1], Hugging Face Transformers 5.20.0 [2].</div>

[TOC]

# PART 0 — Start Here

## 0.1 What a Transformer is

A **Transformer** is a type of neural network that reads a whole piece of text at once, works out which words matter to each other, and uses that to understand the text or write the next word. Its core trick is called **attention**: every word "looks at" every other word and pulls in the information it needs. It was introduced in the 2017 paper *Attention Is All You Need* [9], and it is the engine inside ChatGPT, Claude, Gemini, Llama, Qwen, DeepSeek, BERT, Whisper and most image and video generators.

**Why people use it:** it learns long-range relationships ("*it*" refers to "*the animal*" ten words earlier), it trains in parallel on GPUs (fast), and it keeps getting better as you add data and compute.

**Where it is used in the real world:**

| Area | What the Transformer does | Example products |
|---|---|---|
| Chat and assistants | Generates answers word by word | ChatGPT, Claude, Gemini |
| Search and RAG | Turns text into vectors so similar meaning = nearby vectors | Google Search, enterprise RAG bots |
| Code | Writes and reviews code | GitHub Copilot, Claude Code, Cursor |
| Translation | Reads one language, writes another | Google Translate, DeepL |
| Speech | Audio in, text out | Whisper, phone voice typing |
| Images / video | Attention over image patches | Vision Transformers, diffusion transformers |

**Brutally honest career note:** knowing Transformer internals will not get you a job on its own — every AI engineer interview assumes it. What it *does* is let you debug, fine-tune, quantize, serve and explain models instead of only calling APIs. That is the difference between a "prompt person" and an engineer companies pay well for. Every lab here maps to a common interview question (see Part 13).

## 0.2 The 10 topics and where they live in this guide

![Big picture](diagrams/d01_big_picture.svg)

| # | Your topic | Guide section |
|---|---|---|
| 1 | Embeddings: from discrete to continuous space | Part 4 · 4.1 |
| 2 | The attention mechanism | Part 4 · 4.2 |
| 3 | Self-attention | Part 4 · 4.3 |
| 4 | Multi-head attention | Part 5 · 5.1 |
| 5 | Masked multi-head attention | Part 5 · 5.2 |
| 6 | Positional encoding | Part 5 · 5.3 (+ 5.4 full block) |
| 7 | Encoder–decoder Transformers | Part 6 · 6.1 (built from scratch in 6.4) |
| 8 | Encoder-only Transformers | Part 6 · 6.2 |
| 9 | Decoder-only Transformers | Part 6 · 6.3 |
| 10 | Cross-attention | Part 6 · 6.4 |
| — | How elite labs build them in 2026 | Part 7 |

## 0.3 Plan and time

| Part | What you will do | Time estimate |
|---|---|---|
| 0 Start Here | Understand the map | 20 min |
| 1 Prerequisites | Refresh 7 must-know ideas, prove each with a 1-minute check | 1–2 h |
| 2 Workspace | Set up Google Colab with a free GPU, run a test | 30 min |
| 3 Core Idea | One analogy + the 20% that explains 80% | 30 min |
| 4 Basics | Topics 1–3: embeddings, attention by hand, self-attention in real GPT-2 | 3–4 h |
| 5 Intermediate | Topics 4–6: build multi-head, masks, positional encodings, a full block | 4–5 h |
| 6 Advanced | Topics 7–10: use T5/BERT/Qwen, train your own GPT, build cross-attention | 6–8 h |
| 7 Elite | RoPE + RMSNorm + SwiGLU + GQA + KV cache; find induction heads in GPT-2 | 6–8 h |
| 8–14 | Tools map, maintenance, troubleshooting, projects, cheat sheet, self-test | 3–4 h |
| **Total** | | **≈ 25–35 hours** (2–3 weeks part-time) |

## 0.4 What you will be able to do at the end

1. Explain, with a drawing and numbers, how text becomes vectors and how attention mixes them — the standard whiteboard interview question.
2. Write multi-head attention, causal masking, sinusoidal positions and RoPE from scratch in PyTorch and prove they match PyTorch's official versions.
3. Train your own small GPT on Shakespeare and build an encoder–decoder model with cross-attention that you can inspect.
4. Pick the right family for a job (BERT-style for search/classification, T5-style for input→output, GPT-style for generation) and use real models from Hugging Face.
5. Upgrade a 2017 design to a 2026 "Llama/Qwen-style" design (RoPE, RMSNorm, SwiGLU, GQA, KV cache) and measure the speed-up.
6. Read modern architecture papers and run a mechanistic-interpretability experiment (induction heads).

## 0.5 Environment map

![Environment map](diagrams/d00_environment.svg)

**One-line flow:** your Windows laptop → Chrome → Google Colab notebook running on Google's free T4 GPU → downloads models from Hugging Face Hub and data from GitHub → you save notebooks to Google Drive (and optionally push code to GitHub).

| Place | What you do there |
|---|---|
| Chrome on Windows 11 | Open Colab, Hugging Face, this guide |
| Google Colab (colab.research.google.com) | Write and run every lab, use the free GPU |
| Hugging Face Hub (huggingface.co) | Source of pretrained models (GPT-2, BERT, T5, MiniLM, Qwen3) |
| GitHub raw file | Source of the Tiny Shakespeare training text [28] |
| Google Drive | Where Colab stores your notebooks |
| VS Code on Windows (optional) | Run the same code locally on CPU |

## 0.6 What each piece does

| Component | Its job |
|---|---|
| Tokenizer | Cuts text into tokens (words or word-pieces) and gives each an ID number |
| Embedding table | Turns each token ID into a list of numbers (a vector) that carries meaning |
| Positional encoding | Adds "where in the sentence" information to each vector |
| Attention | Lets each token pull information from other tokens, weighted by relevance |
| Multi-head attention | Runs several attentions in parallel so each can track a different relationship |
| Mask | Blocks attention to tokens that must not be seen (future words, padding) |
| Feed-forward network (MLP) | Lets each token "think" on its own after it has gathered information |
| Residual connection + LayerNorm | Keep training stable when you stack many layers |
| Encoder | A stack of blocks that reads the full input with no mask |
| Decoder | A stack of blocks that writes output one token at a time with a causal mask |
| Cross-attention | The bridge where the decoder reads the encoder's output |
| LM head (Linear + softmax) | Converts the final vector into a probability for every word in the vocabulary |
| PyTorch | The library you use to build and train the networks |
| Hugging Face Transformers | The library that downloads and runs pretrained models in a few lines |
| Google Colab | Free online Python notebook with a GPU |

# PART 1 — Prerequisites

## 1.1 Checklist

| Item | Level |
|---|---|
| Python basics: variables, lists, dicts, functions, classes, loops | MUST |
| Tensors / arrays and their **shape** | MUST |
| Matrix multiplication (and transpose) | MUST |
| Dot product and cosine similarity | MUST |
| Softmax | MUST |
| What a neural network layer, loss and gradient descent are | MUST |
| PyTorch basics: `nn.Module`, `forward`, `loss.backward()`, optimizer | MUST |
| Logarithms and probability (why loss = −log p) | NICE-TO-HAVE |
| Git and GitHub | NICE-TO-HAVE |
| A Hugging Face account | NICE-TO-HAVE (needed only to upload your own models) |

## 1.2 Refreshers and 1-minute checks

Do each check in a Colab cell (you will set Colab up in Part 2 — if you have never used it, do Part 2 first and come back).

**1. Python classes.** A class bundles data and functions. In PyTorch every model is a class with an `__init__` (create layers) and a `forward` (use them).

*1-minute check* — predict the output, then run it:

```python
class Doubler:
    def __init__(self, k):
        self.k = k
    def forward(self, x):
        return x * 2 + self.k

d = Doubler(1)
print(d.forward(5))
```

Correct answer: `11`. If you guessed wrong, revise Python classes for one hour before continuing.

**2. Shapes.** A tensor is an n-dimensional array. Its *shape* says how many numbers sit on each axis. In this guide you will constantly see `(B, T, d)` = (batch of sentences, tokens per sentence, numbers per token).

*1-minute check:*

```python
import torch
x = torch.zeros(2, 6, 32)
print(x.shape, x[0].shape, x[:, -1].shape)
```

Correct answer: `torch.Size([2, 6, 32]) torch.Size([6, 32]) torch.Size([2, 32])`.

**3. Matrix multiplication.** `(n × k) @ (k × m)` gives `(n × m)`. The inner sizes must match. Transpose `.T` swaps rows and columns.

*1-minute check:* by hand, what is `[[1, 2], [3, 4]] @ [[1], [1]]`? Answer: `[[3], [7]]`. Then verify:

```python
import torch
print(torch.tensor([[1., 2.], [3., 4.]]) @ torch.tensor([[1.], [1.]]))
```

**4. Dot product and cosine similarity.** Dot product `a·b` = multiply matching numbers and add. Large positive = the vectors point the same way. Cosine similarity = dot product divided by both lengths, so it is always between −1 and 1.

*1-minute check:* `[1, 2] · [3, 4]` = 1×3 + 2×4 = **11**.

```python
import torch, torch.nn.functional as F
a, b = torch.tensor([1., 2.]), torch.tensor([3., 4.])
print(a @ b, F.cosine_similarity(a, b, dim=0))
```

Correct answer: `tensor(11.) tensor(0.9839)`.

**5. Softmax.** Turns any list of scores into positive numbers that add up to 1: `softmax(x)_i = e^(x_i) / Σ e^(x_j)`. Bigger score → bigger share.

*1-minute check:* softmax of `[0, 0]` is `[0.5, 0.5]`. Softmax of `[1, 0, 1]` ≈ `[0.42, 0.16, 0.42]` — you will reuse this exact example in Part 4.

```python
import torch
print(torch.softmax(torch.tensor([1., 0., 1.]), dim=0))
```

**6. Layer, loss, gradient descent.** A linear layer computes `y = xW + b`. The loss measures how wrong the output is. `loss.backward()` computes how each weight should change; the optimizer nudges the weights a little in that direction. Repeat thousands of times = training.

*1-minute check:* in words, what does `optimizer.zero_grad()` do? Answer: it clears the gradients from the previous step so they do not add up.

**7. PyTorch training loop.** Every training loop in this guide has the same 5 lines: get a batch → forward → loss → backward → optimizer step.

*1-minute check* — run it and confirm the loss goes down:

```python
import torch, torch.nn as nn
torch.manual_seed(0)
model = nn.Linear(1, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
x = torch.randn(64, 1); y = 3 * x + 1
for step in range(51):
    loss = ((model(x) - y) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 25 == 0: print(step, round(loss.item(), 4))
print(model.weight.item(), model.bias.item())
```

You should see the loss shrink towards 0 and the weight/bias approach 3 and 1.

## 1.3 What you must install, buy or sign up for

| Item | Official link | Cost | Which option to choose |
|---|---|---|---|
| Google account | accounts.google.com | Free | Any personal Gmail account |
| Google Colab | colab.research.google.com [3] | Free tier: T4 GPU when available, ~12 h max session, idle disconnect ~90 min [3][4] | **Free tier.** Colab Pro is $9.99/month for 100 compute units (verify on the signup page) — not needed for this guide |
| Chrome browser | google.com/chrome | Free | Latest version |
| Hugging Face account | huggingface.co/join [6] | Free (PRO is $9/month — not needed) | Free; only needed to upload your own models later |
| GitHub account (optional) | github.com/signup | Free | Free; to keep your notebooks under version control |
| Kaggle account (backup GPU) | kaggle.com [5] | Free, ~30 GPU-hours/week (floating, check your settings page) | Use only if Colab says no GPU is available |
| Python for Windows (optional) | python.org/downloads/windows [7] | Free | **Python 3.13** 64-bit. Newest is 3.15.0 (9 Oct 2026) but PyTorch wheels for a brand-new Python version can lag — 3.13 is the safe choice (verify on pytorch.org/get-started) |
| VS Code (optional) | code.visualstudio.com [8] | Free | "Windows x64 User Installer" + the **Python** and **Jupyter** extensions |

Nothing in Parts 2–7 costs money.

# PART 2 — Set Up Your Workspace

## SECTION 2.1 — Open Google Colab

**WHERE:** Chrome on your Windows laptop
**GOAL:** Get a new, empty Colab notebook in front of you.

**Step 1.** Open Chrome.

**Step 2.** Click the address bar at the top.

**Step 3.** Type exactly and press **Enter**:

```text
colab.research.google.com
```

**Step 4.** If you are not signed in, click **Sign in** (top-right) and sign in with your Google account.

**Step 5.** In the "Open notebook" window, click **+ New notebook** (bottom-left). If no window appears, click **File > New notebook in Drive**.

> **You should see:** a page titled `Untitled0.ipynb` with one empty grey code cell and a ▶ button on its left.

**Step 6.** Click the title `Untitled0.ipynb` at the top-left.

**Step 7.** Type a new name and press **Enter**:

```text
T101_Part2_Setup.ipynb
```

> **Watch out:** Colab notebooks live in Google Drive under **My Drive > Colab Notebooks**. If you open a notebook from someone else's link, it is *not* saved for you until you click **File > Save a copy in Drive**.

**CHECKPOINT:** The tab shows `T101_Part2_Setup.ipynb` and there is one empty code cell.
*If not:* you are probably on the Colab welcome page — click **File > New notebook in Drive**.

## SECTION 2.2 — Switch on the free GPU

**WHERE:** Colab, the notebook from 2.1
**GOAL:** Run your code on an NVIDIA T4 GPU instead of the CPU.

**Step 1.** Click **Runtime** in the top menu.

**Step 2.** Click **Change runtime type**.

**Step 3.** Fill the dialog using this table:

| Field | Value | Why |
|---|---|---|
| Runtime type | Python 3 | All code in this guide is Python |
| Hardware accelerator | **T4 GPU** | Free GPU, 15–16 GB memory, enough for every lab |

**Step 4.** Click **Save**.

**Step 5.** Click **Connect** (top-right) if it does not connect automatically.

> **You should see:** a green tick and "RAM / Disk" bars at the top-right.

| If you see... | It means... | Do this next |
|---|---|---|
| "Cannot connect to GPU backend" / "GPU not available" | Free GPUs are all busy right now | Click **Connect without GPU** (Parts 4–5 run fine on CPU), try again later, or use Kaggle (Part 10) |
| "T4 GPU" is greyed out | You hit the free usage limit for today | Wait several hours or use Kaggle |

**CHECKPOINT:** The runtime is connected and **Runtime > Change runtime type** shows **T4 GPU** selected.

## SECTION 2.3 — The "run a cell" routine (you will use it 100+ times)

**WHERE:** Colab notebook
**GOAL:** Learn the exact 3 actions used for every code block in this guide.

**Step 1.** Click **+ Code** (top-left, under the menu). A new empty cell appears.

**Step 2.** Click inside the new cell.

**Step 3.** Paste the code block from the guide (**Ctrl+V**).

**Step 4.** Press **Shift+Enter** to run the cell.

> **You should see:** a spinning circle on the cell's ▶ button while it runs, then the output under the cell.

> **Watch out:** cells share memory **in the order you run them**, not the order on screen. If you get `NameError: name 'X' is not defined`, you skipped a cell — click **Runtime > Run before** on the failing cell, or run the missing cell.

**CHECKPOINT:** You know where **+ Code** is and that **Shift+Enter** runs a cell.

## SECTION 2.4 — Install / upgrade the libraries and run the test

**WHERE:** Colab notebook `T101_Part2_Setup.ipynb`
**GOAL:** Prove PyTorch, the GPU and Hugging Face Transformers all work.

**Step 1.** Click **+ Code**.

**Step 2.** Paste exactly:

```python
!pip install -q -U transformers
```

**Step 3.** Press **Shift+Enter**.

> **You should see:** nothing, or a few quiet lines, after 10–60 seconds. A red "pip's dependency resolver" warning is normal and harmless.

**Step 4.** Click **Runtime > Restart session**, then click **Yes** in the dialog. (Older Colab versions call this "Restart runtime".)

**Step 5.** Click **+ Code**.

**Step 6.** Paste exactly:

```python
import sys, torch, transformers
print("Python      :", sys.version.split()[0])
print("PyTorch     :", torch.__version__)
print("Transformers:", transformers.__version__)
print("GPU         :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU only)")
```

**Step 7.** Press **Shift+Enter**.

> **You should see** something like this (exact versions may differ — that is fine):

```text
Python      : 3.12.x
PyTorch     : 2.x.x+cu12x
Transformers: 5.20.0
GPU         : Tesla T4
```

**Step 8.** Click **+ Code**.

**Step 9.** Paste exactly — this is the "tiny test that proves everything works": it downloads GPT-2 from Hugging Face and asks it to continue a sentence.

```python
from transformers import pipeline
gen = pipeline("text-generation", model="openai-community/gpt2", device=0 if torch.cuda.is_available() else -1)
print(gen("A transformer is a neural network that", max_new_tokens=20, do_sample=False)[0]["generated_text"])
```

**Step 10.** Press **Shift+Enter**.

> **You should see:** download progress bars (about 550 MB), then this text (greedy decoding, so it should match exactly):

```text
A transformer is a neural network that can be used to generate a neural network.

The neural network is a neural network that can
```

> The text is repetitive and a bit silly — GPT-2 is small and from 2019. That is fine; the point is that everything works. A yellow "unauthenticated requests to the HF Hub" warning is harmless.

| If you see... | It means... | Do this next |
|---|---|---|
| `Transformers: 4.x` | The upgrade did not apply | Repeat Steps 1–4, making sure you **restarted the session** |
| `GPU: none (CPU only)` | GPU not attached | Redo Section 2.2 |
| `HTTPError 429` or slow download | Hugging Face rate limit | Wait 1 minute and run again |

**CHECKPOINT:** Transformers version is 5.x, GPU says `Tesla T4` (or you consciously chose CPU), and GPT-2 printed a sentence.
*If not:* see Part 10, rows 1–3.

## SECTION 2.5 — Optional: local setup on Windows 11 (CPU)

**WHERE:** Windows 11 — browser, then **Windows Terminal / PowerShell**, then **VS Code**
**GOAL:** Run Parts 4–5 locally without internet GPU. (Parts 6–7 train faster on Colab's GPU.)

**Step 1.** In Chrome go to `https://www.python.org/downloads/windows/` [7].

**Step 2.** Click the newest **Python 3.13.x** link, then **Windows installer (64-bit)**.

**Step 3.** Run the downloaded installer.

**Step 4.** Tick **Add python.exe to PATH** at the bottom of the first screen.

**Step 5.** Click **Install Now** and wait for "Setup was successful", then click **Close**.

**Step 6.** Press the **Windows** key, type `PowerShell`, press **Enter**.

**Step 7.** Type and press **Enter**:

```powershell
mkdir $HOME\t101; cd $HOME\t101; python -m venv .venv
```

**Step 8.** Type and press **Enter**:

```powershell
.\.venv\Scripts\Activate.ps1
```

> **Watch out:** if you get "running scripts is disabled on this system", run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, answer `Y`, then repeat Step 8.

**Step 9.** Type and press **Enter** (CPU-only PyTorch is smaller and works on any laptop):

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Step 10.** Type and press **Enter**:

```powershell
pip install transformers matplotlib numpy jupyter
```

**Step 11.** Type and press **Enter**:

```powershell
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

> **You should see:** `2.14.1+cpu 5.20.0` (or newer).

**Step 12.** Install VS Code from `https://code.visualstudio.com/` [8], open it, press **Ctrl+Shift+X**, search **Python**, click **Install**; search **Jupyter**, click **Install**.

**Step 13.** In VS Code click **File > Open Folder…**, choose `C:\Users\YOUR_USER\t101`, then **File > New File… > Jupyter Notebook**, and at the top-right click **Select Kernel > Python Environments > .venv**.

| Placeholder | Replace with | Example |
|---|---|---|
| `YOUR_USER` | Your Windows user folder name | `karthik` |

**CHECKPOINT:** Step 11 prints both versions, and a VS Code notebook cell running `print(1+1)` prints `2`.

# PART 3 — The Core Idea

## 3.1 One analogy: the meeting room

Imagine every word in a sentence is a **person in a meeting**.

- Each person holds a **card with facts about themselves** — that is their **embedding** (Topic 1).
- Each person wears a **name badge** saying what they can offer — the **Key**.
- Each person has a **question** they want answered — the **Query**.
- Each person also has the **actual information** they will share if asked — the **Value**.

When the meeting starts, every person reads every badge, decides how relevant each person is to their own question (Query·Key), and then takes notes from everyone, **mostly from the most relevant people** (softmax-weighted sum of Values). That is **self-attention** (Topics 2–3).

Now give everyone **several different questions at once** — "who is the subject?", "what is the verb?", "what was said nearby?" — each answered by a separate mini-meeting. That is **multi-head attention** (Topic 4).

If this meeting is for *writing a story one word at a time*, nobody may read notes from people who have not spoken yet — that is the **causal mask** (Topic 5). Because people in this room have no chairs in a row, you must also write their **seat numbers** on their cards — that is **positional encoding** (Topic 6).

Finally there are three kinds of meetings: one that only *understands* (encoder-only, BERT — Topic 8), one that only *writes* (decoder-only, GPT — Topic 9), and one where a *reading team* briefs a *writing team* (encoder–decoder, T5 — Topic 7). The briefing between teams is **cross-attention** (Topic 10).

## 3.2 The 20% of ideas that explain 80%

1. **Everything is vectors.** Text → token IDs → vectors (embeddings). Meaning = direction in space; similar meaning = high cosine similarity.
2. **Attention formula — memorise it:**
   `Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V`
   Q·Kᵀ = "how relevant is each word to each other word"; √d_k keeps the numbers calm; softmax turns scores into percentages; ·V blends information.
3. **Self-attention:** Q, K and V are all made from the *same* sentence by three learned matrices `W_Q, W_K, W_V`.
4. **Multi-head = split, attend, concat.** `d_model` is cut into `h` heads of size `d_model/h`; each head attends separately; results are glued back and mixed by `W_O`.
5. **Masks = set forbidden scores to −∞** before softmax so they become exactly 0.
6. **Attention has no sense of order** unless you add position information (sinusoidal, learned, or RoPE).
7. **A block = attention (words talk) + MLP (each word thinks) + residuals + norms.** Stack N blocks.
8. **Three families:** encoder-only (understand), decoder-only (generate), encoder–decoder (transform input to output). Cross-attention connects encoder and decoder.
9. **Training objective decides the family's skill:** BERT fills in masked words; GPT predicts the next token; T5 rewrites input into output.
10. **Cost:** attention compares every token with every token → memory and time grow with T² (sequence length squared). Most elite research (FlashAttention, GQA, MLA, KV cache, sparse/linear attention) attacks this cost.

That is the theory. Everything else in this guide is you proving these ten lines with your own code.
