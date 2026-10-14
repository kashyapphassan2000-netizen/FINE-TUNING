# P03 — Step 2: How much hardware, how many hours, how many dollars

**Time:** 1 hour. **Cost:** $0. Everything here runs on your laptop from `config.json` only.
**GOAL:** for ANY model, know: VRAM per method, number of GPUs and which type, single-node or multi-node, host RAM and disk, training hours, and cost on HyperStack, RunPod and Vast.ai, first by hand and then with the planner.

> **Brutal honesty:** every number below is an **estimate** (±20-30% on memory, ±2× on time). The planner picks the GPU; a **20-step dry run** on that GPU (P10 Section 5) measures the truth before you commit hours. Never buy a 50-hour reservation from a formula.

---

## SECTION 1 — The formulas, by hand (works for any model ever released)

### 1.1 Parameters from config.json

For a dense decoder (Llama/Qwen/Mistral-style) with `h = hidden_size`, `L = num_hidden_layers`, `i = intermediate_size`, `V = vocab_size`, `n_q = num_attention_heads`, `n_kv = num_key_value_heads`, `d = head_dim` (default h / n_q):

```text
attention per layer  = h·(n_q·d)  [q]  +  2·h·(n_kv·d)  [k, v]  +  (n_q·d)·h  [o]
MLP per layer        = 3·h·i                  (gate, up, down)
per layer            ≈ attention + MLP        (norms are negligible)
embeddings           = V·h,   lm_head = V·h (0 if tie_word_embeddings = true)
N                    ≈ L · per_layer + embeddings + lm_head
```

**Worked: Qwen3-8B** (h 4096, L 36, i 12288, V 151936, n_q 32, n_kv 8, d 128, untied):
attention = 16.8M + 8.4M + 16.8M = 41.9M; MLP = 3 × 4096 × 12288 = 151.0M; per layer = 193.0M; × 36 = 6.95B; embeddings + head = 2 × 151936 × 4096 = 1.245B → **N ≈ 8.19B** (the planner's exact count: 8.19B ✓).

**Mixture-of-Experts** (E routed experts, k active per token, S shared experts, expert MLP size i_e):

```text
N_total  = L · ( attention + E·3·h·i_e + S·3·h·i_e + router ) + embeddings      ← decides MEMORY
N_active = L · ( attention + k·3·h·i_e + S·3·h·i_e ) + embeddings              ← decides COMPUTE (time)
```

DeepSeek-V3: 671B total, 37.6B active. **All 671B must sit in GPU memory**; time is computed with 37.6B.

### 1.2 Training memory (per method)

```text
M_train ≈ 1.10 × (weights + optimizer/grad state + activations + logits) + 1.5 GB
  full FT : weights 2N + state 14N                           = 16 B/param
  LoRA    : weights 2N + state 16 × N_adapter
  QLoRA   : weights ≈ 0.516 × N_linear + 4 × N_other + state 10 × N_adapter
  N_adapter (LoRA, all-linear) = Σ over every targeted matrix of r × (d_in + d_out)
  activations (grad checkpointing on) ≈ L × 2 × seq × batch × h × 2 B   (+ one layer's working set)
  logits  = batch × seq × V × 4 B
```

### 1.3 GPU count

```text
GPUs = ceil( M_train / (VRAM_per_GPU × 0.9) )        ← only if the model is SHARDED (FSDP / DeepSpeed ZeRO-3)
plain DDP: every GPU holds a full copy → more GPUs = FASTER, never SMALLER
1 node = up to 8 GPUs; > 8 GPUs = multi-node (InfiniBand-class network needed, P10 Section 9)
```

### 1.4 Host (CPU-side) RAM and disk: the part everyone forgets

```text
disk  ≥ downloaded checkpoint (2N bytes) + merged copy (2N) + checkpoints × save_total_limit + ~40 GB envs/caches
        full FT checkpoints include optimizer state: ~16N bytes EACH
RAM   min ≈ 2 shards + 8 GB (streaming load);  recommended ≈ 1.25 × 2N + 16 GB
        (CPU merging, FSDP rank-0 loading and saving a full model need the WHOLE model in RAM)
```

### 1.5 Tokens, time, cost

```text
tokens_total = (tokens in train.jsonl, with the chat template) × epochs        (counted by --dataset)
FLOPs        = 6 × N_active × tokens_total        PEFT, frozen base, gradient checkpointing
             = 8 × N × tokens_total                full fine-tuning, gradient checkpointing
GPU-hours    = FLOPs / (peak dense bf16 FLOP/s × MFU) / 3600
wall-clock   = GPU-hours / number of GPUs          (near-linear inside one node)
cost         = GPU-hours × price per GPU-hour
peak bf16 (dense): RTX 4090 165T* | L4 121T | A100 312T | L40S 362T | H100 SXM 989T | H200 989T | B200 2250T
MFU you can expect: full FT 0.35-0.45 | LoRA 0.30-0.40 | QLoRA/DoRA 0.20-0.30     (* fp16 accumulate)
after the dry run: GPU-hours = tokens_total / measured_tokens_per_sec / 3600   ← replaces the MFU guess
```

**Worked: Qwen3-8B, QLoRA, 12M tokens × 3 epochs on 1 × H100:**
FLOPs = 6 × 8.19e9 × 3.6e7 = 1.77e18. GPU-hours = 1.77e18 / (989e12 × 0.25) / 3600 = **1.99 h**. At HyperStack's H100 PCIe $2.50/h → **≈ $5**. Add ~30-50% for evaluation, checkpoint saves, idle time and the dry run → budget **$7-8**.

---

## SECTION 2 — Count your training tokens (exactly, with the model's own tokenizer)

**WHERE:** PowerShell in `FINE-TUNING\peft-universal`, `(.venv)` active
**GOAL:** the real `tokens` number for the time/cost formula, and the p95 length for `max_seq_length`

Step 1. Paste and press Enter (once you have `train.jsonl` from P06/P07; until then use the lesson file):

```powershell
python scripts/count_tokens.py data/lesson/train.jsonl outputs/tiny
```

**YOU SHOULD SEE:** `rows`, `tokens_total`, `mean_tokens`, `p95_tokens`, `max_tokens`.

> **WATCH OUT:** characters ÷ 4 is a rough guess for English; for code, other languages or chat templates it can be off by 2×. Always count with the real tokenizer (the planner's `--dataset` flag does it for you; it says `exact` or `chars/4 estimate`).

## SECTION 3 — Run the planner (the one command that does all of Section 1)

**WHERE:** PowerShell
**GOAL:** the full table for your model, data and prices

Step 1. Paste and press Enter (replace `MODEL_ID`, the dataset path, epochs, and put in the prices YOU are quoted: Section 4):

```powershell
python -m peftkit.plan MODEL_ID --seq-len 2048 --dataset data/processed/train.jsonl --epochs 3 --gpu-price RTX4090-24=0.53 L40S-48=0.73 A100-80=1.60 H100-80=2.50 H200-141=3.99 B200-180=6.00
```

No dataset yet? Use `--dataset-tokens 1.2e7` (12M tokens) instead of `--dataset`.

**YOU SHOULD SEE (real output, `Qwen/Qwen3-8B`, `--dataset-tokens 1.2e7 --epochs 3`):**

```text
MODEL Qwen/Qwen3-8B  type=qwen3  family=dense  attention=gqa
  params total=8.19B active/token=8.19B [exact (meta-device instantiation)]
TRAINING (per replica, estimates)
  lora      weights      15.3 GB  optim    0.7  acts    0.7  logits   1.2  TOTAL      21.1 GB  -> 1x RTX4090-24; 1x L4-24; 1x A10G-24
  qlora     weights       8.0 GB  optim    0.4  acts    0.7  logits   1.2  TOTAL      12.8 GB  -> 1x RTX4090-24; 1x L4-24; 1x A10G-24
  full      weights      15.3 GB  optim  106.8  acts    0.7  logits   1.2  TOTAL     137.8 GB  -> 1x B200-180; 2x A100-80; 2x H100-80
  trainable LoRA params (all-linear): 43,646,976
COMPUTE tokens=3.6e+07  FLOPs=1.77e+18
TIME for all epochs: hours = k x active x tokens / (GPUs x peak bf16 x MFU), k=6 PEFT, 8 full FT.
  lora     RTX4090 1x 8.5h $5 | L40S 1x 3.9h $3 | A100 1x 4.5h $7 | H100 1x 1.4h $4 | H200 1x 1.4h $6 | B200 1x 0.6h $4
  qlora    RTX4090 1x 11.9h $6 | L40S 1x 5.4h $4 | A100 1x 6.3h $10 | H100 1x 2.0h $5 | H200 1x 2.0h $8 | B200 1x 0.9h $5
  full     RTX4090 7x 1.4h $5 | L40S 4x 1.1h $3 | A100 2x 2.6h $8 | H100 2x 0.8h $4 | H200 2x 0.8h $7 | B200 1x 0.7h $4
HOST (training VM): disk >= 72 GB, RAM min 18 GB, recommended 35 GB
  full fine-tuning checkpoints: disk >= 437 GB (optimizer state in every checkpoint)
SERVING weights bf16=15.3 GB  KV 0.1406 MB/token  KV for your load=18.0 GB  total=38.5 GB
  L40S-48           1 GPUs  max concurrent tokens ~192,556  single GPU
  H100-80           1 GPUs  max concurrent tokens ~402,271  single GPU
```

**How to read it, line by line:**

| Line | Read it as |
|---|---|
| `TOTAL 21.1 GB → 1x RTX4090-24` | LoRA fits one 24 GB card **with only ~0.5 GB spare**: tight. Prefer 48 GB, or QLoRA on 24 GB |
| `qlora … 12.8 GB` | comfortable on any 24 GB card |
| `full … 137.8 GB → 2x H100-80` | full fine-tuning needs 2 × 80 GB **sharded** (ZeRO-3) |
| `TIME … L40S 1x 3.9h $3` | 3.9 GPU-hours on one L40S at your price: the **cheapest** LoRA option here |
| `H100 1x 1.4h $4` | the **fastest** single-GPU option: costs about the same, finishes 3× sooner |
| `HOST disk >= 72 GB` | order a VM disk of ≥ 100 GB (round up; datasets and logs grow) |
| `SERVING … total 38.5 GB` | to serve at the default load you need a 48 GB card (P15) |

### Worked examples at other sizes (real planner output)

**Llama-3.1-70B, 12M tokens × 3 epochs:**

```text
  lora      TOTAL     153.7 GB  -> 1x B200-180; 2x H100NVL-94; 2x H200-141
  qlora     TOTAL      52.6 GB  -> 1x A100-80; 1x H100-80; 1x H100NVL-94
  full      TOTAL    1162.2 GB  -> 8x B200-180; 10x H200-141; 14x H100NVL-94  [1 node, 8 GPUs]
  qlora     A100 1x 54.3h $87 | H100 1x 17.1h $43 | B200 1x 7.5h $45
HOST (training VM): disk >= 304 GB, RAM recommended 180 GB
```

Reading: QLoRA on **one** H100 80 GB, ≈17 h, ≈$43 (+30-50%). The VM needs ≥ 300 GB disk and ~180 GB RAM if you merge on the CPU. Full fine-tuning of 70B needs a full 8 × B200 node and 3.5 TB of checkpoint disk: not a learning project.

**DeepSeek-V3 (671B MoE), QLoRA, attention-only adapters, `--no-upcast`:**

```text
  qlora     weights    1228.9 GB  ...  TOTAL    1360.9 GB  -> 9x B200-180; 11x H200-141; 17x H100NVL-94  [2 nodes x 8 GPUs]
  trainable LoRA params (attention): 97,006,592   all-linear=7645254656
```

Reading: **QLoRA does not shrink this MoE much**: its fused expert tensors are not `nn.Linear`, so bitsandbytes leaves them in 16-bit. Two 8-GPU nodes minimum. Putting LoRA on all experts would add 7.6B trainable parameters, so target attention only. Details: P17 §2.

## SECTION 4 — Provider prices (snapshot read from each provider's page on 14 October 2026)

**On-demand $ per GPU-hour** (USD; single GPU unless stated; VRAM in GB):

| GPU (VRAM) | HyperStack [6] | RunPod Pods [7] | Vast.ai [8] (marketplace: from / median) |
|---|---|---|---|
| RTX 4090 (24) | — | 0.89 | 0.36 / 0.53 |
| RTX 5090 (32) | — | 1.19 | 0.37 / 0.67 |
| L4 (24) | — | 0.59 | 0.20 / 0.20 |
| RTX A6000 (48) | 0.50 (spot 0.40) | 0.59 | 0.28 / 0.40 |
| L40 (48) | 1.00 (spot 0.80) | 0.82 | 0.36 / 0.36 |
| L40S (48) | — | 1.09 | 0.47 / 0.73 |
| RTX 6000 Ada (48) | — | 0.99 | 0.47 / 0.57 |
| RTX Pro 6000 (96) | 1.85 (spot 1.48) | 2.49 | 1.17 / 1.47 |
| A100 80 GB PCIe | 1.35 (spot 1.08) | 1.79 | 0.38 / 0.60 |
| A100 80 GB SXM | 1.60 | 1.79 | 0.40 / 0.72 |
| H100 80 GB PCIe | 2.50 (spot 2.00) | 2.89 | 3.13 / 3.14 |
| H100 80 GB SXM | 3.20 | 3.99 | 1.73 / 2.93 |
| H100 NVL (94) | 2.60 (80 GB "NVLink") | 3.19 | 2.35 / 2.35 |
| H200 (141) | 3.99 | 5.29 | 4.00 / 5.00 |
| B200 (180-192) | 6.00 | 7.99 | 6.25 / 8.38 |

**Billing and the small print that changes the real cost:**

| | HyperStack | RunPod | Vast.ai |
|---|---|---|---|
| what you rent | full VM (Ubuntu, root via `sudo`) | container "Pod" | container on a marketplace host |
| billing | per minute, prepaid | per second | per second |
| storage | ~$0.10 per TB-hour for volumes | volume $0.10/GB/month running, $0.20 idle; network storage $0.05-0.07 | per host, shown in the offer |
| egress | free | free for Pods (verify for your region) | varies per host (shown in the offer) |
| reliability | data-centre SLA | Secure Cloud (data centres) vs Community Cloud | **varies per host**: check reliability score, use interruptible only with checkpoints |
| best for | stable VMs, NVLink/SXM nodes, long jobs | quick start, templates, per-second | lowest price for 24-80 GB cards |

> **WATCH OUT:** prices move monthly and differ by region and availability; Vast.ai prices move **hourly**. Read the price on the deploy screen and put **that** number into `--gpu-price`. Spot/interruptible is 20-50% cheaper but can be stopped at any time: use it only with `save_steps` checkpoints and `--resume true` (P10).

## SECTION 5 — Choose: the decision in 4 lines

1. **Fit:** take the method whose `TOTAL` fits one GPU with ≥ 15% headroom (QLoRA if LoRA is tight).
2. **Price per job, not per hour:** compare the `$` column (GPU-hours × price), not the hourly price. A faster, pricier GPU is often the same total.
3. **Interconnect:** 1 GPU → any provider. 2-8 GPUs → one node with NVLink/SXM. > 8 GPUs → multi-node with InfiniBand (HyperStack SXM clusters, RunPod Clusters).
4. **Host:** disk ≥ the HOST line rounded up to the next 100 GB; RAM ≥ recommended if you will merge on that machine.

**For your first real run (8B class model, ≤ 50M tokens):** QLoRA or LoRA on **one 48 GB card** (L40S / A6000 / RTX 6000 Ada) or **one 80 GB card** (A100/H100), 100-200 GB disk, 64+ GB RAM. Expect $3-15.

## SECTION 6 — After the dry run: replace the estimate with a measurement

**WHERE:** your laptop, after P10 Section 5 printed `train_seconds` and `peak_vram_gb`
**GOAL:** exact hours and dollars

Step 1. Tokens per second measured = (tokens in the 20 steps) / `train_seconds`. Tokens in 20 steps ≈ 20 × per_device_batch × grad_accum × mean_tokens (from Section 2).

Step 2. Paste and press Enter with your measured number:

```powershell
python -m peftkit.plan MODEL_ID --method qlora --dataset data/processed/train.jsonl --epochs 3 --tokens-per-sec 2400 --price-per-gpu-hour 2.50
```

**YOU SHOULD SEE:** `GPU-hours = …` and `cost = …` computed from **your** measured speed, not the MFU guess.

---

## CHECKPOINT

- [ ] I computed N, M_train and GPU-hours for my model by hand within ~20% of the planner.
- [ ] I know: method, GPU type, GPU count, single/multi-node, VM disk, VM RAM.
- [ ] I have a cost estimate per provider with prices I read today, plus a 30-50% margin.
- [ ] I will confirm memory and speed with the 20-step dry run before the real run.
