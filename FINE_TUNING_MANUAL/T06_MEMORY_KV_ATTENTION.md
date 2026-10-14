# T06 — Memory, KV Cache and Attention: where every gigabyte goes

**WHERE:** read anywhere; labs on your PC.
**GOAL:** be able to compute, by hand, the GPU memory of training and serving for any model, and understand how the KV cache, attention variants, FlashAttention, PagedAttention and RoPE relate to each other. P03 uses every formula here.

---

## 1. Training memory: the five terms

```text
M_train ≈ 1.10 × ( M_weights + M_optimizer + M_activations + M_logits ) + 1.5 GB      (1.10 = allocator slack,
                                                                                      1.5 GB = CUDA context)
```

| Term | Full fine-tuning (bf16 mixed precision, AdamW) | LoRA (bf16 base) | QLoRA (NF4 base) |
|---|---|---|---|
| **weights** | 2 B × N | 2 B × N (frozen) | 0.516 B × N_linear + 4 B × N_other (upcast to fp32) |
| **gradients** | 2 B × N | 4 B × N_adapter | 4 B × N_adapter |
| **optimizer** | fp32 master 4 B + Adam m 4 B + v 4 B = **12 B × N** | 4 + 4 + 4 = 12 B × N_adapter (incl. fp32 copy) | 8-bit paged Adam: ≈ 2 B × N_adapter + fp32 weight/grad |
| **per parameter total** | **16 B × N** (the famous rule) | 2 B × N + 16 B × N_adapter | ≈0.55 B × N + 10 B × N_adapter |
| **activations** | ≈ layers × 2 × seq × batch × hidden × 2 B with gradient checkpointing (one boundary tensor per layer) + one layer's full set | same | same |
| **logits** | batch × seq × vocab × 4 B (fp32 copy for the loss) | same | same |

N = total parameters; N_adapter = LoRA parameters = Σ over targeted matrices of r × (d_in + d_out).

**Worked by hand: Qwen3-8B, LoRA r=16 all-linear, seq 2048, batch 1** (planner output in P03):
- weights: 8.19B × 2 B = 16.4 GB = **15.3 GiB**;
- adapter: 43.6M params × 16 B = 0.70 GB → **0.7 GiB**;
- activations (checkpointed) ≈ **0.7 GiB**; logits 1 × 2048 × 151,936 × 4 B = **1.2 GiB**;
- total = 1.1 × (15.3 + 0.7 + 0.7 + 1.2) + 1.5 ≈ **21.1 GiB** → fits a 24 GB card with almost no headroom (48 GB comfortable).
- Full fine-tuning: 8.19B × 16 B = 131 GB (**≈122 GiB**) before activations → 2 × 80 GB GPUs, **sharded** (ZeRO-3/FSDP).

**Why logits matter:** vocab 152k × seq 8192 × 4 B = 5 GB per sequence. Long-context training on big-vocab models is often logit-bound, not weight-bound (TRL's chunked loss reduces it).

**Gradient checkpointing** (on by default): keep only each layer's input during the forward pass and recompute the rest during backward. Cuts activations by ~10-20× for ~30% more compute (the "+2N" in the FLOPs formula).

## 2. Serving memory: weights + KV cache

```text
M_serve = weights (bytes/param × N)  +  KV cache  +  ~1.5 GB workspace,   plan at ≤ 90% of VRAM
bytes/param: fp32 4 | bf16/fp16 2 | fp8/int8 1 | int4/NF4/AWQ/GPTQ ≈ 0.5-0.55 | MXFP4 ≈ 0.53
```

### 2.1 The KV cache formula (memorise this one)

When generating token n, attention needs the **K**ey and **V**alue vectors of all n−1 earlier tokens in every layer. Recomputing them every step is O(n²); **caching** them is O(n) work and O(n) **memory**.

```text
KV bytes per token = 2 (K and V) × layers × kv_heads × head_dim × bytes_per_element
KV for a load      = KV per token × tokens per sequence × concurrent sequences
max concurrent sequences = (VRAM × 0.9 − weights − 1.5 GB) / (KV per token × context)
```

**By hand, Qwen3-8B** (36 layers, 8 KV heads, head_dim 128, bf16): 2 × 36 × 8 × 128 × 2 B = **147,456 B = 144 KB/token**. One 32k-token conversation = 4.5 GB. On one 80 GB GPU: (72 − 15.3 − 1.5) GB ≈ 63 GB KV budget → **~14 conversations of 32k tokens** at once.

```powershell
python labs/lab30_naive_decoding.py
python labs/lab31_kv_cache_math.py
python -m peftkit.kv Qwen/Qwen3-8B --context 32768 --gpu-gb 80
```

**YOU SHOULD SEE (real output):**

```text
use_cache=False 1.668 s
use_cache=True  0.065 s          (25.7x faster, same tokens)
formula  2*L*KV*HD*T*bytes = 2*2*2*16*8*4 = 4,096 bytes
measured allocation                              = 4,096 bytes
formula matches reality: True
...
Qwen/Qwen3-8B  attention=gqa layers=36 heads=32 kv_heads=8 head_dim=128
KV per token (bf16): 144.0 KB   one 32768-token sequence: 4.5 GB
weights 15.26 GB on 1x80.0 GB -> KV budget 63.24 GB
fits ~460,525 cached tokens = 14 sequences at 32768 tokens
what each attention design would cost per token for this shape:
  MHA (kv_heads = heads)               576.00 KB
  GQA (kv_heads = 8)                   144.00 KB
  MQA (kv_heads = 1)                    18.00 KB
```

### 2.2 How the pieces relate (the chain of cause and effect)

```text
attention variant (MHA/GQA/MQA/MLA)  -> kv_heads (or latent size)  -> KV bytes per token
KV bytes per token × context × users -> KV memory                  -> how many users per GPU
users per GPU × GPU price            -> cost per user / per 1M tokens
KV dtype (bf16 -> fp8)               -> halves KV memory           -> doubles users per GPU
sliding-window layers                -> those layers cache only the window, not the full context
MoE                                  -> HUGE weights, but KV depends only on attention -> small cache
PagedAttention                       -> no reserved-but-unused KV -> 3-5x more sequences in practice
prefix caching                       -> shared system prompt cached once for all users
FlashAttention                       -> no N×N score matrix in memory -> long context trainable/servable
RoPE scaling                         -> longer context possible -> but KV grows linearly with it
```

**For fine-tuning:** the KV cache is **off** during training (`use_cache=False`: it is useless with teacher forcing and breaks gradient checkpointing). It matters for (a) **evaluation by generation** (a long eval with big batches can OOM), and (b) **serving cost**, which decides whether the fine-tuned model is worth deploying.

## 3. Attention variants: MHA, MQA, GQA, MLA

```powershell
python labs/lab34_attention_variants.py
```

**YOU SHOULD SEE (real output):**

```text
variant                      kv heads  attn params  KV bytes/token   vs MHA
MHA  (kv_heads = 16)               16    4,194,304           4,096     1.0x
GQA  (kv_heads = 4)                 4    2,621,440           1,024     4.0x
MQA  (kv_heads = 1)                 1    2,228,224             256    16.0x
MLA  (latent rank 128)              -    2,506,752             288    14.2x
```

| Variant | What is cached | Used by | Trade |
|---|---|---|---|
| **MHA** (multi-head) | K, V for every head | GPT-2/3 era, BERT | best expressiveness, biggest cache |
| **MQA** (multi-query) | one K, V shared by all heads | PaLM, Falcon-7B | smallest cache, some quality loss |
| **GQA** (grouped-query) | one K, V per group of query heads (kv_heads 4-8) | Llama 3, Qwen 2/3, Mistral, Gemma | near-MHA quality, MQA-like cost: **the industry default** |
| **MLA** (multi-head latent) | one compressed latent (kv_lora_rank) + a small decoupled RoPE key, per layer | DeepSeek-V2/V3/R1, Kimi-K2 | tiny cache at 128 heads (DeepSeek-V3: 576 numbers/token/layer = 68.6 KB/token); extra decode matmuls |

**You cannot change the attention variant by fine-tuning.** It is fixed at pre-training and it fixes your serving economics. If you need long context cheaply, choose the base model by reading `num_key_value_heads` (or `kv_lora_rank`) first.

## 4. FlashAttention and SDPA

Standard attention materialises an N × N score matrix per head: at N = 32,768 that is **2 GB per head** in bf16. FlashAttention computes exact attention **tile by tile** in on-chip SRAM with an online softmax (keep a running max and sum; rescale previous partial outputs when the max changes). Nothing is approximated; memory becomes O(N) instead of O(N²), and it is faster because attention is **memory-bandwidth bound**.

```powershell
python labs/lab32_flash_attention.py
```

```text
online-softmax result matches standard attention: True (max diff 2.09e-07)
peak score memory in the streaming version: one block only -> 128 KB instead of 1024 KB
```

In practice: `attn_implementation: sdpa` (PyTorch's scaled-dot-product attention picks a fused FlashAttention/memory-efficient kernel automatically; the toolkit default) or `flash_attention_2` if the `flash-attn` package is installed. T4/V100 do not support FlashAttention-2: use `sdpa`.

## 5. PagedAttention (vLLM)

Serving engines used to reserve `max_len` KV slots per request. PagedAttention allocates KV in **blocks of 16 tokens on demand**, like virtual-memory pages, and shares blocks between sequences (copy-on-write).

```powershell
python labs/lab35_paged_attention.py
```

```text
strategy                            KV slots held   GB @128KB/token   waste
contiguous, reserve max_len=2048           65,536             8.0 GB   81.0%
paged, blocks on demand                    12,688             1.5 GB    1.9%
prefix sharing: forking a 300-token prompt for a second sequence allocated 0 new blocks (copy-on-write).
```

## 6. RoPE (rotary position embeddings) and context extension

RoPE rotates pairs of query/key dimensions by an angle proportional to the position, so the attention score depends only on the **distance** (i − j). Pair m rotates at θ_m = base^(−2m/d): fast pairs encode local order, slow pairs long range.

```powershell
python labs/lab36_rope.py
```

```text
scores: [-19.1987, -19.1987, -19.1987, -19.1987]  -> all equal: True   (same distance, different positions)
available rope_type values: ['dynamic', 'linear', 'llama3', 'longrope', 'proportional', 'yarn']
```

**Fine-tuning consequence:** extending context (e.g. 8k → 32k) = change `rope_parameters` (YaRN / linear / dynamic NTK) **and** fine-tune on long examples, then re-evaluate with needle-in-a-haystack tests. It is a training job, not a serving flag. And KV memory grows linearly with the new context (§2).

---

## The formula card (also in P19)

```text
TRAIN  full FT        ≈ 16 B × N + activations + logits
TRAIN  LoRA           ≈ 2 B × N + 16 B × N_adapter + activations + logits
TRAIN  QLoRA          ≈ 0.55 B × N_linear + (2-4) B × N_other + 10 B × N_adapter + act + logits
N_adapter (LoRA)      = Σ_targeted r × (d_in + d_out)        (× experts if experts are targeted)
GPUs (sharded)        = ceil( M_train / (VRAM × 0.9) )       DDP does NOT reduce per-GPU memory
SERVE                 = bytes/param × N + KV + 1.5 GB, at ≤ 90% VRAM
KV per token          = 2 × layers × kv_heads × head_dim × bytes      (MLA: layers × (kv_rank + rope_dim) × bytes)
users per GPU         = (0.9 × VRAM − weights − 1.5 GB) / (KV per token × context)
FLOPs                 = 6 × N_active × tokens (PEFT, ckpt) | 8 × N × tokens (full FT, ckpt)
hours                 = FLOPs / (GPUs × peak FLOP/s × MFU) / 3600         MFU 0.25-0.45
cost                  = GPU-hours × price per GPU-hour
```

## CHECKPOINT

- [ ] I computed Qwen3-8B's LoRA training memory and its KV per token by hand and got ≈21 GiB and 144 KB.
- [ ] I can explain why an MoE model can have huge weights but a small KV cache.
- [ ] I can explain what FlashAttention and PagedAttention each save.
- [ ] I know why the KV cache is off during training, and when it matters for my project.
