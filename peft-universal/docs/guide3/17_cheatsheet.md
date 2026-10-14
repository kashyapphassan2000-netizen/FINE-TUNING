# 17. One-page cheat sheet

```text MEMORISE THIS
KV bytes/token = 2 x layers x kv_heads x head_dim x bytes_per_element      (K and V)
MLA instead:     layers x (kv_lora_rank + qk_rope_head_dim) x bytes
KV total       = bytes/token x (sum of all sequence lengths in flight)
KV budget      = GPUs x VRAM x utilisation - weights - workspace
concurrency    ~ KV budget / (bytes/token x avg tokens per sequence)

VARIANTS (same model width, lab 34 measured bytes/token)
  MHA 4096   GQA(4) 1024   MQA 256   MLA(rank128) 288
  MHA = max quality, max cache | MQA = min cache, quality risk
  GQA = the default (kv_heads 4-8) | MLA = best at high head counts

PHASES
  prefill  = whole prompt, compute-bound, big matmuls
  decode   = one token/step, bandwidth-bound, re-reads weights + cache

KERNELS
  SDPA(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False)
  backends: FLASH_ATTENTION | EFFICIENT_ATTENTION | CUDNN_ATTENTION | MATH (CPU)
  transformers attn_implementation: sdpa | eager | flash_attention_2/3/4 | flex_attention
  decode step -> is_causal=False (one query may see all cached keys)

FLASH = exact. online softmax: m=max, l=sum, rescale by exp(m_old-m_new); no N x N in HBM

PAGING (vLLM 0.29 defaults: prefix caching ON, gpu_memory_utilization 0.92, cache_dtype auto)
  blocks + per-sequence block table -> waste bounded by block_size-1 (lab 35: 81% -> 1.9%)
  copy-on-write blocks -> shared prompts, n>1 sampling, beam search are cheap

POSITION
  RoPE: rotate pairs by p * base^(-2m/d); score depends on (i-j); applied to K BEFORE caching
  scaling rope_type: default | linear | dynamic | yarn | llama3 | longrope | proportional
  transformers 5.17 keeps them under config["rope_parameters"]
  ALiBi = score bias, no params | learned absolute = hard stop at max_pos | sliding window = bounded cache

LEVERS WHEN YOU RUN OUT OF MEMORY (cheapest first)
  shorter max_model_len -> fp8 KV -> prefix caching -> more GPUs (TP <= kv_heads) -> different base model

COMMANDS
  python -m peftkit.kv <model> --context 32768 --gpu-gb 80 --gpus 1     # bytes/token, concurrency, variants
  python -m peftkit.plan <model>                                        # weights + KV + GPU count (Volume 1)
  python labs/lab30..37.py                                              # every number in this book
```

:::check You have finished this book when
- You can size any model's KV cache from its config in under a minute.
- You can explain to a colleague why GQA exists, what MLA compresses, and what paging fixes.
- You can look at a serving OOM and name the three levers you would pull, in order.
:::
