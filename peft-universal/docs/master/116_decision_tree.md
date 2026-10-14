# 116. The universal fine-tuning decision tree and checklist

## 116.1 Decision tree

```text
I HAVE A NEW MODEL
 │
 ├─ Step 0: uadapt.report / compat / plan  -> files, signature, loss, parameters, memory
 ├─ Step 1: classify
 │
 ├─ DECODER-ONLY (text) ......... peftkit SFT path: chat template + assistant-only loss -> LoRA/QLoRA -> vLLM
 ├─ ENCODER-ONLY .................. new head (train fully) + LoRA on encoder; CE on labels; FastAPI/ONNX serving
 ├─ ENCODER-DECODER ............... labels -> shifted decoder inputs; LoRA incl. cross-attention; generate() to evaluate
 ├─ VISION ........................ processor (size/mean/std); conv-LoRA or ViT attention LoRA; head; ONNX/TensorRT
 ├─ AUDIO ......................... sampling rate; CTC (labels <= frames) or seq2seq; WER/CER; streaming-aware serving
 ├─ MULTIMODAL .................... adapt projector -> LM -> (encoder last); mask image+prompt tokens; engine support check
 ├─ MoE ........................... attention-only LoRA; routers frozen; fused experts not bnb-quantizable; load-balance loss
 ├─ DIFFUSION ..................... epsilon-MSE over timesteps; cross-attention LoRA; evaluate by sampling; job-queue serving
 ├─ SSM ........................... adapt in_proj/out_proj; skip narrow dt_proj; no KV cache (fixed state)
 ├─ RNN ........................... adapt Linears around the cell or unfreeze the cell (packed weights)
 ├─ HYBRID ........................ per-block treatment; size memory from attention blocks
 ├─ CUSTOM ........................ uadapt: discover -> inject -> verify -> loop -> merge; your loss, your server
 └─ UNSUPPORTED BY FRAMEWORKS ..... ask WHY (ch 82.3): Linear/Conv present? inject custom LoRA. Packed params? re-implement
                                    the layer with Linears. Nothing adaptable? selective unfreezing. Truly nothing -> IMPOSSIBLE
                                    WITH CURRENT APPROACH, with the reason written down
 │
 ▼ for every branch
SMOKE TEST (gradients + one-batch overfit) -> TRAIN -> EVALUATE vs BASELINE -> EXPORT (prove merge) -> SERVE
  -> SERVED PARITY -> LOAD TEST -> RELEASE GATE -> KEYS + GATEWAY -> AUTOSCALING -> MONITOR -> PRODUCTION
```

## 116.2 Universal model adaptation checklist

```text IDENTIFY
[ ] config, modeling file, processor/tokenizer read; weights' names/shapes listed without loading
[ ] parameters counted (total, trainable); forward signature and outputs known
[ ] family classified WITH evidence; conflicts resolved from forward() source
OBJECTIVE AND TARGETS
[ ] loss found in source or written; label format matches it (-100, lengths, dtypes)
[ ] targets discovered from a trace; heads, routers, embeddings, norms decided deliberately
[ ] strategy chosen from constraints (memory, data, tasks, serving), not habit
ADAPT AND VERIFY
[ ] adapters injected; base frozen; trainable share printed
[ ] wrappers are called or read (verify_called) AND every adapter receives a gradient
[ ] one batch overfits; loss is finite
DATA
[ ] eval sets built first, frozen, excluded from training (contamination check)
[ ] processor/collator verified by decoding one batch; masks and labels inspected
TRAIN AND EVALUATE
[ ] baseline measured on the same suites; dry run gives peak memory and speed; cost written down
[ ] best checkpoint by eval, not last; compare verdict PASS; regressions read; forgetting checked
EXPORT AND SERVE
[ ] merge proven exact (or adapter shipped, with reason); published with revision pinned
[ ] engine chosen by architecture; served parity PASS; load test meets SLO; release gate RELEASE
[ ] keys, limits, metering; autoscaling thresholds from the load test; min replicas and cooldown set
[ ] showcase report generated from the files; everything reproducible from the repo
```
