# P04 — Step 3: The copy-paste prompts (replace one line, size any model)

**WHERE:** any AI assistant (Claude, ChatGPT, Gemini, a local model) in your browser.
**GOAL:** a prompt you reuse forever: change the model link (or paste your local model's `config.json`) and get the full sizing, GPU count, hours and cost, with the formulas shown so you can check every number.

> **Honest rule:** an assistant can mis-read a config or do arithmetic wrong. Use the prompt to **think and compare**, and use `python -m peftkit.plan` (P03) as the **referee**. If they disagree by more than ~20%, trust the planner (it counts parameters exactly from the real model code) and ask the assistant to find its mistake.

---

## PROMPT 1 — Full sizing and cost (the main one)

**Step 1.** Fill the block at the top (only the lines in CAPITALS). For a local or private model, paste the full contents of its `config.json` where shown.
**Step 2.** Copy everything between the lines into the assistant.

```text
====================================================================================================
YOU ARE: a senior ML infrastructure engineer who sizes LLM fine-tuning jobs for a living.
Be exact. Show every formula with the numbers substituted. Never skip a step. If a value is missing,
state the assumption you used in a table at the top. No marketing language.

MY INPUTS
- MODEL: <PASTE A HUGGING FACE LINK, e.g. https://huggingface.co/Qwen/Qwen3-8B, OR "local model, config below">
- CONFIG.JSON (only if local/private; otherwise read it from the repo's Files tab):
  <PASTE config.json HERE, or write "read it from the link">
- TASK: <e.g. "customer-support chat for industrial pumps", "extract fields from invoices", "clinic no-show prediction with explanation">
- DATASET: <rows> rows, mean <tokens> tokens per row, p95 <tokens> tokens   (from scripts/count_tokens.py; if unknown write "unknown")
- EPOCHS: <1-3>
- METHODS TO COMPARE: full fine-tuning, LoRA (r=16, all-linear), QLoRA (NF4 + double quant), DoRA
- SEQUENCE LENGTH FOR TRAINING: <p95 rounded up, e.g. 2048>
- SERVING TARGET (optional): <N> concurrent users, <context> tokens each, dtype <bf16|fp8>
- PROVIDERS AND MY QUOTED PRICES ($/GPU-hour):
  HyperStack: <H100 PCIe 2.50, H100 SXM 3.20, A100 80GB 1.35, L40 1.00, A6000 0.50, H200 3.99, B200 6.00>
  RunPod:     <RTX 4090 0.89, L40S 1.09, A100 80GB 1.79, H100 PCIe 2.89, H100 SXM 3.99, H200 5.29, B200 7.99>
  Vast.ai:    <RTX 4090 0.53, L40S 0.73, A100 80GB 0.60, H100 SXM 2.93, H200 5.00>

DO THIS, IN ORDER, AND SHOW THE WORK
1. ARCHITECTURE: from config.json list hidden_size, num_hidden_layers, intermediate_size (or moe_intermediate_size),
   vocab_size, num_attention_heads, num_key_value_heads, head_dim, tie_word_embeddings, experts
   (n_routed_experts / num_local_experts / num_experts), experts per token, shared experts, kv_lora_rank (MLA),
   sliding_window / layer_types, quantization_config. Classify: dense or MoE; attention MHA/GQA/MQA/MLA/sliding/hybrid.
2. PARAMETERS: N_total and N_active with the formulas
   attention/layer = h*(n_q*d) + 2*h*(n_kv*d) + (n_q*d)*h ;  MLP/layer = 3*h*i  (MoE: E*3*h*i_e + shared)
   N = L*(attention + MLP) + V*h (+ V*h if not tied). Cross-check with the model card's stated size.
3. LoRA TRAINABLE PARAMETERS at r=16 on all linear layers: sum r*(d_in+d_out) over q,k,v,o,gate,up,down per layer
   (MoE: say what happens if experts are included vs attention-only).
4. TRAINING MEMORY per method (GiB), with:
   full FT  = 16 bytes/param ; LoRA = 2*N + 16*N_adapter ; QLoRA = 0.516*N_linear + 4*N_other + 10*N_adapter
   activations (gradient checkpointing) ~ L*2*seq*batch*h*2 bytes ; logits = batch*seq*vocab*4 bytes
   total = 1.10*(sum) + 1.5 GB. Warn if bitsandbytes cannot quantize fused MoE expert tensors (they stay 16-bit).
5. GPU COUNT per GPU type (24, 48, 80, 94, 141, 180 GB): ceil(total / (VRAM*0.9)), assuming FSDP/ZeRO-3 sharding;
   state single GPU / 1 node / multi-node (>8 GPUs) and the interconnect needed.
6. HOST REQUIREMENTS: disk = 2N + 2N (merged copy) + checkpoints*3 + 40 GB (full FT checkpoints ~16N bytes each);
   RAM recommended = 1.25*2N + 16 GB.
7. COMPUTE AND TIME: tokens_total = rows*mean_tokens*epochs. FLOPs = 6*N_active*tokens (PEFT) or 8*N*tokens (full FT).
   Peak dense bf16 FLOP/s: RTX4090 165e12, L4 121e12, A100 312e12, L40S 362e12, H100 SXM 989e12 (PCIe ~756e12),
   H200 989e12, B200 2250e12. MFU: full 0.40, LoRA 0.35, QLoRA/DoRA 0.25.
   GPU-hours = FLOPs/(peak*MFU)/3600 ; wall-clock = GPU-hours/GPUs.
8. COST TABLE: rows = method x provider x GPU type that FITS; columns = GPUs, wall-clock h, GPU-hours, $ (GPU-hours*price),
   $ with +40% margin for eval/dry-run/idle. Sort by $.
9. SERVING (if a target was given): weights at the serving dtype; KV bytes/token = 2*layers*kv_heads*head_dim*bytes
   (MLA: layers*(kv_lora_rank+qk_rope_head_dim)*bytes); KV for the load; GPUs per replica (tensor-parallel 1/2/4/8);
   max concurrent sequences per GPU type.
10. RECOMMENDATION: (a) cheapest option that fits, (b) fastest option under 2x the cheapest cost, (c) the safe
   beginner option, each with: provider, GPU, count, method, hours, $, disk, RAM, and the top 3 risks.
11. VERIFY: give me the exact command to cross-check:
   python -m peftkit.plan <MODEL> --seq-len <S> --dataset-tokens <rows*mean> --epochs <E> --gpu-price <...>
   and list which of your numbers I should compare first.
====================================================================================================
```

**YOU SHOULD SEE:** an assumptions table, 11 numbered sections with formulas filled in, a cost table sorted by price, three recommendations, and the verification command.

| If the assistant… | Do this |
|---|---|
| gives totals without formulas | reply: "Show step 2 and step 4 with the numbers substituted." |
| says it cannot open the link | paste the `config.json` contents (Hub → Files and versions → `config.json` → raw) |
| differs from `peftkit.plan` by > 20% | paste the planner's output and say: "Find the term you computed differently." Usually: tied embeddings, MoE experts, or the fp32 upcast in QLoRA |
| ignores MoE | add: "This is MoE: memory uses N_total, time uses N_active." |

---

## PROMPT 2 — Design the dataset for my domain

```text
YOU ARE: a data lead who builds fine-tuning datasets that pass evaluation, not demos.
DOMAIN: <e.g. clinic operations / insurance claims / pump support>
USERS AND TASK: <who asks what; what a perfect answer looks like; output format (JSON? short text? citations?)>
WHAT I HAVE: <e.g. 300 PDFs of manuals, an Excel export of 12,000 tickets, nothing>
CONSTRAINTS: <privacy/PII rules, licence, languages, budget for LLM API calls>

GIVE ME:
1. Behaviour vs facts: which parts should be fine-tuned and which should be retrieved (RAG). Be blunt.
2. A taxonomy: intents (10-30) x entities x difficulty, with the % of the dataset for each.
3. 5 example conversations per top intent in {"messages": [...]} JSONL, including refusals, "not in the
   documents", clarifying questions and multi-turn cases.
4. The EVAL set design (built FIRST): 100-200 human-checked questions, held-out entities, a general suite and
   a safety suite; how to score each (exact, must_include, regex, judge).
5. How many rows to start with (and why), and the stopping rule for adding more.
6. The top 10 ways this dataset could fail (leakage, imbalance, template bias, PII) and the check for each.
```

## PROMPT 3 — Write seed tasks for Self-Instruct (P07 Section 3)

```text
Write 25 diverse seed tasks for training an assistant in: <DOMAIN>.
Vary the verb (explain, compare, extract, classify, draft, troubleshoot, summarise, convert), the format
(list, table, JSON, one sentence, step-by-step), and the difficulty. Each must be answerable with text only.
Output JSONL, one per line: {"instruction": "...", "input": "", "output": "..."}.
The outputs must be correct for <DOMAIN>; if unsure, write a task whose answer you are sure of.
```

---

## CHECKPOINT

- [ ] I saved Prompt 1 with my provider prices filled in.
- [ ] I ran it once for my `MODEL_ID` and the planner agrees within ~20% (or I found the reason).
