# T01 — The Modern LLM Fine-Tuning Lifecycle, and Pre-Training Deep Dive

**WHERE:** read on any device; run the labs on your PC (P01 must be done) from `FINE-TUNING\peft-universal`.
**GOAL:** know exactly what you start from (a base or instruct model), what pre-training already did, and when you need to continue pre-training versus going straight to fine-tuning.

---

## 1. Modern LLM fine-tuning development lifecycle

### 1.1 Pre-training vs post-training — the two-phase view

```text
 PHASE A: PRE-TRAINING (done by the lab that released the model; you almost never do this)
   trillions of tokens of web/books/code  ->  next-token prediction  ->  BASE model
   cost: 10^5 - 10^7 GPU-hours  (lab 28 computes it)

 PHASE B: POST-TRAINING (this is "fine-tuning"; what you do)
   [optional] continued pre-training (CPT) on raw domain text
   -> SFT  (supervised fine-tuning on prompt -> answer examples)          stage 1, always
   -> preference alignment (RLHF-PPO / DPO / ORPO / KTO)                  stage 2, optional
   -> RL with verifiable rewards (GRPO) for maths/code/tool use           stage 3, optional
   -> evaluation, safety, quantization, serving
```

| | Pre-training | Post-training |
|---|---|---|
| data | raw text, 1-20+ trillion tokens | curated examples: 1k-1M conversations, preference pairs, prompts with checkers |
| objective | predict the next token everywhere | predict the **answer** given the prompt; prefer good answers over bad |
| what it builds | knowledge, grammar, reasoning patterns | behaviour: following instructions, format, tone, safety, domain style |
| who does it | model labs | you |
| cost | millions of dollars | dollars to thousands of dollars |

### 1.2 What pre-training produces (the base model)

A **base model** (e.g. `Qwen/Qwen3-8B-Base`, `meta-llama/Llama-3.1-8B`) is a next-token predictor over internet-shaped text. Given "The capital of France is", it continues "Paris." Given "How do I reset my pump?", it may continue with **another question**, a forum thread, or a list of similar questions, because that is what follows such text on the web. It knows a lot; it does not know it is supposed to **answer you**.

An **instruct / chat model** (e.g. `Qwen/Qwen3-8B`, `meta-llama/Llama-3.1-8B-Instruct`) is a base model **after** the lab's own post-training (SFT + alignment). It has a **chat template** (special tokens that mark who is speaking) and answers questions.

### 1.3 Why base models are not useful out of the box

1. **No turn-taking:** no notion of "user asks, assistant answers, assistant stops". It does not reliably emit an end-of-turn token, so it rambles.
2. **No format control:** ask for JSON and you get prose; ask for a short answer and you get a page.
3. **No refusal/safety behaviour:** it continues any text, harmful or not.
4. **Distribution mismatch:** your users write questions; the web mostly contains statements.

**Decision rule for you:** start from the **instruct** model in 90% of projects (it already behaves; you only teach your domain). Start from the **base** model only when (a) you do CPT first, (b) you need a completely different chat format, or (c) the instruct model's alignment fights your task (rare: some classification/extraction jobs).

### 1.4 Training objectives recap: CLM, MLM, Prefix-LM

All three use the same transformer. They differ only in **which positions may look at which** (the attention mask) and **which positions carry loss** (the labels).

| Objective | Mask | Loss on | Model families | Generates text? |
|---|---|---|---|---|
| **CLM** (causal LM) | each token sees only the past | every next token | GPT, Llama, Qwen, Mistral, DeepSeek, Gemma | yes |
| **MLM** (masked LM) | bidirectional: sees everything | only the ~15% masked tokens | BERT, RoBERTa, DeBERTa, ModernBERT | no (understanding/embeddings) |
| **Prefix-LM** | prefix bidirectional, suffix causal | only the suffix | T5/UL2 (encoder-decoder), GLM, PaLI | yes |

**Run it:**

```powershell
python labs/lab27_objectives.py
```

**YOU SHOULD SEE (real output, abridged):**

```text
=== CLM  (decoder-only: every position predicts the next token, sees only the past) ===
input :    The   pump   must   stay  below      6    bar      .
labels:   pump   must   stay  below      6    bar      .      -
     The 1 . . . . . . .
    pump 1 1 . . . . . .
   ...
=== MLM  (encoder-only: bidirectional, loss ONLY on masked positions) ===
input :    The [MASK]   must   stay  below [MASK]    bar      .
labels:      -   pump      -      -      -      6      -      -
=== Prefix-LM (first 4 tokens bidirectional 'context', suffix generated causally) ===
labels:      -      -      -  below      6    bar      .      -
loss on the same random network:
  CLM       4.2658  trained positions: 7/8
  MLM       4.3000  trained positions: 2/8
  Prefix-LM 4.3919  trained positions: 4/8
```

**The connection to fine-tuning you must remember:** SFT on a chat model with *assistant-only loss* has the **Prefix-LM label pattern** (loss only on the answer) but keeps the **CLM mask**. And you must fine-tune with the objective the model was pre-trained with: CLM data and collator for Llama/Qwen, MLM or classification heads for BERT, seq2seq for T5. Mixing them is the classic "loss goes down, model is junk" failure.

---

## 2. Pre-training deep dive

### 2.1 Data curation and filtering at scale

What the labs do to 100+ TB of raw web text before training (and the same steps, at small scale, are in your data pipeline in P06/P07):

| Stage | What it removes | Typical tool / method |
|---|---|---|
| URL / domain filtering | spam, adult, malware, blocklisted sites | blocklists |
| text extraction | HTML boilerplate, menus, cookie banners | trafilatura, resiliparse |
| language identification | wrong-language pages | fastText LID (keep score > threshold) |
| heuristic quality rules | too short/long docs, symbol-heavy lines, repetition, "lorem ipsum" | Gopher / C4 rules (mean word length, stop-word ratio, % lines ending in punctuation, duplicate n-gram fraction) |
| **exact dedup** | identical documents/lines | hashing (SHA-1/MD5) of normalised text |
| **near-dup dedup** | templated pages, mirrors, boilerplate variants | **MinHash + LSH** on 5-gram shingles, Jaccard ≥ ~0.8 |
| model-based quality | low-educational-value text | classifier trained on LLM-scored samples (FineWeb-Edu style) |
| PII removal | emails, phones, IDs | regex + NER |
| toxicity / safety | hate, CSAM | classifiers + hash lists |
| **decontamination** | benchmark questions leaking into training | 13-gram overlap against eval sets |
| **mixture design** | the proportions of web / code / maths / books / multilingual | ablation runs on small models |

Your toolkit implements the same ideas at your scale: `peftkit.data.quality` = exact hash dedup + **MinHash-LSH near-dup (Jaccard 0.8)** + heuristic filters (length, repetition ratio, non-printable characters, refusal boilerplate) + PII scrub + **13-gram contamination check** against your eval set + a poisoning/injection scan. See T02 §4.

### 2.2 Continued Pre-Training (CPT) for domain adaptation

**What:** keep doing plain next-token prediction, now on **raw domain text** (manuals, papers, tickets, code, a new language). No chat template, no masking, **every token is trained**.
**Teaches:** vocabulary, notation and style of the field; facts as statistical patterns. **Does not teach:** answering questions. You still need SFT afterwards.

```powershell
python labs/lab22_continued_pretraining.py
```

The lab shows held-out domain-text loss dropping after CPT on related raw text: the domain's language becomes less "surprising" to the model.

| | CPT | SFT |
|---|---|---|
| data | raw text | prompt → answer pairs |
| loss on | every token | assistant tokens only |
| typical size | 100M - 10B+ tokens | 1k - 100k examples |
| LoRA rank if PEFT | high (64-256) or full fine-tuning | 8-32 |
| risk | forgetting (high), cost | overfitting on small data |
| evaluate with | perplexity on **held-out domain text** | task scores on held-out questions |

### 2.3 The compute budget problem (scaling laws)

The formulas (Kaplan et al. 2020; Hoffmann et al. 2022 "Chinchilla"):

```text
training compute        C ≈ 6 · N · D          N = parameters, D = training tokens
                                                (forward 2ND + backward 4ND)
Chinchilla loss fit     L(N, D) = E + A / N^α + B / D^β
                        E = 1.69, A = 406.4, B = 410.7, α = 0.34, β = 0.28
compute-optimal rule    D* ≈ 20 · N            (tokens ≈ 20 × parameters)
GPU-hours               C / (peak FLOP/s × MFU) / 3600     MFU = model-FLOPs utilisation, 0.3-0.5
```

```powershell
python labs/lab28_scaling_laws.py
```

**YOU SHOULD SEE (real output, abridged):**

```text
   model    tokens   FLOPs (6ND)   H100-hours @40% MFU  Chinchilla loss
      8B    15.00T      7.20e+23               505,561            1.949
     70B    15.00T      6.30e+24             4,423,660            1.857
  -> an 8B model on 15T tokens is ~506k H100-hours: at $2-3/h that is $1.0-1.5M of GPU time
  CPT 8B on 1B domain tokens: full-FT 4.80e+19 FLOPs = 34 H100-h
  20,000 examples x 600 tokens x 3 epochs = 36M tokens -> 1.73e+18 FLOPs = 1.2 H100-hours at 40% MFU
  -> HONEST NOTE: this parametric fit (Chinchilla 'approach 3') gives D/N of ~50-100. The famous
     'D ~ 20 x N' comes from the paper's approaches 1 and 2; a 2024 replication (Besiroglu et al.) found
     the published approach-3 constants inconsistent and, refitted, close to ~20.
```

**What this means for you:**

1. Pre-training: 10^5-10^7 GPU-hours. You start from a released model. Always.
2. CPT: tens to thousands of GPU-hours. A real project decision (next section).
3. SFT: 1-100 GPU-hours. **Compute is not your constraint; data quality is.**
4. Modern labs "over-train" small models far past 20 tokens/parameter (Llama-3 8B: ~1,900 tokens/parameter) because **serving cost depends on N, not D**. This is why 7-14B models are so strong today and why they are the sweet spot to fine-tune.
5. LoRA does **not** cut FLOPs much (the forward/backward through the frozen base still runs); it cuts **memory**. Plan time by tokens × parameters, plan GPUs by memory.

### 2.4 When CPT is needed vs jumping straight to SFT

```text
Do you have >= 100M tokens of raw domain text the base model has clearly not seen?
├── no  -> SKIP CPT. Go straight to SFT (P05-P10). This is 90% of projects.
└── yes -> Does the base model fail on the domain's LANGUAGE (perplexity on your held-out text
           is much higher than on general text; it mangles terms, units, codes)?
           ├── no  -> SKIP CPT. SFT + retrieval (RAG) for facts.
           └── yes -> CPT (rank 64-256 LoRA or full FT, 5-30% general text mixed in to limit forgetting,
                      evaluate perplexity before/after AND a general suite), then SFT.
```

Typical "yes" cases: a low-resource language, a proprietary programming language or DSL, dense specialised notation (chemistry, legal citation systems, medical coding), a company's internal codebase. Typical "no" cases: "make it answer like our support team", "extract fields from invoices", "classify tickets": that is SFT.

### 2.5 Multi-Token Prediction (MTP)

**Idea:** besides predicting token t+1, also predict t+2, t+3… from the same hidden state. Two published designs:

- **Parallel heads on a shared trunk** (Gloeckle et al., Meta 2024): k output heads, one per future offset. Gains grow with model size, strongest on code and maths.
- **Sequential MTP modules** (DeepSeek-V3, 2024): each extra depth chains the previous depth's representation, keeping the causal chain. One extra depth (D = 1), loss weight λ ≈ 0.1-0.3. Reported ~85-90% acceptance of the second token when used for speculative decoding, ≈1.8× decoding speed.

**Why it matters:** (1) denser training signal (each position supervises several futures, encouraging "planning"); (2) at inference the extra heads are a **free draft model** for speculative decoding.

```powershell
python labs/lab29_multi_token_prediction.py
```

**YOU SHOULD SEE (real output):**

```text
vocab 39 words, context 24, corpus 4916 tokens

A) next-token only          t+1 acc 0.721
B) MTP with 3 heads         t+1 acc 0.721  t+2 acc 0.707  t+3 acc 0.692
                            self-speculative acceptance of the t+2 draft: 0.798  (expected tokens per main-head step ~ 1.80)
```

**Fine-tuning note (important for DeepSeek-V3/R1-style checkpoints):** if a base model ships with MTP modules, most fine-tuning recipes drop or freeze them. Serving engines can use them as the speculative draft **only if you keep them in the checkpoint** you publish. Check the model's `config.json` for `num_nextn_predict_layers` (DeepSeek) before merging and publishing.

---

## CHECKPOINT

- [ ] I can say what a base model is, and why I will usually start from the instruct model.
- [ ] I can draw the CLM, MLM and Prefix-LM masks, and say which my model uses.
- [ ] I can compute C = 6ND for my fine-tuning run and convert it to GPU-hours.
- [ ] I can decide CPT vs straight-to-SFT for my project with the decision tree in 2.4.
- [ ] I ran labs 27, 28, 29 (and 22) and my output matches the shape above.
