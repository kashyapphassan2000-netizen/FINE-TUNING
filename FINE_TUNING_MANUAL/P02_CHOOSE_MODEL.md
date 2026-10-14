# P02 — Step 1: Take the model (Hugging Face, any open source, or your own)

**Time:** 30 minutes. **Cost:** $0 (only `config.json` is downloaded here, a few KB).

## What "a model" must be for fine-tuning

Fine-tuning needs the **weights**. An API-only model (you can only call it over HTTP) **cannot** be fine-tuned by you; only the vendor's own tuning service can. A model is fine-tunable with this manual when you have:

| Item | File(s) | Why |
|---|---|---|
| architecture description | `config.json` | the planner, the loader and PEFT read it |
| weights | `*.safetensors` (preferred) or `pytorch_model*.bin` | what gets loaded and adapted |
| tokenizer | `tokenizer.json` / `tokenizer.model`, `tokenizer_config.json` | text → token ids, and the **chat template** |
| licence | `LICENSE` + the model card | what you may do with it and with your fine-tuned copy |

## SECTION 1 — Pick a candidate (decision table)

**WHERE:** browser → https://huggingface.co/models
**GOAL:** one `MODEL_ID` to size in P03

| Your need | Start with (examples, October 2026; check for newer releases) | Why |
|---|---|---|
| learn the flow, laptop or 1 cheap GPU | `HuggingFaceTB/SmolLM2-135M-Instruct`, `Qwen/Qwen3-0.6B` | minutes per run, cents |
| a real assistant on 1 × 24-48 GB GPU | `Qwen/Qwen3-8B`, `meta-llama/Llama-3.1-8B-Instruct` (gated), `mistralai/Mistral-7B-Instruct-v0.3` | the sweet spot: strong, cheap to tune and serve |
| best quality on 1-2 × 80 GB | 14-32B instruct models (QLoRA on one 80 GB, LoRA on two) | more capacity, 2-4× the cost |
| very large / MoE | `Qwen/Qwen3-235B-A22B`, DeepSeek-V3, Kimi-K2 class | multi-node; read P03 and P17 before spending anything |
| classification / embeddings / NER, not chat | BERT-family encoders (ModernBERT, DeBERTa) | 10-100× cheaper than an LLM for the same job; see P17 |

**Filters on the Hub page:** Tasks → *Text Generation*; Libraries → *Transformers*; Licenses → the ones you can use; sort by *Trending* or *Most downloads*. Prefer the **Instruct** variant (T01 §1.3).

## SECTION 2 — Is it open, and may I use it as I intend?

**WHERE:** PowerShell in `FINE-TUNING\peft-universal`, `(.venv)` active
**GOAL:** a PROCEED / STOP / UNKNOWN decision, with the licence recorded

Step 1. Paste and press Enter (replace `MODEL_ID`; `--intended-use` is `commercial`, `internal` or `research`):

```powershell
python -m peftkit.openness MODEL_ID --intended-use commercial
```

**YOU SHOULD SEE (real output for `Qwen/Qwen3-8B`):**

```json
{
  "model": "Qwen/Qwen3-8B",
  "license": "apache-2.0",
  "license_family": "apache-2.0",
  "open_weights": true,
  "gated": false,
  "commercial_use": true,
  "derivatives": true,
  "redistribute_weights": true,
  "blockers": [],
  "decision": "PROCEED",
  "disclaimer": "engineering check, not legal advice"
}
```

| If you see… | It means… | Do this |
|---|---|---|
| `STOP` … `not open-weights` | no weight files: API-only | pick another model |
| `STOP` … `does not permit commercial use` | e.g. CC-BY-NC | pick another model, or change intended use |
| `GATED` in actions | you must accept terms | P01 Section 3 Step 3, with the same account as your token |
| `UNKNOWN` … `could not read the Hub` | offline, no token, typo in the id | check spelling, set `HF_TOKEN` in `.env`, retry |
| `licence note: community licence…` | Llama-style licence | allowed with duties: acceptable-use policy, attribution, a user-count threshold |

Step 2. Write the decision, the licence and the model **revision** (the commit hash on the model page → *Files and versions* → history) into your project notes. "We used Qwen" is not an audit answer; `Qwen/Qwen3-8B@<commit>` is.

## SECTION 3 — Read the architecture (any model, from config.json only)

**WHERE:** PowerShell
**GOAL:** know the family (dense/MoE), attention type, size and LoRA target names before sizing

Step 1. Paste and press Enter:

```powershell
python -m peftkit.plan MODEL_ID --method lora --seq-len 2048 | Select-Object -First 6
```

**YOU SHOULD SEE (real output for `Qwen/Qwen3-8B`):**

```text
MODEL Qwen/Qwen3-8B  type=qwen3  family=dense  attention=gqa
  layers=36 hidden=4096 heads=32 kv_heads=8 vocab=151936 experts=0 top-k=0 shared=0
  params total=8.19B active/token=8.19B [exact (meta-device instantiation)]
  LoRA target names: ['down_proj', 'gate_proj', 'k_proj', 'o_proj', 'q_proj', 'up_proj', 'v_proj']  fused expert tensors: []
```

**How this works (and why it works for models released next year):** the planner downloads only `config.json`, builds the model on PyTorch's **meta device** (shapes, no memory, no weights) with the real `transformers` code, and counts every parameter exactly. Validated against published sizes: DeepSeek-V3 671.0B (37.6B active), Llama-3.1-405B 405.9B, Qwen3-235B-A22B 235.1B/22.2B active, Kimi-K2 1026.4B/32.9B, Mixtral 46.7B (`tests/test_archs_plan.py`). If `transformers` cannot build a brand-new architecture yet, add `--trust-remote-code` (runs the repo's own Python code: only for repos you trust), or it falls back to a formula estimator and says so (`[formula fallback]`).

| Field | What it tells you |
|---|---|
| `family=moe` | all experts must be in GPU memory; compute uses only active experts (P03, P17) |
| `attention=gqa/mqa/mla/sliding-mix/hybrid-linear` | KV-cache cost class (T06) |
| `kv_heads` | the number that decides serving cost |
| `NOTE: checkpoint is pre-quantized (fp8/…)` | PEFT usually needs a bf16 copy: dequantize first or use a bf16 repo |
| `LoRA target names` | the exact `target_modules` you may use |

## SECTION 4 — Any open-source model that is NOT on Hugging Face

**WHERE:** PowerShell
**GOAL:** a local folder the toolkit can use exactly like a Hub id

Models also come from GitHub releases, ModelScope, Kaggle Models, a vendor download page, or a colleague's disk. The rule: **get it into a folder that looks like a Hugging Face repo** (`config.json` + `*.safetensors` + tokenizer files). Then use the **folder path as `MODEL_ID`** everywhere.

| What you received | Do this |
|---|---|
| a Hugging Face-layout folder | use the path directly: `python -m peftkit.plan D:\models\my-model` |
| ModelScope repo | `pip install modelscope`, then `modelscope download --model ORG/NAME --local_dir D:\models\NAME` |
| `.gguf` file (llama.cpp) | GGUF is an inference format. Find the original safetensors repo (the GGUF's model card names it). `transformers` can dequantize a GGUF with `gguf_file=` but you lose precision: last resort |
| `.pt` / `.pth` state dict of a known architecture | load it into the matching transformers class and `save_pretrained()` (P17 §4) |
| your own PyTorch architecture (not in transformers) | P17 §5: the `uadapt` toolkit finds Linear/Conv layers by tracing, attaches LoRA, trains and merges exactly |
| a model you fine-tuned earlier (merged) | it is already a Hub-layout folder: use it (iterative fine-tuning) |
| an adapter you trained earlier | base model id + adapter folder: `python -m peftkit.generate --model outputs/RUN/adapter …` loads both |

**Download a Hub model to a local folder** (useful to freeze the exact revision, or to work offline):

```powershell
hf auth login
hf download MODEL_ID --local-dir D:\models\SLUG --revision main
python -m peftkit.plan D:\models\SLUG --method lora
```

> **WATCH OUT:** `hf download` fetches all weights (an 8B model ≈ 16 GB). Do this on the GPU machine, not the laptop, unless you really need it locally. For sizing, the laptop only ever needs `config.json`.

## SECTION 5 — Your OWN model (trained by you, or a private repo)

| Case | MODEL_ID to use |
|---|---|
| private Hub repo under your account | `HF_USER/my-model` (the `HF_TOKEN` in `.env` gives access) |
| local folder | the folder path |
| a model you fine-tuned with this manual and published | `HF_USER/SLUG` (fine-tune it again on new data = continual learning: keep replay data, P05 §5) |

---

## CHECKPOINT

- [ ] `MODEL_ID` chosen (Hub id or local folder) and its revision recorded.
- [ ] `peftkit.openness` says PROCEED (or I recorded why UNKNOWN is fine for a local model I own).
- [ ] I know: dense or MoE, attention type, kv_heads, LoRA target names.
