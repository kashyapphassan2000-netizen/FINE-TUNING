# P14 — Step 13: Merge, publish to your Hugging Face account, and open a Space anyone can use

**Time:** 1 hour. **Cost:** a few minutes of GPU time to merge; Hub storage is free within your plan's limits; a Space on free CPU hardware is free.
**GOAL:** (1) a merged model in `HF_USER/SLUG` with a model card built from your evaluation reports, (2) the dataset in `HF_USER/DATASET_NAME`, (3) a Space at `huggingface.co/spaces/HF_USER/SLUG-demo` where anyone can chat with your model, (4) the GPU machine deleted.

## What goes where

| Artifact | Repo type | Visibility (start) | Why |
|---|---|---|---|
| merged model (weights + tokenizer + card) | model: `HF_USER/SLUG` | private → public when ready | what people download and what the Space loads |
| adapter only (tens of MB) | model: `HF_USER/SLUG-adapter` | private | multi-LoRA serving (P15), cheap versioning |
| training dataset | dataset: `HF_USER/DATASET_NAME` | private (public only if no personal or licensed data) | reproducibility and lineage |
| eval set | dataset: `HF_USER/SLUG-eval` | **always private** | a public test set leaks into future training data |
| demo | space: `HF_USER/SLUG-demo` | private → public | "anyone can see my model" |

---

## SECTION 1 — Merge the best checkpoint into the bf16 base

**WHERE:** GPU terminal, `.venv-train`, `.env` loaded
**GOAL:** one ordinary model folder, proven identical to base + adapter

Step 1. Paste and press Enter (`checkpoint-BEST` from P11 Section 3; QLoRA adapters are merged into the **bf16** base, never into the 4-bit one):

```bash
python -m peftkit.merge --adapter outputs/RUN/checkpoint-BEST --out outputs/RUN/merged --base MODEL_ID
```

**YOU SHOULD SEE (real output on the tiny model, fp32):**

```text
{"base": "outputs/tiny", "adapter": "outputs/ckpt_check/checkpoint-6", "peft_type": "LORA", "use_dora": false,
 "max_abs_logit_diff": 2.384185791015625e-07, "out": "outputs/ckpt_check/merged"}
```

`max_abs_logit_diff` ≤ 1e-2 in bf16 (≤ 1e-3 in fp32) = the merged model gives the same outputs as base + adapter. The command exits with an error above the tolerance.

| If you see… | It means… | Do this |
|---|---|---|
| diff large (> 1e-2) | merged onto a different base or revision than you trained on | pass the exact `--base MODEL_ID` (and the same revision) |
| `CUDA out of memory` | the bf16 base does not fit the GPU | merge on CPU: `CUDA_VISIBLE_DEVICES="" python -m peftkit.merge ... --dtype bfloat16` (needs RAM ≈ 1.25 × 2N, P03 HOST line) |
| full fine-tuning | there is nothing to merge | publish `outputs/RUN/model` directly |

Step 2. Is the **adapter** servable as-is by vLLM (multi-LoRA, P15)?

```bash
python -m peftkit.serve_check outputs/RUN/checkpoint-BEST
```

DoRA and AdaLoRA adapters: **no**, so serve the merged model.

## SECTION 2 — Write the model card from your reports

**WHERE:** same terminal
**GOAL:** `outputs/RUN/merged/README.md`: base model, method, data, evaluation numbers, limits, licence

```bash
python -m peftkit.card --run outputs/RUN/run_meta.json --compare reports/compare.json --bench reports/bench_gate.json --dataset HF_USER/DATASET_NAME --license LICENCE_OF_BASE --base-model MODEL_ID --description "WHAT IT DOES" --intended-use "WHO SHOULD USE IT FOR WHAT, AND WHAT NOT" --out outputs/RUN/merged/README.md
```

**YOU SHOULD SEE:** `wrote outputs/RUN/merged/README.md (… chars)`. The card has YAML metadata (`license`, `base_model`, `datasets`, `pipeline_tag`), the training table, the base-vs-tuned evaluation table, the benchmark table, a usage snippet and limitations. Open it and read it: you are publishing it under your name.

> **WATCH OUT:** `--license` is the **base model's** licence (from P02). Community licences (Llama, Gemma) bind your derivative too: keep their naming/attribution rules and acceptable-use policy in the card.

## SECTION 3 — Push the model (private first), with a version

**WHERE:** same terminal
**GOAL:** `HF_USER/SLUG` on the Hub, version branch `v1`

Step 1. Paste and press Enter:

```bash
python -m peftkit.publish --src outputs/RUN/merged --hub HF_USER/SLUG --private
python -m peftkit.publish --src outputs/RUN/merged --hub HF_USER/SLUG --revision v1
```

**YOU SHOULD SEE:** `hub: https://huggingface.co/HF_USER/SLUG/commit/...` twice (main, then the `v1` branch). Optimizer and RNG state files are skipped automatically.

Step 2. (Optional) push the adapter as its own small repo:

```bash
python -m peftkit.card --run outputs/RUN/run_meta.json --compare reports/compare.json --license LICENCE_OF_BASE --base-model MODEL_ID --description "WHAT IT DOES (adapter)" --intended-use "..." --adapter-only --out outputs/RUN/checkpoint-BEST/README.md
python -m peftkit.publish --src outputs/RUN/checkpoint-BEST --hub HF_USER/SLUG-adapter --private
```

Step 3. **Prove the round trip** (download from the Hub into a fresh cache and generate):

```bash
HF_HOME=/tmp/fresh-cache MID="HF_USER/SLUG" python - <<'PY'
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
mid = os.environ["MID"]
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="auto")
enc = tok.apply_chat_template([{"role": "user", "content": "A QUESTION FROM YOUR TEST SET"}], add_generation_prompt=True,
                              return_tensors="pt", return_dict=True).to(model.device)
print(tok.decode(model.generate(**enc, max_new_tokens=64)[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
PY
```

**YOU SHOULD SEE:** the fine-tuned behaviour, from the Hub copy.

## SECTION 4 — Push the dataset and the (private) eval set

```bash
python -m peftkit.publish --src data/datasets/DATASET_NAME-v1.0 --hub HF_USER/DATASET_NAME --repo-type dataset --private
python -m peftkit.publish --src data/eval --hub HF_USER/SLUG-eval --repo-type dataset --private
```

The dataset folder's `README.md` (written by `peftkit.data.records`) becomes the dataset card. For data built with `peftkit.data.build`, copy `train.jsonl`/`eval.jsonl` into a folder with a short `README.md` (source, licence, generator, row counts) before publishing.

> **WATCH OUT:** make a dataset public only if it contains **no personal data** and its sources' licences allow redistribution (`peftkit.data.lineage` from P06 tells you). Never make the eval set public.

## SECTION 5 — A Space: a web page where anyone can chat with your model

**WHERE:** first on the GPU machine (test locally), then the Hub
**GOAL:** `https://huggingface.co/spaces/HF_USER/SLUG-demo`

**Choose the backend first (brutally honest about hardware):**

| Backend | The Space… | Works for | Cost |
|---|---|---|---|
| `local` on free **cpu-basic** | loads the model itself on 2 vCPU / 16 GB RAM | **small models only** (≈ ≤ 1-2B; an 8B model does not fit or takes minutes per answer) | free |
| `local` on paid GPU hardware (`t4-small`, `l4x1`, `a10g-small`, `l40sx1`, `a100-large`, …) | loads the model on a GPU | 7-8B on 24 GB (bf16 tight; better 48 GB), bigger on more | billed per hour while running: set a sleep time |
| `local` on `zero-a10g` (ZeroGPU) | borrows a GPU per request | small/medium models; requires a PRO/organisation plan and `@spaces.GPU` code (not in this template) | plan-dependent |
| `endpoint` | is only the UI; calls **your** OpenAI-compatible server (vLLM/gateway, P15) with a key stored as a Space **secret** | any size | free Space + your server's cost |

Step 1. Test the demo on the GPU machine before deploying (port-forward from your PC: `ssh -L 7860:localhost:7860 gpu`):

```bash
source .venv-train/bin/activate && uv pip install gradio==6.30.0
cd serving/space && MODEL_ID=../../outputs/RUN/merged BACKEND=local SYSTEM_PROMPT="YOUR TRAINED SYSTEM PROMPT" python app.py
```

Open http://localhost:7860 on your PC and ask a test question. (`Ctrl+C` to stop.) The app was launched and called over HTTP while writing this manual: its API endpoint is `/respond`.

Step 2. Deploy (private first). Small model, Space loads it itself:

```bash
cd WORK/FINE-TUNING/peft-universal
python -m peftkit.space --space HF_USER/SLUG-demo --model HF_USER/SLUG --backend local --system "YOUR TRAINED SYSTEM PROMPT" --private --private-model
```

Any size, the Space calls your server (the key goes into a Space **secret**, never into the code):

```bash
export DEMO_API_KEY=THE_KEY_FOR_YOUR_SERVER
python -m peftkit.space --space HF_USER/SLUG-demo --model SERVED_NAME --backend endpoint --endpoint-url https://YOUR-API/v1 --endpoint-key-env DEMO_API_KEY --private
```

Paid GPU hardware for the local backend: add `--hardware l4x1` (or `t4-small`, `a10g-small`, `l40sx1`, …; check prices on the Hub's Spaces pricing page first).

**YOU SHOULD SEE:**

```text
{ "space": "HF_USER/SLUG-demo", "variables": {"MODEL_ID": "...", "BACKEND": "local", "SYSTEM_PROMPT": "..."},
  "secrets": ["HF_TOKEN"], "url": "https://huggingface.co/spaces/HF_USER/SLUG-demo" }
open https://huggingface.co/spaces/HF_USER/SLUG-demo - the first build takes a few minutes (Logs tab shows progress)
```

| What `peftkit.space` does | Detail |
|---|---|
| creates the Space repo | `sdk: gradio`, `sdk_version: 6.30.0`, `app_file: app.py` in the README metadata |
| uploads | `app.py` (Gradio `ChatInterface`, streaming) + `requirements.txt` |
| sets **variables** | `MODEL_ID`, `BACKEND`, `SYSTEM_PROMPT`, `ENDPOINT_URL` |
| sets **secrets** | `ENDPOINT_API_KEY` (endpoint backend), `HF_TOKEN` (only with `--private-model`, so the Space can load your private model) |
| `--dry-run` | writes the Space folder locally and uploads nothing (verified in the test suite) |

Step 3. Open the URL → **Logs** tab until "Running" → ask 5 questions from your test set.

Step 4. Make it public when you are happy, either in the Space's **Settings → Change visibility**, or:

```bash
python -c "from huggingface_hub import HfApi; HfApi().update_repo_settings('HF_USER/SLUG-demo', private=False, repo_type='space')"
```

(The model repo must also be public, or keep `--private-model` so the Space loads it with your token.)

> **WATCH OUT:** a public demo means strangers send it inputs. The Space description already warns that answers can be wrong and asks for no personal data. For paid hardware set a sleep time (Space **Settings → Sleep time**) so an idle Space stops billing.

## SECTION 6 — Close the session (stop paying)

1. Everything you need is on the Hub (model, adapter, dataset, eval) or copied to your PC (`reports/`, `outputs/RUN/run_meta.json`): `scp -r gpu:WORK/FINE-TUNING/peft-universal/reports .`
2. `exit`
3. Delete / terminate / destroy the machine (P08 Section 8). Refresh the console and confirm nothing is running.

---

## CHECKPOINT

- [ ] Merge diff below tolerance; card written from reports and read by me.
- [ ] `HF_USER/SLUG` (main + `v1`) downloads and generates from a fresh cache.
- [ ] Dataset and eval set uploaded, eval set private.
- [ ] The Space runs and answers test questions; public only when ready.
- [ ] The GPU machine is deleted.
