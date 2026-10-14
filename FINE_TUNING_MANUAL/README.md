# FINE-TUNING LAB MANUAL: ZERO TO ELITE

**Any open model (Hugging Face, any open-source repo, or your own local model) → size it (VRAM, RAM, disk, GPU count, hours, cost) → rent GPUs on HyperStack / RunPod / Vast.ai → build the dataset (your PDFs, Excel, Word, CSV, or synthetic data that mimics reality) → JSONL → train → evaluate → tune hyperparameters → publish to your Hugging Face account and a public Space anyone can open.**

*By Karthik Kashyap HV. Notes built on the `peft-universal` toolkit in this repo (`../peft-universal`).*

Assumptions (stated once): you are a beginner; your PC runs Windows 11; you rent NVIDIA GPUs by the hour; your budget is a few dollars per learning session. Versions used: torch 2.14.0, transformers 5.17.0, peft 0.21.0, trl 1.13.0, accelerate 1.15.0, datasets 5.0.1, bitsandbytes 0.50.2, vLLM 0.29.0 (separate env), gradio 6.30.0, huggingface_hub 1.31.0. Prices were read from the providers' own pages on **14 October 2026**. They change monthly, so always re-check.

> **Brutal honesty, before you start.** (1) Fine-tuning teaches **behaviour** (format, tone, task, domain language). It is a poor way to teach **facts that change**: for those, use retrieval (RAG). (2) 500 correct, varied examples beat 50,000 sloppy ones. Data is 80% of the result. (3) "The loss went down" proves nothing. Only a **before/after score on unseen questions** proves the model improved. (4) Every number in Part P03 is an **estimate**: a 20-step dry run on the real GPU is the only truth. (5) What was executed while writing this manual and what was not is stated in every Part.

---

## How to read every section

| Box | Meaning |
|---|---|
| **WHERE / GOAL** | The exact place you work, and what the section achieves |
| **Step N.** | Exactly one action. Do them in order |
| Code box | Copy the whole box and press Enter. Never type the `$` or `>` prompt characters. `powershell` boxes run on your **PC**; `bash` boxes run on the **GPU machine** (or WSL) unless the step says otherwise |
| **YOU SHOULD SEE** | What success looks like. Values in `<angle brackets>` depend on your machine |
| **WATCH OUT** | A mistake people often make at that point |
| **If you see… table** | What to do when the result differs |
| **CHECKPOINT** | Must all be true before you continue |
| **[n]** | Source number; the full list is in [P21_SOURCES.md](P21_SOURCES.md) |

## Placeholders used everywhere

Wherever you see one of these CAPITAL_NAMES, replace it with your own value.

| Placeholder | Replace with | Example |
|---|---|---|
| `MODEL_ID` | Hugging Face id, **or** a local folder that contains `config.json` | `Qwen/Qwen3-8B` or `D:\models\my-model` |
| `SLUG` | short lower-case name for folders and repos | `qwen3-8b-pump` |
| `HF_USER` | your Hugging Face user name | `karthik-ai` |
| `DOMAIN` | a short description of the domain | `industrial pump support` / `clinic operations` |
| `RUN` | the run folder name you choose | `pump-sft-v1` |
| `DATASET_NAME` | the name you give your dataset | `clinic-noshow` |
| `VM_IP` | public IP of the rented GPU machine | `203.0.113.45` |
| `SSH_PORT` | SSH port (22 on HyperStack; RunPod/Vast.ai show their own, e.g. `17445`) | `22` |
| `SSH_USER` | login user (`ubuntu` on HyperStack, `root` on RunPod/Vast.ai) | `ubuntu` |
| `WORK` | work folder on the GPU machine | `/home/ubuntu` (HyperStack) or `/workspace` (RunPod/Vast.ai) |
| `REPO_BRANCH` | branch of this repo that holds the manual | `main` (after you merge) or `claude/youthful-hamilton-c4pf9t` |

## The parts of this manual

### THEORY (learn what each step does; every claim has a runnable lab)

| Part | What you learn | Time |
|---|---|---|
| [T01 Lifecycle + Pre-training](T01_LIFECYCLE_PRETRAINING.md) | pre- vs post-training, base models, CLM/MLM/Prefix-LM, data curation at scale, CPT, scaling laws & compute budget, CPT vs SFT, Multi-Token Prediction | 2 h |
| [T02 Data preparation](T02_DATA_PREPARATION.md) | formats, chat templates (ChatML, Llama-3, Mistral), loss masking, dedup/filter pipelines, Self-Instruct & Alpaca synthetic data | 1.5 h |
| [T03 PEFT](T03_PEFT.md) | intrinsic dimension, LoRA (math, rank, alpha, targets), QLoRA (NF4, double quant, paged optimizers), DoRA, AdaLoRA, LoRA+ | 2 h |
| [T04 SFT](T04_SFT.md) | SFT as stage 1, instruction tuning (FLAN, Alpaca, OpenHermes), chat FT, Chain-of-Thought FT, domain best practice | 1.5 h |
| [T05 Preference alignment](T05_ALIGNMENT.md) | why SFT is not enough, RLHF with PPO (reward model, critic, KL), DPO, GRPO, ORPO, KTO | 2 h |
| [T06 Memory, KV cache, attention](T06_MEMORY_KV_ATTENTION.md) | every byte in training and serving memory, KV cache maths, MHA/GQA/MQA/MLA, FlashAttention, PagedAttention, RoPE | 1.5 h |

### PRACTICAL (one action per step: your flow, in order)

| Part | Your flow step | Time |
|---|---|---|
| [P00 Start here](P00_START_HERE.md) | the map: your 14 steps and where each one lives | 15 min |
| [P01 Laptop setup](P01_LAPTOP_SETUP.md) | tools, accounts, tokens, the project, tests, full CPU rehearsal | 1-2 h |
| [P02 Choose the model](P02_CHOOSE_MODEL.md) | **1. take a model** from Hugging Face / any open source / your own local files; licence; architecture | 30 min |
| [P03 GPU sizing + cost](P03_GPU_SIZING_COST.md) | **2. how many GPUs, VRAM, RAM, disk, hours, dollars**: the formulas by hand + the planner | 1 h |
| [P04 The sizing prompt](P04_SIZING_PROMPT.md) | **3. a copy-paste AI prompt**: replace one link/model and get the full sizing | 10 min |
| [P05 Data strategy](P05_DATA_STRATEGY.md) | **4. what data, how much, for which domain**, eval set first, RAG vs fine-tune | 45 min |
| [P06 Documents → JSONL](P06_DOCUMENTS_TO_JSONL.md) | **5. your PDFs / Excel / Word / CSV / slides → train.jsonl** | 1 h |
| [P07 Synthetic data](P07_SYNTHETIC_DATA.md) | **6. no data? generate it**: LLM-grounded, Self-Instruct, or mimic real data (profile → drivers → contract → generate → gates → name it → JSONL) | 2 h |
| [P08 Rent + connect](P08_RENT_AND_CONNECT.md) | **7. HyperStack / RunPod / Vast.ai**: rent, SSH key, VS Code, WinSCP, tmux | 1 h |
| [P09 Install the stack](P09_INSTALL_STACK.md) | **8. drivers, CUDA, torch, libraries per GPU; load the model and prove it runs** | 45 min |
| [P10 Train](P10_TRAIN.md) | **9. upload train/eval files, configs, doctor, baseline, dry run, real run**, multi-GPU, full FT | 2-6 h |
| [P11 Evaluate](P11_EVALUATE.md) | **10. evaluation**: before/after, regressions, public benchmarks, forgetting gate | 1-2 h |
| [P12 Hyperparameter tuning](P12_HYPERPARAMETER_TUNING.md) | **11. HPO that ends**: what to tune, in which order, seed noise, stopping rule, benchmark targets | 2-4 h |
| [P13 Alignment in practice](P13_ALIGNMENT_PRACTICE.md) | **12. optional: DPO / ORPO / GRPO on your SFT model** | 2-4 h |
| [P14 Publish: Hub + Spaces](P14_PUBLISH_HF_SPACES.md) | **13. merge → model card → your Hugging Face model repo → dataset repo → a public Space anyone can use** | 1 h |
| [P15 Serve (optional)](P15_SERVE.md) | **14. new session: vLLM API, parity, load test** | 1-2 h |
| [P16 Domain playbooks](P16_DOMAIN_PLAYBOOKS.md) | healthcare, legal, finance, support, code, manufacturing, education: what changes | reference |
| [P17 Any architecture](P17_ANY_ARCHITECTURE.md) | dense, MoE, MLA, hybrid/SSM, encoder, encoder-decoder, vision/audio, your own model, future models | reference |
| [P18 Troubleshooting](P18_TROUBLESHOOTING.md) | the problems you will hit, with exact fixes | reference |
| [P19 Cheat sheet](P19_CHEATSHEET.md) | every command and every formula on one page | reference |
| [P20 Self-test](P20_SELFTEST.md) | 20 questions + answer key | 30 min |
| [P21 Sources](P21_SOURCES.md) | every paper, doc and price page cited | reference |
| [TOOLKIT_CHANGES](TOOLKIT_CHANGES.md) | what was wrong in the uploaded toolkit and what was fixed or added, with the evidence | 10 min |

## Environment map: every place you will work

| Place | What happens there |
|---|---|
| Browser | huggingface.co (models, tokens, your repos, Spaces), the GPU provider console, github.com |
| PC: PowerShell + VS Code | install tools, plan, build data, rehearse on CPU, SSH to the GPU |
| PC: WinSCP | drag-and-drop files between your PC and the GPU machine |
| GPU machine: VS Code Remote-SSH terminal (bash) + tmux | install, train, evaluate, tune, merge, publish |
| Hugging Face Hub | source of base models; **your** model repo, dataset repo and Space |

**ONE-LINE FLOW**

```text
PC: choose model -> size + cost -> build/clean/generate data -> JSONL -> rehearse on CPU
  -> rent GPU (HyperStack | RunPod | Vast.ai) -> SSH + WinSCP -> install -> load model
  -> baseline eval -> dry run -> train (tmux) -> evaluate -> tune -> (DPO/ORPO/GRPO)
  -> merge -> push model + dataset to HF -> deploy Space -> DELETE the GPU machine
```

## What was executed while writing this manual

- **Executed (CPU build machine, real output shown):** the 119-test suite of the toolkit (117 passed, 2 skipped: the skips need an NVIDIA GPU), the full CPU rehearsal (data → train → eval → merge → serve → API key → load test → release gate: `REHEARSAL PASSED`), `peftkit.openness` and `peftkit.plan`/`kv` against the live Hugging Face Hub for `Qwen/Qwen3-8B`, the documents → JSONL builder on PDF + Excel + Word, the synthetic-data pipeline with gates, records → named dataset, the Hugging Face Space app (launched locally and called over HTTP), and every new lab (T01-T05).
- **Not executed here (no GPU, no paid accounts):** training on a real GPU, renting machines, pushing to your Hugging Face account, deploying a Space to huggingface.co. Those steps are written from the official docs [1-12] and the toolkit's source; where a menu or flag differs on your screen, look for the same idea and mark it (verify).
