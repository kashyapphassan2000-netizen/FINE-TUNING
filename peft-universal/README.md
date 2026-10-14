# peft-universal

Config-driven fine-tuning for any Hugging Face causal LM with **LoRA, QLoRA, DoRA, AdaLoRA, LoRA+**, a tested
synthetic-data pipeline, exact adapter merging, and vLLM serving behind an API-key gateway with autoscaling.

## Quick start

**Step-by-step manual for this repo: [`../FINE_TUNING_MANUAL/README.md`](../FINE_TUNING_MANUAL/README.md)**: theory (T01-T06: lifecycle,
pre-training, data, PEFT, SFT, alignment, KV cache) and practicals (P00-P21: model choice → GPU sizing and cost on
HyperStack/RunPod/Vast.ai → documents/synthetic data → JSONL → train → evaluate → tune → publish to the Hub and Spaces).
What was changed in this toolkit and why: [`../FINE_TUNING_MANUAL/TOOLKIT_CHANGES.md`](../FINE_TUNING_MANUAL/TOOLKIT_CHANGES.md).

The original reference book is in `docs/` as Markdown (`docs/manual_src` = lab manual, `docs/master_src` = reference). The PDF edition is NOT shipped in this repo; build it on Linux with the DejaVu fonts installed (`sudo apt-get install fonts-dejavu`): `python docs/assemble_master.py && python docs/build_pdf.py --master Universal_Fine_Tuning_Complete_Guide.pdf`.
The Basics lab runs on any laptop: `python scripts/make_tiny_model.py`, then Part 4 of the manual.
Before renting a GPU, prove the whole pipeline on your laptop:

```bash
python scripts/rehearsal_cpu.py   # data -> train -> evaluate -> merge -> serve -> endpoint eval -> load test -> release gate
```


```bash
# laptop (CPU): run the test suite
uv venv .venv --python 3.11 && source .venv/bin/activate      # Windows: .\.venv\Scripts\Activate.ps1
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements-dev.txt && uv pip install -e .
python -m pytest -q

# size ANY model (millions to trillions): memory, GPUs, multi-node, KV cache, compute, time, cost
python -m peftkit.plan meta-llama/Llama-3.1-8B-Instruct --seq-len 4096 --serve-len 32768 --concurrency 32
python -m peftkit.plan deepseek-ai/DeepSeek-V3 --targets attention --serve-dtype fp8
python scripts/count_tokens.py data/processed/train.jsonl <model>

# build data (needs an OpenAI-compatible API in .env)
DOMAIN="your domain" bash scripts/build_dataset.sh

# GPU VM
bash scripts/gpu_preflight.sh && bash scripts/setup_gpu_vm.sh
METHOD=qlora MODEL_CFG=configs/models/llama-3.1-8b.yaml SKIP_SYNTH=1 bash scripts/pipeline.sh
.venv-train/bin/python -m peftkit.publish --src outputs/llama31-8b/merged --hub <you>/<repo> --private

# serving VM: one base, many adapters
python -m peftkit.adapters validate configs/adapters.yaml
.venv-serve/bin/vllm $(python -m peftkit.adapters vllm-args configs/adapters.yaml | cut -d" " -f2-) --api-key $VLLM_API_KEY
```

## Layout

| Path | What |
|---|---|
| `configs/` | base + method + model YAML, accelerate configs |
| `src/synthkit/` | contracts, profiling, generation (copula/simulator/LLM), perturbation, fidelity metrics, privacy tests, multi-judge, utility (TSTR), gates, forecasting, drift monitoring, vendor catalogue |
| `src/uadapt/` | model-agnostic toolkit: inspection, evidence-based classification, hook tracing, name-free target discovery, custom LoRA for Linear/Conv with exact merge, generic training loop, strategy ranking, UniversalModelAdapter, compatibility report, task registry (19 tasks: schema/loss/collator/metrics), metrics engine (F1, AUROC, ECE, Recall@K, MRR, nDCG, WER, CER), universal processor + collator + loss derivation + UniversalPipeline |
| `examples/standard/` | the same training and merge/publish with transformers + peft + trl only (no peftkit) |
| enterprise additions | `configs/deepspeed/zero2.json`, `zero3.json`, `configs/accelerate/deepspeed_zero3.yaml`, `multinode_fsdp2.yaml`; `peftkit.batchmath` (global batch across GPUs/nodes); `peftkit.bench_gate` (lm-eval forgetting gate, optional 4th release-gate check); `serving/nginx/nginx.conf` + `serving/docker-compose.prod.yml` + `scripts/tls_bootstrap.sh` (TLS with Let's Encrypt renewal, JSON logs, per-IP limits, retries across replicas) |
| flow tools | `peftkit.openness` (licence/weights), `peftkit.gpu_stack` (per-GPU install plan), `peftkit.data.build` (documents → JSONL in one command), `peftkit.hpo` (successive halving + seed-noise floor) |
| `src/peftkit/` | train, merge, generate, publish, serve_local, eval_endpoint, loadtest, release_gate, capacity, keys, showcase, hardware estimator, serve_check, archs + plan (any model 100M-1T+), kv (cache calculator), adapters (registry), compat, diagnostics, evaluate, experiments, tokenize_inspect |
| `src/peftkit/data/` | ingest, synth, self_instruct, judge, preference, quality, templates, format |
| `serving/` | gateway (API keys, rate limits, per-key adapter ACL), docker-compose, Kubernetes + KEDA, VM scaler |
| `src/peftkit/archs.py`, `plan.py` | universal architecture analysis (dense/MoE/MLA/hybrid) and the resource planner |
| `src/peftkit/adapters.py`, `configs/adapters.yaml` | multi-adapter registry: validate, vLLM args, runtime load/unload, routing |
| `labs/` | 52 runnable labs with recorded outputs (PEFT maths, data, debugging, evaluation, KV cache, attention variants, paging, RoPE; new: objectives CLM/MLM/Prefix-LM, scaling laws, Multi-Token Prediction, ORPO, GRPO, RLHF-PPO from scratch) |
| added for the manual | `peftkit.align` (DPO/ORPO/KTO/GRPO), `method: full` (full fine-tuning), `data.completion_only_loss` (any chat template), `peftkit.card` (model card from reports), `peftkit.space` + `serving/space/` (Hugging Face Space), `peftkit.data.records` (records/Excel/CSV → named chat dataset), `synthkit.profile` CLI (feature → target drivers), `ConditionalLabelGenerator`, planner: full FT, host RAM/disk, hours and $ per GPU type, `--dataset` token counting |
| `tests/` | 48 tests (tiny offline model, fake LLM/vLLM/S3) |
| `docs/` | the merged complete guide (`master/`, built by `assemble_master.py`) plus five books: `guide/` (PEFT build guide), `guide2/` (fundamentals & evaluation), `guide3/` (KV cache, attention variants, positional encodings), `guide4/` (enterprise synthetic data engine), `guide5/` (supervised fine-tuning), plus the PDF builder |

## Environments

Training (`requirements-train.txt`, torch 2.14) and serving (`requirements-serve.txt`, vllm 0.29.0 pins torch 2.13)
must be separate virtual environments. PyPI torch wheels here are CUDA 13 builds: the GPU driver must support CUDA 13.

## Results

Fill in with your own measured numbers: base model, method, data size, eval metric vs baseline, peak VRAM, serving latency.

## Licence notes

Check the base model licence and your data/API provider terms before publishing models or datasets.
