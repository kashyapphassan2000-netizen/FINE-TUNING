# P15 — Step 14 (optional): New session — serve the model as an API, prove parity, measure capacity

**Time:** 1-2 hours. **Cost:** GPU time while the server runs.
**GOAL:** your Hub model behind an OpenAI-compatible API (vLLM), proven to score like the offline model, with a measured capacity per GPU. This is what the Space's `endpoint` backend (P14) and real users call.

> This Part is the short path. The toolkit's full serving chapters (API keys, rate limits, NGINX + TLS, Docker, Kubernetes + KEDA autoscaling, multi-LoRA) are in `peft-universal/docs/manual_src/M06_advanced.md`, `M06b_gateway_prod.md` and `docs/master_src/F14_step13_serve.md`.

## SECTION 1 — Start vLLM from the Hub

**WHERE:** a GPU machine (P08-P09: `setup_gpu_vm.sh` already created `.venv-serve`)
**GOAL:** `http://127.0.0.1:8000/v1` answering with your model

```bash
cd WORK/FINE-TUNING/peft-universal && set -a; . ./.env; set +a
tmux new -s serve
MODEL=HF_USER/SLUG SERVED_NAME=SLUG MAX_MODEL_LEN=8192 bash scripts/serve_vllm.sh
```

Second tmux window (`Ctrl+b c`):

```bash
curl -s localhost:8000/health && echo healthy
curl -s localhost:8000/v1/chat/completions -H "Authorization: Bearer $VLLM_API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "SLUG", "messages": [{"role": "user", "content": "A QUESTION FROM YOUR TEST SET"}], "max_tokens": 128}'
```

**Many adapters on one base** (one GPU serves several customers' fine-tunes; LoRA/LoRA+ only, merge DoRA/AdaLoRA):

```bash
MODEL=MODEL_ID ADAPTERS="support=HF_USER/SLUG-adapter legal=HF_USER/OTHER-adapter" MAX_LORA_RANK=16 bash scripts/serve_vllm.sh
```

| `serve_vllm.sh` variable | Meaning |
|---|---|
| `MAX_MODEL_LEN` | max context per request: KV memory grows linearly with it (T06) |
| `GPU_UTIL` (default 0.90) | share of VRAM vLLM may take (weights + KV cache) |
| `TP` (default: all GPUs) | tensor-parallel size: GPUs per replica |

## SECTION 2 — Parity: is the served model the model you evaluated?

```bash
source .venv-train/bin/activate
python -m peftkit.eval_endpoint --url http://127.0.0.1:8000/v1 --model SLUG --api-key $VLLM_API_KEY --suite domain=data/eval/test.jsonl general=data/eval/general.jsonl --max-tokens 256 --out reports/served.json --offline reports/tuned.json --parity-out reports/parity.json
```

Use the **same** `--max-tokens` as the offline evaluation (256 in P11), or the comparison measures your settings, not the model. **PASS** = served scores match offline scores. **FAIL** = serving changed the model: check, in this order, the chat template, dtype, `MAX_MODEL_LEN` (truncation), the adapter name, and sampling settings.

## SECTION 3 — Capacity: how many concurrent users per GPU

```bash
python -m peftkit.loadtest --url http://127.0.0.1:8000/v1 --model SLUG --api-key $VLLM_API_KEY --prompts data/eval/test.jsonl --levels 1 2 4 8 16 32 --metrics-url http://127.0.0.1:8000/metrics --slo-p95 2.0 --out reports/loadtest.json
python -m peftkit.release_gate --compare reports/compare.json --parity reports/parity.json --loadtest reports/loadtest.json --bench reports/bench_gate.json   # drop --bench if you skipped P11 Section 4
python -m peftkit.capacity --model HF_USER/SLUG --users 2000 --active-share 0.1 --requests-per-active-user-per-min 2 --prompt-tokens 600 --output-tokens 250 --latency-s 4 --loadtest reports/loadtest.json --prices configs/providers.yaml
```

**Release only on `decision: RELEASE`** (quality + parity + capacity + benchmarks). `capacity` turns the measured per-replica capacity into replicas and monthly cost with Little's law (L = λ × W). Fill `price_per_gpu_hour` in `configs/providers.yaml` with your quotes first.

## SECTION 4 — When you are done

Stop the server (`Ctrl+C` in the serve window) and delete the machine (P08 Section 8). A public API needs the gateway (API keys, limits) and TLS: `M06b_gateway_prod.md`.

---

## CHECKPOINT

- [ ] vLLM answers with my model; parity PASS.
- [ ] I know the measured concurrency per replica at my latency target, and the monthly cost.
