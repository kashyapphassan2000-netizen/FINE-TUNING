# 114. PART L — The universal one-shot template: any model, any domain

Fill the angle brackets, run top to bottom. Every command is in the ZIP and explained in the chapter shown.

```text 0. DECIDE WHAT YOU ARE BUILDING (write these down first)
  task:        <what the model must do>            domain: <e.g. finance support, pump maintenance>
  success:     <metric> on <eval set> must go from <baseline> to at least <target>
  users:       <registered users>, <active share>, <requests/min>, latency SLO <seconds>
  budget:      <GPU price per hour you are quoted>, <max per month>
```

```any 1. MODEL: identify, classify, size (ch 83-83, 19, 22)
python -m uadapt.report --model <org/model> --head <causal-lm|seq-cls|seq2seq|ctc|image-cls|base>
python -m peftkit.compat <org/model>                                   # decoder-only LMs: tokenizer + template checks
python -m peftkit.plan   <org/model> --seq-len <p95 tokens> --serve-len <context> --concurrency <n>
```

```any 2. DATA: real seed + synthetic + quality + eval sets (ch 38-51, 86, 88)
# eval FIRST: data/eval/test.jsonl (domain), general.jsonl, safety.jsonl  - real, human-checked, frozen
DOMAIN="<domain>" bash scripts/build_dataset.sh
python -m peftkit.data.quality --inputs data/interim/fmt/train.jsonl --out data/interim/clean.jsonl --report reports/quality.json --contamination data/eval/test.jsonl
python -m peftkit.data.format  --inputs data/interim/clean.jsonl --out-dir data/processed --tokenizer <org/model> --max-tokens <p95>
python -m peftkit.tokenize_inspect --tokenizer <org/model> --file data/processed/train.jsonl --index 0
```

```any 3. PRE-FLIGHT and REHEARSAL (ch 54, 2)
python -m peftkit.diagnostics --config configs/base.yaml configs/methods/<lora|qlora|dora|adalora|loraplus>.yaml configs/models/<model>.yaml
python scripts/rehearsal_cpu.py
```

```gpu 4. GPU, BASELINE, DRY RUN, TRAIN (ch 55-57, 91)
bash scripts/gpu_preflight.sh && bash scripts/setup_gpu_vm.sh && source .venv-train/bin/activate
python -m peftkit.evaluate --model <org/model> --suite domain=data/eval/test.jsonl general=data/eval/general.jsonl safety=data/eval/safety.jsonl --out reports/base.json
python -m peftkit.train --config configs/base.yaml configs/methods/<method>.yaml configs/models/<model>.yaml --set train.max_steps=20 train.output_dir=outputs/dryrun
python -m peftkit.train --config configs/base.yaml configs/methods/<method>.yaml configs/models/<model>.yaml
# non-decoder or custom architectures: the uadapt path instead (ch 98)
#   UniversalModelAdapter(model, batch) -> discover -> inject -> validate_training -> train(loader, loss_fn)
```

```gpu 5. EVALUATE, COMPARE, BENCHMARK (ch 58-61, 88)
python -m peftkit.evaluate --model outputs/<run>/adapter --suite domain=data/eval/test.jsonl general=data/eval/general.jsonl safety=data/eval/safety.jsonl --out reports/tuned.json
python -m peftkit.evaluate --compare reports/base.json reports/tuned.json --out reports/compare.json
MODEL=outputs/<run>/merged BASE=<org/model> bash scripts/run_benchmarks.sh          # optional public benchmarks
```

```gpu 6. MERGE, CHECK, PUBLISH TO HUGGING FACE (ch 65)
python -m peftkit.merge --adapter outputs/<run>/checkpoint-<best> --out outputs/<run>/merged --base <org/model>
python -m peftkit.serve_check outputs/<run>/adapter
python -m peftkit.publish --src outputs/<run>/merged --hub <you>/<repo> --private
```

```gpu 7. NEW GPU SESSION: LOAD FROM THE HUB AND SERVE WITH vLLM (ch 66, 93)
MODEL=<you>/<repo> SERVED_NAME=<name> MAX_MODEL_LEN=<context> bash scripts/serve_vllm.sh
curl -s localhost:8000/health
```

```gpu 8. EVALUATE THE SERVED MODEL, LOAD TEST, GATE (ch 67-69)
python -m peftkit.eval_endpoint --url http://127.0.0.1:8000/v1 --model <name> --api-key $VLLM_API_KEY --suite domain=data/eval/test.jsonl general=data/eval/general.jsonl --out reports/served.json --offline reports/tuned.json --parity-out reports/parity.json
python -m peftkit.loadtest --url http://127.0.0.1:8000/v1 --model <name> --api-key $VLLM_API_KEY --prompts data/eval/test.jsonl --levels 1 2 4 8 16 32 --metrics-url http://127.0.0.1:8000/metrics --slo-p95 <seconds> --out reports/loadtest.json
python -m peftkit.release_gate --compare reports/compare.json --parity reports/parity.json --loadtest reports/loadtest.json
```

```gpu 9. CAPACITY AND COST FOR YOUR USERS (ch 107-108)
python -m peftkit.capacity --model <you>/<repo> --users <n> --active-share <a> --requests-per-active-user-per-min <r> --prompt-tokens <p> --output-tokens <o> --latency-s <w> --loadtest reports/loadtest.json --prices configs/providers.yaml --json > reports/capacity.json
```

```gpu 10. PUBLIC API: GATEWAY, KEYS, AUTOSCALING (ch 110-110, 56)
MODEL=<you>/<repo> docker compose -f serving/docker-compose.yml up -d --build
python -m peftkit.keys remote-create --gateway https://<your-api> --admin-key $ADMIN --name <customer> --rpm <limit> --models <name>
kubectl apply -f serving/k8s/vllm.yaml && kubectl apply -f serving/k8s/keda-scaledobject.yaml     # threshold from loadtest.json
```

```any 11. SHOW IT (ch 113)
python -m peftkit.showcase --title "<project>" --compare reports/compare.json --tuned reports/tuned.json --parity reports/parity.json --loadtest reports/loadtest.json --capacity reports/capacity.json --run outputs/<run>/run_meta.json --out reports/showcase.html
```

:::check The template is complete when
- The release gate printed RELEASE.
- A new customer key receives an answer through the public URL.
- `reports/showcase.html` exists and every number in it came from a file.
:::
