# LAB MANUAL — PART 12: Cheat Sheet

```text PART 2 - WORKSPACE (laptop, PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv python install 3.11
winget install --id Git.Git -e ; winget install --id Microsoft.VisualStudioCode -e ; winget install --id WinSCP.WinSCP -e
uv venv .venv --python 3.11 ; .\.venv\Scripts\Activate.ps1
uv pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements-dev.txt ; uv pip install -e .
python -m pytest -q ; python scripts/rehearsal_cpu.py
```

```text PART 4 - BASICS (laptop)
python scripts/make_tiny_model.py
python -m peftkit.tokenize_inspect --tokenizer outputs/tiny --file data/lesson/train.jsonl --index 0
python -m peftkit.diagnostics --config configs/base.yaml configs/methods/lora.yaml configs/lesson.yaml
python -m peftkit.evaluate --model outputs/tiny --suite domain=data/lesson/test.jsonl --max-new-tokens 16 --out reports/lesson_base.json
python -m peftkit.train --config configs/base.yaml configs/methods/lora.yaml configs/lesson.yaml
python -m peftkit.evaluate --model outputs/lesson/adapter --suite domain=data/lesson/test.jsonl --max-new-tokens 16 --out reports/lesson_tuned.json
python -m peftkit.evaluate --compare reports/lesson_base.json reports/lesson_tuned.json
python -m peftkit.merge --adapter outputs/lesson/adapter --out outputs/lesson/merged --dtype float32
```

```text PART 5 - INTERMEDIATE
python -m peftkit.openness MODEL_ID --intended-use commercial
python -m peftkit.plan MODEL_ID --seq-len 4096 --batch 1
python -m peftkit.data.build --docs data/raw --out data/processed --tokenizer MODEL_ID --domain "YOUR_DOMAIN" --eval data/eval/test.jsonl
python -m peftkit.diagnostics --config configs/base.yaml configs/methods/qlora.yaml configs/models/my-model.yaml
ssh-keygen -t ed25519 -f $HOME\.ssh\hyperstack ; ssh hs-train
bash scripts/gpu_preflight.sh ; python3 -m peftkit.gpu_stack ; bash scripts/setup_gpu_vm.sh ; source .venv-train/bin/activate
tmux new -s train   |  Ctrl-b d  |  tmux attach -t train
python -m peftkit.train --config ... --set train.max_steps=20 train.output_dir=outputs/dryrun
python -m peftkit.train --config ...        [--resume true after a crash]
python -m peftkit.evaluate --compare reports/base.json reports/tuned.json --out reports/compare.json
```

```text PART 6 - ADVANCED
python -m peftkit.merge --adapter outputs/RUN/checkpoint-BEST --out outputs/RUN/merged --base MODEL_ID
python -m peftkit.serve_check outputs/RUN/adapter
python -m peftkit.publish --src outputs/RUN/merged --hub HF_USER/REPO --private
MODEL=HF_USER/REPO SERVED_NAME=SERVED_NAME MAX_MODEL_LEN=8192 bash scripts/serve_vllm.sh
python -m peftkit.eval_endpoint --url http://127.0.0.1:8000/v1 --model SERVED_NAME --api-key $VLLM_API_KEY --suite domain=data/eval/test.jsonl --offline reports/tuned.json --parity-out reports/parity.json
python -m peftkit.loadtest --url http://127.0.0.1:8000/v1 --model SERVED_NAME --api-key $VLLM_API_KEY --prompts data/eval/test.jsonl --levels 1 2 4 8 16 32 --slo-p95 2.0 --out reports/loadtest.json
python -m peftkit.release_gate --compare reports/compare.json --parity reports/parity.json --loadtest reports/loadtest.json
MODEL=HF_USER/REPO docker compose -f serving/docker-compose.yml up -d --build
python -m peftkit.keys remote-create --gateway http://127.0.0.1:8080 --admin-key ADMIN --name CUSTOMER --rpm 60 --models SERVED_NAME
python -m peftkit.capacity --model HF_USER/REPO --users N --loadtest reports/loadtest.json --prices configs/providers.yaml
kubectl apply -f serving/k8s/vllm.yaml ; kubectl apply -f serving/k8s/keda-scaledobject.yaml
```

```text PART 7 - ELITE
python -m peftkit.hpo --config ... --space '{"train.lr": [1e-4, 2e-4, 5e-4], "method.r": [8, 16, 32]}' --budget 12 --rungs 20 60 180 --seeds
python labs/lab23_dpo.py            python labs/lab63_universal_pipeline.py
UniversalPipeline(model, batch).plan() -> adapt() -> smoke_test(batch) -> fit(loader) -> export(path)
```

```text PART 4B - WSL
wsl --list --online ; wsl --install -d Ubuntu-24.04 ; wsl --list --verbose
cd ~/code && code .                       (VS Code in WSL; bottom-left shows "WSL: ...")
curl -LsSf https://astral.sh/uv/install.sh | sh ; source $HOME/.local/bin/env
docker run --rm hello-world ; docker compose version      (needs v2.24+ for Part 6B)
```

```text PART 5B - STANDARD LIBRARIES ONLY
python examples/standard/train_sft_raw.py --model M --train T.jsonl --eval E.jsonl --out OUT [--qlora]
python examples/standard/merge_publish_raw.py --base M --adapter OUT/adapter --out OUT/merged [--hub USER/REPO --private]
vllm serve USER/REPO --served-model-name NAME --max-model-len 8192 --api-key $VLLM_API_KEY --host 0.0.0.0 --port 8000
```

```text PART 5C - BENCHMARKS
uv venv .venv-eval --python 3.11 ; source .venv-eval/bin/activate ; uv pip install "lm-eval[vllm]==0.4.13"
lm_eval --model vllm --model_args "pretrained=MODEL,dtype=bfloat16,gpu_memory_utilization=0.85" --tasks mmlu,gsm8k --batch_size auto --apply_chat_template --output_path reports/benchmarks/base [--limit 200]
lm_eval --model hf --model_args "pretrained=MODEL,peft=ADAPTER,dtype=bfloat16" --tasks mmlu,gsm8k --batch_size auto --apply_chat_template --output_path reports/benchmarks/tuned
python -m peftkit.bench_gate --base reports/benchmarks/base --tuned reports/benchmarks/tuned --max-drop 0.02 --out reports/bench_gate.json
python -m peftkit.release_gate --compare ... --parity ... --loadtest ... --bench reports/bench_gate.json
```

```text PART 6B - PRODUCTION FRONT DOOR
nslookup YOUR_DOMAIN
bash scripts/tls_bootstrap.sh YOUR_DOMAIN YOUR_EMAIL
docker compose -f serving/docker-compose.yml -f serving/docker-compose.prod.yml up -d
docker compose -f serving/docker-compose.yml -f serving/docker-compose.prod.yml exec nginx tail -f /var/log/nginx/access.json
nginx -t                                   (validate after every edit)
```

```text PART 7B - DISTRIBUTED
python -m peftkit.batchmath --global-batch 256 --per-device 2 --gpus-per-node 8 --nodes 2
nvidia-smi --list-gpus ; nvidia-smi topo -m ; hostname -I
accelerate launch --config_file configs/accelerate/deepspeed_zero3.yaml examples/standard/train_sft_raw.py ...
accelerate launch --config_file configs/accelerate/multinode_fsdp2.yaml --machine_rank 0|1 examples/standard/train_sft_raw.py ...
```

|Formula|Meaning|
|---|---|
|weights GB ≈ params × bytes/param (bf16 2, fp8 1, NF4 ~0.55)|can it load|
|training GB ≈ 1.1 × (weights + 16 × trainable + activations + logits)|can it train (Adam)|
|KV bytes/token = 2 × layers × kv_heads × head_dim × bytes|memory per user token|
|λ = users × active share × req/min ÷ 60; L = λ × latency|Little's law: concurrent requests|
|replicas = ceil(L × peak × (1 + headroom) ÷ capacity per replica)|fleet size|
|monthly = replicas × GPUs/replica × price/h × 730|serving bill|
|effective batch = per-device batch × grad accumulation × GPUs|stability|
|a gain smaller than the seed spread is noise|when to stop tuning|
|global batch = per-device × accumulation × GPUs/node × nodes; keep it fixed, change accumulation|scaling out|
|benchmark delta below −max_drop (e.g. −0.02) on any general task = forgetting|release gate, 4th check|
