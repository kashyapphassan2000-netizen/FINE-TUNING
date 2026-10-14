# 2. The complete runbook: every command, in order

:::why How to use this chapter
Do the steps in order. Do not skip a CHECK. Each step names the chapter that explains *why*. If a step fails, the troubleshooting tables are in chapters 63 and 40. Copy commands from the ZIP's `docs/master/02_runbook.md` if the PDF wraps a line.
:::

## Stage 1 — Accounts (browser, once)

1. Create accounts and keys: Hugging Face (write token, accept licences of gated models), Hyperstack (billing, environment, SSH key), your LLM API provider (base URL + key), GitHub. **Chapter 16.**

```browser
huggingface.co -> Settings -> Access Tokens -> New token (write)      -> save as HF_TOKEN
console.hyperstack.cloud -> Billing (add credit) -> Environments -> SSH keys
your provider dashboard -> API key + OpenAI-compatible base URL      -> OPENAI_BASE_URL / OPENAI_API_KEY
```

## Stage 2 — Laptop from zero (Windows)

2. Install the tools (**chapter 17**):

```laptop
winget install Git.Git
winget install Microsoft.VisualStudioCode
winget install WinSCP.WinSCP
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# close and reopen the terminal, then:
uv --version
uv python install 3.11
wsl --install -d Ubuntu          # optional: needed for the .sh scripts and FakeCloud
```

3. VS Code extensions: Python, Pylance, Remote - SSH, YAML.

4. Unzip the project and create the virtual environment:

```laptop
mkdir $HOME\code; cd $HOME\code
# unzip peft-universal.zip here
cd peft-universal
uv venv .venv --python 3.11
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
uv pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements-dev.txt
uv pip install -e .
```

5. Secrets file (**chapter 17**, section on `.env`):

```laptop
Copy-Item .env.example .env
notepad .env        # HF_TOKEN, OPENAI_BASE_URL, OPENAI_API_KEY, SYNTH_MODEL, VLLM_API_KEY
```

6. Prove the code works on your machine:

```laptop
python -m pytest -q
```

```out
74 passed, 1 skipped
```

7. Put it under version control (**chapter 17**):

```laptop
git init -b main
git add .
git commit -m "peft-universal: initial"
git remote add origin git@github.com:<you>/peft-universal.git
git push -u origin main
git status          # .env must NOT be listed
```

## Stage 3 — Understand the model before paying for anything

8. Can this model be fine-tuned here? (**chapter 19**)

```laptop
python -m peftkit.compat meta-llama/Llama-3.1-8B-Instruct
```

9. What will it need and cost? (**chapters 21-23**)

```laptop
python -m peftkit.plan meta-llama/Llama-3.1-8B-Instruct --seq-len 4096 --batch 2 --serve-len 32768 --concurrency 32
python -m peftkit.kv   meta-llama/Llama-3.1-8B-Instruct --context 32768 --gpu-gb 48
```

10. Write your model config from the snippet `compat` printed (**chapter 18**): `configs/models/my-model.yaml` with `name_or_path`, a pinned `revision`, `max_seq_length`.

## Stage 4 — Rehearse the ENTIRE pipeline on the laptop (new, do not skip)

11. This runs every stage of this book on a tiny model — train, evaluate, merge, serve, evaluate the served API, load test, release gate — in about two minutes on CPU:

```laptop
python scripts/rehearsal_cpu.py
```

@lab rehearsal_cpu

12. CHECK: the last line says `REHEARSAL PASSED`. If it does not, fix it here, for free, before a GPU is involved.

## Stage 5 — Data (laptop; API calls only)

13. Write the **test set first**: 50-200 real, human-checked items in `data/eval/test.jsonl`, plus 50+ general items in `data/eval/general.jsonl`. Never train on them. (**chapters 25, 44**)

14. Put your documents in `data/raw/` and 20+ real seed tasks in `data/seeds.jsonl`. (**chapters 32, 37**)

15. Build the dataset. The script is bash, so on Windows use WSL or Git Bash, or run its `python -m` lines one by one in PowerShell:

```wsl
DOMAIN="your domain" bash scripts/build_dataset.sh
```

16. Clean and guard against leakage (**chapters 30-31**):

```laptop
python -m peftkit.data.quality --inputs data/interim/fmt/train.jsonl --out data/interim/clean.jsonl --report reports/quality.json --contamination data/eval/test.jsonl
python -m peftkit.data.format --inputs data/interim/clean.jsonl --out-dir data/processed --tokenizer meta-llama/Llama-3.1-8B-Instruct --max-tokens 4096 --eval-ratio 0.05
```

17. Look at one example exactly as the model will (**chapter 8**):

```laptop
python -m peftkit.tokenize_inspect --tokenizer meta-llama/Llama-3.1-8B-Instruct --file data/processed/train.jsonl --index 0
```

18. CHECK: trained fraction between ~2% and ~90%, `EOS trained: True`, no truncation. Read 30 rows by hand.

19. Pre-flight doctor — **zero FAIL** before you rent anything (**chapter 40**):

```laptop
python -m peftkit.diagnostics --config configs/base.yaml configs/methods/qlora.yaml configs/models/my-model.yaml
```

20. Commit and push the data pipeline and configs (not the data if it is private).

## Stage 6 — GPU machine

21. Deploy a Hyperstack VM with the GPU `plan` recommended, an image whose driver supports CUDA 13, SSH enabled (**chapter 41**).

22. Connect: SSH config + VS Code Remote-SSH (**chapter 41**):

```laptop
ssh-keygen -t ed25519 -f $HOME\.ssh\hyperstack
notepad $HOME\.ssh\config      # Host hs-train / HostName <IP> / User ubuntu / IdentityFile ~/.ssh/hyperstack
ssh hs-train
```

23. On the VM: preflight, then the two environments (train and serve are separate because vLLM pins an older torch):

```gpu
git clone git@github.com:<you>/peft-universal.git && cd peft-universal
bash scripts/gpu_preflight.sh          # must print OK: driver can run CUDA 13 wheels
bash scripts/setup_gpu_vm.sh           # creates .venv-train and .venv-serve
cp .env.example .env && nano .env
```

24. Copy the processed data and eval sets to the VM (git, or WinSCP drag-and-drop).

25. CHECK: `cuda available: True` printed by the setup script.

## Stage 7 — Baseline, dry run, train

26. **Baseline** — the number to beat, on the same suites the final model will face (**chapter 44**):

```gpu
source .venv-train/bin/activate
python -m peftkit.evaluate --model meta-llama/Llama-3.1-8B-Instruct --suite domain=data/eval/test.jsonl general=data/eval/general.jsonl --out reports/base.json
```

27. 20-step dry run: peak VRAM and speed (**chapters 42, 23**):

```gpu
tmux new -s train
python -m peftkit.train --config configs/base.yaml configs/methods/qlora.yaml configs/models/my-model.yaml --set train.max_steps=20 train.output_dir=outputs/dryrun
python -c "import json;m=json.load(open('outputs/dryrun/run_meta.json'));print(m['peak_vram_gb'],m['train_seconds'])"
```

28. Price the real run with the measured speed:

```gpu
python -m peftkit.plan meta-llama/Llama-3.1-8B-Instruct --dataset-tokens <tokens> --epochs 2 --tokens-per-sec <measured> --price-per-gpu-hour <your price>
```

29. The real run (inside tmux; `Ctrl-b d` to detach; `--resume true` after a crash) (**chapter 42**):

```gpu
python -m peftkit.train --config configs/base.yaml configs/methods/qlora.yaml configs/models/my-model.yaml
```

## Stage 8 — Evaluate the trained model (the stage the build guide skipped)

30. Same suites, trained adapter:

```gpu
python -m peftkit.evaluate --model outputs/<run>/adapter --suite domain=data/eval/test.jsonl general=data/eval/general.jsonl --out reports/tuned.json
```

31. Compare with the baseline — domain up, general not down, regressions named (**chapters 44-45**):

```gpu
python -m peftkit.evaluate --compare reports/base.json reports/tuned.json --out reports/compare.json
```

32. CHECK: verdict `PASS`. If `FAIL` or `WARNING ... catastrophic forgetting`: stop, read chapters 45-47, fix the data or training, go back to step 29. Do **not** continue to serving.

33. Label 30 failures by cause and fix the biggest bucket first (**chapter 47**).

34. Pick the **best** checkpoint by eval score, not the last one (**chapter 43**).

## Stage 9 — Merge and publish

35. Merge the best checkpoint into the bf16 base, check exactness, check vLLM compatibility (**chapter 51**):

```gpu
python -m peftkit.merge --adapter outputs/<run>/checkpoint-<best> --out outputs/<run>/merged --base meta-llama/Llama-3.1-8B-Instruct
python -m peftkit.serve_check outputs/<run>/adapter
python -m peftkit.publish --src outputs/<run>/merged --hub <you>/<repo> --private
```

36. Delete or hibernate the training VM once the upload is verified.

## Stage 10 — Serve, then evaluate what users actually get

37. New (or same) GPU VM, serving env only, start vLLM (**chapter 52**):

```gpu
MODEL=<you>/<repo> SERVED_NAME=mymodel MAX_MODEL_LEN=8192 bash scripts/serve_vllm.sh
curl -s localhost:8000/health && echo healthy
```

38. Evaluate the **served** model and check parity with the offline result (**chapter 53**):

```gpu
python -m peftkit.eval_endpoint --url http://127.0.0.1:8000/v1 --model mymodel --api-key $VLLM_API_KEY --suite domain=data/eval/test.jsonl general=data/eval/general.jsonl --out reports/served.json --offline reports/tuned.json --parity-out reports/parity.json
```

39. CHECK: parity `PASS` (gap within 0.05, zero errors). If it fails: chat template, dtype/quantization, `max_model_len`, adapter name, sampling — in that order.

## Stage 11 — Capacity, gate, gateway, autoscaling

40. Load test vLLM with realistic prompts and your latency SLO (**chapter 54**):

```gpu
python -m peftkit.loadtest --url http://127.0.0.1:8000/v1 --model mymodel --api-key $VLLM_API_KEY --prompts data/eval/test.jsonl --levels 1 2 4 8 16 32 --requests-per-level 64 --metrics-url http://127.0.0.1:8000/metrics --slo-p95 2.0 --out reports/loadtest.json
```

41. The release gate — all three must pass (**chapter 55**):

```gpu
python -m peftkit.release_gate --compare reports/compare.json --parity reports/parity.json --loadtest reports/loadtest.json
```

42. Gateway with API keys and rate limits in front of vLLM (**chapter 56**):

```gpu
python -c "import secrets,hashlib;k=secrets.token_urlsafe(32);print('client key:',k);print('hash:',hashlib.sha256(k.encode()).hexdigest())"
# put the hash in .env as GATEWAY_KEY_HASHES, then on a "with Docker" image:
set -a; . ./.env; set +a
MODEL=<you>/<repo> docker compose -f serving/docker-compose.yml up -d --build
curl -s localhost:8080/healthz
```

43. Autoscaling, using the **measured** threshold from step 40 (**chapter 56**): put `suggested_scale_up_threshold` into `serving/k8s/keda-scaledobject.yaml` (`threshold`) or `serving/scaler.py --dry-run` first; replicas needed = peak concurrent users ÷ `max_concurrency_per_replica`.

```any
kubectl apply -f serving/k8s/vllm.yaml
kubectl apply -f serving/k8s/keda-scaledobject.yaml
kubectl get hpa,pods -w
```

44. Optional: several domain adapters on one base (**chapter 57**); artifacts to S3 / FakeCloud (**chapter 58**).

## Stage 12 — Public API, capacity, proof

45. Issue API keys to users (only hashes are stored; each key gets its own rate limit and model access) (**chapter 96**):

```gpu
python -m peftkit.keys remote-create --gateway https://<your-api> --admin-key $ADMIN --name <customer> --rpm 60 --models mymodel
```

46. Size the fleet and the bill from the MEASURED load test (**chapters 94-95**):

```gpu
python -m peftkit.capacity --model <you>/<repo> --users 2000 --active-share 0.1 --requests-per-active-user-per-min 2 --prompt-tokens 600 --output-tokens 250 --latency-s 4 --loadtest reports/loadtest.json --prices configs/providers.yaml --json > reports/capacity.json
```

47. Optional public benchmarks on the merged model (MMLU-style tasks need the vllm/hf backend, not a chat API) (**chapter 89**):

```gpu
MODEL=outputs/<run>/merged BASE=meta-llama/Llama-3.1-8B-Instruct bash scripts/run_benchmarks.sh
```

48. Build the stakeholder report from the files (**chapter 99**):

```any
python -m peftkit.showcase --title "My model" --compare reports/compare.json --tuned reports/tuned.json --parity reports/parity.json --loadtest reports/loadtest.json --capacity reports/capacity.json --run outputs/<run>/run_meta.json --out reports/showcase.html
```

49. Not a decoder-only LLM? Replace steps 8-10 and 26-35 with the universal path: `python -m uadapt.report ...`, then `UniversalModelAdapter` → discover → inject → validate → train → evaluate → export (**chapters 68-90, 99**).

## Stage 13 — Close the loop

50. Monitor quality monthly with `eval_endpoint` on a fresh real slice; re-run the load test after any model, GPU or context-length change; write the one-page record (**chapters 59-62**).

:::check The runbook is finished when
- The release gate printed `RELEASE`.
- A request through the gateway with a client key returns an answer; without a key it returns 401.
- Your autoscaling threshold came from `reports/loadtest.json`, not from this book.
:::
