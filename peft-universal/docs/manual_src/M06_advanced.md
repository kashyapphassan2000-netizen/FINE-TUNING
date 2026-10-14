# LAB MANUAL — PART 6: ADVANCED — Publish, serve, open an API to users, scale

:::why Concept (what, why, when)
Turn the trained adapter into a service strangers can use: merge it, publish it, load it on a fresh GPU with vLLM, **prove** the served model equals the evaluated one, measure how much load one server takes, then put API keys, limits and autoscaling in front. **Why:** a model nobody can call is a science project. **When:** after Part 5 ends with a PASS verdict.
:::

:::honest What was and was not executed
The whole chain — merge, serve behind an OpenAI API, gateway with keys, served-model evaluation, load test, release gate, issuing a key to a new customer — ran end to end on CPU in `scripts/rehearsal_cpu.py` (Part 2, Section 7). The vLLM, Docker and Kubernetes commands below were checked against source and configuration, not executed on a GPU here.
:::

### Placeholders used in this Part

|Placeholder|Replace with|Example|
|---|---|---|
|RUN|your run folder|pump-sft|
|BEST|the best checkpoint number from Part 5 Section 9|100|
|HF_USER|your Hugging Face user name|karthik-ai|
|REPO|the model repository name|pump-assistant-8b|
|SERVED_NAME|the name users put in `model=`|pump|
|CUSTOMER|a name for the person you issue a key to|acme|
|API_URL|your public gateway URL|https://api.example.com|

## SECTION 1 — Merge the best checkpoint

**WHERE:** VS Code terminal on hs-train, `.venv-train` active

**GOAL:** one merged model, proven equal to base + adapter

Step 1. Type and press Enter:

```gpu
python -m peftkit.merge --adapter outputs/RUN/checkpoint-BEST --out outputs/RUN/merged --base MODEL_ID
```

**You should see:** `max_abs_logit_diff` near 1e-6 (the laptop run gave 5.1e-07).

Step 2. Type and press Enter:

```gpu
python -m peftkit.serve_check outputs/RUN/adapter
```

**You should see:** `servable_as_adapter: true` for LoRA/QLoRA (DoRA/AdaLoRA say false: serve the merged model).

:::check
- Merge difference below 1e-4; serve_check read.
:::

## SECTION 2 — Publish to Hugging Face

**WHERE:** same terminal

**GOAL:** the model in a private Hub repository, downloadable from anywhere

Step 1. Type and press Enter:

```gpu
python -m peftkit.publish --src outputs/RUN/merged --hub HF_USER/REPO --private
```

Step 2. In the browser, open `https://huggingface.co/HF_USER/REPO`.

Step 3. Click **Files and versions** and confirm `*.safetensors`, `config.json` and the tokenizer files are present.

Step 4. Click **Model card** → **Edit** and write: base model + revision, method, dataset description, the before/after table from `reports/compare.json`, known limits, licence duties.

:::tip Under the hood (standard libraries)
`merged.push_to_hub("HF_USER/REPO", private=True)` and `tokenizer.push_to_hub(...)` from transformers/huggingface_hub — see `examples/standard/merge_publish_raw.py --hub`.
:::

:::check
- The repository shows the weights and tokenizer; the card has your evaluation numbers.
:::

## SECTION 3 — Stop paying for the training GPU

**WHERE:** browser, Hyperstack console

**GOAL:** no idle billing

Step 1. Click **Virtual Machines**.

Step 2. Click your VM → **Delete** (or **Hibernate** if you will return within hours; storage still bills).

Step 3. Confirm.

:::check
- The VM list no longer shows it as ACTIVE.
:::

## SECTION 4 — New session: bring up vLLM

**WHERE:** a new Hyperstack VM (Part 5, Sections 5-7, using the serving GPU from Section 9 below), VS Code terminal

**GOAL:** an OpenAI-compatible server on port 8000

Step 1. Type and press Enter:

```gpu
source .venv-serve/bin/activate
```

Step 2. Type and press Enter:

```gpu
MODEL=HF_USER/REPO SERVED_NAME=SERVED_NAME MAX_MODEL_LEN=8192 bash scripts/serve_vllm.sh
```

**You should see:** vLLM loading the weights, then a line that the server is listening on port 8000. Loading time ≈ download + disk + engine start (Reference chapter 108).

:::tip Under the hood
The script runs `vllm serve HF_USER/REPO --served-model-name SERVED_NAME --max-model-len 8192 --api-key $VLLM_API_KEY --host 0.0.0.0 --port 8000` (Part 5B, Section 3).
:::

Step 3. Open a second terminal (VS Code: **Terminal > New Terminal**) and type:

```gpu
curl -s localhost:8000/health && echo healthy
```

**You should see:** `healthy`.

:::warn Watch out: versions
This book verified vLLM **0.29.0**; PyPI showed **0.30.0** on 28 Sep 2026 [6]. Keep the pinned version unless you re-run the served-model evaluation after upgrading.
:::

:::check
- `healthy` prints.
:::

## SECTION 5 — Prove the served model is the model you evaluated

**WHERE:** second terminal on the serving VM, `.venv-train` active (the evaluation tools live there)

**GOAL:** parity PASS

Step 1. Type and press Enter:

```gpu
python -m peftkit.eval_endpoint --url http://127.0.0.1:8000/v1 --model SERVED_NAME --api-key $VLLM_API_KEY --suite domain=data/eval/test.jsonl --out reports/served.json --offline reports/tuned.json --parity-out reports/parity.json
```

**You should see:** per-suite scores, then `parity` with `"verdict": "PASS"` (the laptop rehearsal: gap 0.0, zero errors).

|If you see…|It means…|Do this next|
|---|---|---|
|gap > 0.05|serving changed the model|check chat template, dtype/quantization, `MAX_MODEL_LEN`, adapter name, sampling — in that order|
|errors > 0|requests failed|check the API key and that SERVED_NAME matches|

:::check
- Parity PASS.
:::

## SECTION 6 — Load test, then the release gate

**WHERE:** same terminal

**GOAL:** measured capacity per replica, and RELEASE

Step 1. Type and press Enter:

```gpu
python -m peftkit.loadtest --url http://127.0.0.1:8000/v1 --model SERVED_NAME --api-key $VLLM_API_KEY --prompts data/eval/test.jsonl --levels 1 2 4 8 16 32 --requests-per-level 64 --metrics-url http://127.0.0.1:8000/metrics --slo-p95 2.0 --out reports/loadtest.json
```

**You should see:** a table of p50/p95/p99 latency per concurrency level and a recommendation with `max_concurrency_per_replica` and `suggested_scale_up_threshold`.

Step 2. Type and press Enter:

```gpu
python -m peftkit.release_gate --compare reports/compare.json --parity reports/parity.json --loadtest reports/loadtest.json
```

:::tip Add the forgetting check
If you ran Part 5C, append `--bench reports/bench_gate.json` to this command: the gate then also blocks a release whose MMLU/GSM8K scores dropped.
:::

**You should see:**

```out
[PASS] quality   PASS: domain improved, no large general regression
[PASS] parity    PASS
[PASS] capacity  <your number>
decision: RELEASE
```

:::check
- `decision: RELEASE`. On BLOCK, the line names which check failed and where to fix it — do not continue.
:::

## SECTION 7 — Open it to users: gateway and API keys

**WHERE:** the serving VM (an image with Docker)

**GOAL:** a public endpoint that requires a key per user

Step 1. Type and press Enter (prints a client key once and its hash):

```gpu
python -c "import secrets,hashlib;k=secrets.token_urlsafe(32);print('client key:',k);print('hash:',hashlib.sha256(k.encode()).hexdigest())"
```

Step 2. Open `.env` in VS Code and set `GATEWAY_KEY_HASHES` to the printed hash, and `GATEWAY_ADMIN_KEY_HASH` to the hash of a separate admin key made the same way. Save.

Step 3. Type and press Enter:

```gpu
set -a; . ./.env; set +a
```

Step 4. Type and press Enter:

```gpu
MODEL=HF_USER/REPO docker compose -f serving/docker-compose.yml up -d --build
```

Step 5. Type and press Enter:

```gpu
curl -s localhost:8080/healthz
```

Step 6. Issue a key to a user (only its hash is stored; the plaintext is shown once):

```gpu
python -m peftkit.keys remote-create --gateway http://127.0.0.1:8080 --admin-key YOUR_ADMIN_KEY --name CUSTOMER --rpm 60 --models SERVED_NAME
```

Step 7. Give the user this snippet (it is the standard OpenAI SDK; only `base_url` changes):

```py what your user runs
from openai import OpenAI
client = OpenAI(base_url="API_URL/v1", api_key="THE_KEY_YOU_ISSUED")
r = client.chat.completions.create(model="SERVED_NAME", messages=[{"role": "user", "content": "Hello"}])
print(r.choices[0].message.content)
```

**You should see (rehearsal, real):** the customer's request returned HTTP 200 with an answer, and `/admin/usage` showed their requests and tokens.

:::warn Watch out before going public
TLS certificate and a domain; open ports only as needed; never put a key inside a public web page; a sign-up flow with abuse controls; terms of use. The gateway gives you the mechanics, not the legal and product work (Reference chapter 110).
:::

:::tip Going public?
Put the production front door of Part 6B (NGINX with automatic Let's Encrypt TLS, JSON logs, per-IP limits and retries across replicas) in front of this gateway before you share the URL.
:::

:::check
- A request with a user key returns an answer; without a key it returns 401.
:::

## SECTION 8 — Size the fleet and the bill

**WHERE:** laptop or serving VM

**GOAL:** replicas and monthly cost from YOUR measurements

Step 1. Open `configs/providers.yaml` and fill `price_per_gpu_hour` for the GPU you use (from the provider's page).

Step 2. Type and press Enter:

```any
python -m peftkit.capacity --model HF_USER/REPO --users 2000 --active-share 0.1 --requests-per-active-user-per-min 2 --prompt-tokens 600 --output-tokens 250 --latency-s 4 --loadtest reports/loadtest.json --prices configs/providers.yaml
```

**You should see:** the Little's-law line (e.g. `L = lambda x W = 6.667 req/s x 4 s = 26.67 concurrent requests`), replicas per GPU type, and $/month and $/1M output tokens where you entered a price.

:::check
- You can state: replicas needed at peak, GPUs per replica, monthly cost.
:::

## SECTION 9 — Autoscale with Kubernetes and KEDA

**WHERE:** a Kubernetes cluster with GPU nodes, `kubectl` configured

**GOAL:** replicas added when users queue, removed when they leave

Step 1. Open `serving/k8s/keda-scaledobject.yaml` and set `threshold` to `suggested_scale_up_threshold` from `reports/loadtest.json`. Save.

Step 2. Type and press Enter:

```any
kubectl apply -f serving/k8s/vllm.yaml
```

Step 3. Type and press Enter:

```any
kubectl apply -f serving/k8s/keda-scaledobject.yaml
```

Step 4. Type and press Enter:

```any
kubectl get hpa,pods -w
```

**You should see:** the HPA created by KEDA, and pods increasing when the queue passes the threshold.

|Setting|Value|Why|
|---|---|---|
|minReplicaCount|1 or more|scale-to-zero means the first user waits for a cold start|
|threshold|from the load test|a number measured on your model and GPU|
|cooldownPeriod|longer than the cold start|so replicas are not killed while still loading|

:::check
- Under a load test from another machine, pods increase; after it stops, they return to the minimum.
:::

## 6.10 Common mistake at this level

**Scaling a model that was never evaluated as served.** Teams load-test and autoscale a server whose chat template or dtype differs from training, and ship a model that is quietly worse. **How to spot it:** no `reports/parity.json` exists, or nobody looked at it. The release gate refuses to pass without it.
