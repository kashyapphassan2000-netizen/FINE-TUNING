# 16. Step 13 — New session: load from the Hub, serve, open it to users, scale

:::why This is a different day and a different machine
Training is done. Now the model has to answer strangers, fast, without you.
:::

## 16.1 Bring up the server

```gpu
git clone <your repo> && cd peft-universal
bash scripts/gpu_preflight.sh && bash scripts/setup_gpu_vm.sh
source .venv-serve/bin/activate
MODEL=<you>/<repo> SERVED_NAME=mymodel MAX_MODEL_LEN=8192 bash scripts/serve_vllm.sh
curl -s localhost:8000/health && echo healthy
```

Loading takes as long as the maths says (size ÷ bandwidth + disk + engine init) — pre-download to a volume to turn cold starts into warm starts.

## 16.2 Prove it is the same model, and measure what it can take

```gpu
python -m peftkit.eval_endpoint --url http://127.0.0.1:8000/v1 --model mymodel --api-key $VLLM_API_KEY \
   --suite domain=data/eval/test.jsonl general=data/eval/general.jsonl \
   --out reports/served.json --offline reports/tuned.json --parity-out reports/parity.json
python -m peftkit.loadtest --url http://127.0.0.1:8000/v1 --model mymodel --api-key $VLLM_API_KEY \
   --prompts data/eval/test.jsonl --levels 1 2 4 8 16 32 --metrics-url http://127.0.0.1:8000/metrics \
   --slo-p95 2.0 --out reports/loadtest.json
python -m peftkit.release_gate --compare reports/compare.json --parity reports/parity.json --loadtest reports/loadtest.json
```

**No RELEASE, no traffic.** Parity failure means serving changed the model (template, dtype, adapter, sampling).

## 16.3 Open it to users

```gpu
set -a; . ./.env; set +a
MODEL=<you>/<repo> docker compose -f serving/docker-compose.yml up -d --build      # vLLM + gateway + metrics
python -m peftkit.keys remote-create --gateway https://<your-api> --admin-key $ADMIN \
    --name <customer> --rpm 60 --models mymodel
```

Your users then use any OpenAI SDK with `base_url=https://<your-api>/v1` and their key. The gateway enforces per-key limits and model access, meters tokens per key, and skips unhealthy upstreams. A browser front end is optional (`serving/ui/index.html`, hostable free on a static host).

## 16.4 Scale with the users

```any
kubectl apply -f serving/k8s/vllm.yaml
kubectl apply -f serving/k8s/keda-scaledobject.yaml     # threshold = suggested_scale_up_threshold from loadtest.json
kubectl get hpa,pods -w
```

Replicas = peak concurrency × (1 + headroom) ÷ measured capacity per replica. Keep `minReplicas ≥ 1` for a public API (cold starts), and cooldowns longer than the cold start.

## 16.5 Infrastructure as code, and the cloud

```text deploy/terraform/
main.tf         provider, GPU node pool, registry, secrets
vllm.tf         the serving deployment + service
gateway.tf      the gateway deployment + ingress + TLS
monitoring.tf   Prometheus + the KEDA ScaledObject
```

Terraform makes the fleet reproducible: `terraform plan` before every change, state in remote storage, and one variable file per environment. Artifacts (adapters, merged weights, reports) go to S3-compatible storage — rehearse locally against FakeCloud, then switch the endpoint to the real cloud. Budget alerts and a cleanup script are part of the deployment, not an afterthought.

:::check
- Release gate printed RELEASE; a customer key gets an answer; without a key it is 401.
- Autoscaling threshold came from `reports/loadtest.json`.
- The whole fleet can be destroyed and rebuilt from code.
:::
