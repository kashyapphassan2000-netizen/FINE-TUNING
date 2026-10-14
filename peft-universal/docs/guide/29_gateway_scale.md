# 29. API gateway, keys, and autoscaling

## 29.1 Gateway

@file serving/gateway/app.py
@file serving/gateway/Dockerfile

Tested behaviour: missing/wrong key → 401; the client key is never forwarded (vLLM receives its own key); a dead upstream is skipped and marked unhealthy for 30 s; per-key token bucket → 429; streaming passes through; `/metrics` exposes counters; `/admin/upstreams` (admin key only) changes the node list.

```any Create client keys (store only hashes on the server)
python -c "import secrets,hashlib;k=secrets.token_urlsafe(32);print('give to client:',k);print('hash for .env:',hashlib.sha256(k.encode()).hexdigest())"
```

Put hashes, comma-separated, in `GATEWAY_KEY_HASHES`; the admin key hash in `GATEWAY_ADMIN_KEY_HASH`.

:::warn Single-replica limits
Rate-limit buckets live in memory. With more than one gateway replica, move them to Redis. Add TLS with a reverse proxy (Caddy/Nginx) or a cloud load balancer; do not send keys over plain HTTP on the internet.
:::

## 29.2 One VM with Docker Compose

@file serving/docker-compose.yml
@file serving/prometheus.yml

```gpu On a Hyperstack "with Docker" image
set -a; . ./.env; set +a
MODEL=<you>/llama31-8b-pump docker compose -f serving/docker-compose.yml up -d --build
docker compose -f serving/docker-compose.yml logs -f vllm      # wait for healthy
curl -s localhost:8080/healthz
```

The vLLM image tag `v0.29.0` could not be checked from here; confirm it on Docker Hub and never use `latest` in production.

## 29.3 Autoscaling

Autoscaling means adding GPU replicas when requests queue and removing them when idle. The signal is vLLM's `vllm:num_requests_waiting` metric (name read from the v0.29.0 source).

### Option A: Kubernetes + KEDA (recommended)

@file serving/k8s/vllm.yaml
@file serving/k8s/keda-scaledobject.yaml

Requirements: a GPU node pool with the NVIDIA device plugin, Prometheus scraping the pods, KEDA installed, and a node autoscaler so Pending pods get new GPU nodes. Hyperstack offers Kubernetes clusters (its API reference has a Clusters section); EKS/GKE work the same. Not executed here.

```any
kubectl create secret generic llm-secrets --from-literal=vllm-api-key=... --from-literal=hf-token=...
kubectl apply -f serving/k8s/vllm.yaml
kubectl apply -f serving/k8s/keda-scaledobject.yaml
kubectl get hpa,pods -w
```

### Option B: plain VMs with the reference scaler

@file serving/scaler.py
@file scripts/serve_node_bootstrap.sh

The decision logic is unit-tested (patience, cooldown, min/max, metric parsing). VM creation uses the request body from Hyperstack's official API quickstart. Scale-down calls DELETE on the VM path; that path was not verifiable, so it only runs with `--allow-delete` after you confirm it in the API reference. Start with `--dry-run`.

```any
GATEWAY_ADMIN_KEY=... HYPERSTACK_API_KEY=... python serving/scaler.py --dry-run \
  --gateway http://<gw>:8080 --nodes "123=http://10.0.0.5:8000" --environment <env> --keypair <key> \
  --flavor <flavor> --image "<image with CUDA 13 capable driver>" --ssh-key ~/.ssh/hyperstack --allowed-cidr <gw-ip>/32
```

:::warn Cold starts
A new GPU VM needs minutes to boot, install and load weights. Autoscaling absorbs slow load growth, not second-level spikes. Keep `min_nodes ≥ 1`, a long cooldown, and consider a pre-baked snapshot with the env and weights (Hyperstack supports VM snapshots).
:::

:::check
- Gateway rejects a wrong key and serves a right one.
- Metrics are scraped.
- Scaler runs in `--dry-run` and prints sensible decisions.
:::
