# 97. Autoscaling as users grow

## 97.1 The loop

```text
users grow -> λ grows -> requests queue in vLLM (vllm:num_requests_waiting rises)
   -> queue above the MEASURED threshold for N checks -> add a replica (cold/warm start, chapter 94)
   -> gateway gets the new upstream -> queue drains -> latency back under SLO
traffic falls -> queue ~0 for a long cooldown -> remove a replica (never below the minimum)
```

## 97.2 The numbers come from your measurements

|Setting|Source|
|---|---|
|scale-up threshold (queue depth)|`reports/loadtest.json → suggested_scale_up_threshold`|
|capacity per replica|`max_concurrency_per_replica` from the same file|
|max replicas|`peftkit.capacity` at your forecast peak, plus budget|
|min replicas|≥ 1 for a public API (chapter 94)|
|cooldown|longer than the cold start (chapter 94)|
|GPUs per replica|tensor-parallel size from `peftkit.capacity` (`fits_on_tp`)|

## 97.3 Implementations (chapter 56 has the manifests)

- **Kubernetes + KEDA** (recommended): the ScaledObject reads `vllm:num_requests_waiting` from Prometheus; the cluster autoscaler adds GPU nodes when pods are pending.
- **Serverless GPU platforms**: autoscaling is built in; you trade control and cold starts for simplicity.
- **Plain VMs with `serving/scaler.py`**: works for small fleets; its VM-delete call is unverified (chapter 1.3) — keep `--dry-run` until you have confirmed it.

## 97.4 Growth stages, honestly

|Stage|Traffic|Setup|
|---|---|---|
|1 you alone|a few requests|laptop `serve_local`, or one small GPU VM on demand; turn it off when idle|
|2 a few testers|light|one GPU replica + gateway + keys; manual start/stop|
|3 first customers|steady|min 1-2 replicas, KEDA autoscaling, monitoring, measured thresholds|
|4 growth|busy hours|multiple replicas across availability zones, reserved capacity for the base load, spot/on-demand for peaks|
|5 many tenants|many domains|one base + multi-LoRA adapters (chapter 57), per-key model access|
