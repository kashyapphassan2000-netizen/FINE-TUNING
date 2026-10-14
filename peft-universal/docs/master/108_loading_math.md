# 108. Loading the model at serve time: the maths

:::why In one line
Every new replica must download and load the weights before serving anyone. That delay decides how early autoscaling must react, and whether scale-to-zero is acceptable.
:::

## 108.1 The terms

$$ weights (GB) = parameters × bytes per parameter     (bf16 2, fp8 1, int4 ≈ 0.5)
$$ download (s) = weights × 8 / network (Gbit/s)
$$ disk→GPU (s) = weights / disk read (GB/s)
$$ cold start = download + disk→GPU + engine init (graph capture, KV allocation)
$$ warm start = disk→GPU + engine init      (weights already on local disk / in the image)

## 108.2 Worked: an 8B model in bf16 (from `peftkit.capacity`)

weights 14.96 GB; at 1 Gbit/s: download ≈ **120 s**; at 2 GB/s disk: ≈ **7 s**; engine init assumed 60 s (measure yours in the vLLM start-up log) → **cold start ≈ 187 s, warm ≈ 67 s**.

A 70B model in bf16 (~141 GB) at the same network speed needs about 19 minutes just to download. That is why production images bake the weights into a volume or snapshot.

## 108.3 Consequences for autoscaling

- Scale up **before** the queue hurts: trigger at a queue depth reached while current replicas still meet the SLO (chapter 68's `suggested_scale_up_threshold`), and set cooldowns longer than the cold start.
- Keep a warm minimum (`minReplicaCount ≥ 1`) for a public API.
- Pre-download weights to a shared volume (`HF_HOME` on persistent disk) or bake them into a snapshot — turns cold starts into warm starts.
- Verify at start-up: log the model revision and a checksum, and fail the health check if they differ from the release.

```any measure it rather than trust the formula
time python -c "from huggingface_hub import snapshot_download; snapshot_download('<you>/<repo>')"
# then read vLLM's start-up log for model load time and CUDA graph capture time
```
