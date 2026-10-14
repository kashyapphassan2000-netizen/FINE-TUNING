# 71. Multi-adapter plug-and-play: one base, many domains

:::why The idea
An adapter is a few megabytes. A base model is tens of gigabytes. So keep **one** base model in GPU memory and clip on many adapters: finance, legal, support, one per customer. vLLM routes each request to the adapter named in the `model` field. Adding a new domain costs a training run and a registry line, not another GPU.
:::

## 71.1 What can and cannot be mixed

|Rule|Why|
|---|---|
|all adapters must share the **same base model and revision**|an adapter is a delta against exact weights; a different revision silently degrades quality|
|adapter format must be plain LoRA (lora / qlora / loraplus)|vLLM 0.29 rejects DoRA and AdaLoRA adapters (verified in its source)|
|no `modules_to_save` (e.g. resized embeddings)|vLLM rejects it: merge and deploy that model separately|
|rank ≤ `--max-lora-rank`, which must be one of 1, 8, 16, 32, 64, 128, 256, 320, 512|vLLM's allowed values|
|`--max-loras` = how many different adapters can be in **one batch**; `--max-cpu-loras` = how many are cached|memory vs flexibility|
|QLoRA adapters serve on the bf16 base|they were trained against a 4-bit view of the same weights; if you see quality loss, merge instead|

## 71.2 The registry

@file configs/adapters.yaml
@file src/peftkit/adapters.py

```any Validate before you deploy anything
python -m peftkit.adapters validate configs/adapters.yaml
```

```out Real output (executed)
OK
```

```any Generate the exact vLLM command
python -m peftkit.adapters vllm-args configs/adapters.yaml
```

```out Real output (executed): note it picked rank 32 (the largest adapter) and loaded the shadow adapter too
vllm serve meta-llama/Llama-3.1-8B-Instruct --enable-lora --max-lora-rank 32 --max-loras 3 --max-cpu-loras 3 --lora-modules finance=your-user/llama31-8b-finance-lora legal=your-user/llama31-8b-legal-lora support=your-user/llama31-8b-support-lora
```

The validator catches, with tests proving each one: bad names, duplicates, DoRA/AdaLoRA adapters, rank over 512, a base mismatch between `adapter_config.json` and the registry, and a rank that disagrees with the adapter file.

`status:` controls the lifecycle: `active` (loaded and routable), `shadow` (loaded so you can evaluate it side by side, never routed automatically), `retired` (not loaded at all). That is how you roll out `finance-v2` without touching `finance`.

## 71.3 Serving many adapters

```gpu Static: everything listed at start-up
.venv-serve/bin/vllm $(python -m peftkit.adapters vllm-args configs/adapters.yaml | cut -d' ' -f2-) \
   --api-key $VLLM_API_KEY --max-model-len 8192
```

```gpu Dynamic: add or drop an adapter without restarting the server
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True     # set before starting vLLM
python -m peftkit.adapters load   --url http://localhost:8000 --name finance-v2 --path your-user/fin-lora-v2
python -m peftkit.adapters unload --url http://localhost:8000 --name finance-v2
```

Those two endpoints (`/v1/load_lora_adapter`, `/v1/unload_lora_adapter`) and their request fields were read from the vLLM v0.29.0 source, and the client is tested here against a fake server.

```py Calling a specific adapter: it is just the model name
client.chat.completions.create(model="finance", messages=[...])   # adapter
client.chat.completions.create(model="legal",   messages=[...])   # different adapter, same GPU
client.chat.completions.create(model="base",    messages=[...])   # untouched base model
```

## 71.4 Routing when the caller does not know which adapter to use

```any Keyword router (baseline, tested)
python -m peftkit.adapters route configs/adapters.yaml "Can you explain the indemnity clause in this contract?"
```

```out Real output (executed)
legal
```

The rule: an explicit adapter name always wins; otherwise keyword scores decide; ties and no-hits fall back to the base model. Keywords are a baseline that you can read and debug. For production, train a small classifier on your own traffic, or ask a cheap model to pick, and keep the fallback. Never let routing fail closed into a wrong domain: answering finance questions with the legal adapter is worse than answering with the base model.

## 71.5 Per-customer access control

The gateway supports an `ADAPTER_ACL` map from a client key hash to the adapters it may use (tested: a key allowed only `finance` gets 200 for `finance` and 403 for `legal`).

```any .env
ADAPTER_ACL={"<sha256 of client key>": ["finance", "base"], "<other key hash>": ["legal"]}
```

That is how you run one GPU for several customers without one customer reaching another's model.

## 71.6 Using several adapters locally (PEFT, no server)

```py One base object, many named adapters
from peftkit.adapters import load_many
model = load_many(base_model, {"finance": "user/fin-lora", "legal": "user/legal-lora"})
model.set_adapter("legal")                       # switch
with model.disable_adapter():                    # plain base model
    ...
out = model(input_ids=batch, adapter_names=["finance", "legal"])   # different adapter per row of one batch
model.add_weighted_adapter(["finance", "legal"], [0.5, 0.5], "fin_legal", combination_type="cat")
model.delete_adapter("legal")
```

All of this is covered by `tests/test_adapters.py`, which proves that switching to an adapter gives **exactly** the same logits as loading that adapter alone, that `disable_adapter()` reproduces the base model, and that a mixed batch applies the right adapter to each row.

:::warn On merging adapters together
`add_weighted_adapter` combines adapters (`linear`, `cat`, `ties`, `dare_ties`, ...). It is useful, but it is not magic: a combined adapter is a new artifact that must be evaluated like any other. Two adapters trained on conflicting styles usually produce a worse third one. Measure before shipping.
:::

## 71.7 Operating a fleet of adapters

|Practice|Why|
|---|---|
|version every adapter (`finance-v3`) and never overwrite|rollback in seconds by flipping `status`|
|keep the base revision pinned in the registry|the single most common silent breakage|
|one eval set per domain, plus a shared regression set|catches an adapter that hurts general behaviour|
|ship new adapters as `shadow` first, compare on real traffic|no-risk rollout|
|store `run_meta.json` next to each adapter|you can rebuild it exactly|
|watch per-adapter request counts and latency in the gateway metrics|find dead adapters and hot ones|
|cap `--max-loras` to what the GPU can hold|too many active adapters in one batch costs throughput|

:::check
- `peftkit.adapters validate` passes for my registry.
- Every adapter lists the same base + revision, and `serve_check` passes for each.
- I can add and remove an adapter at runtime without restarting vLLM.
- Routing has a safe fallback, and keys are restricted with `ADAPTER_ACL`.
:::
