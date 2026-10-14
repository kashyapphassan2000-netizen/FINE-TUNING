# 32. The code: configuration system

:::why
One YAML per concern: `base.yaml` (defaults), `methods/<m>.yaml` (method math and its best defaults), `models/<m>.yaml` (model id and batch sizes). Files merge left to right; `--set a.b=c` overrides last. `pydantic` validates everything before a GPU minute is spent.
:::

@file src/peftkit/__init__.py
@file src/peftkit/config.py
@file configs/base.yaml
@file configs/methods/lora.yaml
@file configs/models/smollm2-135m.yaml
@file configs/models/llama-3.1-8b.yaml
@file configs/models/qwen-large.yaml

## 32.1 Adding a new model (the universal recipe)

```text configs/models/<your-model>.yaml
run_name: <short-name>
model:
  name_or_path: <org>/<repo>          # any causal LM on the Hub, or a local folder
  revision: <commit-sha>              # pin for production
  trust_remote_code: false            # only true for repos you have read
  max_seq_length: <p95 tokens of your data>
train:
  output_dir: outputs/<short-name>
  per_device_batch_size: <from dry run>
  grad_accum: <target effective batch / per_device / gpus>
```

If `all-linear` is not what you want (e.g., MoE routers), set `method.target_modules` from `inspect_model` output.

## 32.2 Model and PEFT factories

@file src/peftkit/modeling.py
@file src/peftkit/peft_factory.py
@file src/peftkit/inspect_model.py
