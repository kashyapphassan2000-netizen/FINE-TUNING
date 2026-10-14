# 27. Merge, check, and publish (Hugging Face, Google Drive, local, S3)

## 27.1 Merge and prove it

@file src/peftkit/merge.py

:::why
Merging bakes W_{0} + (α/r)BA (or the DoRA/AdaLoRA equivalent) into normal weights. The script runs the same prompt through the adapter model and the merged model and fails if the logits differ by more than a tolerance. In tests all four methods merged with a maximum difference ≤ 5e-6 (float32).
:::

```gpu
source .venv-train/bin/activate
python -m peftkit.merge --adapter outputs/llama31-8b/adapter --out outputs/llama31-8b/merged \
    --base meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16
```

```out
{"base": "meta-llama/Llama-3.1-8B-Instruct", ..., "max_abs_logit_diff": <small>, "out": "outputs/llama31-8b/merged"}
```

In bf16 the tolerance is 1e-2 because rounding differs; the float32 test proves the math.

## 27.2 Sanity generation

@file src/peftkit/generate.py

```gpu
python -m peftkit.generate --model outputs/llama31-8b/merged --prompt "What is the maximum pump pressure?"
```

In the recorded CPU test, a LoRA adapter trained 180 steps on the tiny model answered "The maximum pressure is 6 bar." exactly as in its data.

## 27.3 Adapter or merged? Ask the checker

@file src/peftkit/serve_check.py

```gpu
python -m peftkit.serve_check outputs/llama31-8b/adapter
```

|Method|servable_as_adapter (tested)|Serve|
|---|---|---|
|lora, qlora, loraplus|true|adapter (multi-LoRA) or merged|
|dora|false: "DoRA adapters are not supported by vLLM 0.29.0"|merged|
|adalora|false: peft_type ADALORA|merged|

## 27.4 Publish

@file src/peftkit/publish.py

```gpu Hugging Face (private)
python -m peftkit.publish --src outputs/llama31-8b/merged --hub <you>/llama31-8b-pump --private
python -m peftkit.publish --src outputs/llama31-8b/adapter --hub <you>/llama31-8b-pump-lora --private
```

```gpu Google Drive with rclone (one-time setup on a headless VM)
rclone config        # n (new) → name: gdrive → storage: drive → leave client id empty → scope 1
                     # "Use web browser to automatically authenticate?" → n
# on the LAPTOP run the command rclone prints:   rclone authorize "drive"
# sign in in the browser, copy the token back into the VM prompt
bash scripts/sync_gdrive.sh outputs/llama31-8b/adapter peft/llama31-8b/adapter
```

@file scripts/sync_gdrive.sh

```gpu Local / mounted folder, or S3-compatible storage
python -m peftkit.publish --src outputs/llama31-8b/merged --local /mnt/backup/llama31-8b
AWS_ENDPOINT_URL=<endpoint or empty for AWS> python -m peftkit.publish --src outputs/llama31-8b/merged --s3 s3://models/llama31-8b
```

Optimizer states and training args are skipped on upload (tested). Hyperstack also offers S3-compatible object storage (per its API reference); the `--s3` path works with any S3-compatible endpoint through `AWS_ENDPOINT_URL`.

:::warn Delete the training VM
After the upload is verified (open the repo page on the Hub), delete or hibernate the VM in the console. An idle GPU still bills.
:::

:::check
- Merge report shows a small `max_abs_logit_diff`.
- Model repo visible on the Hub (private); adapter copy in Drive.
- Training VM deleted or hibernated.
:::
