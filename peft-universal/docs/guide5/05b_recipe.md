# 6. The full SFT run on a real model

Everything above is mechanics. This is the sequence you actually type, using the tools from Volume 1, with the gates from Volume 2.

## 6.1 Before any GPU

```any 1. can this model be fine-tuned here, and what will it cost?
python -m peftkit.compat meta-llama/Llama-3.1-8B-Instruct
python -m peftkit.plan  meta-llama/Llama-3.1-8B-Instruct --seq-len 4096 --batch 2
```

```any 2. build the data (Volume 1 ch 15-20, Volume 4 for synthetic)
DOMAIN="industrial pump support" bash scripts/build_dataset.sh
python -m peftkit.data.quality --inputs data/interim/all.jsonl --out data/interim/clean.jsonl \
    --report reports/quality.json --contamination data/eval/test.jsonl
python -m peftkit.data.format --inputs data/interim/clean.jsonl --out-dir data/processed \
    --tokenizer meta-llama/Llama-3.1-8B-Instruct --max-tokens 4096 --eval-ratio 0.05
```

```any 3. look at ONE example exactly as the model will see it
python -m peftkit.tokenize_inspect --tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --file data/processed/train.jsonl --index 0 --max-length 4096
```

```any 4. pre-flight: every FAIL must be fixed before you rent anything
python -m peftkit.diagnostics --config configs/base.yaml configs/methods/qlora.yaml configs/models/llama-3.1-8b.yaml
```

## 6.2 The SFT config

```yaml configs/models/my-sft.yaml
run_name: pump-sft
model:
  name_or_path: meta-llama/Llama-3.1-8B-Instruct
  revision: <commit-sha>                 # pin it
  max_seq_length: 4096                   # p95 of your token lengths
data:
  train_file: data/processed/train.jsonl
  eval_file: data/processed/eval.jsonl
  assistant_only_loss: true              # chapter 1
  packing: false
train:
  output_dir: outputs/pump-sft
  epochs: 2
  per_device_batch_size: 2
  grad_accum: 8                          # effective batch 16
  lr: 2.0e-4
  warmup_steps: 0.03
  eval_steps: 50
  save_steps: 50
  save_total_limit: 3
```

## 6.3 Baseline, dry run, real run

```gpu
# the number to beat, measured BEFORE training
python -m peftkit.evaluate --model meta-llama/Llama-3.1-8B-Instruct \
    --suite domain=data/eval/test.jsonl general=data/eval/general.jsonl --out reports/base.json
# 20 steps: peak VRAM and tokens/sec, so you can price the full run
python -m peftkit.train --config configs/base.yaml configs/methods/qlora.yaml configs/models/my-sft.yaml \
    --set train.max_steps=20 train.output_dir=outputs/dryrun
# the real run, inside tmux
python -m peftkit.train --config configs/base.yaml configs/methods/qlora.yaml configs/models/my-sft.yaml
```

## 6.4 Did it work?

```gpu
python -m peftkit.evaluate --model outputs/pump-sft/adapter \
    --suite domain=data/eval/test.jsonl general=data/eval/general.jsonl --out reports/tuned.json
python -m peftkit.evaluate --compare reports/base.json reports/tuned.json --out reports/compare.json
```

The comparison prints per-suite deltas, the named regressions, and a verdict: PASS, FAIL (the domain did not improve), or WARNING (domain up but general down — the forgetting case measured in chapter 5).

## 6.5 Ship

```gpu
python -m peftkit.merge --adapter outputs/pump-sft/checkpoint-<best>/ --out outputs/pump-sft/merged \
    --base meta-llama/Llama-3.1-8B-Instruct
python -m peftkit.serve_check outputs/pump-sft/adapter
python -m peftkit.publish --src outputs/pump-sft/merged --hub <you>/pump-sft --private
MODEL=<you>/pump-sft bash scripts/serve_vllm.sh
```

:::warn Merge the BEST checkpoint, not the last
Small domain sets overfit inside one epoch. `save_steps` + `eval_steps` at 50 gives you checkpoints to choose from; pick the one with the best eval score (Volume 2, chapter 11).
:::

:::check
- compat + plan + doctor all clean before renting a GPU.
- Baseline measured on the same suites as the final model.
- Best checkpoint merged, `serve_check` passed, one real answer served.
:::
