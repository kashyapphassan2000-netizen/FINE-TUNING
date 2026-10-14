# 13. Step 10 — Transfer, configure, run

## 13.1 Push code and data

```laptop
# WinSCP: drag the folder, or:
rsync -avz --exclude .venv --exclude outputs ./ hs-train:~/peft-universal/
scp -r data/processed data/eval hs-train:~/peft-universal/data/
```

## 13.2 The config you are about to run

```yaml configs/models/my-model.yaml
run_name: pump-sft
model: {name_or_path: <org/model>, revision: <commit-sha>, max_seq_length: <p95 tokens>}
data:  {train_file: data/processed/train.jsonl, eval_file: data/processed/eval.jsonl, assistant_only_loss: true}
train: {output_dir: outputs/pump-sft, epochs: 2, per_device_batch_size: 2, grad_accum: 8,
        lr: 2.0e-4, warmup_steps: 0.03, eval_steps: 50, save_steps: 50, save_total_limit: 3}
```

## 13.3 Run it

```gpu
tmux new -s train
source .venv-train/bin/activate
python -m peftkit.diagnostics --config configs/base.yaml configs/methods/qlora.yaml configs/models/my-model.yaml
python -m peftkit.evaluate --model <org/model> --suite domain=data/eval/test.jsonl general=data/eval/general.jsonl --out reports/base.json
python -m peftkit.train --config configs/base.yaml configs/methods/qlora.yaml configs/models/my-model.yaml --set train.max_steps=20 train.output_dir=outputs/dryrun
python -m peftkit.train --config configs/base.yaml configs/methods/qlora.yaml configs/models/my-model.yaml
# Ctrl-b d to detach; tmux attach -t train to return
```

|Watch|Healthy|
|---|---|
|`trainable params` in the log|not 0, and the share you expected|
|loss in the first 20 steps|falling|
|`grad_norm`|bounded, not spiking|
|eval every 50 steps|improving, then flattening|
|`nvidia-smi`|memory stable, utilisation high|

If it crashes: add `--resume true` to the same command; it continues from the last checkpoint. If the loss is flat or NaN, stop and use the debugging table — do not "try another learning rate" blindly.
