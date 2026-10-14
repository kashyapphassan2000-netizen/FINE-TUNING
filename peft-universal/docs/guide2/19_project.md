# 19. The full project: a domain assistant from zero to an API

This is the complete run-through. Every command exists in the repository, and every stage has a gate you must pass before spending money on the next one.

```text
documents + 30 hand-written questions
   │ 1 test set FIRST (never used for training)
   │ 2 compat + plan          -> is it supported, which GPU, what will it cost
   │ 3 data pipeline          -> ingest, synth, self-instruct, quality, judge, format
   │ 4 doctor                 -> every FAIL fixed
   │ 5 baseline eval          -> the number to beat
   │ 6 dry run (20 steps)     -> peak VRAM, tokens/sec, real cost estimate
   │ 7 train                  -> adapter + run_meta.json
   │ 8 eval + compare         -> domain up? general down? regressions?
   │ 9 failure analysis       -> label 30 failures, fix the biggest bucket, repeat from 3
   │10 merge + serve_check    -> exact merge, servable format
   │11 publish                -> Hub (private) + backup
   │12 vLLM + gateway         -> API with keys, limits, metrics
   ▼
a service, with a measured before/after number
```

## 19.1 The commands, in order

```any 1-2: is this model usable, and what will it cost
python -m peftkit.compat <model>
python -m peftkit.plan <model> --seq-len 4096 --dataset-tokens <tokens> --epochs 2 \
    --tokens-per-sec <after dry run> --price-per-gpu-hour <your price>
```

```any 3: data (laptop, API calls only)
DOMAIN="<your domain>" bash scripts/build_dataset.sh
python -m peftkit.data.quality --inputs data/interim/all.jsonl --out data/interim/clean.jsonl \
    --report reports/quality.json --contamination data/eval/test.jsonl
```

```any 4-6: gates before the long run
python -m peftkit.diagnostics --config configs/base.yaml configs/methods/qlora.yaml configs/models/<m>.yaml
python -m peftkit.evaluate --model <base> --suite domain=data/eval/test.jsonl general=data/eval/general.jsonl --out reports/base.json
python -m peftkit.train --config ... --set train.max_steps=20 train.output_dir=outputs/dryrun
```

```any 7-9: train, measure, improve
python -m peftkit.train --config configs/base.yaml configs/methods/qlora.yaml configs/models/<m>.yaml
python -m peftkit.evaluate --model outputs/<run>/adapter --suite domain=... general=... --out reports/tuned.json
python -m peftkit.evaluate --compare reports/base.json reports/tuned.json --out reports/compare.json
```

```any 10-12: ship
python -m peftkit.merge --adapter outputs/<run>/adapter --out outputs/<run>/merged --base <hub id>
python -m peftkit.serve_check outputs/<run>/adapter
python -m peftkit.publish --src outputs/<run>/merged --hub <you>/<repo> --private
MODEL=<you>/<repo> bash scripts/serve_vllm.sh      # then the gateway, Volume 1 ch 29
```

## 19.2 The gates (do not skip one)

|Gate|Pass condition|
|---|---|
|test set|written before any generation, never in training, `--contamination` clean|
|doctor|zero FAIL|
|baseline|measured on the same suites as the final model|
|dry run|fits in VRAM; tokens/sec measured; cost written down|
|after training|domain score up, general score not down more than 0.05, regressions reviewed|
|merge|`max_abs_logit_diff` tiny|
|serving|`/health` 200 and one real answer through the gateway with an API key|

## 19.3 What to write down at the end

One paragraph: base model and revision, dataset size and sources, method and key hyperparameters, the before/after numbers on N unseen examples, peak VRAM, total GPU-hours and cost, and the known limitations. That paragraph is your model card, your resume bullet, and your handover document.

:::check
- Every gate above passed, in order.
- I can state one measured improvement on unseen data.
- Someone else could repeat my run from the repo and `run_meta.json`.
:::
