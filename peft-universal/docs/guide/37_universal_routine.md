# 37. The universal routine: one page, any model, start to finish

Every step is a command that already exists in this repo. Replace `<MODEL>` and the numbers; nothing else changes between a 135M model and a 671B one.

```text The 14 steps
 0  decide: prompt / RAG / fine-tune                        (ch 33.3)
 1  laptop: pytest                                          python -m pytest -q
 2  know the model                                          python -m peftkit.plan <MODEL> --seq-len <L>
 3  know the data size                                      python scripts/count_tokens.py <train.jsonl> <MODEL>
 4  build the dataset                                       DOMAIN="..." bash scripts/build_dataset.sh
 5  pick method + targets                                   configs/methods/<m>.yaml  (+ target_modules for MoE/MLA)
 6  rent the GPU that the plan says                         Hyperstack console (ch 24)
 7  preflight + install                                     bash scripts/gpu_preflight.sh && bash scripts/setup_gpu_vm.sh
 8  dry run 20 steps, read peak VRAM + speed                --set train.max_steps=20
 9  compute time and cost, decide to continue               python -m peftkit.plan ... --tokens-per-sec <measured> --price-per-gpu-hour <p>
10  full run in tmux                                        python -m peftkit.train --config ...
11  merge (if needed) + verify + sanity generate            python -m peftkit.merge ... ; python -m peftkit.generate ...
12  publish + DELETE the training VM                        python -m peftkit.publish --hub <you>/<repo> --private
13  serve                                                   MODEL=<you>/<repo> bash scripts/serve_vllm.sh   (+ gateway)
14  evaluate, register the adapter, iterate                 ch 38 registry, ch 17 judges, human eval set
```

## 37.1 The decision tree

```text Which method, how many GPUs
Does the model fit in bf16 on one GPU (peftkit.plan "LoRA" column)?
├─ yes ─ is quality your bottleneck?  ── yes ─> DoRA (merge before serving)
│                                     └─ no ── LoRA  (+ LoRA+ if you want faster convergence)
└─ no  ─ is it DENSE?
         ├─ yes ─ does QLoRA fit on one GPU? ── yes ─> QLoRA
         │                                    └─ no ─> LoRA + FSDP2 over N GPUs (ch 26.3)
         └─ no (MoE) ─ QLoRA will NOT help (ch 34.8)
                       ├─ fits sharded on one node ─> LoRA, attention-only targets, FSDP2
                       └─ needs multi-node ────────> use an expert-parallel framework (ch 36.6)

Serving: adapter-servable (lora/qlora/loraplus) -> multi-LoRA on one base (ch 38)
         dora/adalora or added tokens          -> merge, serve as a normal model
```

## 37.2 The numbers to write down for every run

|Field|Where it comes from|
|---|---|
|base model + revision (commit sha)|model card|
|method, r, alpha, targets, lr, epochs, effective batch, max_seq_length|your YAML|
|dataset: rows, tokens, synthetic share, filters, judge threshold|`reports/quality.json`, `count_tokens.py`|
|peak VRAM, steps, train loss, eval metric|`run_meta.json`|
|GPU type and count, hours, cost|provider console + chapter 35|
|merge check, serve mode (adapter/merged), adapter name in the registry|`merge_report.json`, `serve_check`, `configs/adapters.yaml`|

If you cannot fill this table, you cannot reproduce the run, and neither can anyone else on your team.
