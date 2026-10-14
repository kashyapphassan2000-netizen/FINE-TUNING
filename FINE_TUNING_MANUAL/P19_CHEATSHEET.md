# P19 — Cheat sheet: every command and every formula

All commands run from `FINE-TUNING/peft-universal` with the right environment active (`.venv` on the PC, `.venv-train` on the GPU) and `.env` loaded on the GPU (`set -a; . ./.env; set +a`).

## The flow in 30 commands

```bash
# ---------- PC (laptop, $0) ----------
python -m pytest -q && python scripts/rehearsal_cpu.py                                   # prove the pipeline
python -m peftkit.openness MODEL_ID --intended-use commercial                            # 1 open? licence?
python -m peftkit.compat MODEL_ID                                                        #   supported path?
python -m peftkit.plan MODEL_ID --seq-len 2048 --dataset data/processed/train.jsonl --epochs 3 \
       --gpu-price RTX4090-24=0.53 L40S-48=0.73 A100-80=1.60 H100-80=2.50                # 2 VRAM, GPUs, hours, $
python -m peftkit.kv MODEL_ID --context 32768 --gpu-gb 80                                #   KV cache / users
python scripts/count_tokens.py data/processed/train.jsonl MODEL_ID                       #   exact tokens, p95
python -m peftkit.data.build --docs data/raw --out data/processed --tokenizer MODEL_ID \
       --domain "DOMAIN" --mode grounded --per-chunk 3 --eval data/eval/test.jsonl --system "..."   # 5 docs -> JSONL
python -m peftkit.data.records --input faq.xlsx --name NAME --task template --user "{question}" --assistant "{answer}"
python -m peftkit.data.self_instruct --seeds data/seeds.jsonl --domain "DOMAIN" --n 300 --out data/interim/si.jsonl
python -m synthkit.profile --real real.jsonl --numeric ... --categorical ... --target TARGET    # 6 drivers
python -m synthkit.pipeline --real real.jsonl --contract contracts/X.yaml --numeric ... --categorical ... \
       --label TARGET --n 5000 --smoothing 0.5 --out reports/synth                       #   generate + 10 gates
python -m peftkit.data.records --input reports/synth/synthetic.jsonl --name NAME --task predict --target TARGET
python -m peftkit.data.quality --inputs a.jsonl --out clean.jsonl --near-dup 0.8 --pii scrub --contamination data/eval/test.jsonl
python -m peftkit.data.lineage --inputs train.jsonl --stamp --source-id X --license owned --source-type synthetic
python -m peftkit.tokenize_inspect --tokenizer MODEL_ID --file data/processed/train.jsonl --index 0 --completion-only
# ---------- GPU machine ----------
bash scripts/gpu_preflight.sh && bash scripts/setup_gpu_vm.sh                            # 8 stack
python -m peftkit.diagnostics --config configs/base.yaml configs/methods/qlora.yaml configs/models/SLUG.yaml   # doctor
python -m peftkit.evaluate --model MODEL_ID --suite domain=data/eval/test.jsonl general=data/eval/general.jsonl --max-new-tokens 256 --out reports/base.json
python -m peftkit.train --config ... --set train.max_steps=20 train.output_dir=outputs/dryrun     # dry run
python -m peftkit.train --config configs/base.yaml configs/methods/qlora.yaml configs/models/SLUG.yaml   # 9 train
python -m peftkit.train --config ... --resume true                                       #   after a crash
python -m peftkit.evaluate --model outputs/RUN/adapter --suite ... --out reports/tuned.json       # 10 evaluate
python -m peftkit.evaluate --compare reports/base.json reports/tuned.json --out reports/compare.json
python -m peftkit.bench_gate --base reports/benchmarks/base --tuned reports/benchmarks/tuned --max-drop 0.02
python -m peftkit.hpo --config ... --space '{"train.lr":[5e-5,1e-4,2e-4,5e-4],"method.r":[8,16,32]}' --budget 12 --seeds  # 11
python -m peftkit.align dpo --model outputs/RUN/merged --data preference.jsonl --out outputs/RUN-dpo     # 12 (optional)
python -m peftkit.merge --adapter outputs/RUN/checkpoint-BEST --out outputs/RUN/merged --base MODEL_ID  # 13 publish
python -m peftkit.card --run outputs/RUN/run_meta.json --compare reports/compare.json --license L --base-model MODEL_ID --description "..." --intended-use "..." --out outputs/RUN/merged/README.md
python -m peftkit.publish --src outputs/RUN/merged --hub HF_USER/SLUG --private
python -m peftkit.publish --src data/datasets/NAME-v1.0 --hub HF_USER/NAME --repo-type dataset --private
python -m peftkit.space --space HF_USER/SLUG-demo --model HF_USER/SLUG --backend local --private --private-model
```

## Connection (PowerShell)

```powershell
ssh-keygen -t ed25519 -f $HOME\.ssh\gpu_ed25519        # once
notepad $HOME\.ssh\config                              # Host gpu / HostName VM_IP / Port SSH_PORT / User SSH_USER / IdentityFile ~/.ssh/gpu_ed25519
ssh gpu                                                # terminal
scp -r data\processed gpu:WORK/FINE-TUNING/peft-universal/data/     # files (or WinSCP, SFTP)
ssh -L 7860:localhost:7860 -L 6006:localhost:6006 gpu  # see the demo / TensorBoard on the PC
```

`tmux new -s train` · detach `Ctrl+b d` · `tmux attach -t train` · new window `Ctrl+b c`.

## Formulas

```text
PARAMETERS (dense)   N ≈ L·[h·n_q·d + 2·h·n_kv·d + n_q·d·h + 3·h·i] + V·h·(2 if untied else 1)
PARAMETERS (MoE)     N_total uses all E experts (MEMORY); N_active uses top-k + shared (COMPUTE)
LoRA params          Σ_targets r·(d_in + d_out)          LoRA update  W' = W + (α/r)·B·A,  B₀ = 0
TRAIN MEMORY         full FT 16 B/param | LoRA 2N + 16·N_lora | QLoRA ≈0.55·N + 10·N_lora
                     + activations (≈ L·2·seq·batch·h·2 B, checkpointed) + logits (batch·seq·V·4 B); × 1.1 + 1.5 GB
GPUs                 ceil(M / (VRAM × 0.9)) with sharding (FSDP/ZeRO-3); DDP does not reduce per-GPU memory
HOST                 disk ≥ 2N + 2N + checkpoints + 40 GB;  RAM ≈ 1.25·2N + 16 GB (CPU merge)
FLOPs                PEFT 6·N_active·tokens | full FT 8·N·tokens (gradient checkpointing)
HOURS                FLOPs / (GPUs · peak · MFU) / 3600     MFU: full .40 | LoRA .35 | QLoRA/DoRA .25
COST                 GPU-hours × price (+30-50% for eval, dry run, idle)
SERVE                weights (2 B bf16 | 1 fp8 | ~0.5 int4) + KV + 1.5 GB at ≤ 90% VRAM
KV per token         2·layers·kv_heads·head_dim·bytes   (MLA: layers·(kv_rank + rope_dim)·bytes)
USERS per GPU        (0.9·VRAM − weights − 1.5 GB) / (KV per token × context)
GLOBAL BATCH         per_device × grad_accum × GPUs × nodes  (keep it fixed when adding GPUs)
SCALING              C ≈ 6ND;  compute-optimal D ≈ 20·N;  Chinchilla L = 1.69 + 406.4/N^0.34 + 410.7/D^0.28
LITTLE'S LAW         concurrent requests L = arrival rate λ × latency W
DPO                  −log σ(β[log π(y_w)/π_ref(y_w) − log π(y_l)/π_ref(y_l)])
GRPO advantage       (r_i − mean_group) / std_group;   PPO clip  min(ρA, clip(ρ, 1±ε)A)
ORPO                 L_SFT + λ·(−log σ(log odds(y_w) − log odds(y_l)))
```

## Starting hyperparameters

| | LoRA / QLoRA | DoRA | full FT | DPO (LoRA) | GRPO (LoRA) |
|---|---|---|---|---|---|
| lr | 2e-4 | 1e-4 | 1e-5 | 5e-5 | 1e-5 |
| r / α | 16 / 32 | 16 / 32 | — | 16 / 32 | 16 / 32 |
| epochs | 1-3 | 1-3 | 1-3 | 1 | — (steps) |
| effective batch | 16-64 | 16-64 | 64-256 | 16-64 | G = 8 per prompt |
| other | `completion_only_loss: true` | merge before vLLM | ZeRO-3 | β 0.1 | β 0 (TRL default) |
