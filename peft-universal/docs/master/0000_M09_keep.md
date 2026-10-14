# LAB MANUAL — PART 9: Keep It Working

## SECTION 1 — Save and back up

**WHERE:** laptop PowerShell and the GPU VM

**GOAL:** nothing important lives only on a rented machine

Step 1. Commit and push code and configs after every change: `git add .` → `git commit -m "what changed"` → `git push`.

Step 2. Push every model you keep to the Hub (`peftkit.publish`, Part 6 Section 2).

Step 3. Copy `reports/` and `outputs/RUN/run_meta.json` back to the laptop with WinSCP after every run.

Step 4. Keep datasets versioned: a folder per version (`data/processed/v1`, `v2`) plus the lineage report.

:::check
- Deleting the GPU VM right now would lose nothing you care about.
:::

## SECTION 2 — Update safely

1. Pin versions: the book's verified set is torch 2.14.0, transformers 5.17.0, peft 0.21.0, TRL 1.13.0, vLLM 0.29.0.
2. Newer releases exist (TRL 1.14.0, vLLM 0.30.0, huggingface-hub 2.0.0 on 28 Sep 2026 [6]). Upgrade **one package at a time**, then re-run `python -m pytest -q` and `python scripts/rehearsal_cpu.py`.
3. After upgrading anything on the serving side, re-run the served-model evaluation and the load test (Part 6 Sections 5-6).
4. Update uv itself with `uv self update`.

## SECTION 3 — Restart and resume

|Situation|Command|
|---|---|
|SSH dropped during training|`ssh hs-train` then `tmux attach -t train`|
|training crashed|same training command with `--resume true` added|
|vLLM stopped|re-run `scripts/serve_vllm.sh`; weights are cached in `HF_HOME`|
|gateway stopped|`docker compose -f serving/docker-compose.yml up -d`|

## SECTION 4 — Undo

- **Code or config:** `git log` to find the good commit, `git revert <commit>`.
- **A worse model:** re-deploy the previous Hub revision (every push is a commit on the Hub).
- **A bad adapter in multi-LoRA serving:** unload it at runtime (Reference chapter 71) and load the previous one.
- **A leaked key:** revoke it (`python -m peftkit.keys revoke --file serving/keys.json --key-id <id>` or `DELETE /admin/keys/<id>`), issue a new one.

## SECTION 5 — Stop the bill

Every session ends with: VM deleted or hibernated in the console, `kubectl scale --replicas=0` for demo deployments, and a look at the provider's billing page.
