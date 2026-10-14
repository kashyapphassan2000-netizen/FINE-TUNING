# P00 — START HERE

## What fine-tuning is (one paragraph)

A pre-trained model already understands language. **Fine-tuning** continues its training on **your** examples so it adopts **your** behaviour: your format, your domain words, your tone, your tasks. With **PEFT** (LoRA and relatives) you train a small add-on (0.1-2% of the weights) instead of the whole model, so an 8B model fits on one rented 24-48 GB GPU and costs a few dollars to train. The output is an **adapter** (tens of MB) that you merge into the base model and publish.

**Why people do it:** a fine-tuned 8B model can match a much bigger general model on one narrow job, for a fraction of the running cost, on hardware and data you control.
**Where it is used:** support assistants, structured extraction (invoices, medical notes, contracts), classification/routing, code assistants on a company codebase, domain chat in regulated industries (healthcare, finance, legal) where data cannot leave the company.

## Your flow, mapped (every step you asked for → where it is done)

| # | Your step (as you described it) | What it really involves | Part | Main command |
|---|---|---|---|---|
| 1 | "take the model from Hugging Face, any open-source model, or my own model" | is it open? licence? which architecture? load from Hub id **or** local folder | P02 | `python -m peftkit.openness MODEL_ID` |
| 2 | "calculate what GPU is required … VRAM, RAM, memory … number of GPUs … mathematical formula" | weights + optimizer + activations + logits (+ KV cache to serve); GPU count; host RAM; disk | P03 | `python -m peftkit.plan MODEL_ID ...` |
| 3 | "HyperStack, Vast.ai or RunPod … cost analysis … exact number of hours" | FLOPs = 6 × params × tokens → hours per GPU type → × price | P03 | `--dataset train.jsonl --gpu-price ...` |
| 4 | "create a prompt so I can replace any link or model" | a copy-paste prompt for any AI assistant that does the full sizing with the same formulas | P04 | — |
| 5 | "create a data set … change the domain of the model" | eval set first; what data teaches behaviour vs facts; how much | P05 | — |
| 6 | "PDFs or any kind of data … convert to .json / .jsonl, whether PDF, Excel, whatever" | ingest → chunk → grounded Q&A → dedup/PII/contamination → chat JSONL | P06 | `python -m peftkit.data.build ...` |
| 7 | "if the synthetic data is not generated … map the real-time data … correlations with independent and dependent … create your own dataset … give it a name … JSONL" | profile real sample → feature→target drivers → contract → generate → measure gap → privacy/utility gates → name + version → chat JSONL | P07 | `python -m synthkit.profile`, `synthkit.pipeline`, `peftkit.data.records` |
| 8 | "connect the GPUs to my laptop through WinSCP or SSH … HyperStack, Vast.ai or RunPod" | SSH key, `~/.ssh/config`, VS Code Remote-SSH, WinSCP (SFTP), tmux | P08 | `ssh gpu` |
| 9 | (install everything) | driver ↔ CUDA ↔ torch match, two environments | P09 | `bash scripts/setup_gpu_vm.sh` |
| 10 | "dump the train / evaluation files" | upload `train.jsonl`, `eval.jsonl`, `test.jsonl`, configs | P10 | WinSCP / `scp` |
| 11 | (train) | doctor → baseline → 20-step dry run → real run in tmux | P10 | `python -m peftkit.train ...` |
| 12 | "after the training … evaluation" | before/after on unseen questions; regressions; benchmarks; forgetting gate | P11 | `python -m peftkit.evaluate ...` |
| 13 | "hyperparameter tuning to map the things perfectly as per the benchmarks" | search lr/rank/epochs with successive halving; stop when gains < seed noise | P12 | `python -m peftkit.hpo ...` |
| 14 | "any domain — healthcare or any domain in the world" | the domain is a parameter; what changes per domain | P16 | — |
| 15 | "feed the model into Hugging Face … my Spaces so anyone can see" | merge → model card → push model + dataset → deploy a Gradio Space | P14 | `python -m peftkit.publish`, `python -m peftkit.space` |
| 16 | "KV cache … how the relations work … theory and practicals" | T01-T06 theory, each with a runnable lab | T01-T06 | `python labs/...` |

## The rules that hold at every step

1. **Nothing proceeds past a gate that failed:** licence → data gate → doctor → evaluation → (benchmarks) → publish.
2. **Every number that decides money comes from a measurement**, not an estimate (dry run for memory and speed, evaluation for quality).
3. **Everything that will run on the GPU is rehearsed on the laptop first** (P01 Section 7). A failed rehearsal costs $0; a failed GPU run costs money.
4. **Every artifact carries its lineage:** model id + revision, dataset name + version, config, seed.
5. **The GPU machine is a cost, not a home.** Weights go to the Hub; the machine is deleted at the end of every session.

## What you will be able to do at the end

- Size **any** model (dense or Mixture-of-Experts, 100M to 1T+ parameters) from its `config.json`: VRAM per method, GPU count, multi-node or not, host RAM, disk, hours, and cost per provider.
- Turn any pile of PDFs, spreadsheets, Word files or slides into a clean, leak-free chat JSONL dataset in one command.
- Generate synthetic data that is **measured** against a real sample, keeps the feature→target relationships, and passes privacy and utility gates.
- Fine-tune with LoRA / QLoRA / DoRA / AdaLoRA / LoRA+ (or full fine-tuning) on HyperStack, RunPod or Vast.ai, and **prove** it improved.
- Tune hyperparameters with a search that ends, and align with DPO / ORPO / GRPO.
- Publish the model and the dataset to your Hugging Face account, and a Space anyone can open.

## Cost of learning this end to end (realistic)

| Item | Cost |
|---|---|
| Laptop parts (P01-P07, P20) | $0 (Hugging Face free account; synthetic data via an API key costs cents to a few dollars) |
| One QLoRA run of an 8B model on ~10-40M tokens | about $3-15 of GPU time (P03 computes yours) |
| Hyperparameter search (8-12 short runs) | 3-5× one run |
| Hugging Face model + dataset repos | free (private storage limits apply [5]) |
| A Space on free CPU hardware | free (small models only; bigger models: paid GPU hardware or the endpoint backend, P14) |
| Forgetting to delete the GPU machine overnight | $20-100. **Set a calendar alarm.** |
