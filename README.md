# FINE-TUNING
MY-OWN FINETUNING NOTES

| Folder | What |
|---|---|
| [`FINE_TUNING_MANUAL/`](FINE_TUNING_MANUAL/README.md) | **the step-by-step lab manual**: theory (T01-T06) + practicals (P00-P21), any model, any domain: size → rent GPUs (HyperStack / RunPod / Vast.ai) → data (documents, Excel, synthetic) → JSONL → train → evaluate → tune → publish to Hugging Face + Spaces. Also as one Word file: `FINE_TUNING_MANUAL/FINE_TUNING_LAB_MANUAL.docx` |
| [`peft-universal/`](peft-universal/README.md) | the toolkit every command in the manual runs (`peftkit`, `synthkit`, `uadapt`, labs, tests) |

Start here: [`FINE_TUNING_MANUAL/P00_START_HERE.md`](FINE_TUNING_MANUAL/P00_START_HERE.md).

> This repository is **public**. Never commit `.env`, tokens, real documents or real user data: `peft-universal/.gitignore` excludes them; keep it that way.
