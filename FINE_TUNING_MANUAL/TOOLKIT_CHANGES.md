# TOOLKIT_CHANGES — what was wrong in the uploaded `peft-universal`, what was fixed, what was added

The uploaded toolkit (`peft-universal.zip`) was read file by file and run end to end. Its baseline was strong: **105 tests passed** once two missing test dependencies were installed. Running the manual's flow on it surfaced the defects below. Every fix has a test or a re-run as evidence. Final state: **117 passed, 2 skipped (the 2 need an NVIDIA GPU)** and `REHEARSAL PASSED`.

## Bugs fixed (each reproduced before fixing)

| # | Where | Defect | Evidence | Fix |
|---|---|---|---|---|
| 1 | `peftkit/data/build.py` + `synth.py` | grounded mode wrote the **generator's instruction** ("You write training data… Return JSON") as the system prompt of **every training row**: the model would be trained with an instruction it never sees at inference | code reading; `test_build_never_writes_generator_prompt_into_training_rows` | separate `gen_system` (to the LLM) from the row system prompt (`--system`, default "You are a helpful DOMAIN assistant.") |
| 2 | `peftkit/data/ingest.py` | every CSV row was its own text and was then dropped by `min_chars=200`: **most spreadsheet rows silently vanished** | reproduced with short rows | table rows are grouped into blocks and split only between rows; Excel/TSV/PowerPoint added |
| 3 | `ingest.chunk` | overlapping chunks began mid-word/mid-row (`"f the relief valve…"`) | sample PDF/Excel run | overlap now starts at a paragraph/sentence/word boundary |
| 4 | `build.py` / `synth.py` | all builds shared `<out parent>/interim/`; a grounded build after a chunks-only build crashed (`KeyError: 'meta'`) | reproduced | interim folder per output folder, per-mode file names, resume ignores foreign rows |
| 5 | `scripts/pipeline.sh` | `SKIP_SYNTH=1` (the README's own GPU command) still formatted a non-existent `synthetic.jsonl` and would overwrite uploaded data | code reading | SKIP_SYNTH now skips the whole data stage and checks `train.jsonl` exists |
| 6 | requirements | `scikit-learn` (imported by `synthkit.utility`) and `reportlab` (tests) were missing: `synthkit.pipeline` crashed on a clean install | `ModuleNotFoundError` on a fresh venv | added (plus `openpyxl`, `python-pptx`, `numpy`) |
| 7 | `diagnostics`, `tokenize_inspect`, `evaluate`, `compat` | chat-template markers detected with `"generation" in template`, which is **always true** (`add_generation_prompt` appears in every template) → all-zero masks: 0% trained, doctor FAIL, NaN answer-NLL on Qwen/Llama/Mistral | test with a marker-less template | `has_generation_markers()` regex for real `{% generation %}` blocks |
| 8 | training path | `assistant_only_loss` needs `{% generation %}` markers; **none of Qwen3, Qwen2.5, Mistral-7B-v0.3, SmolLM2 official templates have them** (measured), yet the toolkit's manual recommended it for real models | downloaded templates | new `data.completion_only_loss: true` (TRL prompt-completion masking; works with any template; trains the last answer + EOS), doctor + `tokenize_inspect --completion-only` |
| 9 | `synthkit/pipeline.py` | utility gate used **accuracy**: on a 14% no-show target it reported 0.995 "utility" while the AUC fell 0.63 → 0.53 | clinic example | AUC for binary labels, metric recorded in the report |
| 10 | `synthkit` generator | the copula lost the feature → target relationship when category groups are small: utility FAILED at every smoothing level | clinic example table (P07 §6) | `ConditionalLabelGenerator` (label from P(target \| features), sampled): SHIP at smoothing 0.5; `--label-mode conditional` default |
| 11 | `scripts/setup_gpu_vm.sh` | used `sudo` → fails in RunPod/Vast.ai root containers (no sudo); no way to install torch for an older driver | code reading | `SUDO` only when not root; `TORCH_INDEX=` option; `gpu_preflight.sh` explains the 3 fixes, `ALLOW_OLD_CUDA=1` |
| 12 | docs (F11, M05, M09, M12) | `--set train.resume=true` is **silently ignored** (no such config key) | config model reading | `--resume true` |
| 13 | docs (F09, F13) | `apply_chat_template(..., return_tensors="pt")` then `model.generate(ids)` crashes on transformers 5 (BatchEncoding) | the toolkit's own troubleshooting row 12 | `return_dict=True`, `generate(**enc)` |
| 14 | `peftkit/data/lineage.py` | no way to stamp lineage onto rows from `data.build`; "proprietary" blocked your **own** data for commercial use | run on build output | `--stamp` mode; `owned` licence family |
| 15 | `peftkit/publish.py` | `--revision v1` failed for a branch that does not exist yet; no dataset repos | code reading | branch created first; `--repo-type dataset` |
| 16 | `.env.example` | `HF_HOME=/ephemeral/...` exists only on HyperStack (RunPod/Vast: `/workspace`) | provider docs | documented per provider |
| 17 | `README.md` | pointed to a PDF that is not in the upload | file listing | says how to build it |

## Added for the manual

| Addition | Why (the user's flow) |
|---|---|
| `peftkit.plan`: `full` method (16 B/param, 8·N·D FLOPs), host RAM/disk, hours + $ per GPU type (MFU-based until measured), `--gpu-price GPU=PRICE`, `--dataset train.jsonl` exact token counting, `--tokenizer` | "exact number of GPUs, RAM, VRAM, memory, hours, cost on HyperStack/Vast.ai/RunPod" |
| `method: full` (+ `configs/methods/full.yaml`, doctor support) | full fine-tuning, not only PEFT |
| `peftkit.data.records` (JSONL/JSON/CSV/TSV/Excel → named, versioned chat dataset + card + lineage, `--label-map`, `--hub`) | "give the dataset a name… convert into JSONL"; Excel FAQs directly |
| `synthkit.profile` CLI with **feature → target drivers** (Pearson/Spearman, η², correlation ratio, Cramér's V) and time-series flags | "correlations with each of the independent and dependent" |
| `peftkit.align` (DPO, ORPO, KTO, GRPO with the eval-format verifier, gated length reward) | the alignment syllabus, runnable on a real model |
| `peftkit.card` (model card from run/eval/benchmark reports) | publishing to the Hub properly |
| `peftkit.space` + `serving/space/app.py` (Gradio 6.30 chat; local or endpoint backend; secrets for keys; backend-specific requirements) | "feed the model into Hugging Face… Spaces so anyone can see" |
| labs 24-29 (ORPO, GRPO, RLHF-PPO from scratch, CLM/MLM/Prefix-LM, scaling laws, MTP) with recorded outputs | the theory syllabus: every claim runnable |
| `examples/real_sample` (clinic no-show simulator + contract), `examples/sample_docs` (PDF/Excel/Word/FAQ), `examples/custom_model` (LoRA on your own nn.Module) | every practical runs on a laptop |
| `tests/test_additions.py` (12 tests) | evidence for every change above |

## Found and documented, not changed (so you know)

- TRL 1.13 has **no PPOTrainer** (lab 26 implements PPO from scratch); ORPO lives in `trl.experimental.orpo`; `GRPOConfig` defaults are `beta=0.0`, `loss_type="dapo"`.
- Passing the model object returned by `SFTTrainer` straight into `GRPOTrainer` in one process gave `grad_norm = 0`; save → reload fixes it (lab 25).
- On Linux, `requirements-dev.txt` replaces the CPU torch wheel with PyPI's CUDA build (works on CPU, bigger download).
- lm-evaluation-harness, multi-GPU (DeepSpeed/FSDP) and real-GPU training were not executed here (no GPU); their configs were validated by the libraries' own parsers by the toolkit's authors, and the manual marks those steps "(verify)".
