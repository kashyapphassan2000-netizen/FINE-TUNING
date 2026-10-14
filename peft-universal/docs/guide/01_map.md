# 1. Requirement → implementation map

Every line of your brief, where it is taught, and which files implement it.

|Your requirement|Chapters|Files|
|---|---|---|
|Teach PEFT from zero, math included, short and clear|3-9|labs/lab1-7|
|Intrinsic dimensionality insight|3|labs/lab1_intrinsic_dimension.py|
|LoRA: math, rank, alpha, target modules|4|src/peftkit/peft_factory.py, labs/lab2|
|QLoRA: NF4, double quantization, paged optimizers|5|modeling.py, configs/methods/qlora.yaml, labs/lab3|
|DoRA: magnitude + direction|6|configs/methods/dora.yaml, labs/lab4|
|AdaLoRA: adaptive rank allocation|7|callbacks.py, configs/methods/adalora.yaml, labs/lab5|
|LoRA+: separate learning rates for A and B|8|train.py, configs/methods/loraplus.yaml, labs/lab6|
|Works from PyTorch, JAX, "whatever"|4.7, 9|labs/lab7_jax_lora.py (+ honest scope note)|
|Universal template for any open model, small → big mapping|9, 23, 26|configs/*, hardware.py, inspect_model.py|
|Decide hardware: params → VRAM → GPU|23|src/peftkit/hardware.py|
|Dataset formats: instruction pairs, chat|11|data/format.py|
|Chat templates: ChatML, Llama-3, Mistral|12|data/templates.py, labs/lab8|
|Loss masking: why, what breaks|13|data/templates.py, labs/lab9|
|Dedup and filtering pipelines|14|data/quality.py|
|Synthetic data: self-instruct, Alpaca-style|15|data/self_instruct.py, data/synth.py|
|Why data is the leverage point; taxonomy|10|-|
|Instruction dataset generation|15|data/synth.py, data/self_instruct.py|
|Preference dataset generation|16|data/preference.py|
|LLM-as-judge, judge scoring|17|data/judge.py|
|Tools: distilabel, DataDreamer, Argilla|18|examples/*, data/argilla_export.py|
|Risks: model collapse, data poisoning|19|data/quality.py, labs/lab10|
|End-to-end domain SFT dataset|20|scripts/build_dataset.sh|
|10,000+ documents of any format → training file|14, 20|data/ingest.py (streaming, dedup, skip bad files)|
|Hyperstack: credits, VM, connect GPU to laptop|24|scripts/gpu_preflight.sh, setup_gpu_vm.sh|
|What to install and why (one line each)|22, 24|requirements-*.txt|
|Train|26|train.py, pipeline.sh, configs/accelerate/*|
|Save to Hugging Face / Google Drive / local|27|publish.py, sync_gdrive.sh|
|New GPU session → pull model → vLLM inference + serving|28|serve_vllm.sh, serve_check.py|
|API for many users|29|serving/gateway/app.py|
|Autoscaling|29|serving/k8s/keda-scaledobject.yaml, serving/scaler.py|
|Enterprise extras ("masala")|26-31|merge verification, run_meta.json, rate limits, PII scrub, contamination guard, poisoning scan, Prometheus|
|FakeCloud first, then AWS|30|publish.py (S3), docs|
|Resume bullets, README, GitHub|31|README.md|
|Verification loop + troubleshooting|32|tests/*|
|Decide GPUs for ANY model, 100M to 1T+ (addendum)|34|src/peftkit/archs.py, src/peftkit/plan.py|
|VRAM vs KV cache, dense vs MoE, MLA/sliding/hybrid|34|archs.kv_bytes_per_token|
|Compute, time and cost mapping|35|plan.py (--tokens-per-sec, --gpu-flops, --mfu)|
|Map any architecture, now or in the future|36|archs.py, inspect_model.py, labs/outputs/arch_matrix.txt|
|One page, start to finish, any model|37|scripts/*|
|Many adapters (finance, legal, support) on one base|38|src/peftkit/adapters.py, configs/adapters.yaml|
|Plain-English primer for beginners|33|-|
