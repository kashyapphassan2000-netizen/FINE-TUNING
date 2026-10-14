# LAB MANUAL — PART 8: Tools and Resources Map (researched live)

Verified on **28 September 2026**: versions from PyPI [6], star counts and licences read from each GitHub repository page [7-19], prices from the providers' own pages where available [3][4][5]. Prices change; confirm at checkout.

## 8.1 The map

|Tool|What it does|Free / OSS / Paid|Cost + free-tier limits|Who uses it|Platform|Link (stars)|Best for|Verdict|
|---|---|---|---|---|---|---|---|---|
|transformers 5.17.0|load and run almost every open model|OSS, Apache-2.0|free|industry and research; maintained by Hugging Face|Linux/Win/Mac|github.com/huggingface/transformers (166,736)|the model layer under everything|essential|
|peft 0.21.0|LoRA, QLoRA, DoRA, AdaLoRA, IA3, prompt tuning|OSS, Apache-2.0|free|maintained by Hugging Face|all|github.com/huggingface/peft (21,729)|adapters on supported models|essential|
|TRL 1.14.0 (book verified 1.13.0)|SFT, DPO, reward modelling trainers|OSS, Apache-2.0|free|maintained by Hugging Face|all|github.com/huggingface/trl (19,402)|chat SFT and preference tuning|essential|
|vLLM 0.30.0 (book verified 0.29.0)|high-throughput serving, OpenAI-compatible API, multi-LoRA|OSS, Apache-2.0|free|started at UC Berkeley; widely deployed|Linux + GPU|github.com/vllm-project/vllm (92,811)|serving decoder LLMs|essential for LLM serving|
|SGLang|serving engine alternative to vLLM|OSS, Apache-2.0|free|research + industry (verify specific users)|Linux + GPU|github.com/sgl-project/sglang (36,491)|structured / agentic serving|worth benchmarking against vLLM|
|Unsloth 2026.9.11|fast, memory-efficient fine-tuning|OSS core, Apache-2.0|free (paid tiers exist; verify)|very popular with individual practitioners|Linux/Win + GPU|github.com/unslothai/unsloth (76,881)|single-GPU speed|excellent for one-GPU runs|
|LLaMA-Factory|config/UI-driven fine-tuning of many models|OSS, Apache-2.0|free|academic origin; broad community|Linux + GPU|github.com/hiyouga/LLaMA-Factory (75,098)|many models, little code|great for breadth|
|Axolotl 0.19.0|YAML-driven fine-tuning, multi-GPU|OSS, Apache-2.0|free|community and startups (verify)|Linux + GPU|github.com/axolotl-ai-cloud/axolotl (12,508)|reproducible multi-GPU configs|strong for teams|
|torchtune|PyTorch-native fine-tuning recipes|OSS, BSD-3-Clause|free|PyTorch team (Meta)|Linux + GPU|github.com/pytorch/torchtune (5,811)|hackable recipes|good for learning internals|
|lm-evaluation-harness 0.4.13|public benchmarks (MMLU, GSM8K, IFEval, …)|OSS, MIT|free|EleutherAI; common benchmark runner|all|github.com/EleutherAI/lm-evaluation-harness (14,089)|general-capability checks|essential; note chat backends cannot run loglikelihood tasks|
|distilabel 1.5.3|synthetic data pipelines|OSS, Apache-2.0|free (LLM calls cost)|Argilla / Hugging Face|all|github.com/argilla-io/distilabel (3,400)|generate → judge → filter|good complement to `synthkit`|
|Argilla 2.8.0|human review and annotation|OSS, Apache-2.0|free self-hosted|Argilla / Hugging Face|all|github.com/argilla-io/argilla (5,125)|human-in-the-loop review|use for the 50-item human check|
|uv 0.12.19|Python + environments, fast|OSS, MIT|free|Astral; widely adopted|all|github.com/astral-sh/uv (90,227)|every environment in this book|essential|
|KEDA|event-driven autoscaling on Kubernetes|OSS, Apache-2.0|free|CNCF project|Kubernetes|github.com/kedacore/keda (10,548)|queue-based GPU autoscaling|recommended over custom scalers|
|Hugging Face Hub|host models/datasets, gated access, private repos|freemium|Free: 100 GB private storage; PRO $9/month: 1 TB private, more ZeroGPU (quota figures vary by source — verify) [5][31]|everyone|web|huggingface.co|publishing and versioning models|free tier is enough to start|
|Hyperstack|GPU VMs by the hour|paid|H100 PCIe on-demand from $2.50/h; spot: H100 PCIe $2.00, A100 $1.08, L40 $0.80, A6000 $0.40; volumes ≈ $0.10/TB/h [3][4]|—|web + SSH|hyperstack.cloud|single-GPU to multi-node training and serving|good value; verify region availability|
|Other GPU clouds|the same, different prices|paid|mid-2026 H100 ranged ~$2.46/h (specialised spot) to ~$10-12/h (hyperscaler on-demand) [23]|—|web + SSH|RunPod, Lambda, Vast.ai, CoreWeave, AWS, GCP, Azure|compare with `configs/providers.yaml`|price-check every project|

## 8.2 Recommended setups

|Setup|What|Monthly cost|
|---|---|---|
|**Best free**|laptop + this project + Hugging Face free + Kaggle/Colab free GPUs for experiments (limits vary; verify)|$0|
|**Best paid (individual)**|laptop + Hyperstack on-demand GPU only while training/serving + HF PRO|$9 + GPU hours|
|**Within this manual's budget**|L40 48 GB or A100 80 GB spot for training runs of a few hours; one L40/A6000-class GPU for a demo API, stopped when idle|typically tens of dollars per project at the listed spot rates; compute yours with `peftkit.capacity`|

## 8.3 Repositories worth studying

1. **huggingface/trl** — read `SFTTrainer` and `DPOTrainer`: how masking, packing and reference models are really implemented.
2. **vllm-project/vllm** — the scheduler and paged KV cache: why continuous batching makes serving cheap.
3. **unslothai/unsloth** — how kernel-level work cuts memory and time on one GPU.
4. **EleutherAI/lm-evaluation-harness** — how benchmark prompts and scoring are actually defined.
5. **hiyouga/LLaMA-Factory** — one configuration system spanning many model families: a working example of "universal methodology, model-specific code".
