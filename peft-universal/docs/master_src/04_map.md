# 4. Requirement → chapter map

Every requirement from the original briefs, and where this edition covers it.

|Requirement|Chapters|Main files|
|---|---|---|
|**THE FLOW (Part O): the 13 steps in working order**|**3-16**|-|
|Is the model open? licence, weights, gating|4|peftkit/openness.py|
|GPUs to load vs GPUs to fine-tune, the arithmetic|5|plan.py, kv.py, capacity.py|
|No data: build it from documents and reality|6|data/build.py, synthkit|
|Synthetic data that mimics reality (measured gap loop)|7|synthkit (profile→generate→metrics→gates)|
|The data gate before any GPU|8|data/quality.py, data/lineage.py, diagnostics.py|
|Rent a GPU, VS Code Remote-SSH, WinSCP, tmux|9|scripts/|
|Install the stack per GPU (driver/CUDA/torch/libs)|10|peftkit/gpu_stack.py|
|Load the model on the GPU and verify|11|-|
|Any documents → train.jsonl, any scale, any domain|12|peftkit/data/build.py|
|Transfer, configure, run|13|configs/, scripts/|
|Evaluate, then hyperparameter search with a stopping rule|14|evaluate.py, hpo.py|
|Merge the best checkpoint and publish to Hugging Face|15|merge.py, publish.py|
|New session: serve with vLLM, API keys, autoscale, Terraform, cloud|16|serving/, keys.py|
|Beginner can follow from VS Code and a venv to autoscaling|2 (runbook), 16-18|scripts/rehearsal_cpu.py|
|Fundamentals: forward, loss, backprop, optimizer|6|labs/lab11|
|Transformer internals, tokenization, masking|7, 8, 27-29|labs/lab12-13, peftkit/tokenize_inspect.py|
|LoRA, QLoRA, DoRA, AdaLoRA, LoRA+ with maths|9-15|labs/lab1-7, peft_factory.py|
|Any model: compatibility, architecture mapping|19-20|compat.py, archs.py|
|Hardware for any size (1M to 1T+), KV cache, cost|21-23|plan.py, kv.py, hardware.py|
|Data: formats, templates, dedup, quality, synthetic, judges, preference, risks|24-37|peftkit/data/*|
|Hyperparameters, experiments, debugging|38-40|experiments.py, diagnostics.py|
|Hyperstack GPU and training|41-43|scripts/*, train.py|
|**Evaluation after training, baseline, forgetting**|44-47|evaluate.py|
|Fine-tuning vs RAG, CPT, DPO|48-50|labs/lab21-23|
|Merge, publish (Hub, Drive, S3)|51|merge.py, publish.py|
|vLLM serving|52|scripts/serve_vllm.sh|
|**Evaluate the served model (new)**|53|eval_endpoint.py, serve_local.py|
|**Load test and capacity (new)**|54|loadtest.py|
|**Release gate before scaling (new)**|55|release_gate.py|
|Gateway, API keys, autoscaling|56|serving/*|
|Multi-adapter, cloud|57-58|adapters.py, publish.py|
|Project, capstone, deliverables, verification|59-64|-|
|The brief this book follows|65|-|
|**Universal adaptation brief (Part J)**|||
|Which concepts generalise, and which do not (5-way classification)|67|-|
|Universal algorithm; methodology ≠ code; "why unsupported?" chain|68|-|
|Step 0 identify: print(model), config, signature, source, modules, params, meta device|69|uadapt/inspect_model.py|
|Step 1 classify: decision tree, detection, what changes|70|uadapt/inspect_model.py|
|Step 2 forward pass: hooks, shapes, backward hooks, fx, torchinfo, activations, gradients|71|uadapt/trace.py|
|Step 3 objective: CE, BCE, MSE, Huber, contrastive, triplet, CTC, diffusion, preference, ranking, multi-task|72|-|
|Step 4 what to train: full, LoRA family, IA3, prefix/prompt, adapters, selective, head, projector, custom|73|uadapt/strategy.py|
|Step 5 target discovery by computation|74|uadapt/targets.py, labs/lab60|
|Custom LoRA, freeze, save, merge maths; silent-failure bugs|75|uadapt/lora.py|
|Encoder-only, encoder-decoder, vision, audio, multimodal, MoE, SSM/RNN/hybrid, diffusion|76-83|labs/lab60|
|Custom architectures: loop, AMP, accumulation, checkpointing, DDP, FSDP, quantization|84|uadapt/loop.py|
|UniversalModelAdapter design; automated vs judgement|85|uadapt/adapter.py|
|Compatibility report script and every field|86|uadapt/report.py|
|Data adaptation for every modality|87|-|
|Universal debugging (symptom → cause → command → expected → fix → verify)|88|labs/lab61|
|Evaluation for any model; own eval sets; public benchmarks (lm-eval)|89|scripts/run_benchmarks.sh|
|Serving stack chosen by architecture|90|-|
|**Hardware, cost, public API (Part K)**|||
|GPU providers analysed; how to choose|91|configs/providers.yaml|
|GPUs needed to fine-tune (maths)|92|plan.py, capacity.py|
|GPUs needed to serve; users → Little's law → replicas|93|capacity.py|
|Loading the model at serve time (maths); cold start|94|capacity.py|
|Cost: train once, serve monthly, per 1M tokens, per user|95|capacity.py|
|OpenAI-style API: anyone gets a key; limits, access, metering|96|serving/gateway/app.py, keys.py|
|GPUs grow with users: autoscaling stages|97|serving/k8s/*|
|Frontend or not; free hosting for one user|98|serving/ui/index.html, serve_local.py|
|Showcase to owners / public|99|showcase.py|
|**Mastery (Part L)**|||
|One-shot universal template, any model, any domain|100|-|
|Unknown-model challenge, progressively harder|101, 103|-|
|Universal decision tree + adaptation checklist|102|-|
|**Task abstraction and review response (Part M)**|||
|External review answered point by point|104|-|
|Support levels 1-5 for any model|105|-|
|Modality × architecture × task × method matrix|106|uadapt/tasks.py|
|Task registry: 19 tasks with schema, loss, collator, metrics, serving|107|uadapt/tasks.py, labs/lab62|
|Metrics engine: F1, AUROC, ECE, Recall@K, MRR, nDCG, WER, CER|107|uadapt/metrics.py|
|Classification, token classification, regression, ranking|108|labs/lab62|
|Embedding and retrieval models (contrastive, hard negatives)|108|uadapt/tasks.py|
|Reward modelling vs DPO/IPO/KTO/ORPO|108|uadapt/tasks.py|
|Image generation in depth (VAE, schedules, SNR, bucketing, DreamBooth, ControlNet)|109|-|
|TTS and video: recipes, metrics, and what is not shipped|110|uadapt/tasks.py|
|Multimodal depth: validation, resolution, missing modality, placeholders|111|-|
|Corrected scope claim: what "universal" honestly means|112|-|
|Data provenance, lineage, licence conflict controls|113|peftkit/data/lineage.py|
|The engine architecture, with implemented parts marked|114|-|
|Unknown-model challenge v2: 16-point determination|115|-|
|**Universal algorithm as code (Part N)**|||
|Second review answered; what was genuinely missing|116|-|
|Universal processor layer (any modality, incl. new ones)|117|uadapt/processors.py|
|First-principles derivation, worked on BERT + 4 families|118|labs/lab63|
|Universal loss derivation from contracts (10 rules)|119|uadapt/derive.py|
|Modality-independent pipeline + universal collator + adapter degradation|120|uadapt/pipeline.py, uadapt/collate.py|
|A modality that does not exist, trained end to end|121|labs/lab63|

