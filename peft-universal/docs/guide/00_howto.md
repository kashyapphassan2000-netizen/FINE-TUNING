# 0. How to use this guide

Type: **PROJECT** (with full CONCEPT chapters for every method and every data topic). You get one repository, `peft-universal`, that fine-tunes any open causal language model on the Hugging Face Hub with LoRA, QLoRA, DoRA, AdaLoRA or LoRA+, builds the training data (from your documents or synthetically), merges, publishes, and serves with vLLM behind an API gateway that can autoscale.

The idea of "universal": nothing in the code is tied to one model. You change three YAML files (base, method, model) and the same commands run a 135M model on a laptop CPU or a 70B model on rented GPUs. Chapter 23 teaches how to size the hardware for any model before you pay for it.

## 0.1 Label legend

Every command block has a coloured header that tells you where to run it.

|Label|Where|Shell|
|---|---|---|
|LAPTOP · PowerShell|your Windows laptop|PowerShell 7 or Windows PowerShell|
|LAPTOP · WSL bash|your laptop, inside WSL Ubuntu|bash (only for FakeCloud)|
|GPU VM · bash|the rented Hyperstack VM, through VS Code Remote-SSH|bash|
|LAPTOP or GPU VM · bash|either machine|bash|
|BROWSER|a web page (Hugging Face, Hyperstack console, ...)|clicks|
|FILE path|a file in the repo. Copy it from the ZIP, not from the PDF|-|
|EXPECTED OUTPUT|what you should see|-|

Long code lines are wrapped in the PDF and marked with `↳`. The ZIP contains the byte-identical files. **Always copy from the ZIP.**

## 0.2 What was actually executed

:::honest Read this before trusting any step
- EXECUTED (CPU, Linux container, 17 Sep 2026): the full pytest suite, 32 passed, 1 skipped (the QLoRA GPU test). It trains LoRA, DoRA, AdaLoRA and LoRA+ on a tiny random Llama built offline, merges each adapter and proves the merge is exact, runs the vLLM compatibility check, generates text, tests every data stage against fake OpenAI-compatible servers, tests S3 upload/download against a mocked S3 (moto), and tests the API gateway (auth, rate limit, failover, streaming) against a fake vLLM.
- EXECUTED: all 10 concept labs. Every "REAL OUTPUT" box in this PDF is copied from a real run.
- EXECUTED: `py_compile` on every Python file, `bash -n` on every shell script, YAML parsing of every config.
- CHECKED AGAINST SOURCE, NOT RUN (needs GPU / network / accounts): QLoRA 4-bit training (bitsandbytes CUDA kernels), vLLM 0.29.0 serving (flags read from the v0.29.0 source), multi-GPU accelerate configs (keys read from accelerate 1.15.0), distilabel / DataDreamer / Argilla example code (class names read from their wheels), Hyperstack API calls (from the official API quickstart), Docker Compose and Kubernetes/KEDA manifests.
- NOT VERIFIED: the Docker tag `vllm/vllm-openai:v0.29.0` (Docker Hub was not reachable), the Hyperstack DELETE VM path (the reference page body is rendered by JavaScript), GPU memory numbers from the estimator (they are formulas; confirm with a dry run), any price.
- No benchmark number in this PDF is invented. Resume bullets use placeholders.
:::

## 0.3 Versions (PyPI, checked 17 Sep 2026)

|Package|Version|Used in|
|---|---|---|
|torch|2.14.0 (CUDA 13.0 build)|training env|
|transformers|5.17.0|training env|
|peft|0.21.0|training env|
|trl|1.13.0|training env|
|accelerate|1.15.0|training env|
|datasets|5.0.1|training env|
|bitsandbytes|0.50.2|training env|
|huggingface_hub|1.31.0|training env|
|vllm|0.29.0 (pins torch==2.13.0)|serving env only|
|distilabel / argilla|1.5.3 / 2.8.0|data env (optional)|
|datadreamer.dev|0.46.0 (pins datasets<4, hub<1.0, numpy<2)|its own env|

## 0.4 Why several virtual environments

:::why Dependency conflicts found while building this guide
- vLLM 0.29.0 pins `torch==2.13.0`. The training stack uses torch 2.14.0. One environment cannot hold both, so the GPU VM gets `.venv-train` and `.venv-serve`.
- DataDreamer 0.46.0 pins `datasets<4`, `huggingface-hub<1.0`, `numpy<2`. The training env has datasets 5.0.1 and hub 1.31.0. If you use DataDreamer, give it `.venv-dd`.
- distilabel 1.5.3 and argilla 2.8.0 have loose pins and install next to the training stack, but a separate `.venv-data` keeps the training env clean.
:::

## 0.5 Reading order

Part A (chapters 3-9) is theory for the five methods. Part B (10-20) is data: formats, templates, loss masking, dedup, synthetic generation, judges, preference data, tools, risks and the end-to-end pipeline. Part C (21-32) is the build, in the order you type it. If you only want to ship, do Part C and come back to A and B when a checkpoint box asks you to.
