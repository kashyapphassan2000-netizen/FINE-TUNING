# LAB MANUAL — PART 0: Start Here

**Assumptions** (the brief's blank fields): complete beginner · Windows 11 laptop plus a rented Hyperstack GPU · goal: fine-tune any open model on your own data and serve it to real users through an API · budget: free software, pay-as-you-go GPU at roughly $0.40-$2.50 per GPU-hour [3][4].

:::why How this book is organised
This **Lab Manual (Parts 0-14)** is the hand-holding path: one action per step, exactly what to type, what you should see, and a checkpoint at the end of every section. The **Reference** that follows (chapters 0-135) explains why everything works and covers every architecture. The manual tells you when to go to the reference.
:::

## 0.1 What it is

Fine-tuning takes a model that already understands language and teaches it **your** behaviour: your format, your domain, your tone, your tasks. With parameter-efficient fine-tuning (LoRA and its relatives) you train a small add-on file instead of the whole model, so it fits on one rented GPU. Serving then turns the result into an API that other people and programs can call.

**Why people use it:** a small fine-tuned model can match a much bigger general model on one narrow job, for a fraction of the running cost, on hardware you control.

**Where it is used:** customer-support assistants, structured extraction (invoices, medical notes, contracts), code assistants trained on a company's codebase, classification and routing, and domain chat models in regulated industries where data cannot leave the company.

## 0.2 The route

|Part|What you will do|Time estimate|
|---|---|---|
|0 Start Here|understand the route and the pieces|20 min|
|1 Prerequisites|check what you know; create the accounts|1-2 h|
|2 Set Up Your Workspace|install everything on the laptop; prove it works|1-2 h|
|3 The Core Idea|the 20% of ideas that explain 80%|30 min|
|4 BASICS|fine-tune a tiny model on your laptop, measure it, merge it|1 h|
|4B The Linux Bridge|WSL 2, ten Linux commands, VS Code in WSL, Docker — before touching a remote VM|1-2 h|
|5 INTERMEDIATE|your data, a real model, a rented GPU|1 day|
|5B Under the Hood|the same steps with transformers/peft/trl only — no peftkit|1 h|
|5C Benchmarks|lm-evaluation-harness (MMLU, GSM8K) and a forgetting gate before release|2-4 h|
|6 ADVANCED|publish, serve, open an API to users, scale|1 day|
|6B Production Front Door|NGINX, Let's Encrypt TLS with renewal, JSON logs, retries on replica failure|2-3 h|
|7 ELITE|search hyperparameters honestly, preference-tune, adapt any architecture|2-3 days|
|7B Distributed Training|DeepSpeed ZeRO-2/3 and FSDP2, 8×H100 and multi-node, batch maths|1 day|
|8 Tools and Resources Map|what the field uses today, verified|30 min|
|9 Keep It Working|save, back up, update, restart, undo|30 min|
|10 Troubleshooting|the ten problems you will hit|reference|
|11 Practice Projects|easy / medium / hard|1-3 weeks|
|12 Cheat Sheet|every command in one place|reference|
|13 Self-Test|10 questions, answer key at the end|30 min|
|14 Sources|every link cited|reference|

## 0.3 At the end you will be able to

1. Decide whether a given model can legally and technically be fine-tuned, and on how many of which GPUs.
2. Turn any pile of documents into a clean, leak-free training dataset in one command.
3. Fine-tune an open model with LoRA/QLoRA on a rented GPU and **prove** it improved against a baseline.
4. Publish it to Hugging Face, serve it with vLLM, and give other people API keys to call it.
5. Size the fleet and the monthly bill from measurements, and autoscale as users grow.
6. Adapt an architecture nobody has written a tutorial for.

## 0.4 Environment map

```text where you will work, and how the pieces connect
BROWSER  ── huggingface.co (models, tokens) · console.hyperstack.cloud (GPU VMs) · github.com (code)
   │
LAPTOP (Windows 11)
   ├── PowerShell ........ install tools, run the laptop commands
   ├── VS Code ........... edit files; Remote-SSH opens the GPU VM as if it were local
   ├── WinSCP ............ drag-and-drop files to and from the GPU VM
   └── project folder .... peft-universal (the ZIP), with a .venv virtual environment
          │  ssh / sftp
GPU VM (Ubuntu, rented) ── bash · tmux · .venv-train (training) · .venv-serve (vLLM)
          │  push weights                     │  serve
HUGGING FACE HUB (your model repo)   ──>   vLLM + gateway (API keys) ──> your users (OpenAI SDK / browser)
                                                   │
                                           Docker · Kubernetes · KEDA autoscaling
```

**One-line flow:** laptop (prepare + rehearse) → rented GPU (train + evaluate) → Hugging Face (publish) → GPU again (serve) → gateway (keys) → users.

## 0.5 What each piece does

|Component|Its job|
|---|---|
|Python 3.11 + uv|the language and a fast tool that creates isolated environments|
|VS Code|the editor; its Remote-SSH extension edits files directly on the GPU VM|
|WinSCP|copies files between Windows and the GPU VM|
|Git + GitHub|version history and backup of your code|
|Hugging Face Hub|where models and datasets are downloaded from and published to|
|Hyperstack (or any GPU cloud)|rents the GPU by the hour|
|PyTorch|the maths engine that runs the model on the GPU|
|transformers / peft / trl|load models, attach LoRA adapters, run the training loop|
|`peftkit` (this project)|the commands in this manual: planning, data, training, evaluation, merging, serving checks|
|`uadapt` (this project)|the same method for any architecture, including your own models|
|`synthkit` (this project)|synthetic data that is measured against real data|
|vLLM|the high-throughput server that answers many users at once|
|gateway (this project)|API keys, per-user limits, usage metering in front of vLLM|
|Docker / Kubernetes / KEDA|package the server, run many copies, add copies when users queue|
