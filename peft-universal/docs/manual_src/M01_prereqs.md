# LAB MANUAL — PART 1: Prerequisites

## 1.1 Checklist

|Item|Level|
|---|---|
|Use a terminal: change folder, run a command, read an error|MUST|
|Basic Python: run a script, read a traceback, install a package|MUST|
|Files and folders on Windows: paths, extensions, hidden files|MUST|
|JSON and JSONL: what a record looks like|MUST|
|What a GPU and VRAM are|MUST|
|What SSH keys are|MUST|
|Git basics: commit, push|NICE-TO-HAVE|
|Docker basics|NICE-TO-HAVE|
|Linear algebra (matrix multiply)|NICE-TO-HAVE — Reference chapters 20-21 teach what you need|

## 1.2 Refreshers and one-minute checks (MUST items)

**Terminal.** A text window where you type commands. `cd` changes folder, `dir` (PowerShell) or `ls` (Linux) lists files.
*Check:* open PowerShell and run the command below; it must print your home folder.

```laptop
cd $HOME; pwd
```

**Python.** You run `python file.py` or `python -m module`; errors end with a line naming the problem.
*Check:* explain what `ModuleNotFoundError: No module named 'peft'` means. (Answer: the package is not installed in the environment you are running.)

**Paths.** Windows uses `C:\Users\you\...`; Linux uses `/home/ubuntu/...`. Forward and back slashes are not interchangeable in every tool.
*Check:* name the folder your Downloads live in.

**JSON / JSONL.** JSON is one object `{"key": "value"}`; JSONL is one JSON object **per line**, the format training data uses.
*Check:* is this valid JSONL? `{"a": 1}` on line one, `{"a": 2}` on line two. (Yes.)

**GPU and VRAM.** The GPU does the maths; VRAM is its memory. If the model plus training state does not fit in VRAM, you get "CUDA out of memory".
*Check:* an 8-billion-parameter model in bf16 needs about how many GB just for weights? (8 × 2 bytes = ~16 GB.)

**SSH keys.** A private key stays on your laptop; the public key goes to the server. Together they let you log in without a password.
*Check:* which of the two files may you share? (The `.pub` one only.)

## 1.3 Everything to install or sign up for

|Name|Official link|Cost|Which option to choose|
|---|---|---|---|
|uv (installs Python too)|https://docs.astral.sh/uv/ [1]|free|the standalone PowerShell installer|
|Git for Windows|https://git-scm.com [22]|free|defaults are fine|
|Visual Studio Code|https://code.visualstudio.com [20]|free|User Installer, 64-bit|
|WinSCP|https://winscp.net [21]|free|standard installer|
|Hugging Face account|https://huggingface.co|free (100 GB private storage); PRO $9/month optional [5][31]|free is enough for this manual|
|Hyperstack account|https://www.hyperstack.cloud|pay-as-you-go; e.g. H100 PCIe from $2.50/h on-demand, L40 $0.80/h and A6000 $0.40/h spot [3][4]|add a small credit first; verify prices at checkout|
|GitHub account|https://github.com|free|for backup of your project|
|LLM API key (OpenAI-compatible)|your provider|pay-per-token|only needed for synthetic Q&A generation in Part 5|
