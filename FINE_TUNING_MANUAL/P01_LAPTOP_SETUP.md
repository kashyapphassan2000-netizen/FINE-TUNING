# P01 — Laptop Setup: tools, accounts, the project, and a full rehearsal

**Time:** 1-2 hours, once. **Cost:** $0.

## Prerequisites checklist

| Item | Level | One-minute check |
|---|---|---|
| Use a terminal: change folder, run a command, read an error | MUST | open PowerShell, run `cd $HOME; pwd` → prints your home folder |
| Basic Python: run a script, read a traceback | MUST | what does `ModuleNotFoundError: No module named 'peft'` mean? (package not installed in the active environment) |
| JSON / JSONL | MUST | is `{"a": 1}` on line 1 and `{"a": 2}` on line 2 valid JSONL? (yes) |
| What a GPU and VRAM are | MUST | an 8B model in bf16 needs how many GB just for weights? (8 × 2 = 16 GB) |
| SSH keys | MUST | which of the two key files may you share? (the `.pub` one only) |
| Git basics | NICE-TO-HAVE | `git status` |
| Linear algebra | NICE-TO-HAVE | T03 teaches what you need |

## Accounts and tools

| Name | Link | Cost | Choose |
|---|---|---|---|
| uv (installs Python too) | https://docs.astral.sh/uv/ [13] | free | standalone PowerShell installer |
| Git for Windows | https://git-scm.com | free | defaults |
| Visual Studio Code | https://code.visualstudio.com | free | User Installer, 64-bit |
| WinSCP | https://winscp.net | free | standard installer |
| Hugging Face | https://huggingface.co | free; PRO optional | free is enough to start |
| GPU provider (one of) | HyperStack https://console.hyperstack.cloud · RunPod https://console.runpod.io · Vast.ai https://cloud.vast.ai | prepaid credit | add $10-20 first (P08) |
| OpenAI-compatible LLM API (optional) | any provider | pay per token | only for LLM-generated data (P06 grounded mode, P07 Self-Instruct) |

---

## SECTION 1 — Install uv and Python 3.11

**WHERE:** Windows Start menu → PowerShell
**GOAL:** `uv` and Python 3.11 available in any terminal

Step 1. Press the Windows key, type `PowerShell`, click **Windows PowerShell**.

Step 2. Paste and press Enter:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Step 3. Close PowerShell and open a new one (Step 1 again).

Step 4. Paste and press Enter:

```powershell
uv --version
uv python install 3.11
```

**YOU SHOULD SEE:** `uv 0.x.y`, then a line saying Python 3.11.x was installed.

> **WATCH OUT:** some antivirus products flag the uv installer (a known false positive). If blocked, download `uv-x86_64-pc-windows-msvc.zip` from the uv GitHub releases page and put `uv.exe` in a folder on your PATH.

**CHECKPOINT** — `uv --version` works in a NEW terminal. If not: sign out of Windows and back in.

## SECTION 2 — Install Git, VS Code, WinSCP

**WHERE:** PowerShell
**GOAL:** the three desktop tools

Step 1. Paste and press Enter (one line at a time; accept the prompts):

```powershell
winget install --id Git.Git -e
winget install --id Microsoft.VisualStudioCode -e
winget install --id WinSCP.WinSCP -e
```

Step 2. Close and reopen PowerShell. Paste and press Enter:

```powershell
git --version
```

**YOU SHOULD SEE:** `git version 2.x`.

| If you see… | It means… | Do this |
|---|---|---|
| `winget` is not recognized | App Installer missing | install "App Installer" from the Microsoft Store, repeat |
| `git` not recognized after install | terminal opened before install | close and reopen PowerShell |
| WinSCP Id not found | the Id changed | run `winget search WinSCP` and use the Id it shows |

Step 3. Open VS Code → press `Ctrl+Shift+X` → install **Python** (Microsoft), **Remote - SSH** (Microsoft), **YAML** (Red Hat).

**CHECKPOINT** — `git --version` works; VS Code shows the three extensions as installed.

## SECTION 3 — Hugging Face account and tokens

**WHERE:** browser
**GOAL:** an account and two tokens (read for downloading, write for publishing)

Step 1. Go to https://huggingface.co/join, create the account, confirm the email. Your user name is `HF_USER`.

Step 2. Go to https://huggingface.co/settings/tokens → **Create new token** → type **Write** → name `ft-write` → **Create**. Copy it now (it is shown once) into a password manager.

Step 3. (Gated models such as Llama) open the model page, e.g. https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct, click **Agree and access repository** while logged in. Approval can take minutes to days.

> **WATCH OUT:** a token is a password. Never paste it into code, a chat, a notebook you share, or a commit. Your repo `FINE-TUNING` is **public**: anything you commit is world-readable.

**CHECKPOINT** — you have `HF_USER` and a write token stored safely.

## SECTION 4 — Put this repository on your PC

**WHERE:** PowerShell
**GOAL:** `C:\Users\YOUR_NAME\code\FINE-TUNING\peft-universal` exists

Step 1. Paste and press Enter (replace `REPO_BRANCH`, see README placeholders):

```powershell
mkdir $HOME\code -Force; cd $HOME\code
git clone -b REPO_BRANCH https://github.com/kashyapphassan2000-netizen/FINE-TUNING.git
cd FINE-TUNING\peft-universal
dir
```

**YOU SHOULD SEE:** folders `configs  contracts  data  docs  examples  labs  scripts  serving  src  tests`.

**CHECKPOINT** — `dir` shows `src` and `configs`.

## SECTION 5 — Create the environment and install

**WHERE:** PowerShell in `FINE-TUNING\peft-universal`
**GOAL:** an isolated `.venv` with everything the laptop parts need

Step 1. Paste and press Enter:

```powershell
uv venv .venv --python 3.11
.\.venv\Scripts\Activate.ps1
```

**YOU SHOULD SEE:** `(.venv)` at the start of the prompt.

| If you see… | It means… | Do this |
|---|---|---|
| "running scripts is disabled on this system" | PowerShell execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, answer `Y`, repeat Step 1 |

Step 2. Paste and press Enter (CPU build of PyTorch first, so pip does not pull a 2.5 GB CUDA wheel onto your laptop):

```powershell
uv pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements-dev.txt
uv pip install -e .
python -c "import peftkit, uadapt, synthkit; print('ok')"
```

**YOU SHOULD SEE:** `ok`.

> **WATCH OUT (Linux / WSL laptops only):** on Linux, the `requirements-dev.txt` step can replace the CPU wheel with PyPI's CUDA build of torch (`2.14.0+cu130`; observed on the build machine of this manual). It still runs on CPU (all tests passed with it); it is only a ~2.5 GB bigger download. On Windows, PyPI torch is CPU-only, so this does not happen.

> **WATCH OUT:** every new PowerShell window needs `.\.venv\Scripts\Activate.ps1` again (inside `FINE-TUNING\peft-universal`). Most "module not found" errors are a forgotten activation.

**CHECKPOINT** — `(.venv)` is shown and the import prints `ok`.

## SECTION 6 — Your secrets file

**WHERE:** VS Code → File → Open Folder → `FINE-TUNING\peft-universal`
**GOAL:** `.env` with your keys, never committed

Step 1. In the Explorer panel right-click `.env.example` → **Copy**, right-click empty space → **Paste**, rename the copy to `.env`.

Step 2. Fill what you have now; save with `Ctrl+S`:

| Field | Value | Why |
|---|---|---|
| `HF_TOKEN` | your write token | download gated models, publish yours |
| `HF_HOME` | leave as is on the GPU; on the PC delete the line or set `C:\hf-cache` | where models are cached |
| `OPENAI_BASE_URL` | your provider's OpenAI-compatible URL ending in `/v1` | LLM-generated data (optional) |
| `OPENAI_API_KEY` | that provider's key | same |
| `SYNTH_MODEL` | the generator model name at that provider | same |

Step 3. Paste and press Enter. It must print nothing (an empty result means git ignores `.env`):

```powershell
git status --short | Select-String "\.env$"
```

**CHECKPOINT** — `.env` exists and `git status` does not list it.

## SECTION 7 — Prove everything works on your PC (tests + full rehearsal)

**WHERE:** PowerShell, `(.venv)` active
**GOAL:** the whole pipeline runs on CPU before you pay for any GPU

Step 1. Paste and press Enter (about 4 minutes):

```powershell
python -m pytest -q
```

**YOU SHOULD SEE (real output from this repo):**

```text
117 passed, 2 skipped, 3 warnings in 219.96s (0:03:39)
```

Step 2. Paste and press Enter (about 2-3 minutes): data → train → evaluate → merge → serve → API key → load test → release gate, all on a tiny model:

```powershell
python scripts/rehearsal_cpu.py
```

**YOU SHOULD SEE (last lines, real output):**

```text
=== STEP 10: RELEASE GATE: quality + parity + capacity
    [PASS] quality   PASS: domain improved, no large general regression
    [PASS] parity    PASS
    [PASS] capacity  4
    decision: RELEASE
...
REHEARSAL PASSED - every stage ran end to end
```

Step 3. Make the tiny model the labs and later sections use:

```powershell
python scripts/make_tiny_model.py
```

**YOU SHOULD SEE:** `tiny model ready: ...\outputs\tiny`

| If you see… | It means… | Do this |
|---|---|---|
| a test fails with `ModuleNotFoundError` | a dependency is missing | `uv pip install -r requirements-dev.txt` again, then `uv pip install -e .` |
| `2 skipped` | those tests need an NVIDIA GPU | expected |
| rehearsal hangs at a server step | a port is busy (8000-8100) | close other servers or reboot, rerun |

**CHECKPOINT** — `117 passed, 2 skipped` and `REHEARSAL PASSED`. **Do not rent a GPU until this is true.**

## SECTION 8 (optional, recommended for Linux beginners) — WSL 2

Everything on the GPU machine is Linux `bash`. Practise it locally for free:

```powershell
wsl --install -d Ubuntu-24.04
```

Reboot, open **Ubuntu** from the Start menu, create a user, then practise: `ls`, `cd`, `pwd`, `nano file.txt`, `cat file.txt`, `tmux new -s t` (detach with `Ctrl+b` then `d`, return with `tmux attach -t t`). The toolkit's full WSL chapter is `peft-universal/docs/manual_src/M04b_wsl.md`.

---

## CHECKPOINT (end of P01)

- [ ] uv, Python 3.11, Git, VS Code (+3 extensions), WinSCP installed.
- [ ] Hugging Face account and write token; gated licences accepted if needed.
- [ ] Repo cloned; `.venv` active; `.env` filled and not tracked by git.
- [ ] `117 passed, 2 skipped` and `REHEARSAL PASSED`; `outputs\tiny` exists.
