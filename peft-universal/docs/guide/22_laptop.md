# 22. Laptop setup from zero (Windows)

## 22.1 Install table

|Tool|Why (one line)|Install|Verify|
|---|---|---|---|
|Git for Windows|version control + Git Bash to run the .sh scripts|`winget install Git.Git`|`git --version`|
|VS Code|editor + Remote-SSH to work on the GPU VM|`winget install Microsoft.VisualStudioCode`|`code --version`|
|uv|fast Python + virtualenv manager|see block below|`uv --version`|
|Python 3.11|runtime (installed by uv)|`uv python install 3.11`|`uv run --python 3.11 python -V`|
|OpenSSH client|connect to the VM (built into Windows 10/11)|Settings → Optional features|`ssh -V`|
|WinSCP|drag-and-drop file transfer to the VM|`winget install WinSCP.WinSCP`|opens|
|rclone (optional)|Google Drive sync|`winget install Rclone.Rclone`|`rclone version`|
|WSL Ubuntu (optional)|runs FakeCloud and bash tools|`wsl --install -d Ubuntu`|`wsl -l -v`|
|Docker Desktop (optional)|build the gateway image locally|`winget install Docker.DockerDesktop`|`docker version`|

```laptop Install uv (official installer)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# open a NEW terminal, then:
uv --version
uv python install 3.11
```

VS Code extensions: **Python**, **Pylance**, **Remote - SSH**, **YAML**, **Jupyter** (optional), **GitLens** (optional).

## 22.2 Get the project and create the environment

```laptop
mkdir $HOME\code; cd $HOME\code
# unzip peft-universal.zip here (from this guide), then:
cd peft-universal
uv venv .venv --python 3.11
.\.venv\Scripts\Activate.ps1
# CPU build of torch for the laptop (PyPI Windows wheels are CPU; the explicit index makes it certain)
uv pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements-dev.txt
uv pip install -e .
python -m pytest -q
```

```out
32 passed, 1 skipped
```

:::warn Windows notes
- If `Activate.ps1` is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
- The recorded run was on Linux. If a torch 2.14.0 CPU wheel for your Python is missing on that index, install the newest torch that has one; the laptop only runs tests.
- `jax[cpu]` is only for lab 7; remove it from `requirements-dev.txt` if it fails on Windows.
- vLLM does not run on Windows. Serving happens on the GPU VM.
:::

## 22.3 Folder tree

```text peft-universal/
configs/            base.yaml, methods/*.yaml, models/*.yaml, accelerate/*.yaml
data/raw/           your documents (git-ignored)      data/seeds.jsonl   human seed tasks
data/processed/     train.jsonl, eval.jsonl (git-ignored)
docs/               build_pdf.py + guide/*.md (this PDF)
examples/           distilabel + DataDreamer reference pipelines
labs/               10 concept labs + outputs/
scripts/            gpu_preflight, setup_gpu_vm, pipeline, build_dataset, serve_vllm, sync_gdrive, ...
serving/            gateway/, k8s/, docker-compose.yml, prometheus.yml, scaler.py
src/peftkit/        config, hardware, modeling, peft_factory, callbacks, train, merge, generate,
                    inspect_model, serve_check, publish, data/{ingest,synth,llm,format,templates,
                    quality,self_instruct,judge,preference,argilla_export}
tests/              pytest suite + tiny model + fake servers
```

## 22.4 Housekeeping files

@file .gitignore
@file .gitattributes
@file pyproject.toml
@file requirements-train.txt
@file requirements-serve.txt
@file requirements-dev.txt

## 22.5 .env and why

:::why
Secrets (tokens, keys, URLs) change per machine and must never be committed. Code reads them from the environment; `.env` is a local file that fills the environment. `.gitignore` excludes it.
:::

@file .env.example

```laptop
Copy-Item .env.example .env
notepad .env
```

## 22.6 Git init and first push

```laptop
git init -b main
git add .
git commit -m "peft-universal: initial template"
# create an EMPTY private repo on GitHub named peft-universal, then:
git remote add origin git@github.com:<you>/peft-universal.git
git push -u origin main
git status   # .env must NOT be listed
```

:::check
- `python -m pytest -q` passes on my laptop.
- `.env` exists and is not tracked by git.
- The repo is on GitHub (private).
:::
