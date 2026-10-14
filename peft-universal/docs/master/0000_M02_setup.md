# LAB MANUAL — PART 2: Set Up Your Workspace

### Placeholders used in this Part

|Placeholder|Replace with|Example|
|---|---|---|
|YOUR_NAME|your Windows user name|karthik|
|YOUR_EMAIL|the email you use on GitHub|you@example.com|
|YOUR_GITHUB_USER|your GitHub user name|karthik-ai|

## SECTION 1 — Install uv (and Python with it)

**WHERE:** Windows Start menu → PowerShell

**GOAL:** have `uv` and Python 3.11 available in any terminal

Step 1. Press the Windows key.

Step 2. Type `PowerShell`.

Step 3. Click **Windows PowerShell**.

Step 4. Paste this command and press Enter:

```laptop
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**You should see:** lines ending with a message that uv was installed and where.

:::warn Watch out: antivirus
Some antivirus products flag this installer as suspicious (a known false positive reported on the uv issue tracker [2]). If yours blocks it, download the binary from the uv GitHub releases page instead [1].
:::

Step 5. Close the PowerShell window.

Step 6. Open a new PowerShell window (Steps 1-3 again).

Step 7. Type and press Enter:

```laptop
uv --version
```

**You should see:** `uv 0.12.19` or newer [6].

Step 8. Type and press Enter:

```laptop
uv python install 3.11
```

**You should see:** a line saying Python 3.11.x was installed.

:::check
- `uv --version` prints a version in a NEW terminal.
- If not: the installer did not update PATH. Sign out of Windows and back in, then retry Step 7.
:::

## SECTION 2 — Install Git, VS Code and WinSCP

**WHERE:** PowerShell

**GOAL:** the three desktop tools installed with one command each

Step 1. Type and press Enter (confirms the package ID before installing):

```laptop
winget search Git.Git
```

**You should see:** a row whose **Id** column reads `Git.Git`.

Step 2. Type and press Enter:

```laptop
winget install --id Git.Git -e
```

Step 3. Type and press Enter:

```laptop
winget install --id Microsoft.VisualStudioCode -e
```

Step 4. Type and press Enter:

```laptop
winget search WinSCP
```

**You should see:** an Id such as `WinSCP.WinSCP`. Use exactly the Id shown in the next step.

Step 5. Type and press Enter (replace the Id if Step 4 showed a different one):

```laptop
winget install --id WinSCP.WinSCP -e
```

Step 6. Close and reopen PowerShell.

Step 7. Type and press Enter:

```laptop
git --version
```

**You should see:** `git version 2.x`.

|If you see…|It means…|Do this next|
|---|---|---|
|`winget` is not recognized|App Installer is missing|install "App Installer" from the Microsoft Store, then repeat|
|`git` is not recognized after install|the terminal was opened before the install|close and reopen PowerShell|

:::check
- `git --version` works; VS Code and WinSCP appear in the Start menu.
:::

## SECTION 3 — Add the VS Code extensions

**WHERE:** VS Code

**GOAL:** Python support and Remote-SSH for the GPU VM

Step 1. Open VS Code from the Start menu.

Step 2. Press `Ctrl+Shift+X` to open **Extensions**.

Step 3. Type `Python` in the search box.

Step 4. Click **Install** on **Python** (publisher: Microsoft).

Step 5. Type `Remote - SSH` in the search box.

Step 6. Click **Install** on **Remote - SSH** (publisher: Microsoft).

Step 7. Type `YAML` in the search box.

Step 8. Click **Install** on **YAML** (publisher: Red Hat).

:::check
- The three extensions show **Installed** / a gear icon instead of an Install button.
:::

## SECTION 4 — Put the project on your laptop

**WHERE:** File Explorer, then PowerShell

**GOAL:** the project folder at `C:\Users\YOUR_NAME\code\peft-universal`

Step 1. Type in PowerShell and press Enter:

```laptop
mkdir $HOME\code
```

Step 2. In File Explorer, right-click `peft-universal.zip` → **Extract All...**

Step 3. In the **Extract** window, set the destination to `C:\Users\YOUR_NAME\code`.

Step 4. Click **Extract**.

Step 5. Back in PowerShell, type and press Enter:

```laptop
cd $HOME\code\peft-universal
```

Step 6. Type and press Enter:

```laptop
dir
```

**You should see:** folders including `src`, `configs`, `scripts`, `tests`, `data`, `docs`, `serving`.

:::warn Watch out: double folder
If you see only one folder called `peft-universal`, the ZIP extracted into a nested folder. Run `cd peft-universal` once more.
:::

:::check
- `dir` shows `src` and `configs` directly.
:::

## SECTION 5 — Create the virtual environment and install

**WHERE:** PowerShell, inside the project folder

**GOAL:** an isolated `.venv` with every library the laptop part needs

Step 1. Type and press Enter:

```laptop
uv venv .venv --python 3.11
```

Step 2. Type and press Enter:

```laptop
.\.venv\Scripts\Activate.ps1
```

**You should see:** `(.venv)` at the start of the prompt.

|If you see…|It means…|Do this next|
|---|---|---|
|"running scripts is disabled on this system"|PowerShell's execution policy|run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, answer `Y`, repeat Step 2|

Step 3. Type and press Enter (the CPU build of PyTorch — the laptop does not train big models):

```laptop
uv pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
```

Step 4. Type and press Enter:

```laptop
uv pip install -r requirements-dev.txt
```

Step 5. Type and press Enter:

```laptop
uv pip install -e .
```

**You should see:** a line ending in `Installed ... peft-universal`.

:::check
- The prompt shows `(.venv)`.
- `python -c "import peftkit, uadapt, synthkit; print('ok')"` prints `ok`.
- If not: you are in a new terminal without the environment — repeat Step 2.
:::

## SECTION 6 — Create your secrets file

**WHERE:** VS Code

**GOAL:** a `.env` file that holds keys and never goes to GitHub

Step 1. In VS Code click **File > Open Folder...**

Step 2. Select `C:\Users\YOUR_NAME\code\peft-universal` and click **Select Folder**.

Step 3. In the Explorer panel, right-click `.env.example` → **Copy**.

Step 4. Right-click empty space in the Explorer panel → **Paste**.

Step 5. Right-click the new copy → **Rename**, and name it `.env`

Step 6. Open `.env` and fill the fields you have now (the rest later):

|Field|Value|Why|
|---|---|---|
|HF_TOKEN|a **write** token from huggingface.co → Settings → Access Tokens|download gated models, publish yours|
|OPENAI_BASE_URL|your LLM provider's OpenAI-compatible URL|synthetic Q&A generation (Part 5)|
|OPENAI_API_KEY|that provider's key|same|
|VLLM_API_KEY|any long random string you make up|protects your vLLM server (Part 6)|

Step 7. Press `Ctrl+S` to save.

:::warn Watch out
Never paste keys into code files or chat messages. `.gitignore` already excludes `.env`.
:::

## SECTION 7 — Prove the code works on your machine

**WHERE:** PowerShell with `(.venv)` active

**GOAL:** the full test suite and the full-pipeline rehearsal pass

Step 1. Type and press Enter:

```laptop
python -m pytest -q
```

**You should see (real output from this build):**

```out
101 passed, 1 skipped
```

Step 2. Type and press Enter (runs data → train → evaluate → merge → serve → API key → load test → release gate, on CPU, in about two minutes):

```laptop
python scripts/rehearsal_cpu.py
```

**You should see:** the last line `REHEARSAL PASSED - every stage ran end to end`.

:::check
- 101 passed, 1 skipped (the skip needs an NVIDIA GPU).
- REHEARSAL PASSED.
- If a test fails: re-run Section 5 Step 5, then read the failing test's last lines — they name the missing piece.
:::

## SECTION 8 — Put it under version control

**WHERE:** PowerShell, then browser

**GOAL:** your project backed up on GitHub, without secrets

Step 1. Type and press Enter:

```laptop
git config --global user.name "YOUR_NAME"
```

Step 2. Type and press Enter:

```laptop
git config --global user.email "YOUR_EMAIL"
```

Step 3. Type and press Enter:

```laptop
git init -b main
```

Step 4. Type and press Enter:

```laptop
git add .
```

Step 5. Type and press Enter:

```laptop
git status
```

**You should see:** many new files, and **not** `.env`.

Step 6. Type and press Enter:

```laptop
git commit -m "peft-universal: initial"
```

Step 7. In the browser, go to https://github.com/new, name the repository `peft-universal`, choose **Private**, click **Create repository**.

Step 8. Type and press Enter:

```laptop
git remote add origin https://github.com/YOUR_GITHUB_USER/peft-universal.git
```

Step 9. Type and press Enter:

```laptop
git push -u origin main
```

:::check
- The repository page on GitHub shows your files and no `.env`.
:::
