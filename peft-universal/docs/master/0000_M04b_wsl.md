# LAB MANUAL — PART 4B: The Linux Bridge — WSL 2 crash course for Windows users

:::why Concept (what, why, when)
From Part 5 on, everything that runs on the GPU runs on **Linux** (Ubuntu): bash, tmux, apt, Docker, Kubernetes. WSL 2 gives you a real Ubuntu inside Windows, so you learn those commands on your laptop where mistakes are free. **Why:** the jump from PowerShell to a remote Ubuntu VM is where beginners get lost; with WSL the remote VM in Part 5 feels identical to what you already practised. **When:** before Part 5.
:::

:::honest What was and was not executed
WSL cannot run inside the Linux build machine used for this manual. The Windows-side steps follow Microsoft's `wsl` command and VS Code's WSL extension as documented; the Linux-side commands (uv, the project, the rehearsal) are the same ones executed on Linux throughout this book. Verify menu names on your Windows build [32][33].
:::

## SECTION 1 — Install WSL 2 and Ubuntu

**WHERE:** Windows Start menu → **Terminal** or **PowerShell**, opened **as Administrator** (right-click → **Run as administrator**)

**GOAL:** an Ubuntu Linux you can open from Windows

Step 1. Type and press Enter (lists the Linux distributions you can install):

```laptop
wsl --list --online
```

**You should see:** a table with a NAME column containing entries such as `Ubuntu` and versioned names like `Ubuntu-24.04`.

Step 2. Type and press Enter, using a NAME exactly as shown in Step 1:

```laptop
wsl --install -d Ubuntu-24.04
```

Step 3. Restart Windows when asked.

Step 4. After the restart an Ubuntu window opens. Type a **Linux user name** (lowercase, no spaces) and press Enter.

Step 5. Type a **password** and press Enter (nothing is shown while you type — that is normal).

Step 6. Type the password again and press Enter.

**You should see:** a prompt like `YOUR_NAME@LAPTOP:~$`.

Step 7. In PowerShell (not Ubuntu), type and press Enter:

```laptop
wsl --list --verbose
```

**You should see:** your Ubuntu with `VERSION` = **2**.

|If you see…|It means…|Do this next|
|---|---|---|
|VERSION 1|old WSL mode|`wsl --set-version Ubuntu-24.04 2`|
|"virtualization is not enabled"|disabled in firmware|enable Intel VT-x / AMD-V in BIOS/UEFI, restart|
|install hangs at 0%|Store/network problem|`wsl --update`, then retry Step 2|

:::check
- Ubuntu opens and `wsl --list --verbose` shows VERSION 2.
:::

## SECTION 2 — Ten Linux commands you will use on the GPU VM

**WHERE:** the Ubuntu window

**GOAL:** type the commands the rest of this manual assumes, once, where nothing can break

Run each line, one at a time, and read the result:

```wsl
pwd
ls -la
mkdir -p ~/code && cd ~/code
echo "hello" > note.txt && cat note.txt
nano note.txt
cp note.txt copy.txt && mv copy.txt renamed.txt && rm renamed.txt
df -h ~
free -h
sudo apt update
tmux new -s practice
```

|Command|What it does|Where you will need it|
|---|---|---|
|`pwd`, `ls -la`, `cd`|where am I, what is here, go there|every GPU session|
|`nano FILE`|edit a file in the terminal (`Ctrl+O` save, `Ctrl+X` exit)|editing `.env` on the VM|
|`df -h`, `free -h`|disk and RAM left|before downloading a model|
|`sudo apt update`|refresh the package list (asks your password)|installing system packages|
|`tmux new -s NAME`|a session that survives a dropped connection; `Ctrl+b` then `d` to leave it|every training run|

Step 11. Inside tmux, press `Ctrl+b`, then `d`.

Step 12. Type and press Enter:

```wsl
tmux attach -t practice
```

**You should see:** you are back inside the same session. Type `exit` to close it.

:::check
- You used nano to edit a file and re-attached to a tmux session.
:::

## SECTION 3 — VS Code inside WSL

**WHERE:** VS Code on Windows, then the Ubuntu window

**GOAL:** edit Linux files in VS Code exactly as you will edit files on the GPU VM

Step 1. In VS Code press `Ctrl+Shift+X`.

Step 2. Type `WSL` in the search box.

Step 3. Click **Install** on **WSL** (publisher: Microsoft).

Step 4. In the Ubuntu window, type and press Enter:

```wsl
cd ~/code && code .
```

**You should see:** a VS Code window whose bottom-left corner shows **WSL: Ubuntu-24.04**.

Step 5. In that VS Code window, click **Terminal > New Terminal**.

**You should see:** a bash terminal (`$` prompt), not PowerShell.

:::tip This is the same experience as Part 5
In Part 5, VS Code's **Remote-SSH** opens the GPU VM the same way: green corner label, bash terminal, Linux paths. If this section felt normal, Part 5 will too.
:::

:::check
- Bottom-left shows **WSL: …** and the integrated terminal is bash.
:::

## SECTION 4 — Run the project inside Linux

**WHERE:** the VS Code WSL terminal

**GOAL:** the same rehearsal from Part 2, now on Linux

Step 1. Type and press Enter (the Linux installer from the uv documentation [1]):

```wsl
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Step 2. Type and press Enter:

```wsl
source $HOME/.local/bin/env
```

Step 3. Copy the project into Linux. Type and press Enter (replace YOUR_NAME with your **Windows** user name):

```wsl
cp -r /mnt/c/Users/YOUR_NAME/code/peft-universal ~/code/
```

Step 4. Type and press Enter:

```wsl
cd ~/code/peft-universal && rm -rf .venv
```

Step 5. Type and press Enter:

```wsl
uv venv .venv --python 3.11 && source .venv/bin/activate
```

Step 6. Type and press Enter:

```wsl
uv pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu && uv pip install -r requirements-dev.txt && uv pip install -e .
```

Step 7. Type and press Enter:

```wsl
python scripts/rehearsal_cpu.py
```

**You should see:** `REHEARSAL PASSED - every stage ran end to end`.

:::warn Watch out: keep Linux projects on the Linux disk
`/mnt/c/...` is your Windows disk seen from Linux; working there is much slower and mixes Windows and Linux line endings. Keep the project under `~/code`. And never reuse the Windows `.venv` in Linux (Step 4 removes it): environments are per operating system.
:::

:::check
- REHEARSAL PASSED inside WSL.
:::

## SECTION 5 — Docker in WSL (needed for Part 6)

**WHERE:** Windows, then the WSL terminal

**GOAL:** `docker` works from your Linux shell

Step 1. Install **Docker Desktop** for Windows from https://www.docker.com/products/docker-desktop [34] (check its licence terms: free for personal use and small organisations, paid for larger companies (verify)).

Step 2. Open Docker Desktop → **Settings** → **Resources** → **WSL integration**.

Step 3. Switch on your Ubuntu distribution.

Step 4. Click **Apply & restart**.

Step 5. In the WSL terminal, type and press Enter:

```wsl
docker run --rm hello-world
```

**You should see:** `Hello from Docker!`

Step 6. Type and press Enter:

```wsl
docker compose version
```

**You should see:** `Docker Compose version v2.x`. Part 6B needs **v2.24 or newer**.

:::check
- `hello-world` ran, and Compose is v2.24+.
:::

## 4B.6 Common mistake at this level

**Mixing the two worlds.** Running `pip` from PowerShell while the project runs in WSL, editing files on `C:\` from Linux, or copying a Windows `.venv` into Linux. **How to spot it:** "module not found" errors for things you just installed, or `\r` characters in scripts. **Fix:** one project copy per operating system, one environment per copy, and check the VS Code corner label before you type.
