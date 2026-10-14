# P08 — Step 7: Rent the GPU (HyperStack, RunPod or Vast.ai) and connect from Windows

**Time:** 1 hour the first time, 5 minutes after. **Cost:** billing starts when the machine is running.
**GOAL:** a GPU machine you reach with `ssh gpu`, edit with VS Code, drag files to with WinSCP, and keep jobs alive with tmux.

> **Cost discipline starts now.** Billing runs while the machine exists in a running state, not while you train. Before renting, set a phone alarm for your planned end time. At the end of every session: weights to the Hub (P14), then **delete** the machine (Section 8).

## Which provider? (one-minute choice)

| You want… | Choose | Why |
|---|---|---|
| a normal Ubuntu VM, stable, NVLink/SXM nodes for multi-GPU | **HyperStack** | full VM with `sudo`, per-minute billing, free egress |
| the quickest start with a ready PyTorch image, per-second billing | **RunPod** (Secure Cloud) | templates with SSH pre-configured, `/workspace` volume |
| the lowest price for 24-80 GB cards | **Vast.ai** | marketplace; check each host's reliability score |

Prices to compare: P03 Section 4. GPU to pick: P03 Section 5.

---

## SECTION 1 — Create your SSH key (once, for all providers)

**WHERE:** PowerShell on your PC
**GOAL:** a key pair: the private key stays on your PC, the public key goes to the provider

Step 1. Paste and press Enter (press Enter at both prompts for no passphrase, or type one and remember it):

```powershell
ssh-keygen -t ed25519 -f $HOME\.ssh\gpu_ed25519 -C "your_email@example.com"
```

Step 2. Print the public key and copy the whole line (it starts with `ssh-ed25519`):

```powershell
Get-Content $HOME\.ssh\gpu_ed25519.pub
```

**YOU SHOULD SEE:** one line like `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAA... your_email@example.com`

> **WATCH OUT:** paste the **whole** line including `ssh-ed25519`, not the `SHA256:` fingerprint. Only the `.pub` file is ever shared. If you add several keys to an account, put each on its own line.

**CHECKPOINT** — `$HOME\.ssh\gpu_ed25519` and `gpu_ed25519.pub` exist.

---

## SECTION 2A — HyperStack (full VM)

**WHERE:** browser, https://console.hyperstack.cloud
**GOAL:** an ACTIVE Ubuntu VM with a public IP; `SSH_USER = ubuntu`, `SSH_PORT = 22`, `WORK = /home/ubuntu`

Step 1. **Billing** → add credit ($10-20 to start).

Step 2. **Environments** → **Create Environment** → pick a region → **Create**. (GPU availability differs per region; if your GPU is missing, try another region.)

Step 3. **SSH Keys** → **Import Key** → choose the environment → name `gpu` → paste the line from Section 1 Step 2 → **Import**.

Step 4. **Virtual Machines** → **Deploy New Virtual Machine**, and fill in:

| Field | Value | Why |
|---|---|---|
| Environment | the one from Step 2 | region and network |
| Flavor (GPU) | from P03 (e.g. 1× L40 48 GB, 1× A100 80 GB, 1× H100 80 GB) | fits your method with headroom |
| Image | an **Ubuntu image with NVIDIA drivers + CUDA** (read the image description: the driver version must support CUDA 13 for the pinned torch 2.14 wheels, i.e. driver R580 or newer; P09 Section 1 checks it) | correct driver = no first-day failure |
| Boot disk / volume | ≥ the `HOST disk` line of P03 rounded up (100 GB minimum for 8B) | weights + merged copy + checkpoints |
| SSH key | `gpu` | passwordless login |
| Public IP | **enabled** | so you can connect |
| Security rules / firewall | allow **SSH (TCP 22)** from your IP | the VM is unreachable otherwise |

Step 5. **Deploy**. Wait until the status is **ACTIVE** (1-5 minutes). Copy the **public IP** → this is `VM_IP`.

> **WATCH OUT:** **Hibernate** keeps the disk (and its storage cost) but stops GPU billing; **Delete** stops all billing and destroys the disk. Spot VMs (cheaper) can be reclaimed at any time: use them only with checkpoints (P10).

## SECTION 2B — RunPod (container "Pod")

**WHERE:** browser, https://console.runpod.io
**GOAL:** a running Pod with **SSH over exposed TCP**; `SSH_USER = root`, `SSH_PORT` = the mapped port shown, `WORK = /workspace`

Step 1. **Billing** → add credit.

Step 2. **Settings → SSH Public Keys** → paste the line from Section 1 Step 2 → **Save**. (Keys are injected into **new** Pods; a Pod created before you added the key does not get it.)

Step 3. **Pods** → **Deploy** → choose **Secure Cloud** (data-centre) or **Community Cloud** (cheaper, less reliable) → pick the GPU from P03.

Step 4. **Template:** choose an official **Runpod PyTorch** template (it runs an SSH daemon and exposes TCP 22). **Edit Template / Pod settings:**

| Field | Value | Why |
|---|---|---|
| Container Disk | 30-50 GB | OS + Python environments (wiped when the Pod stops) |
| Volume Disk | ≥ the `HOST disk` line of P03 (mounted at `/workspace`) | weights, data, checkpoints survive a stop |
| Expose TCP Ports | `22` (already set in official templates) | needed for full SSH/SCP/SFTP |
| Instance pricing | On-Demand (or Spot/interruptible only with checkpoints) | |

Step 5. **Deploy On-Demand**. When the Pod is **Running**, open it → **Connect** tab → copy the command under **"SSH over exposed TCP"**. It looks like:

```text
ssh root@213.173.108.12 -p 17445 -i ~/.ssh/id_ed25519
```

→ `VM_IP = 213.173.108.12`, `SSH_PORT = 17445`.

> **WATCH OUT:** the other command, `ssh <pod-id>@ssh.runpod.io`, is the **proxied** SSH. It works for a terminal but **does not support SCP/SFTP**, so WinSCP and VS Code file transfer fail with it. Always use "SSH over exposed TCP" (needs a Pod with a public IP; official templates provide it).

## SECTION 2C — Vast.ai (marketplace)

**WHERE:** browser, https://cloud.vast.ai
**GOAL:** a rented instance with **direct** SSH; `SSH_USER = root`, `SSH_PORT` = the port shown, `WORK = /workspace`

Step 1. **Billing** → add credit.

Step 2. **Account → Keys** (https://cloud.vast.ai/manage-keys/) → **Add SSH Key** → paste the line from Section 1 Step 2. (Account keys apply to **new** instances only.)

Step 3. **Search** → set filters:

| Filter | Value | Why |
|---|---|---|
| GPU model / count | from P03 | |
| GPU RAM | ≥ the VRAM you need | |
| Disk space | ≥ the `HOST disk` line of P03 | **disk size is fixed when you create the instance** |
| Reliability | ≥ 0.98 | marketplace hosts vary |
| Internet down/up | ≥ 500 Mbps | model downloads and Hub uploads |
| CUDA version (max) | ≥ 13.0 for the pinned torch 2.14 wheels (or see P09 Section 1) | the host's driver decides |
| Instance type | On-demand (interruptible only with checkpoints) | |

Step 4. **Template:** a PyTorch/CUDA template with SSH launch mode (e.g. "PyTorch (Vast)"). Click **Rent** on an offer.

Step 5. **Instances** → when it is running, click the **SSH / Connect** button → choose the **direct** connection if offered. You see:

```text
ssh -p 50983 root@154.64.230.50 -L 8080:localhost:8080
```

→ `VM_IP = 154.64.230.50`, `SSH_PORT = 50983`. (Vast puts you inside a tmux session by default.)

> **WATCH OUT:** `scp` and `sftp` take the port with a **capital** `-P`; `ssh` uses lower-case `-p`. A **stopped** Vast instance still bills for its disk; only **destroy** stops all billing, and also deletes the data.

---

## SECTION 3 — One SSH config for every provider

**WHERE:** PowerShell, then Notepad
**GOAL:** `ssh gpu` works for whatever machine you rent today

Step 1. Paste and press Enter:

```powershell
notepad $HOME\.ssh\config
```

(Click **Yes** to create the file if asked.)

Step 2. Paste this block, replace `VM_IP`, `SSH_PORT`, `SSH_USER`, save (`Ctrl+S`), close:

```text
Host gpu
    HostName VM_IP
    Port SSH_PORT
    User SSH_USER
    IdentityFile ~/.ssh/gpu_ed25519
    ServerAliveInterval 30
    ServerAliveCountMax 10
```

> **WATCH OUT:** Notepad may save it as `config.txt`. In File Explorer enable **View → File name extensions** and rename it to `config` with no extension.

Step 3. Paste and press Enter; answer `yes` to the fingerprint question:

```powershell
ssh gpu
```

**YOU SHOULD SEE:** a Linux prompt like `ubuntu@...:~$` (HyperStack) or `root@...:/#` (RunPod/Vast.ai).

| If you see… | It means… | Do this |
|---|---|---|
| asked for a password | the key is not on the machine | re-check the key in the provider account; for RunPod/Vast, keys only reach machines created **after** you added the key: recreate it |
| `Permission denied (publickey)` | wrong key, user or port | check `IdentityFile`, `User` (ubuntu vs root), `Port`; run `ssh -vv gpu` to see which key is tried |
| `Connection timed out` | firewall / wrong IP / machine not running | HyperStack: allow TCP 22 in security rules; check status ACTIVE/Running |
| `REMOTE HOST IDENTIFICATION HAS CHANGED` | new machine, same IP or alias | `ssh-keygen -R "[VM_IP]:SSH_PORT"` (or `ssh-keygen -R VM_IP` for port 22), retry |
| `bad permissions` on the key | Windows ACL too open | right-click the key → Properties → Security → leave only your user |

Step 4. Type `exit` and press Enter.

**Every new session:** only `HostName` and `Port` change in this file. Edit two lines, done.

## SECTION 4 — VS Code Remote-SSH (edit and run on the GPU as if local)

**WHERE:** VS Code
**GOAL:** a VS Code window whose files and terminal are on the GPU machine

Step 1. Press `F1` → type `Remote-SSH: Connect to Host` → Enter → choose `gpu`.

Step 2. Wait for "Setting up SSH host" to finish (first time: about a minute; it installs a small server on the machine).

Step 3. **File → Open Folder** → `WORK` (e.g. `/home/ubuntu` or `/workspace`) → **OK**.

Step 4. **Terminal → New Terminal**: this terminal is **bash on the GPU machine**.

**YOU SHOULD SEE:** `SSH: gpu` in the bottom-left corner.

## SECTION 5 — WinSCP (drag-and-drop files)

**WHERE:** WinSCP on your PC
**GOAL:** your PC on the left, the GPU machine on the right

Step 1. Open WinSCP → **New Session**:

| Field | Value |
|---|---|
| File protocol | **SFTP** |
| Host name | `VM_IP` |
| Port number | `SSH_PORT` (22 on HyperStack) |
| User name | `SSH_USER` (`ubuntu` or `root`) |
| Password | (empty) |

Step 2. **Advanced…** → **SSH → Authentication** → **Private key file** → `…` → choose `C:\Users\YOUR_NAME\.ssh\gpu_ed25519` (set the file filter to *All files*). WinSCP offers to **convert it to PuTTY format (.ppk)** → **OK** → save the `.ppk` next to it → **OK**.

Step 3. **Save** the session as `gpu` → **Login**.

**YOU SHOULD SEE:** the remote home folder on the right. Drag a test file across.

**Command-line alternatives** (often faster for whole folders; run in PowerShell):

```powershell
scp -P SSH_PORT -i $HOME\.ssh\gpu_ed25519 -r data\processed SSH_USER@VM_IP:WORK/FINE-TUNING/peft-universal/data/
scp -P SSH_PORT -i $HOME\.ssh\gpu_ed25519 SSH_USER@VM_IP:WORK/FINE-TUNING/peft-universal/reports/compare.json .
```

(With the config from Section 3: `scp -r data\processed gpu:WORK/FINE-TUNING/peft-universal/data/`.)

## SECTION 6 — tmux: keep the run alive when your laptop sleeps

**WHERE:** the GPU machine's terminal
**GOAL:** training continues if the connection drops

```bash
tmux new -s train          # start a named session
# ... run your commands inside it ...
# detach: press Ctrl+b, release, then press d
tmux ls                    # list sessions
tmux attach -t train       # come back
```

> **WATCH OUT:** a job started **outside** tmux dies when SSH disconnects. Start every long command inside tmux. Vast.ai already opens a tmux session for you; inside it use `Ctrl+b c` for a new window.

## SECTION 7 — Port forwarding (see a web UI running on the GPU in your browser)

```powershell
ssh -L 7860:localhost:7860 -L 6006:localhost:6006 gpu
```

Then open http://localhost:7860 (a Gradio demo, P14) or http://localhost:6006 (TensorBoard) on your PC. Nothing is exposed to the internet.

## SECTION 8 — End of every session: stop paying

| Provider | Stops GPU billing, keeps disk (disk still billed) | Stops ALL billing (deletes the data) |
|---|---|---|
| HyperStack | Hibernate | **Delete** the VM (and any separate volume) |
| RunPod | Stop the Pod (volume disk billed at the idle rate) | **Terminate** the Pod; delete network volumes you no longer need |
| Vast.ai | Stop (disk still billed) | **Destroy** the instance |

Order of operations: (1) adapter, merged model and reports are on the Hub or your PC (P14), (2) `exit`, (3) delete/terminate/destroy in the console, (4) refresh the console and confirm nothing is running.

---

## CHECKPOINT

- [ ] `ssh gpu` logs in without a password.
- [ ] VS Code shows `SSH: gpu`; its terminal is bash on the GPU machine.
- [ ] WinSCP transferred a test file.
- [ ] `tmux new -s train` works, and I can detach and re-attach.
- [ ] I know exactly which button stops all billing on my provider.
