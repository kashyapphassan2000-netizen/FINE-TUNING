# 9. Step 6 — Rent the GPU and connect from Windows

## 9.1 Rent

```browser
provider console -> deploy VM -> GPU type from chapter 5 -> image with a driver new enough for your torch build
                 -> your SSH key -> open port 22 (and 8000/8080 later, from YOUR IP only)
```

## 9.2 Connect (three tools, one machine)

```laptop
ssh-keygen -t ed25519 -f $HOME\.ssh\hyperstack
notepad $HOME\.ssh\config
# Host hs-train
#   HostName <public ip>
#   User ubuntu
#   IdentityFile ~/.ssh/hyperstack
#   ServerAliveInterval 30
ssh hs-train
```

|Tool|Use|Note|
|---|---|---|
|**VS Code Remote-SSH**|edit and run on the GPU as if it were local|install the extension, then "Connect to Host… hs-train"|
|**WinSCP**|drag-and-drop file transfer from Windows|SFTP, host `hs-train` details, same key file|
|**tmux**|keep the run alive when your laptop sleeps|`tmux new -s train`, detach with `Ctrl-b d`|

```laptop faster than dragging: send the whole project, minus junk
scp -r -i $HOME\.ssh\hyperstack .\peft-universal ubuntu@<ip>:~/
# or from WSL/Git Bash:
rsync -avz --exclude .venv --exclude data/raw --exclude outputs ./ hs-train:~/peft-universal/
```

:::warn Cost discipline starts now
Billing begins when the VM is ACTIVE, not when training starts. Write the shutdown command on a sticky note: `ssh hs-train sudo poweroff` plus deleting the VM in the console — hibernated VMs can still bill for storage.
:::

:::check
- `ssh hs-train` works without a password prompt.
- VS Code shows the remote folder; WinSCP transfers a test file.
- `tmux` session created before anything long starts.
:::
