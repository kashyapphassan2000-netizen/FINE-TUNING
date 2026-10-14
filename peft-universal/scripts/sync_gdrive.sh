#!/usr/bin/env bash
# Copy a folder to Google Drive with rclone. One-time: `rclone config` -> new remote "gdrive", type "drive".
# Headless VM: choose "n" for auto config, run the printed `rclone authorize "drive"` on your laptop, paste the token back.
set -euo pipefail
SRC=${1:?usage: sync_gdrive.sh <local_dir> <gdrive_path>}
DST=${2:?usage: sync_gdrive.sh <local_dir> <gdrive_path>}
rclone copy "$SRC" "gdrive:$DST" --progress --transfers 4 --exclude "optimizer.pt" --exclude "checkpoint-*/**"
rclone check "$SRC" "gdrive:$DST" --one-way --exclude "optimizer.pt" --exclude "checkpoint-*/**"
