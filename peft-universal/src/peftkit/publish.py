"""Ship a model/adapter folder to: Hugging Face Hub, a local/mounted folder (e.g. Google Drive), or S3.

python -m peftkit.publish --src outputs/run/merged --hub your-user/my-model --private
python -m peftkit.publish --src outputs/run/adapter --local /mnt/gdrive/models/run1
python -m peftkit.publish --src outputs/run/merged --s3 s3://my-bucket/models/run1   (AWS_ENDPOINT_URL for FakeCloud/Hyperstack)
python -m peftkit.publish --src data/datasets/pump-tickets-v1.0 --hub your-user/pump-tickets --repo-type dataset --private
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

SKIP = {"optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin"}


def to_hub(src: str, repo_id: str, private: bool, revision: str | None = None, repo_type: str = "model") -> str:
    """repo_type: model (weights/adapters), dataset (your JSONL + card)."""
    from huggingface_hub import HfApi

    api = HfApi()  # token from HF_TOKEN env or `hf auth login`
    api.create_repo(repo_id, private=private, exist_ok=True, repo_type=repo_type)
    if revision:
        api.create_branch(repo_id, branch=revision, repo_type=repo_type, exist_ok=True)
    info = api.upload_folder(repo_id=repo_id, folder_path=src, revision=revision, repo_type=repo_type,
                             commit_message=f"upload {Path(src).name}", ignore_patterns=list(SKIP))
    return str(info)


def to_local(src: str, dst: str) -> str:
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*SKIP))
    return dst


def to_s3(src: str, uri: str) -> list[str]:
    import boto3

    assert uri.startswith("s3://"), "use s3://bucket/prefix"
    bucket, _, prefix = uri[5:].partition("/")
    s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or None)
    keys = []
    for path in sorted(Path(src).rglob("*")):
        if path.is_file() and path.name not in SKIP:
            key = f"{prefix.rstrip('/')}/{path.relative_to(src).as_posix()}".lstrip("/")
            s3.upload_file(str(path), bucket, key)
            keys.append(key)
    return keys


def from_s3(uri: str, dst: str) -> int:
    import boto3

    bucket, _, prefix = uri[5:].partition("/")
    s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or None)
    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix):].lstrip("/")
            target = Path(dst) / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, obj["Key"], str(target))
            n += 1
    return n


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Publish model artifacts")
    ap.add_argument("--src", required=True)
    ap.add_argument("--hub", default=None, help="repo id user/name")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--revision", default=None, help="branch to upload to, e.g. v1 (created if missing)")
    ap.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    ap.add_argument("--local", default=None, help="folder path (mounted Google Drive works)")
    ap.add_argument("--s3", default=None, help="s3://bucket/prefix")
    a = ap.parse_args(argv)
    if not (a.hub or a.local or a.s3):
        ap.error("choose at least one of --hub --local --s3")
    if a.hub:
        print("hub:", to_hub(a.src, a.hub, a.private, a.revision, a.repo_type))
    if a.local:
        print("local:", to_local(a.src, a.local))
    if a.s3:
        print("s3 objects:", len(to_s3(a.src, a.s3)))


if __name__ == "__main__":
    main()
