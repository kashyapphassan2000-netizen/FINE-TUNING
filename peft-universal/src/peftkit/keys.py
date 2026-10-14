"""Manage gateway API keys, offline (edit the key file) or against a running gateway.

python -m peftkit.keys create --file keys.json --name alice --rpm 30 --models mymodel
python -m peftkit.keys list   --file keys.json
python -m peftkit.keys revoke --file keys.json --key-id 3f9a1c0b2d4e
python -m peftkit.keys remote-create --gateway https://gw.example.com --admin-key $ADMIN --name bob
"""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import time
from pathlib import Path


def _load(path: str) -> dict:
    p = Path(path)
    return json.loads(p.read_text() or "{}") if p.exists() else {}


def create(path: str, name: str, rpm: float | None = None, models: list | None = None) -> dict:
    keys = _load(path)
    key = "sk-" + secrets.token_urlsafe(32)
    h = hashlib.sha256(key.encode()).hexdigest()
    keys[h] = {"name": name, "rpm": rpm, "models": models, "created": int(time.time())}
    Path(path).write_text(json.dumps(keys, indent=2))
    return {"api_key": key, "key_id": h[:12]}


def revoke(path: str, key_id: str) -> bool:
    keys = _load(path)
    hits = [h for h in keys if h.startswith(key_id)]
    if len(hits) != 1:
        return False
    keys.pop(hits[0])
    Path(path).write_text(json.dumps(keys, indent=2))
    return True


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="API key management")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--file", required=True)
    c.add_argument("--name", required=True)
    c.add_argument("--rpm", type=float, default=None)
    c.add_argument("--models", nargs="*", default=None)
    lst = sub.add_parser("list")
    lst.add_argument("--file", required=True)
    r = sub.add_parser("revoke")
    r.add_argument("--file", required=True)
    r.add_argument("--key-id", required=True)
    rc = sub.add_parser("remote-create")
    rc.add_argument("--gateway", required=True)
    rc.add_argument("--admin-key", required=True)
    rc.add_argument("--name", required=True)
    rc.add_argument("--rpm", type=float, default=None)
    rc.add_argument("--models", nargs="*", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "create":
        print(json.dumps(create(a.file, a.name, a.rpm, a.models), indent=2))
        print("give api_key to the user now; only its hash is stored")
    elif a.cmd == "list":
        for h, m in _load(a.file).items():
            print(f"{h[:12]}  {m.get('name'):<16} rpm={m.get('rpm')} models={m.get('models')}")
    elif a.cmd == "revoke":
        print("revoked" if revoke(a.file, a.key_id) else "not found / ambiguous")
    else:
        import httpx

        r = httpx.post(f"{a.gateway.rstrip('/')}/admin/keys", headers={"Authorization": f"Bearer {a.admin_key}"},
                       json={"name": a.name, "rpm": a.rpm, "models": a.models}, timeout=30)
        print(r.status_code, r.text)


if __name__ == "__main__":
    main()
