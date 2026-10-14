"""STEP 5 of the flow: which drivers and libraries this GPU needs, and the exact install commands.

The software stack is decided by the GPU's compute capability and the CUDA version its driver supports.
Getting this wrong is the most common first-day failure on a rented machine.

python -m peftkit.gpu_stack                 # detect the GPU in this machine and print the plan
python -m peftkit.gpu_stack --gpu H100      # plan for a GPU you are about to rent
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess

# compute capability -> what the stack must satisfy. Sources: each library's own support matrix.
GPUS = {
    "T4":    {"cc": 7.5, "vram": 16, "bf16": False, "fp8": False, "flash_attn": False, "note": "fp16 only; older"},
    "V100":  {"cc": 7.0, "vram": 32, "bf16": False, "fp8": False, "flash_attn": False, "note": "no bf16, no FA2"},
    "A10G":  {"cc": 8.6, "vram": 24, "bf16": True,  "fp8": False, "flash_attn": True,  "note": "good value"},
    "L4":    {"cc": 8.9, "vram": 24, "bf16": True,  "fp8": True,  "flash_attn": True,  "note": "inference-oriented"},
    "L40S":  {"cc": 8.9, "vram": 48, "bf16": True,  "fp8": True,  "flash_attn": True,  "note": "strong all-round"},
    "A100":  {"cc": 8.0, "vram": 80, "bf16": True,  "fp8": False, "flash_attn": True,  "note": "NVLink, no fp8"},
    "H100":  {"cc": 9.0, "vram": 80, "bf16": True,  "fp8": True,  "flash_attn": True,  "note": "fp8 + FA3"},
    "H200":  {"cc": 9.0, "vram": 141, "bf16": True, "fp8": True,  "flash_attn": True,  "note": "more VRAM than H100"},
    "B200":  {"cc": 10.0, "vram": 180, "bf16": True, "fp8": True, "flash_attn": True,  "note": "newest: needs a very recent torch/CUDA"},
    "RTX4090": {"cc": 8.9, "vram": 24, "bf16": True, "fp8": True, "flash_attn": True, "note": "consumer; check provider terms"},
}


def detect() -> dict:
    """What is actually in this machine."""
    out = {"nvidia_smi": shutil.which("nvidia-smi") is not None}
    if out["nvidia_smi"]:
        try:
            q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
            rows = [r.strip() for r in q.stdout.splitlines() if r.strip()]
            out["gpus"] = rows
            out["count"] = len(rows)
            out["driver"] = rows[0].split(",")[-1].strip() if rows else None
        except Exception as exc:
            out["error"] = str(exc)[:120]
    try:
        import torch

        out.update({"torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
                    "torch_cuda": getattr(torch.version, "cuda", None)})
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            out["compute_capability"] = f"{cap[0]}.{cap[1]}"
            out["device_name"] = torch.cuda.get_device_name(0)
            out["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
    except Exception:
        pass
    return out


def plan(gpu: str | None = None, detected: dict | None = None) -> dict:
    d = detected or {}
    key = None
    name = (gpu or d.get("device_name") or "").upper().replace(" ", "").replace("-", "")
    # longest key first, or "L40S" matches "L4" and you get the wrong capability table
    for k in sorted(GPUS, key=len, reverse=True):
        if k.replace("-", "").upper() in name:
            key = k
            break
    spec = GPUS.get(key or "", None)
    steps = [
        ("driver + CUDA", "nvidia-smi", "the driver must support the CUDA version your torch wheel was built "
         "for (torch 2.14 wheels here are CUDA 13). A too-old driver is the #1 first-day failure"),
        ("python env", "uv venv .venv-train --python 3.11 && source .venv-train/bin/activate", "isolate training"),
        ("torch", "uv pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130",
         "match the wheel to the driver's CUDA"),
        ("core libs", "uv pip install -r requirements-train.txt",
         "transformers, peft, trl, accelerate, datasets"),
        ("quantization", "uv pip install bitsandbytes==0.50.2", "needed only for QLoRA (Linear layers)"),
        ("attention kernels", "uv pip install flash-attn --no-build-isolation",
         "optional; long builds; skip if torch SDPA already selects a fused kernel"),
        ("serving env (separate!)", "uv venv .venv-serve && uv pip install vllm==0.29.0",
         "vLLM pins an older torch: never share the training env"),
        ("verify", "python -c \"import torch;print(torch.__version__, torch.cuda.is_available(), "
         "torch.cuda.get_device_name(0))\"", "must print True and the GPU name"),
    ]
    warn = []
    if spec:
        if not spec["bf16"]:
            warn.append(f"{key} has no bf16: use fp16 with a loss scaler, and expect more instability")
        if not spec["flash_attn"]:
            warn.append(f"{key} does not support FlashAttention-2: use attn_implementation='sdpa'")
        if not spec["fp8"]:
            warn.append(f"{key} has no fp8: do not plan an fp8 KV cache or fp8 weights")
        if spec["cc"] >= 10.0:
            warn.append("very new architecture: verify that your torch/vLLM build supports this compute capability")
    else:
        warn.append("unknown GPU: check bf16/fp8/FlashAttention support before choosing dtypes")
    return {"gpu": key or gpu or "unknown", "spec": spec, "detected": d, "steps": steps, "warnings": warn,
            "extra_for_multi_gpu": ["NCCL comes with the torch wheel",
                                    "accelerate config (DDP or FSDP2) - see the training chapters",
                                    "check NVLink/InfiniBand with `nvidia-smi topo -m` before multi-node runs"]}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="GPU software stack plan")
    ap.add_argument("--gpu", default=None, help="T4, A10G, L4, L40S, A100, H100, H200, B200, RTX4090")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    p = plan(a.gpu, detect())
    if a.json:
        print(json.dumps(p, indent=2))
        return
    print(f"GPU: {p['gpu']}  spec: {p['spec']}")
    print(f"detected: {json.dumps(p['detected'])}\n")
    for i, (name, cmd, why) in enumerate(p["steps"], 1):
        print(f"{i}. {name}\n   $ {cmd}\n   why: {why}")
    for w in p["warnings"]:
        print(f"WARNING: {w}")
    print("multi-GPU:", "; ".join(p["extra_for_multi_gpu"]))


if __name__ == "__main__":
    main()
