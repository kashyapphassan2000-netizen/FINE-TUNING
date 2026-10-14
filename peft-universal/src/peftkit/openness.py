"""STEP 1 of the flow: is this model actually open, and may I use it the way I intend?

Three different questions people confuse:
  1. are the WEIGHTS downloadable?      (open weights)
  2. what does the LICENCE allow?       (commercial use, derivatives, redistribution)
  3. is it GATED / does it need consent? (acceptance forms, usage policies)

python -m peftkit.openness meta-llama/Llama-3.1-8B-Instruct --intended-use commercial
"""
from __future__ import annotations

import argparse
import json

# Licence families that matter in practice. Verify the actual text before shipping a product.
FAMILIES = {
    "apache-2.0": {"commercial": True, "derivatives": True, "redistribute": True, "note": "permissive"},
    "mit": {"commercial": True, "derivatives": True, "redistribute": True, "note": "permissive"},
    "bsd-3-clause": {"commercial": True, "derivatives": True, "redistribute": True, "note": "permissive"},
    "llama3": {"commercial": True, "derivatives": True, "redistribute": True,
               "note": "community licence: acceptable-use policy, naming/attribution duties, and a "
                       "monthly-active-user threshold above which you must ask the licensor"},
    "llama3.1": {"commercial": True, "derivatives": True, "redistribute": True, "note": "as llama3"},
    "gemma": {"commercial": True, "derivatives": True, "redistribute": True,
              "note": "terms of use with prohibited-use policy; restrictions propagate to derivatives"},
    "qwen": {"commercial": True, "derivatives": True, "redistribute": True, "note": "check the repo's LICENSE file"},
    "cc-by-4.0": {"commercial": True, "derivatives": True, "redistribute": True, "note": "attribution required"},
    "cc-by-sa-4.0": {"commercial": True, "derivatives": True, "redistribute": True,
                     "note": "share-alike: derivatives inherit the licence"},
    "cc-by-nc-4.0": {"commercial": False, "derivatives": True, "redistribute": True, "note": "NON-COMMERCIAL"},
    "openrail": {"commercial": True, "derivatives": True, "redistribute": True,
                 "note": "use-based restrictions that must be passed to downstream users"},
    "research-only": {"commercial": False, "derivatives": True, "redistribute": False, "note": "research only"},
    "proprietary": {"commercial": False, "derivatives": False, "redistribute": False, "note": "closed"},
    "unknown": {"commercial": False, "derivatives": False, "redistribute": False,
                "note": "treat as closed until you have read the licence"},
}


def normalise(license_id: str | None) -> str:
    if not license_id:
        return "unknown"
    lic = license_id.strip().lower()
    for key in FAMILIES:
        if lic == key or lic.startswith(key):
            return key
    if "nc" in lic and "cc-by" in lic:
        return "cc-by-nc-4.0"
    if "rail" in lic:
        return "openrail"
    return "unknown"


def fetch(model_id: str, token: str | None = None) -> dict:
    """Ask the Hub what it knows. Needs network; returns an error dict offline."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_id, token=token, files_metadata=True)
        files = [s.rfilename for s in (info.siblings or [])]
        return {"id": info.id, "license": (info.cardData or {}).get("license") or
                next((t.split(":", 1)[1] for t in (info.tags or []) if t.startswith("license:")), None),
                "gated": bool(getattr(info, "gated", False)), "private": bool(getattr(info, "private", False)),
                "files": files,
                "weight_files": [f for f in files if f.endswith((".safetensors", ".bin", ".gguf", ".pt"))],
                "has_config": any(f == "config.json" for f in files),
                "tags": info.tags or []}
    except Exception as exc:                      # offline, gated without token, or the model does not exist
        return {"id": model_id, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}


def assess(info: dict, intended_use: str = "commercial") -> dict:
    lic = normalise(info.get("license"))
    f = FAMILIES[lic]
    blockers, actions = [], []
    if info.get("error"):
        return {"model": info.get("id"), "license": None, "license_family": "unknown", "open_weights": None,
                "gated": None, "commercial_use": None, "derivatives": None, "redistribute_weights": None,
                "blockers": [], "actions": [f"could not read the Hub: {info['error']}",
                                            "check network and HF_TOKEN, or read the model page manually"],
                "decision": "UNKNOWN", "disclaimer": "engineering check, not legal advice"}
    if info.get("error"):
        actions.append(f"could not read the Hub: {info['error']} - check manually on the model page")
    if not info.get("weight_files") and not info.get("error"):
        blockers.append("no weight files: the model is not open-weights (API-only). Fine-tuning is IMPOSSIBLE "
                        "unless the vendor offers a tuning service")
    if intended_use == "commercial" and not f["commercial"]:
        blockers.append(f"licence '{info.get('license')}' ({lic}) does not permit commercial use")
    if info.get("gated"):
        actions.append("GATED: accept the licence on the model page with the same account as your HF_TOKEN")
    if lic == "unknown":
        actions.append("licence not identified: read LICENSE / the model card before training")
    if f.get("note"):
        actions.append(f"licence note: {f['note']}")
    actions.append("record the licence and revision in your dataset/model lineage (see the lineage chapter)")
    return {"model": info.get("id"), "license": info.get("license"), "license_family": lic,
            "open_weights": bool(info.get("weight_files")), "gated": bool(info.get("gated")),
            "commercial_use": f["commercial"], "derivatives": f["derivatives"],
            "redistribute_weights": f["redistribute"], "blockers": blockers, "actions": actions,
            # offline / unreadable Hub: we could not assess, which is NOT the same as "forbidden"
            "decision": ("UNKNOWN" if info.get("error") else "STOP" if blockers else "PROCEED"),
            "disclaimer": "engineering check, not legal advice"}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Is this model open, and may I use it as intended?")
    ap.add_argument("model")
    ap.add_argument("--intended-use", default="commercial", choices=["commercial", "internal", "research"])
    ap.add_argument("--token", default=None)
    a = ap.parse_args(argv)
    r = assess(fetch(a.model, a.token), a.intended_use)
    print(json.dumps(r, indent=2))
    raise SystemExit({"PROCEED": 0, "STOP": 2}.get(r["decision"], 3))


if __name__ == "__main__":
    main()
