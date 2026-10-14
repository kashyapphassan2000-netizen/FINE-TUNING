"""Dataset provenance, lineage and licence controls.

Every training record should be traceable to where it came from and what you are allowed to do with it.
This is engineering control, NOT legal advice: it records facts and flags conflicts for a human to decide.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

REQUIRED = ["source_id", "source_type", "license", "dataset_version", "created_at"]
OPTIONAL = ["generator_model", "generator_version", "prompt_version", "transformation", "quality_score",
            "filter_reason", "parent_batch", "consent", "retention_until"]

# Coarse buckets for the traps that actually bite. Verify each licence's real text before relying on it.
LICENSES = {
    "mit": {"commercial": True, "share_alike": False, "attribution": True},
    "apache-2.0": {"commercial": True, "share_alike": False, "attribution": True},
    "bsd-3-clause": {"commercial": True, "share_alike": False, "attribution": True},
    "cc0": {"commercial": True, "share_alike": False, "attribution": False},
    "cc-by-4.0": {"commercial": True, "share_alike": False, "attribution": True},
    "cc-by-sa-4.0": {"commercial": True, "share_alike": True, "attribution": True},
    "cc-by-nc-4.0": {"commercial": False, "share_alike": False, "attribution": True},
    "odbl": {"commercial": True, "share_alike": True, "attribution": True},
    "gpl-3.0": {"commercial": True, "share_alike": True, "attribution": True},
    "proprietary": {"commercial": False, "share_alike": False, "attribution": True},   # SOMEONE ELSE's closed data
    "owned": {"commercial": True, "share_alike": False, "attribution": False},         # your/your employer's own data
    "unknown": {"commercial": False, "share_alike": False, "attribution": True},
}


def stamp(records: list[dict], *, source_id: str, source_type: str, license: str, dataset_version: str,
          **extra) -> list[dict]:
    """Attach lineage to every record. `source_type`: human | scraped | licensed | synthetic | customer."""
    now = int(time.time())
    out = []
    for r in records:
        body = {k: v for k, v in r.items() if k != "meta"}
        meta = {**(r.get("meta") or {}), "source_id": source_id, "source_type": source_type,
                "license": license.lower(), "dataset_version": dataset_version, "created_at": now,
                "record_hash": hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:16],
                **{k: v for k, v in extra.items() if v is not None}}
        out.append({**r, "meta": meta})
    return out


def verify(records: list[dict], required: list[str] | None = None) -> dict:
    """Every record must carry the required lineage fields before it may be trained on."""
    req = required or REQUIRED
    missing = Counter()
    bad = []
    for i, r in enumerate(records):
        meta = r.get("meta") or {}
        for k in req:
            if not meta.get(k):
                missing[k] += 1
                if len(bad) < 5:
                    bad.append(i)
    return {"n": len(records), "missing_counts": dict(missing), "example_rows": bad, "ok": not missing}


def summarise(records: list[dict]) -> dict:
    lic = Counter((r.get("meta") or {}).get("license", "unknown") for r in records)
    src = Counter((r.get("meta") or {}).get("source_type", "unknown") for r in records)
    gen = Counter((r.get("meta") or {}).get("generator_model") for r in records if (r.get("meta") or {}).get("generator_model"))
    n = max(1, len(records))
    return {"licenses": dict(lic), "source_types": dict(src), "generator_models": dict(gen),
            "synthetic_share": round(src.get("synthetic", 0) / n, 4),
            "customer_data_share": round(src.get("customer", 0) / n, 4)}


def check_licenses(records: list[dict], intended_use: str = "commercial", model_license: str = "apache-2.0") -> dict:
    """Flag conflicts between the data's licences and what you intend to do with the trained model."""
    s = summarise(records)
    blockers, warnings = [], []
    for lic, count in s["licenses"].items():
        info = LICENSES.get(lic)
        if info is None:
            warnings.append(f"{lic}: unknown licence on {count} records - identify it before training")
            continue
        if intended_use == "commercial" and not info["commercial"]:
            blockers.append(f"{lic}: not usable commercially ({count} records)")
        if info["share_alike"]:
            warnings.append(f"{lic}: share-alike ({count} records) - may impose terms on what you distribute")
    if s["synthetic_share"] > 0 and any((r.get("meta") or {}).get("generator_model") for r in records):
        warnings.append("synthetic records present: the GENERATOR's terms of service also apply "
                        "(some vendors restrict training competing models)")
    if s["customer_data_share"] > 0:
        warnings.append("customer data present: check contract, consent, retention and deletion obligations")
    return {"intended_use": intended_use, "model_license": model_license, "summary": s,
            "blockers": blockers, "warnings": warnings,
            "decision": "BLOCK" if blockers else ("REVIEW" if warnings else "OK"),
            "disclaimer": "engineering control only, not legal advice: a human must sign this off"}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Dataset lineage and licence check (or --stamp lineage onto rows)")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--intended-use", default="commercial", choices=["commercial", "internal", "research"])
    ap.add_argument("--stamp", action="store_true", help="WRITE lineage onto every row of the single input file")
    ap.add_argument("--source-id", default=None, help="e.g. pump-manuals-2026-10")
    ap.add_argument("--source-type", default="synthetic", choices=["human", "scraped", "licensed", "synthetic", "customer"])
    ap.add_argument("--license", default="owned",
                    help="licence of the SOURCE data: owned (your own), cc-by-4.0, apache-2.0, proprietary (someone else's)")
    ap.add_argument("--dataset-version", default="1.0")
    ap.add_argument("--generator-model", default=None, help="LLM that generated the rows, if any")
    ap.add_argument("--out", default=None, help="with --stamp: output file (default: overwrite the input)")
    a = ap.parse_args(argv)
    rows = [json.loads(x) for f in a.inputs for x in Path(f).read_text(encoding="utf-8").splitlines() if x.strip()]
    if a.stamp:
        if len(a.inputs) != 1 or not a.source_id:
            ap.error("--stamp needs exactly one --inputs file and --source-id")
        rows = stamp(rows, source_id=a.source_id, source_type=a.source_type, license=a.license,
                     dataset_version=a.dataset_version, generator_model=a.generator_model)
        out = Path(a.out or a.inputs[0])
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"stamped {len(rows)} rows -> {out}")
        return
    v = verify(rows)
    print(json.dumps({"verify": v, "licences": check_licenses(rows, a.intended_use)}, indent=2))
    raise SystemExit(0 if v["ok"] and check_licenses(rows, a.intended_use)["decision"] != "BLOCK" else 1)


if __name__ == "__main__":
    main()
