"""Data contract: the rules a synthetic dataset must obey, written BEFORE generating a single row.

A contract fixes the schema, the allowed ranges/categories, business constraints, privacy policy and
the target marginal distributions. Every generated batch is checked against it and rejected if it fails.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Field_:
    name: str
    dtype: str                      # int | float | category | datetime | text | bool
    min: float | None = None
    max: float | None = None
    categories: list | None = None
    nullable: bool = False
    pii: bool = False               # must never appear in synthetic output
    target_distribution: dict | None = None   # category -> share, or {"mean":..,"std":..}


@dataclass
class Contract:
    name: str
    version: str
    fields: list
    constraints: list = field(default_factory=list)   # python expressions over column names
    privacy: dict = field(default_factory=lambda: {"pii_allowed": False, "max_exact_match_rate": 0.0,
                                                   "min_dcr_ratio": 0.9})
    quality: dict = field(default_factory=lambda: {"max_psi": 0.2, "min_utility_ratio": 0.9,
                                                   "max_duplicate_rate": 0.02})

    @classmethod
    def load(cls, path: str | Path) -> "Contract":
        d = yaml.safe_load(Path(path).read_text())
        return cls(name=d["name"], version=str(d.get("version", "1.0")),
                   fields=[Field_(**f) for f in d["fields"]],
                   constraints=d.get("constraints", []),
                   privacy={**cls.__dataclass_fields__["privacy"].default_factory(), **(d.get("privacy") or {})},
                   quality={**cls.__dataclass_fields__["quality"].default_factory(), **(d.get("quality") or {})})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    def to_dict(self) -> dict:
        return {"name": self.name, "version": self.version,
                "fields": [{k: v for k, v in f.__dict__.items() if v is not None} for f in self.fields],
                "constraints": self.constraints, "privacy": self.privacy, "quality": self.quality}

    def column_names(self) -> list:
        return [f.name for f in self.fields]

    def validate(self, rows: list[dict]) -> dict:
        """Return {violation_type: count} plus the row indices of the first few offenders."""
        counts: dict[str, int] = {}
        examples: dict[str, list] = {}

        def hit(kind: str, i: int) -> None:
            counts[kind] = counts.get(kind, 0) + 1
            examples.setdefault(kind, [])
            if len(examples[kind]) < 3:
                examples[kind].append(i)

        by_name = {f.name: f for f in self.fields}
        for i, r in enumerate(rows):
            for name, f in by_name.items():
                if name not in r:
                    hit(f"missing_column:{name}", i)
                    continue
                v = r[name]
                if v is None:
                    if not f.nullable:
                        hit(f"null_not_allowed:{name}", i)
                    continue
                if f.dtype in ("int", "float"):
                    try:
                        x = float(v)
                    except (TypeError, ValueError):
                        hit(f"not_numeric:{name}", i)
                        continue
                    if f.dtype == "int" and float(v) != int(float(v)):
                        hit(f"not_integer:{name}", i)
                    if f.min is not None and x < f.min:
                        hit(f"below_min:{name}", i)
                    if f.max is not None and x > f.max:
                        hit(f"above_max:{name}", i)
                elif f.dtype == "category" and f.categories is not None and v not in f.categories:
                    hit(f"unknown_category:{name}", i)
                elif f.dtype == "bool" and not isinstance(v, bool):
                    hit(f"not_bool:{name}", i)
            for expr in self.constraints:
                try:
                    if not eval(expr, {"__builtins__": {}}, dict(r)):   # noqa: S307 - contract authors are trusted
                        hit(f"constraint:{expr}", i)
                except Exception:
                    hit(f"constraint_error:{expr}", i)
        return {"n_rows": len(rows), "violations": counts, "example_rows": examples,
                "ok": not counts}


EXAMPLE = """
name: support_tickets_synthetic
version: "1.0"
fields:
  - {name: customer_tier, dtype: category, categories: [free, pro, enterprise]}
  - {name: issue_type, dtype: category, categories: [billing, outage, howto, bug]}
  - {name: severity, dtype: int, min: 1, max: 5}
  - {name: first_response_minutes, dtype: float, min: 0, max: 2880}
  - {name: resolved, dtype: bool}
constraints:
  - "severity <= 3 or issue_type != 'howto'"
  - "first_response_minutes >= 0"
privacy: {pii_allowed: false, max_exact_match_rate: 0.0, min_dcr_ratio: 0.9}
quality: {max_psi: 0.2, min_utility_ratio: 0.9, max_duplicate_rate: 0.02}
"""


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Validate rows against a data contract")
    ap.add_argument("contract")
    ap.add_argument("--rows", required=True, help="jsonl of records")
    a = ap.parse_args(argv)
    rows = [json.loads(x) for x in Path(a.rows).read_text().splitlines() if x.strip()]
    print(json.dumps(Contract.load(a.contract).validate(rows), indent=2))


if __name__ == "__main__":
    main()
