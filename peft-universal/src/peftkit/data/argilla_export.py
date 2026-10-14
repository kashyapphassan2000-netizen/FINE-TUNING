"""Push rows to Argilla for human review (optional; needs a running Argilla server).
Class names checked against argilla 2.8.0 source; this module was NOT run against a live server here."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--workspace", default=None)
    a = ap.parse_args(argv)
    import argilla as rg

    client = rg.Argilla(api_url=os.environ["ARGILLA_API_URL"], api_key=os.environ["ARGILLA_API_KEY"])
    settings = rg.Settings(
        fields=[rg.TextField(name="prompt"), rg.TextField(name="response")],
        questions=[rg.RatingQuestion(name="quality", values=[1, 2, 3, 4, 5]),
                   rg.TextQuestion(name="correction", required=False)],
    )
    ds = rg.Dataset(name=a.dataset, workspace=a.workspace, settings=settings, client=client)
    ds.create()
    records = []
    for line in Path(a.input).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        msgs = json.loads(line)["messages"]
        records.append({"prompt": "\n".join(m["content"] for m in msgs[:-1]), "response": msgs[-1]["content"]})
    ds.records.log(records)
    print(f"logged {len(records)} records to {a.dataset}")


if __name__ == "__main__":
    main()
