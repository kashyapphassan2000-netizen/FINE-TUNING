"""Run every lab cell in order, capture its printed output and figures.

Usage: python run_labs.py part4_basics [part5_intermediate ...]
Writes build/out/<part>.json and build/out/fig_<cell>.png
"""
import contextlib
import io
import json
import os
import re
import sys
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
LABS = os.path.join(HERE, "..", "labs")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

# The build machine is CPU-only, so long training loops are shortened here.
# The guide prints the real (full-length) code; only this run is shortened.
CPU_OVERRIDES = json.loads(os.environ.get("T101_OVERRIDES", "{}"))


def split_cells(path):
    cells, cur_id, cur = [], None, []
    for line in open(path, encoding="utf-8"):
        m = re.match(r"^# %% (\S+)", line)
        if m:
            if cur_id:
                cells.append((cur_id, "".join(cur).strip("\n")))
            cur_id, cur = m.group(1), []
        elif cur_id:
            cur.append(line)
    if cur_id:
        cells.append((cur_id, "".join(cur).strip("\n")))
    return cells


def run(part):
    cells = split_cells(os.path.join(LABS, part + ".py"))
    ns = {"__name__": "__main__"}
    results = {}
    os.chdir(OUT)
    for cid, code in cells:
        figs = []

        def fake_show(*a, **k):
            name = f"fig_{cid.replace('.', '_')}_{len(figs)}.png"
            plt.savefig(os.path.join(OUT, name), dpi=130, bbox_inches="tight")
            plt.close("all")
            figs.append(name)

        plt.show = fake_show
        run_code = code
        for old, new in CPU_OVERRIDES.get(cid, []):
            assert old in run_code, (cid, old)
            run_code = run_code.replace(old, new)
        buf = io.StringIO()
        t0 = time.time()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            try:
                exec(compile(run_code, f"<{cid}>", "exec"), ns)
            except Exception:
                buf.write("!!! ERROR\n" + traceback.format_exc())
        dt = time.time() - t0
        out = buf.getvalue()
        results[cid] = {"code": code, "out": out, "figs": figs, "seconds": round(dt, 1)}
        print(f"--- {cid} ({dt:.1f}s)\n{out}", flush=True)
        if "!!! ERROR" in out:
            print("STOPPING on error", flush=True)
            break
    json.dump(results, open(os.path.join(OUT, part + ".json"), "w"), indent=1)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        run(p)
