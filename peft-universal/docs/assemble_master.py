"""Assemble the single merged edition from Volume 1 (docs/guide) + Volume 2 (docs/guide2) + new chapters
(docs/master_src), in PIPELINE ORDER, renumbering headings and rewriting cross-references.

python docs/assemble_master.py          -> writes docs/master/*.md, then build with:
python docs/build_pdf.py --master Universal_Fine_Tuning_Complete_Guide.pdf
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

DOCS = Path(__file__).resolve().parent
V1, V2, NEW, OUT = DOCS / "guide", DOCS / "guide2", DOCS / "master_src", DOCS / "master"

# (new number, source, old number or file stem)
ORDER = [
    (0, "new", "00_howto"), (1, "new", "01_gaps"), (2, "new", "02_runbook"), (3, "new", "F01_flow_map"), (4, "new", "F02_step1_open"), (5, "new", "F03_step2_hardware"),
    (6, "new", "F04_step3_data_reality"), (7, "new", "F05_step4_mimic"), (8, "new", "F06_step5_data_gate"),
    (9, "new", "F07_step6_connect"), (10, "new", "F08_step7_stack"), (11, "new", "F09_step8_load"),
    (12, "new", "F10_step9_jsonl"), (13, "new", "F11_step10_run"), (14, "new", "F12_step11_eval_tune"),
    (15, "new", "F13_step12_publish"), (16, "new", "F14_step13_serve"),
    (17, "v1", 2), (18, "new", "04_map"),
    (19, "v1", 33), (20, "v2", 1), (21, "v2", 2), (22, "v2", 3),
    (23, "v1", 3), (24, "v1", 4), (25, "v1", 5), (26, "v1", 6), (27, "v1", 7), (28, "v1", 8), (29, "v1", 9),
    (30, "v1", 21), (31, "v1", 22), (32, "v1", 25),
    (33, "v2", 12), (34, "v1", 36), (35, "v1", 23), (36, "v1", 34), (37, "v1", 35),
    (38, "v1", 10), (39, "v2", 5), (40, "v1", 11), (41, "v1", 12), (42, "v1", 13), (43, "v2", 4),
    (44, "v1", 14), (45, "v2", 6), (46, "v1", 15), (47, "v1", 16), (48, "v1", 17), (49, "v1", 18),
    (50, "v1", 19), (51, "v1", 20),
    (52, "v2", 7), (53, "v2", 8), (54, "v2", 13), (55, "v1", 24), (56, "v1", 26), (57, "v2", 11),
    (58, "v2", 9), (59, "v2", 10), (60, "v2", 14), (61, "v2", 15), (62, "v2", 16),
    (63, "v2", 17), (64, "v2", 18),
    (65, "v1", 27), (66, "v1", 28), (67, "new", "53_eval_endpoint"), (68, "new", "54_loadtest"),
    (69, "new", "55_release_gate"), (70, "v1", 29), (71, "v1", 38), (72, "v1", 30),
    (73, "v1", 37), (74, "v2", 19), (75, "v2", 20), (76, "v1", 31), (77, "v1", 32), (78, "v2", 21),
    (79, "new", "65_brief"),
    (80, "new", "66_primer"), (81, "new", "67_scope"), (82, "new", "68_algorithm"), (83, "new", "69_identify"),
    (84, "new", "70_classify"), (85, "new", "71_trace"), (86, "new", "72_objective"), (87, "new", "73_strategy"),
    (88, "new", "74_targets"), (89, "new", "75_custom_lora"), (90, "new", "76_encoder"), (91, "new", "77_encdec"),
    (92, "new", "78_vision"), (93, "new", "79_audio"), (94, "new", "80_multimodal"), (95, "new", "81_moe"),
    (96, "new", "82_ssm_rnn"), (97, "new", "83_diffusion"), (98, "new", "84_custom_pipeline"),
    (99, "new", "85_adapter_class"), (100, "new", "86_report"), (101, "new", "87_data"), (102, "new", "88_debugging"),
    (103, "new", "89_eval_any"), (104, "new", "90_serving_stack"), (105, "new", "91_providers"),
    (106, "new", "92_gpus_finetune"), (107, "new", "93_gpus_serve"), (108, "new", "94_loading_math"),
    (109, "new", "95_cost"), (110, "new", "96_public_api"), (111, "new", "97_autoscale_growth"),
    (112, "new", "98_frontend_hosting"), (113, "new", "99_showcase"), (114, "new", "100_template"),
    (115, "new", "101_challenge"), (116, "new", "102_decision_tree"), (117, "new", "103_answers"),
    (118, "new", "104_review"), (119, "new", "105_levels"), (120, "new", "106_axes"),
    (121, "new", "107_registry"), (122, "new", "108_branches"), (123, "new", "109_diffusion_deep"),
    (124, "new", "110_tts_video"), (125, "new", "111_multimodal_deep"), (126, "new", "112_scope"),
    (127, "new", "113_lineage"), (128, "new", "114_engine"), (129, "new", "115_challenge2"),
    (130, "new", "116_review2"), (131, "new", "117_processor"), (132, "new", "118_bert_derivation"),
    (133, "new", "119_loss_derivation"), (134, "new", "120_pipeline"), (135, "new", "121_invented"),
]
MAP = {"v1": {}, "v2": {}}
for new, src, old in ORDER:
    if src in MAP:
        MAP[src][old] = new

# notes injected right after a chapter's first heading (the gaps this edition fixes)
INJECT = {
    42: ":::warn New in this edition\nTraining is not the end. Before merging, run the baseline-vs-tuned evaluation in chapters 44-45 (runbook steps 30-34) and merge the BEST checkpoint.\n:::\n",
    51: ":::warn New in this edition\nOnly merge after chapter 44's compare verdict is PASS, and merge the best checkpoint (chapter 43), not the last.\n:::\n",
    52: ":::warn New in this edition\nA running server is not a validated server. After this chapter, evaluate the SERVED model (chapter 53) and load test it (chapter 54) before the gateway and autoscaling in chapter 56.\n:::\n",
    56: ":::warn New in this edition: thresholds come from measurement\nThe example KEDA threshold (8) and scaler defaults below are placeholders. Use `suggested_scale_up_threshold` and `max_concurrency_per_replica` from `reports/loadtest.json` (chapter 54), and only scale after the release gate (chapter 55) prints RELEASE.\n:::\n",
}


def source_file(src: str, old) -> Path:
    if src == "new":
        return NEW / f"{old}.md"
    folder = V1 if src == "v1" else V2
    for f in sorted(folder.glob("*.md")):
        first = next((l for l in f.read_text(encoding="utf-8").splitlines() if l.startswith("# ")), "")
        m = re.match(r"# (\d+)\.", first)
        if m and int(m.group(1)) == old:
            return f
    raise FileNotFoundError(f"{src} chapter {old}")


def remap_num(vol: str, n: int) -> int | None:
    return MAP[vol].get(n)


NEW_SHIFT_FROM = 3      # every pre-existing "new" chapter numbered >= 3 moved by +14


def rewrite_new_refs(text: str) -> str:
    """Chapters written before the FLOW part was inserted refer to the old numbering. Shift any
    reference to a chapter that moved (3..121 -> +14); the FLOW chapters 3-16 keep their own numbers.

    ONE pass handles both "chapter N" and "chapters N-M": doing singles first and ranges second
    shifted the first number twice (it produced "chapters 137-124")."""
    def sh(n: int) -> int:
        return n + 14 if NEW_SHIFT_FROM <= n <= 121 else n

    def ref(m):
        head, a, sub, rng_b = m.group(1), int(m.group(2)), m.group(3) or "", m.group(5)
        out = f"{head} {sh(a)}{sub}"
        return out + f"-{sh(int(rng_b))}" if rng_b else out

    text = re.sub(r"\b(chapter|chapters|ch|Chapter|Ch)\s(\d+)(\.\d+)?(\s*-\s*(\d+))?\b", ref, text)

    def cell(m):
        parts = re.split(r"([,\-]\s*)", m.group(1))
        return "|" + "".join(str(sh(int(x.strip()))) if x.strip().isdigit() else x for x in parts) + "|"

    return re.sub(r"\|(\d+(?:\s*[,\-]\s*\d+)*)\|", cell, text)


def rewrite_refs(text: str, vol: str) -> str:
    """Rewrite 'Volume X, chapter N[.m]', 'chapter(s) N[-M]', 'ch N' to the merged numbering."""
    other = {"v1": "v2", "v2": "v1"}[vol]

    def vol_ref(m):
        v = "v1" if m.group(1) == "1" else "v2"
        n = int(m.group(3))
        new = remap_num(v, n)
        return f"chapter {new}{m.group(4) or ''}" if new is not None else m.group(0)

    text = re.sub(r"Volume ([12]),? (chapters?|ch) (\d+)(\.\d+)?", vol_ref, text)

    def range_ref(m):
        a, b = remap_num(vol, int(m.group(2))), remap_num(vol, int(m.group(3)))
        if a is None or b is None:
            return m.group(0)
        lo, hi = min(a, b), max(a, b)
        span = sorted(remap_num(vol, k) for k in range(int(m.group(2)), int(m.group(3)) + 1)
                      if remap_num(vol, k) is not None)
        contiguous = span == list(range(span[0], span[-1] + 1)) if span else False
        return f"{m.group(1)} {lo}-{hi}" if contiguous else f"{m.group(1)} {', '.join(str(s) for s in span)}"

    text = re.sub(r"\b(chapters) (\d+)-(\d+)\b", range_ref, text)

    def single(m):
        new = remap_num(vol, int(m.group(2)))
        return f"{m.group(1)} {new}{m.group(3) or ''}" if new is not None else m.group(0)

    text = re.sub(r"\b(chapter|ch|Chapter|Ch) (\d+)(\.\d+)?\b", single, text)
    return text.replace("Volume 1", "this book").replace("Volume 2", "this book")


def renumber_headings(text: str, old, new: int) -> str:
    lines, out, in_code = text.splitlines(), [], False
    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
        if not in_code and old is not None:
            if re.match(rf"^# {old}\.\s", line):
                line = re.sub(rf"^# {old}\.", f"# {new}.", line, count=1).replace("ADDENDUM: ", "")
            elif re.match(rf"^#{{2,3}} {old}\.(\d+)", line):
                line = re.sub(rf"^(#{{2,3}}) {old}\.", rf"\1 {new}.", line, count=1)
            elif re.match(rf"^#{{2,3}} {old}B?\.", line):
                pass
        out.append(line)
    return "\n".join(out) + "\n"


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    report = []
    for new, src, old in ORDER:
        f = source_file(src, old)
        text = f.read_text(encoding="utf-8")
        if src == "new" and not f.stem.startswith("F"):
            old_num = int(re.match(r"(\d+)", f.stem).group(1)) if re.match(r"\d", f.stem) else None
            text = rewrite_new_refs(text)
            if old_num is not None and old_num != new:
                text = re.sub(rf"^# {old_num}\.", f"# {new}.", text, count=1, flags=re.M)
                text = re.sub(rf"^(#{{2,3}}) {old_num}\.", rf"\1 {new}.", text, flags=re.M)
        if src != "new":
            text = rewrite_refs(text, src)
            first = next(l for l in text.splitlines() if l.startswith("# "))
            old_num = int(re.match(r"# (\d+)\.", first).group(1))
            text = renumber_headings(text, old_num, new)
            if src == "v1" and old == 33:          # the primer used "33." section numbers
                text = re.sub(r"^(#{2,3}) 33\.", rf"\1 {new}.", text, flags=re.M)
        if new in INJECT:
            parts = text.split("\n", 1)
            text = parts[0] + "\n\n" + INJECT[new] + "\n" + (parts[1] if len(parts) > 1 else "")
        (OUT / f"{new:03d}_{f.stem.split('_', 1)[-1]}.md").write_text(text, encoding="utf-8")
        report.append((new, src, old, f.name))
    # The LAB MANUAL (Parts 0-14, hand-holding style) goes IN FRONT of the reference chapters.
    # "0000_" sorts before "000_", so the manual comes first without renumbering anything.
    MANUAL = DOCS / "manual_src"
    manual = sorted(MANUAL.glob("M*.md")) if MANUAL.exists() else []
    for f in manual:
        (OUT / f"0000_{f.stem}.md").write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"assembled {len(manual)} lab-manual parts + {len(report)} reference chapters into {OUT}")


if __name__ == "__main__":
    main()
