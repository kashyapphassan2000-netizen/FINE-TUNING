"""Builds the guide PDF from docs/guide/*.md (a small markup) with ReportLab.
Markup: '# ' chapter (new page, TOC+bookmark), '## ' section (TOC+bookmark), '### ' subsection,
```label fenced code (label: laptop|gpu|browser|out|py|yaml|text|math), ':::kind Title' ... ':::' boxes
(why|tip|warn|check|honest|math), '@file path' embeds a repo file verbatim, '@lab name' embeds lab output,
'|a|b|' tables, '- ' bullets, '1. ' numbered, blank line = paragraph break, **bold**, `code`."""
from __future__ import annotations

import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, NextPageTemplate, PageBreak, PageTemplate, Paragraph,
                                Preformatted, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]

from reportlab.lib.fonts import addMapping  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402

_FD = "/usr/share/fonts/truetype/dejavu/"
for _n, _f in [("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"),
               ("DV-BI", "DejaVuSans-BoldOblique.ttf"), ("DVM", "DejaVuSansMono.ttf"),
               ("DVS-I", "DejaVuSerif-Italic.ttf")]:
    pdfmetrics.registerFont(TTFont(_n, _FD + _f))
addMapping("DV", 0, 0, "DV")
addMapping("DV", 1, 0, "DV-B")
addMapping("DV", 0, 1, "DV-I")
addMapping("DV", 1, 1, "DV-BI")
W, H = A4
M = 16 * mm
ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["BodyText"], fontName="DV", fontSize=9.0, leading=13.0, spaceAfter=4)
H1 = ParagraphStyle("h1", parent=BODY, fontName="DV-B", fontSize=19, leading=24, spaceAfter=10,
                    textColor=colors.HexColor("#0B3D5C"))
H2 = ParagraphStyle("h2", parent=BODY, fontName="DV-B", fontSize=13.5, leading=18, spaceBefore=10,
                    spaceAfter=5, textColor=colors.HexColor("#12587F"))
H3 = ParagraphStyle("h3", parent=BODY, fontName="DV-B", fontSize=11, leading=15, spaceBefore=6, spaceAfter=3)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=12, bulletIndent=3, spaceAfter=2)
CODE = ParagraphStyle("code", fontName="DVM", fontSize=6.9, leading=8.9, textColor=colors.HexColor("#111111"))
MATH = ParagraphStyle("math", parent=BODY, fontName="DV-I", fontSize=11.5, leading=16, alignment=TA_CENTER,
                      spaceBefore=3, spaceAfter=5)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.2, leading=10.4, spaceAfter=0)
CELLH = ParagraphStyle("cellh", parent=CELL, fontName="DV-B", textColor=colors.white)
TOC1 = ParagraphStyle("toc1", fontName="DV-B", fontSize=10, leading=14, leftIndent=0)
TOC2 = ParagraphStyle("toc2", fontName="DV", fontSize=8.6, leading=11, leftIndent=14)

LABELS = {
    "laptop": ("LAPTOP · PowerShell", "#1F6FB2"), "gpu": ("GPU VM · bash", "#2E7D32"),
    "browser": ("BROWSER", "#7B1FA2"), "out": ("EXPECTED OUTPUT", "#555555"),
    "py": ("PYTHON", "#37474F"), "yaml": ("YAML", "#37474F"), "text": ("TEXT", "#37474F"),
    "json": ("JSON", "#37474F"), "any": ("LAPTOP or GPU VM · bash", "#00695C"), "wsl": ("LAPTOP · WSL bash", "#1F6FB2"),
}
BOXES = {
    "why": ("WHY / THEORY", "#E3F2FD", "#1565C0"), "tip": ("PRO TIP", "#E8F5E9", "#2E7D32"),
    "warn": ("HONEST WARNING", "#FFF3E0", "#E65100"), "check": ("CHECKPOINT", "#F3E5F5", "#6A1B9A"),
    "honest": ("VERIFICATION STATUS", "#FFFDE7", "#F9A825"), "math": ("MATH", "#ECEFF1", "#37474F"),
}


def inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`", r'<font face="DVM" color="#8B1A1A">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\^\{([^}]+)\}", r"<super>\1</super>", text)
    text = re.sub(r"_\{([^}]+)\}", r"<sub>\1</sub>", text)
    return text


def code_block(text: str, label: str, title: str | None = None) -> Table:
    name, color = LABELS.get(label, (label.upper(), "#37474F"))
    head = Paragraph(f'<font color="white"><b>{inline(title or name)}</b></font>', CELL)
    lines = text.rstrip("\n").split("\n")
    chunks = ["\n".join(lines[i:i + 12]) for i in range(0, len(lines), 12)] or [""]
    rows = [[head]] + [[Preformatted(c, CODE, maxLineLength=114, newLineChars="  \u21b3 ")] for c in chunks]
    t = Table(rows, colWidths=[W - 2 * M], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(color)),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F6F8FA")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor(color)),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, 0), 2.5), ("BOTTOMPADDING", (0, 0), (-1, 0), 3),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 3), ("TOPPADDING", (0, 1), (-1, 1), 3),
    ]))
    return t


def box(kind: str, title: str, lines: list[str]) -> Table:
    name, bg, fg = BOXES[kind]
    flow = [Paragraph(f'<font color="{fg}"><b>{name}{": " + inline(title) if title else ""}</b></font>', BODY)]
    for ln in lines:
        s = ln.rstrip()
        if not s:
            continue
        if kind == "check" and s.startswith("- "):
            flow.append(Paragraph("&#9744;&nbsp;" + inline(s[2:]), BUL))
        elif s.startswith("- "):
            flow.append(Paragraph(inline(s[2:]), BUL, bulletText="•"))
        elif s.startswith("$$"):
            flow.append(Paragraph(inline(s.strip("$ ")), MATH))
        else:
            flow.append(Paragraph(inline(s), BODY))
    t = Table([[flow]], colWidths=[W - 2 * M])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
                           ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(fg)),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 5),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return t


def table(rows: list[list[str]]) -> Table:
    n = len(rows[0])
    data = [[Paragraph(inline(c), CELLH if i == 0 else CELL) for c in r + [""] * (n - len(r))] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[(W - 2 * M) / n] * n, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12587F")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#B0BEC5")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


class Heading(Paragraph):
    def __init__(self, text, style, level, key):
        super().__init__(text, style)
        self.level, self.key, self.plain = level, key, re.sub("<[^>]+>", "", text)


class Doc(BaseDocTemplate):
    def __init__(self, path, **kw):
        super().__init__(path, pagesize=A4, leftMargin=M, rightMargin=M, topMargin=M, bottomMargin=M, **kw)
        frame = Frame(M, M, W - 2 * M, H - 2 * M, id="f")
        self.addPageTemplates([PageTemplate("cover", [frame], onPage=lambda c, d: None),
                               PageTemplate("main", [frame], onPage=self.decorate)])

    def decorate(self, c, doc):
        c.saveState()
        c.setFont("DV", 7)
        c.setFillColor(colors.HexColor("#78909C"))
        c.drawString(M, 9 * mm, COVER["footer"])
        c.drawRightString(W - M, 9 * mm, f"Page {doc.page}")
        c.restoreState()

    def afterFlowable(self, f):
        if isinstance(f, Heading):
            self.canv.bookmarkPage(f.key)
            self.canv.addOutlineEntry(f.plain, f.key, level=f.level, closed=f.level > 0)
            self.notify("TOCEntry", (f.level, f.plain, self.page, f.key))


def parse(md: str, story: list, counter: list[int]) -> None:
    lines = md.splitlines()
    i, para = 0, []

    def flush():
        if para:
            story.append(Paragraph(inline(" ".join(para)), BODY))
            para.clear()

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s.startswith("```"):
            flush()
            spec = s[3:].strip().split(" ", 1)
            label, title = spec[0] or "text", (spec[1] if len(spec) > 1 else None)
            buf = []
            i += 1
            while not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            story += [code_block("\n".join(buf), label, title), Spacer(1, 4)]
        elif s.startswith(":::") and len(s) > 3:
            flush()
            kind, _, title = s[3:].partition(" ")
            buf = []
            i += 1
            while lines[i].strip() != ":::":
                buf.append(lines[i])
                i += 1
            story += [box(kind, title, buf), Spacer(1, 5)]
        elif s.startswith("# "):
            flush()
            counter[0] += 1
            story += [PageBreak(), Heading(inline(s[2:]), H1, 0, f"k{counter[0]}")]
        elif s.startswith("## "):
            flush()
            counter[0] += 1
            story.append(Heading(inline(s[3:]), H2, 1, f"k{counter[0]}"))
        elif s.startswith("### "):
            flush()
            story.append(Paragraph(inline(s[4:]), H3))
        elif s.startswith("@file "):
            flush()
            rel = s[6:].strip()
            story += [code_block((ROOT / rel).read_text(encoding="utf-8"), "py", f"FILE  {rel}"), Spacer(1, 5)]
        elif s.startswith("@lab "):
            flush()
            name = s[5:].strip()
            txt = (ROOT / "labs/outputs" / f"{name}.txt").read_text(encoding="utf-8").strip("\n")
            story += [code_block(txt, "out", f"REAL OUTPUT  labs/{name}.py  (executed on CPU)"), Spacer(1, 5)]
        elif s.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            story += [table(rows), Spacer(1, 6)]
            continue
        elif s.startswith("$$"):
            flush()
            story.append(Paragraph(inline(s.strip("$ ")), MATH))
        elif s.startswith("- "):
            flush()
            story.append(Paragraph(inline(s[2:]), BUL, bulletText="•"))
        elif re.match(r"\d+\. ", s):
            flush()
            num, _, rest = s.partition(". ")
            story.append(Paragraph(inline(rest), BUL, bulletText=f"{num}."))
        elif not s:
            flush()
        else:
            para.append(s)
        i += 1
    flush()


COVER = {"title": "Universal PEFT Build Guide",
         "lines": ["LoRA \u00b7 QLoRA \u00b7 DoRA \u00b7 AdaLoRA \u00b7 LoRA+",
                   "Data preparation \u00b7 Synthetic data \u00b7 LLM-as-judge \u00b7 Preference data",
                   "Hardware sizing \u00b7 Hyperstack GPUs \u00b7 Merging \u00b7 Hugging Face \u00b7 vLLM \u00b7 Autoscaling",
                   "PROJECT build guide + concept guide. One template for any open causal LM.",
                   "Library versions checked on PyPI: 17 September 2026. Copy code from the ZIP, not from this PDF."],
         "footer": "Universal PEFT Build Guide \u2014 LoRA \u00b7 QLoRA \u00b7 DoRA \u00b7 AdaLoRA \u00b7 LoRA+ \u00b7 Data \u00b7 vLLM",
         "dir": "docs/guide"}


def cover(story):
    title = ParagraphStyle("t", parent=H1, fontSize=30, leading=36, alignment=TA_CENTER)
    sub = ParagraphStyle("s", parent=BODY, fontSize=13, leading=18, alignment=TA_CENTER, textColor=colors.HexColor("#37474F"))
    story += [Spacer(1, 50 * mm), Paragraph(COVER["title"], title), Spacer(1, 8 * mm)]
    for k, line in enumerate(COVER["lines"]):
        gap = 8 * mm if k == len(COVER["lines"]) - 3 else 3 * mm
        story += [Paragraph(line, sub), Spacer(1, gap)]


def build(out: Path) -> None:
    story: list = []
    cover(story)
    story.append(NextPageTemplate("main"))
    story.append(PageBreak())
    story.append(Paragraph("Contents", H1))
    toc = TableOfContents()
    toc.levelStyles = [TOC1, TOC2]
    story.append(toc)
    counter = [0]
    for md in sorted((ROOT / COVER["dir"]).glob("*.md")):
        parse(md.read_text(encoding="utf-8"), story, counter)
    doc = Doc(str(out), title=COVER["title"], author="peftkit")
    doc.multiBuild(story)


VOL2 = {"title": "Fine-Tuning From Zero, Volume 2",
        "lines": ["The missing fundamentals, evaluation and debugging book",
                  "Training mechanics \u00b7 Transformer internals \u00b7 Tokenization \u00b7 Loss masking",
                  "Dataset engineering \u00b7 Hyperparameters \u00b7 Experiments \u00b7 Evaluation \u00b7 Forgetting",
                  "Model compatibility \u00b7 Debugging labs \u00b7 RAG vs fine-tuning \u00b7 CPT \u00b7 DPO \u00b7 Capstone",
                  "Companion to the Universal PEFT Build Guide. Every number here was produced by running the code.",
                  "Library versions checked on PyPI: 17 September 2026. Copy code from the ZIP, not from this PDF."],
        "footer": "Fine-Tuning From Zero, Volume 2 \u2014 fundamentals \u00b7 evaluation \u00b7 debugging \u00b7 capstone",
        "dir": "docs/guide2"}

VOL3 = {"title": "KV Cache, Attention Variants and Positional Encodings",
        "lines": ["From the naive decoding problem to trillion-parameter serving",
                  "KV cache \u00b7 KV memory math \u00b7 FlashAttention \u00b7 PyTorch SDPA",
                  "MHA \u00b7 MQA \u00b7 GQA \u00b7 MLA \u00b7 PagedAttention and vLLM",
                  "Positional encodings \u00b7 RoPE and long-context scaling",
                  "A CONCEPT guide. Beginner primer first, then maths, runnable labs and production practice.",
                  "Every number produced by running the code on 18 September 2026. Copy code from the ZIP."],
        "footer": "KV Cache \u00b7 Attention Variants \u00b7 Positional Encodings \u2014 concept guide with runnable labs",
        "dir": "docs/guide3"}

VOL4 = {"title": "Enterprise Synthetic Data Engine",
        "lines": ["Generate data that mimics reality, prove it, and keep proving it",
                  "Contracts \u00b7 Profiling \u00b7 Copula / simulator / LLM generation \u00b7 Calibrated noise",
                  "Fidelity \u00b7 Privacy and memorization \u00b7 Multi-judge \u00b7 Utility on REAL holdout \u00b7 Gates",
                  "Rare events \u00b7 Model collapse \u00b7 Forecasting \u00b7 Real-time drift \u00b7 Any domain",
                  "A working toolkit (synthkit) plus the method. Assumes you know nothing.",
                  "Every number produced by running the code on 18 September 2026. Copy code from the ZIP."],
        "footer": "Enterprise Synthetic Data Engine \u2014 contracts \u00b7 generation \u00b7 validation \u00b7 privacy \u00b7 utility \u00b7 monitoring",
        "dir": "docs/guide4"}

VOL5 = {"title": "Supervised Fine-Tuning",
        "lines": ["Stage 1 of every post-training pipeline, done properly",
                  "Instruction tuning \u00b7 FLAN \u00b7 Alpaca \u00b7 OpenHermes-style collections",
                  "Chat and multi-turn fine-tuning \u00b7 Chain-of-thought fine-tuning",
                  "Domain-specific practice \u00b7 data size \u00b7 replay \u00b7 forgetting",
                  "A CONCEPT guide with five runnable labs. Every number came from running the code.",
                  "Checked 19 September 2026. Copy code from the ZIP, not from this PDF."],
        "footer": "Supervised Fine-Tuning \u2014 instruction tuning \u00b7 chat \u00b7 chain-of-thought \u00b7 domain practice",
        "dir": "docs/guide5"}

MASTER = {"title": "Universal Fine-Tuning: The Complete Guide",
          "lines": ["LAB MANUAL (Parts 0-14): zero to elite, one action per step",
                    "REFERENCE (chapters 0-135): why it works, for any model and any architecture",
                    "VS Code \u00b7 virtual environment \u00b7 fundamentals \u00b7 LoRA / QLoRA / DoRA / AdaLoRA / LoRA+",
                    "Sizing any model \u00b7 data \u00b7 training \u00b7 EVALUATION \u00b7 merge \u00b7 vLLM",
                    "Served-model evaluation \u00b7 load testing \u00b7 release gate \u00b7 gateway \u00b7 autoscaling",
                    "Merged edition of the Universal PEFT Build Guide and Fine-Tuning From Zero, plus the missing stages.",
                    "Re-verified 28 September 2026: versions, prices, repositories; DeepSpeed, accelerate and nginx configs parsed by their own tools. Copy code from the ZIP."],
          "footer": "Universal Fine-Tuning: The Complete Guide \u2014 laptop to autoscaled API",
          "dir": "docs/master"}

if __name__ == "__main__":
    if "--master" in sys.argv:
        sys.argv.remove("--master")
        COVER.update(MASTER)
    if "--vol5" in sys.argv:
        sys.argv.remove("--vol5")
        COVER.update(VOL5)
    if "--vol4" in sys.argv:
        sys.argv.remove("--vol4")
        COVER.update(VOL4)
    if "--vol3" in sys.argv:
        sys.argv.remove("--vol3")
        COVER.update(VOL3)
    if "--vol2" in sys.argv:
        sys.argv.remove("--vol2")
        COVER.update(VOL2)
    build(Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "Universal_PEFT_Build_Guide.pdf"))
