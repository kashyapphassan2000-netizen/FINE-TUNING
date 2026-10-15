"""Assemble the guide: Markdown parts + real lab outputs -> HTML -> PDF, DOCX, MD, notebooks.

Run after run_labs.py:  python build_guide.py
Outputs go to transformers-101/output/
"""
import glob
import html
import json
import os
import re
import shutil
import subprocess

import markdown
from pygments.formatters import HtmlFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "output")
SITE = os.path.join(HERE, "site")
CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
NAME = "Transformers_101_Zero_to_Elite"
PARTS = ["part4_basics", "part5_intermediate", "part6_advanced", "part7_elite"]


def load_results():
    res = {}
    for p in PARTS:
        res.update(json.load(open(os.path.join(HERE, "out", p + ".json"))))
    return res


def fill(md, res, img_ext):
    def code(m):
        return "\n```python\n" + res[m.group(1)]["code"].strip("\n") + "\n```\n"

    def out(m):
        o = res[m.group(1)]["out"].rstrip("\n")
        assert "!!! ERROR" not in o, m.group(1)
        return "\n```text\n" + o + "\n```\n"

    def fig(m):
        cid = m.group(1)
        figs = res[cid]["figs"]
        assert figs, cid
        return "\n".join(f"![Output of cell {cid}](figs/{f})" for f in figs)

    md = re.sub(r"\{\{code:(\S+?)\}\}", code, md)
    md = re.sub(r"\{\{out:(\S+?)\}\}", out, md)
    md = re.sub(r"\{\{fig:(\S+?)\}\}", fig, md)
    if img_ext == "png":
        md = re.sub(r"\(diagrams/(d\d\d_\w+)\.svg\)", r"(diagrams/\1.png)", md)
    md = md.replace("\n**GOAL:**", "  \n**GOAL:**")
    left = re.findall(r"\{\{\w+:\S+?\}\}", md)
    assert not left, left
    return md


def svg_to_png():
    """Render each SVG diagram to a 2x PNG (for DOCX / Google Docs)."""
    os.makedirs(os.path.join(SITE, "diagrams"), exist_ok=True)
    for svg in sorted(glob.glob(os.path.join(ROOT, "diagrams", "*.svg"))):
        txt = open(svg).read()
        w, h = re.search(r'width="(\d+)" height="(\d+)"', txt).groups()
        page = os.path.join(SITE, "_tmp.html")
        open(page, "w").write(f'<html><body style="margin:0">{txt}</body></html>')
        png = os.path.join(SITE, "diagrams", os.path.basename(svg)[:-4] + ".png")
        subprocess.run([CHROME, "--headless", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
                        "--force-device-scale-factor=2", f"--window-size={w},{h}",
                        f"--screenshot={png}", page], capture_output=True, check=True)
        shutil.copy(svg, os.path.join(SITE, "diagrams"))
    os.remove(os.path.join(SITE, "_tmp.html"))


CSS = """
@page { size: A4; margin: 16mm 14mm 18mm 14mm;
  @bottom-center { content: "Transformers 101 \\2014  page " counter(page); font: 9px Inter, Arial, sans-serif; color: #64748b; } }
@page :first { @bottom-center { content: none; } }
* { font-feature-settings: "calt" 0, "liga" 0, "case" 0; font-variant-ligatures: none; }
body { font-family: Inter, 'Segoe UI', Arial, sans-serif; font-size: 10.6pt; line-height: 1.5; color: #0f172a; background: #fff; }
h1 { font-size: 22pt; color: #1e3a8a; border-bottom: 3px solid #1e3a8a; padding-bottom: 4px; margin-top: 0; break-before: page; }
.doctitle { font-size: 28pt; font-weight: 800; color: #1e3a8a; margin-top: 8mm; line-height: 1.15; }
.subtitle { font-size: 13pt; color: #475569; margin-bottom: 18px; }
h2 { font-size: 15pt; color: #1e40af; margin-top: 22px; break-after: avoid; }
h3 { font-size: 12.5pt; color: #0f172a; margin-top: 18px; break-after: avoid; background: #f1f5f9; padding: 4px 8px; border-left: 4px solid #2563eb; }
p { margin: 6px 0; }
code { font-family: 'Liberation Mono', Consolas, 'Courier New', monospace; font-size: 9pt; background: #f1f5f9; padding: 1px 3px; border-radius: 3px; }
pre { background: #0f172a; color: #e2e8f0; padding: 9px 11px; border-radius: 6px; font-size: 8.4pt; line-height: 1.38; white-space: pre-wrap; word-break: break-word; break-inside: auto; }
pre code { background: none; color: inherit; padding: 0; font-size: inherit; }
.codehilite { background: #0f172a; border-radius: 6px; margin: 6px 0; }
.codehilite pre { margin: 0; }
pre.out { background: #f8fafc; color: #0f172a; border: 1px solid #cbd5e1; border-left: 4px solid #059669; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 8.8pt; break-inside: auto; }
th { background: #1e3a8a; color: #fff; text-align: left; padding: 5px 6px; }
td { border: 1px solid #cbd5e1; padding: 4px 6px; vertical-align: top; }
tr { break-inside: avoid; }
table:has(th:nth-child(8)) { font-size: 7.3pt; }
table:has(th:nth-child(8)) td, table:has(th:nth-child(8)) th { padding: 3px 4px; }
tr:nth-child(even) td { background: #f8fafc; }
blockquote { margin: 8px 0; padding: 6px 10px; border-radius: 6px; border-left: 5px solid #94a3b8; background: #f8fafc; }
blockquote.see { border-left-color: #059669; background: #ecfdf5; }
blockquote.warn { border-left-color: #dc2626; background: #fef2f2; }
p.checkpoint { border: 2px solid #7c3aed; background: #f5f3ff; padding: 6px 10px; border-radius: 6px; margin-top: 10px; break-inside: avoid; }
p.where { background: #eff6ff; border-left: 5px solid #2563eb; padding: 6px 10px; border-radius: 4px; }
p.step { margin: 5px 0 3px 0; }
img { max-width: 100%; display: block; margin: 10px auto; break-inside: avoid; }
img[src^="figs/"] { max-width: 62%; max-height: 105mm; border: 1px solid #e2e8f0; border-radius: 4px; }
.assume { background: #fffbeb; border: 1px solid #f59e0b; padding: 8px 12px; border-radius: 6px; font-size: 10pt; }
.toc { font-size: 9.6pt; columns: 2; column-gap: 24px; }
.toc ul { list-style: none; padding-left: 0; margin: 0; }
.toc > ul > li { margin-top: 6px; font-weight: 700; break-inside: avoid-column; }
.toc li li { font-weight: 400; padding-left: 12px; }
.toc a { color: #0f172a; text-decoration: none; }
.toc .pg { float: right; color: #64748b; }
hr { border: none; border-top: 1px dashed #cbd5e1; margin: 18px 0; }
"""


def style_html(body):
    body = re.sub(r"<blockquote>\s*<p><strong>You should see",
                  '<blockquote class="see">\n<p><strong>You should see', body)
    body = re.sub(r"<blockquote>\s*<p><strong>Watch out",
                  '<blockquote class="warn">\n<p><strong>Watch out', body)
    body = body.replace("<p><strong>CHECKPOINT:</strong>", '<p class="checkpoint"><strong>CHECKPOINT:</strong>')
    body = body.replace("<p><strong>WHERE:</strong>", '<p class="where"><strong>WHERE:</strong>')
    body = re.sub(r'<p><strong>Step (\d+)\.', r'<p class="step"><strong>Step \1.', body)
    # output blocks (```text) get a light style so code and results look different
    body = re.sub(r'<div class="codehilite"><pre><span></span><code>(?=(?:(?!</code>).)*</code></pre></div>)',
                  lambda m: m.group(0), body, flags=re.S)
    return body


def md_to_html(md, toc_pages=None):
    conv = markdown.Markdown(extensions=["extra", "toc", "codehilite", "sane_lists"],
                             extension_configs={"codehilite": {"guess_lang": False, "css_class": "codehilite"},
                                                "toc": {"toc_depth": "1-2", "title": None}})
    # text blocks -> plain <pre class="out"> so they are not highlighted
    blocks = {}

    def stash(m):
        key = f"OUTBLOCK{len(blocks)}X"
        blocks[key] = '<pre class="out"><code>' + html.escape(m.group(1)) + "</code></pre>"
        return "\n\n" + key + "\n\n"

    md2 = re.sub(r"```text\n(.*?)\n```", stash, md, flags=re.S)
    body = conv.convert(md2)
    for k, v in blocks.items():
        body = body.replace(f"<p>{k}</p>", v).replace(k, v)
    toc = conv.toc
    if toc_pages:
        def add_pg(m):
            anchor, title = m.group(1), m.group(2)
            pg = toc_pages.get(anchor, "")
            return f'<a href="#{anchor}">{title}<span class="pg">{pg}</span></a>'
        toc = re.sub(r'<a href="#([^"]+)">(.*?)</a>', add_pg, toc)
    toc = toc.replace('<div class="toc">', '<div class="toc"><h2>Contents</h2>', 1)
    body = body.replace("<p>[TOC]</p>", toc)
    body = style_html(body)
    pyg = HtmlFormatter(style="monokai").get_style_defs(".codehilite")
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>Transformers 101 Lab Manual</title>"
            f"<style>{pyg}{CSS}</style></head><body>{body}</body></html>"), conv.toc_tokens


def print_pdf(html_path, pdf_path):
    subprocess.run([CHROME, "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path}", html_path], capture_output=True, check=True)


def toc_page_numbers(pdf_path, tokens):
    from pypdf import PdfReader
    pages = [re.sub(r"\s+", " ", (p.extract_text() or "")) for p in PdfReader(pdf_path).pages]
    flat = []
    for t in tokens:
        flat.append(t)
        flat.extend(t["children"])
    result, start = {}, 1
    for t in flat:
        name = html.unescape(re.sub(r"<[^>]+>", "", t["name"]))
        probe = re.sub(r"[^A-Za-z0-9]", "", name)[:30]
        for i in range(start, len(pages)):
            if probe in re.sub(r"[^A-Za-z0-9]", "", pages[i]):
                result[t["id"]] = i + 1
                start = i
                break
    return result


def notebooks(res):
    os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
    titles = {"part4_basics": "T101_Part4_Basics", "part5_intermediate": "T101_Part5_Intermediate",
              "part6_advanced": "T101_Part6_Advanced", "part7_elite": "T101_Part7_Elite"}
    for part, title in titles.items():
        data = json.load(open(os.path.join(HERE, "out", part + ".json")))
        cells = [{"cell_type": "markdown", "metadata": {},
                  "source": [f"# {title.replace('_', ' ')}\n",
                             "Transformers 101 lab notebook. Run every cell in order (Shift+Enter). "
                             "Use Runtime > Change runtime type > T4 GPU."]}]
        for cid, c in data.items():
            cells.append({"cell_type": "markdown", "metadata": {}, "source": [f"### Cell {cid}"]})
            src = c["code"].splitlines(keepends=True)
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": src})
        nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                           "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                           "language_info": {"name": "python"}},
              "nbformat": 4, "nbformat_minor": 5}
        json.dump(nb, open(os.path.join(ROOT, "notebooks", title + ".ipynb"), "w"), indent=1)


def main():
    res = load_results()
    shutil.rmtree(SITE, ignore_errors=True)
    os.makedirs(os.path.join(SITE, "figs"), exist_ok=True)
    for f in glob.glob(os.path.join(HERE, "out", "fig_*.png")):
        shutil.copy(f, os.path.join(SITE, "figs"))
    svg_to_png()
    raw = "\n\n".join(open(f, encoding="utf-8").read() for f in sorted(glob.glob(os.path.join(ROOT, "guide", "p*.md"))))

    # 1) HTML + PDF (SVG diagrams), two passes for TOC page numbers
    md_svg = fill(raw, res, "svg")
    page, tokens = md_to_html(md_svg)
    html_path = os.path.join(SITE, "index.html")
    open(html_path, "w", encoding="utf-8").write(page)
    pdf_tmp = os.path.join(SITE, "pass1.pdf")
    print_pdf(html_path, pdf_tmp)
    pages = toc_page_numbers(pdf_tmp, tokens)
    page, _ = md_to_html(md_svg, pages)
    open(html_path, "w", encoding="utf-8").write(page)
    os.makedirs(OUT, exist_ok=True)
    print_pdf(html_path, os.path.join(OUT, NAME + ".pdf"))

    # 2) Markdown + DOCX (PNG diagrams, for Google Docs import)
    md_png = fill(raw, res, "png").replace("[TOC]", "")
    md_png = re.sub(r'<div class="doctitle">(.*?)</div>', r"", md_png)
    md_png = re.sub(r'<div class="subtitle">(.*?)</div>', r"*\1*", md_png)
    md_png = re.sub(r'<div class="assume">(.*?)</div>', r"> \1", md_png, flags=re.S)
    md_path = os.path.join(SITE, NAME + ".md")
    open(md_path, "w", encoding="utf-8").write(md_png)
    subprocess.run(["pandoc", md_path, "-f", "markdown", "-o", os.path.join(OUT, NAME + ".docx"),
                    "--toc", "--toc-depth=2", "--resource-path", SITE,
                    "--metadata", "title=Transformers 101 - Zero to Elite Lab Manual"], check=True, cwd=SITE)
    # standalone markdown bundle with its images
    for sub in ["figs", "diagrams"]:
        dst = os.path.join(OUT, sub)
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(os.path.join(SITE, sub), dst)
    shutil.copy(md_path, os.path.join(OUT, NAME + ".md"))

    notebooks(res)
    print("TOC pages found:", len(pages), "of", sum(1 + len(t["children"]) for t in tokens))
    print("done ->", OUT)


if __name__ == "__main__":
    main()
