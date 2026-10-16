"""Write the rendered guide Markdown into one tab of a Google Doc with native formatting.

Used from a sandbox that has a Docs API caller:
    ops = build_ops(md_text, img_base_url)
    write_ops(ops, api, doc_id, tab_id)       # api(method, url, body=None) -> dict

Supports: headings, paragraphs with **bold** / *italic* / `code` / [links](url), bullet and
numbered lists, blockquotes (coloured callouts), fenced code blocks, pipe tables, images.
"""
import re

FONT_CODE = "Roboto Mono"
COLORS = {
    "code_bg": "#f1f5f9", "out_bg": "#ecfdf5", "see": "#ecfdf5", "warn": "#fef2f2",
    "quote": "#f8fafc", "check": "#f5f3ff", "where": "#eff6ff", "th_bg": "#1e3a8a",
    "link": "#1d4ed8", "h1": "#1e3a8a", "h2": "#1e40af",
}


def rgb(hexstr):
    h = hexstr.lstrip("#")
    return {"color": {"rgbColor": {"red": int(h[0:2], 16) / 255, "green": int(h[2:4], 16) / 255,
                                   "blue": int(h[4:6], 16) / 255}}}


def u16(s):
    return len(s.encode("utf-16-le")) // 2


# ----------------------------------------------------------------------------- inline parsing
INLINE = re.compile(r"(`[^`]+`|\*\*.+?\*\*|\[[^\]]+\]\([^)]+\)|(?<![\w*])\*(?!\s)[^*\n]+?(?<!\s)\*(?![\w*]))")


def parse_inline(text):
    """Return (plain_text, [(start, end, style_dict)]) with offsets in UTF-16 units."""
    out, styles, pos = [], [], 0

    def add(s, style=None):
        nonlocal pos
        if not s:
            return
        n = u16(s)
        if style:
            styles.append((pos, pos + n, style))
        out.append(s)
        pos += n

    i = 0
    for m in INLINE.finditer(text):
        add(text[i:m.start()])
        tok = m.group(0)
        if tok.startswith("`"):
            add(tok[1:-1], {"code": True})
        elif tok.startswith("**"):
            inner_txt, inner_styles = parse_inline(tok[2:-2])
            base = pos
            add(inner_txt, {"bold": True})
            for s, e, st in inner_styles:
                styles.append((base + s, base + e, st))
        elif tok.startswith("["):
            mm = re.match(r"\[([^\]]+)\]\(([^)]+)\)", tok)
            add(mm.group(1), {"link": mm.group(2)})
        else:
            add(tok[1:-1], {"italic": True})
        i = m.end()
    add(text[i:])
    return "".join(out), styles


# ----------------------------------------------------------------------------- block parsing
def parse_blocks(md):
    lines = md.split("\n")
    blocks, i = [], 0
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
            continue
        m = re.match(r"^```(\w*)\s*$", ln)
        if m:
            lang, buf = m.group(1) or "text", []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            blocks.append(("code", lang, "\n".join(buf)))
            continue
        m = re.match(r"^(#{1,3}) (.*)$", ln)
        if m:
            blocks.append(("h", len(m.group(1)), m.group(2).strip()))
            i += 1
            continue
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", ln)
        if m:
            blocks.append(("img", m.group(1), m.group(2)))
            i += 1
            continue
        if re.match(r"^-{3,}\s*$", ln):
            blocks.append(("hr",))
            i += 1
            continue
        if ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.match(r"^:?-{2,}:?$", c) for c in cells):
                    rows.append(cells)
                i += 1
            width = max(len(r) for r in rows)
            rows = [r + [""] * (width - len(r)) for r in rows]
            blocks.append(("table", rows))
            continue
        if ln.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].startswith(">"):
                buf.append(re.sub(r"^>\s?", "", lines[i]))
                i += 1
            paras = [p for p in "\n".join(buf).split("\n\n") if p.strip()]
            for p in paras:
                blocks.append(("quote", p.replace("\n", " ").strip()))
            continue
        m = re.match(r"^(\s*)(- |\d+\. )(.*)$", ln)
        if m:
            ordered = m.group(2)[0].isdigit()
            items = []
            while i < len(lines):
                mm = re.match(r"^(\s*)(- |\d+\. )(.*)$", lines[i])
                if mm and (mm.group(2)[0].isdigit()) == ordered:
                    items.append(mm.group(3))
                    i += 1
                elif lines[i].startswith("   ") and lines[i].strip() and items:
                    items[-1] += "\u000b" + lines[i].strip()
                    i += 1
                else:
                    break
            blocks.append(("list", ordered, items))
            continue
        buf = [ln]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"^(```|#{1,3} |!\[|\||>|\s*- |\s*\d+\. |-{3,})", lines[i]):
            buf.append(lines[i])
            i += 1
        text = ""
        for k, b in enumerate(buf):
            if k == 0:
                text = b.rstrip()
            elif buf[k - 1].endswith("  "):
                text += "\u000b" + b.strip()
            else:
                text += " " + b.strip()
        blocks.append(("p", text))
    return blocks


# ----------------------------------------------------------------------------- ops
def build_ops(md, img_base, title):
    """Turn markdown into a list of ('text', paragraphs) / ('table', rows) / ('img', url) / ('pagebreak',) ops.

    A 'text' op holds paragraphs: (plain, inline_styles, para_kind, extra)."""
    md = re.sub(r"</?b>", "**", md)
    ops, pend = [], []

    def flush():
        if pend:
            ops.append(("text", list(pend)))
            pend.clear()

    pend.append((title, [], "title", None))
    first_h1 = True
    for b in parse_blocks(md):
        kind = b[0]
        if kind == "h":
            txt, st = parse_inline(b[2])
            if b[1] == 1 and not first_h1:
                flush()
                ops.append(("pagebreak",))
            if b[1] == 1:
                first_h1 = False
            pend.append((txt, st, f"h{b[1]}", None))
        elif kind == "p":
            txt, st = parse_inline(b[1])
            pk = "p"
            if b[1].startswith("**CHECKPOINT:**"):
                pk = "check"
            elif b[1].startswith("**WHERE:**"):
                pk = "where"
            pend.append((txt, st, pk, None))
        elif kind == "quote":
            txt, st = parse_inline(b[1])
            pk = "quote"
            if b[1].startswith("**You should see"):
                pk = "see"
            elif b[1].startswith("**Watch out"):
                pk = "warn"
            pend.append((txt, st, pk, None))
        elif kind == "list":
            for k, it in enumerate(b[2]):
                txt, st = parse_inline(it)
                pend.append((txt, st, "ol" if b[1] else "ul", k))
        elif kind == "code":
            lang, code = b[1], b[2]
            pk = "out" if lang == "text" else "code"
            code_lines = code.split("\n") or [""]
            for k, cl in enumerate(code_lines):
                pos = "only" if len(code_lines) == 1 else ("first" if k == 0 else ("last" if k == len(code_lines) - 1 else "mid"))
                pend.append((cl if cl else " ", [], pk, pos))
        elif kind == "hr":
            pend.append(("", [], "p", None))
        elif kind == "table":
            flush()
            ops.append(("table", [[parse_inline(c) for c in row] for row in b[1]]))
        elif kind == "img":
            flush()
            src = b[2]
            url = src if src.startswith("http") else img_base.rstrip("/") + "/" + src
            ops.append(("img", url, "fig" if "figs/" in src else "diagram"))
    flush()
    return ops


def text_style_req(start, end, style, tab):
    ts, fields = {}, []
    if style.get("bold"):
        ts["bold"] = True
        fields.append("bold")
    if style.get("italic"):
        ts["italic"] = True
        fields.append("italic")
    if style.get("code"):
        ts["weightedFontFamily"] = {"fontFamily": FONT_CODE}
        ts["backgroundColor"] = rgb("#eef2f7")
        fields += ["weightedFontFamily", "backgroundColor"]
    if style.get("link"):
        ts["link"] = {"url": style["link"]}
        ts["foregroundColor"] = rgb(COLORS["link"])
        ts["underline"] = True
        fields += ["link", "foregroundColor", "underline"]
    return {"updateTextStyle": {"range": {"startIndex": start, "endIndex": end, "tabId": tab},
                                "textStyle": ts, "fields": ",".join(fields)}}


PARA_STYLE = {
    "title": ({"namedStyleType": "TITLE"}, None),
    "h1": ({"namedStyleType": "HEADING_1"}, None),
    "h2": ({"namedStyleType": "HEADING_2"}, None),
    "h3": ({"namedStyleType": "HEADING_3"}, None),
    "p": ({"namedStyleType": "NORMAL_TEXT"}, None),
    "ul": ({"namedStyleType": "NORMAL_TEXT"}, None),
    "ol": ({"namedStyleType": "NORMAL_TEXT"}, None),
    "see": ({"namedStyleType": "NORMAL_TEXT"}, "see"),
    "warn": ({"namedStyleType": "NORMAL_TEXT"}, "warn"),
    "quote": ({"namedStyleType": "NORMAL_TEXT"}, "quote"),
    "check": ({"namedStyleType": "NORMAL_TEXT"}, "check"),
    "where": ({"namedStyleType": "NORMAL_TEXT"}, "where"),
    "code": ({"namedStyleType": "NORMAL_TEXT"}, "code_bg"),
    "out": ({"namedStyleType": "NORMAL_TEXT"}, "out_bg"),
}
BORDER_COL = {"see": "#059669", "warn": "#dc2626", "quote": "#94a3b8", "check": "#7c3aed",
              "where": "#2563eb", "code_bg": "#334155", "out_bg": "#059669"}


def para_style_req(start, end, pk, pos, tab):
    named, shade = PARA_STYLE[pk]
    ps = dict(named)
    fields = ["namedStyleType", "shading", "borderLeft", "indentStart", "indentFirstLine",
              "spaceAbove", "spaceBelow", "lineSpacing", "alignment"]
    ps["alignment"] = "START"
    if shade:
        ps["shading"] = {"backgroundColor": rgb(COLORS[shade])}
        ps["borderLeft"] = {"color": rgb(BORDER_COL[shade]), "width": {"magnitude": 3, "unit": "PT"},
                            "padding": {"magnitude": 6, "unit": "PT"}, "dashStyle": "SOLID"}
        ps["indentStart"] = {"magnitude": 9, "unit": "PT"}
        ps["indentFirstLine"] = {"magnitude": 9, "unit": "PT"}
    if pk in ("code", "out"):
        ps["lineSpacing"] = 100
        ps["spaceAbove"] = {"magnitude": 6 if pos in ("first", "only") else 0, "unit": "PT"}
        ps["spaceBelow"] = {"magnitude": 6 if pos in ("last", "only") else 0, "unit": "PT"}
    elif pk in ("see", "warn", "quote", "check", "where"):
        ps["spaceAbove"] = {"magnitude": 6, "unit": "PT"}
        ps["spaceBelow"] = {"magnitude": 6, "unit": "PT"}
    return {"updateParagraphStyle": {"range": {"startIndex": start, "endIndex": end, "tabId": tab},
                                     "paragraphStyle": ps, "fields": ",".join(fields)}}


def text_requests(paras, idx, tab):
    """Requests that insert `paras` at idx (start of the final empty paragraph). Returns (requests, new_idx)."""
    full = "".join(p[0] + "\n" for p in paras)
    reqs = [{"insertText": {"location": {"index": idx, "tabId": tab}, "text": full}}]
    total = u16(full)
    reqs.append({"updateTextStyle": {"range": {"startIndex": idx, "endIndex": idx + total, "tabId": tab},
                                     "textStyle": {}, "fields": "bold,italic,underline,weightedFontFamily,fontSize,backgroundColor,foregroundColor,link"}})
    pos, list_run = idx, None
    for plain, styles, pk, extra in paras:
        n = u16(plain)
        reqs.append(para_style_req(pos, pos + n + 1, pk, extra, tab))
        for s, e, st in styles:
            reqs.append(text_style_req(pos + s, pos + e, st, tab))
        if pk in ("code", "out") and n:
            reqs.append({"updateTextStyle": {"range": {"startIndex": pos, "endIndex": pos + n, "tabId": tab},
                                             "textStyle": {"weightedFontFamily": {"fontFamily": FONT_CODE},
                                                           "fontSize": {"magnitude": 8.5, "unit": "PT"}},
                                             "fields": "weightedFontFamily,fontSize"}})
        if pk in ("h1", "h2") and n:
            reqs.append({"updateTextStyle": {"range": {"startIndex": pos, "endIndex": pos + n, "tabId": tab},
                                             "textStyle": {"foregroundColor": rgb(COLORS[pk])},
                                             "fields": "foregroundColor"}})
        if pk in ("ul", "ol"):
            if list_run and list_run[0] == pk and extra != 0:
                list_run[2] = pos + n + 1
            else:
                if list_run:
                    reqs.append(bullets_req(list_run, tab))
                list_run = [pk, pos, pos + n + 1]
        elif list_run:
            reqs.append(bullets_req(list_run, tab))
            list_run = None
        pos += n + 1
    if list_run:
        reqs.append(bullets_req(list_run, tab))
    return reqs, idx + total


def bullets_req(run, tab):
    preset = "NUMBERED_DECIMAL_ALPHA_ROMAN" if run[0] == "ol" else "BULLET_DISC_CIRCLE_SQUARE"
    return {"createParagraphBullets": {"range": {"startIndex": run[1], "endIndex": run[2], "tabId": tab},
                                       "bulletPreset": preset}}


# ----------------------------------------------------------------------------- execution
TF = "table(rows,columns,tableRows(tableCells(content(startIndex))))"


def body_of(api, doc_id, tab_id):
    d = api("GET", f"https://docs.googleapis.com/v1/documents/{doc_id}",
            params={"includeTabsContent": "true",
                    "fields": "tabs(tabProperties(tabId),documentTab(body(content(startIndex,endIndex,%s))),"
                              "childTabs(tabProperties(tabId),documentTab(body(content(startIndex,endIndex,%s)))))" % (TF, TF)})

    def find(tabs):
        for t in tabs:
            if t["tabProperties"]["tabId"] == tab_id:
                return t
            f = find(t.get("childTabs", []))
            if f:
                return f
    return find(d["tabs"])["documentTab"]["body"]["content"]


def write_ops(ops, api, doc_id, tab_id, start_op=0, max_seconds=150, log=print, batch_limit=300):
    """Apply ops[start_op:] until the time budget is used. Returns the next op index (len(ops) when done)."""
    import time
    t0 = time.time()
    url = f"https://docs.googleapis.com/v1/documents/{doc_id}:batchUpdate"
    content = body_of(api, doc_id, tab_id)
    idx = content[-1]["endIndex"] - 1
    queue = []
    last_write = [0.0]

    def post(body):
        wait = 1.05 - (time.time() - last_write[0])
        if wait > 0:
            time.sleep(wait)
        api("POST", url, body=body)
        last_write[0] = time.time()

    def send(force=False):
        nonlocal queue
        while queue and (force or len(queue) >= batch_limit):
            chunk, queue = queue[:batch_limit], queue[batch_limit:]
            post({"requests": chunk})

    n_op = start_op
    while n_op < len(ops):
        if time.time() - t0 > max_seconds:
            break
        op = ops[n_op]
        if op[0] == "text":
            reqs, idx = text_requests(op[1], idx, tab_id)
            queue += reqs
            send()
        elif op[0] == "pagebreak":
            queue.append({"insertPageBreak": {"location": {"index": idx, "tabId": tab_id}}})
            idx += 2
        elif op[0] == "img":
            width = 468 if op[2] == "diagram" else 330
            queue.append({"insertInlineImage": {"location": {"index": idx, "tabId": tab_id}, "uri": op[1],
                                                "objectSize": {"width": {"magnitude": width, "unit": "PT"}}}})
            queue.append({"insertText": {"location": {"index": idx + 1, "tabId": tab_id}, "text": "\n"}})
            queue.append({"updateParagraphStyle": {"range": {"startIndex": idx, "endIndex": idx + 2, "tabId": tab_id},
                                                   "paragraphStyle": {"namedStyleType": "NORMAL_TEXT", "alignment": "CENTER"},
                                                   "fields": "namedStyleType,alignment,shading,borderLeft,indentStart,indentFirstLine"}})
            idx += 2
        elif op[0] == "table":
            rows = op[1]
            R, Cc = len(rows), len(rows[0])
            queue.append({"insertTable": {"rows": R, "columns": Cc, "location": {"index": idx, "tabId": tab_id}}})
            send(force=True)
            content = body_of(api, doc_id, tab_id)
            tbl = [e for e in content if "table" in e and e["startIndex"] >= idx][0]
            S = tbl["startIndex"]
            trows = tbl["table"]["tableRows"]
            cell_reqs = []
            for i in reversed(range(R)):
                for j in reversed(range(Cc)):
                    p = trows[i]["tableCells"][j]["content"][0]["startIndex"]
                    txt, styles = rows[i][j]
                    if not txt:
                        continue
                    n = u16(txt)
                    cell_reqs.append({"insertText": {"location": {"index": p, "tabId": tab_id}, "text": txt}})
                    cell_reqs.append({"updateTextStyle": {"range": {"startIndex": p, "endIndex": p + n, "tabId": tab_id},
                                                          "textStyle": {"fontSize": {"magnitude": 9, "unit": "PT"},
                                                                        "bold": i == 0,
                                                                        "foregroundColor": rgb("#ffffff" if i == 0 else "#0f172a")},
                                                          "fields": "fontSize,bold,foregroundColor,weightedFontFamily,backgroundColor,italic,link,underline"}})
                    for s_, e_, st in styles:
                        cell_reqs.append(text_style_req(p + s_, p + e_, st, tab_id))
            cell_reqs.append({"updateTableCellStyle": {
                "tableRange": {"tableCellLocation": {"tableStartLocation": {"index": S, "tabId": tab_id}, "rowIndex": 0, "columnIndex": 0},
                               "rowSpan": 1, "columnSpan": Cc},
                "tableCellStyle": {"backgroundColor": rgb(COLORS["th_bg"])}, "fields": "backgroundColor"}})
            queue += cell_reqs
            send(force=True)
            content = body_of(api, doc_id, tab_id)
            tbl = [e for e in content if "table" in e and e["startIndex"] == S][0]
            idx = tbl["endIndex"]
        n_op += 1
        if n_op % 20 == 0:
            log(f"op {n_op}/{len(ops)} idx={idx}")
    send(force=True)
    return n_op
