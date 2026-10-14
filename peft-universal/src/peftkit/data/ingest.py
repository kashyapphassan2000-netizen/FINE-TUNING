"""Stage 1: any folder of documents -> chunks.jsonl  ({id, source, text}).

Streams file by file, so 10 or 100,000 documents use the same memory.
Supported: .txt .md .jsonl .json .csv .tsv .xlsx .xlsm .html .htm .pdf .docx .pptx
Spreadsheets: every row becomes "header: value | header: value" text (one row = one record), per sheet.
Old binary .xls / .doc / .ppt are NOT read: open them in Excel/Word/PowerPoint and "Save As" the new format.
Scanned PDFs (images, no text layer) yield empty text: run OCR first (e.g. `ocrmypdf in.pdf out.pdf`).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Iterator

SUPPORTED = {".txt", ".md", ".jsonl", ".json", ".csv", ".tsv", ".xlsx", ".xlsm", ".html", ".htm", ".pdf", ".docx",
             ".pptx"}

TABLES = {".csv", ".tsv", ".xlsx", ".xlsm"}


def _row_text(row: dict) -> str:
    """One table row -> one line of text. Empty cells are skipped so they do not become noise."""
    return row.get("text") or " | ".join(f"{k}: {v}" for k, v in row.items() if k and v not in (None, ""))


def read_file(path: Path) -> Iterator[str]:
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        yield path.read_text(encoding="utf-8", errors="ignore")
    elif ext in (".html", ".htm"):
        raw = path.read_text(encoding="utf-8", errors="ignore")
        raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
        yield re.sub(r"<[^>]+>", " ", raw)
    elif ext == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    yield row.get("text") or row.get("content") or json.dumps(row, ensure_ascii=False)
    elif ext == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else [data]
        for row in rows:
            yield row.get("text", json.dumps(row, ensure_ascii=False)) if isinstance(row, dict) else str(row)
    elif ext in (".csv", ".tsv"):
        with path.open(encoding="utf-8", errors="ignore", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t" if ext == ".tsv" else ","):
                text = _row_text(row)
                if text:
                    yield text
    elif ext in (".xlsx", ".xlsm"):
        from openpyxl import load_workbook  # read_only streams rows: a 1M-row sheet does not load into RAM

        wb = load_workbook(str(path), read_only=True, data_only=True)  # data_only: cached formula VALUES
        try:
            for ws in wb.worksheets:
                rows = ws.iter_rows(values_only=True)
                header = next(rows, None)
                if not header:
                    continue
                names = [str(h).strip() if h is not None else f"col{i + 1}" for i, h in enumerate(header)]
                for values in rows:
                    if values is None or all(v is None for v in values):
                        continue
                    text = _row_text(dict(zip(names, values)))
                    if text:
                        yield f"[{ws.title}] {text}"
        finally:
            wb.close()
    elif ext == ".pptx":
        from pptx import Presentation  # python-pptx

        slides = []
        for i, slide in enumerate(Presentation(str(path)).slides, 1):
            parts = [sh.text_frame.text for sh in slide.shapes if getattr(sh, "has_text_frame", False)]
            if parts:
                slides.append(f"Slide {i}: " + "\n".join(parts))
        yield "\n\n".join(slides)
    elif ext == ".pdf":
        from pypdf import PdfReader

        yield "\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)
    elif ext == ".docx":
        import docx  # python-docx

        yield "\n".join(p.text for p in docx.Document(str(path)).paragraphs)


def clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk(text: str, size: int = 2000, overlap: int = 200) -> list[str]:
    """Character chunks that prefer to break on paragraph/sentence boundaries."""
    if size <= overlap:
        raise ValueError("size must be larger than overlap")
    out, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start:end]
            cut = max(window.rfind("\n\n"), window.rfind(". "))
            if cut > size // 2:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= n:
            break
        nxt = max(end - overlap, start + 1)
        # begin the overlap at a clean boundary (paragraph > sentence > word), never mid-word or mid-row:
        # a chunk that starts with "f the relief valve" teaches the model to write fragments
        window = text[nxt:end]
        for sep in ("\n\n", ". ", "\n", " "):
            i = window.find(sep)
            if i != -1:
                nxt += i + len(sep)
                break
        start = nxt if nxt < end else end
    return out


def ingest(src: Path, out: Path, size: int = 2000, overlap: int = 200, min_chars: int = 200) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    stats = {"files": 0, "skipped_files": 0, "chunks": 0, "dupes": 0}
    with out.open("w", encoding="utf-8") as fo:
        for path in sorted(src.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED:
                continue
            try:
                texts = list(read_file(path))
                if path.suffix.lower() in TABLES:
                    # one row is usually < min_chars and would be dropped: group rows into blocks the
                    # chunker can cut BETWEEN rows (blank line = preferred break), never inside one
                    texts = ["\n\n".join(texts)] if texts else []
            except Exception as exc:  # a bad file must not kill a 10k-file run
                print(f"[skip] {path}: {exc}")
                stats["skipped_files"] += 1
                continue
            stats["files"] += 1
            for doc_i, text in enumerate(texts):
                for ch_i, piece in enumerate(chunk(clean(text), size, overlap)):
                    if len(piece) < min_chars:
                        continue
                    h = hashlib.sha1(piece.encode()).hexdigest()
                    if h in seen:
                        stats["dupes"] += 1
                        continue
                    seen.add(h)
                    rec = {"id": h[:16], "source": f"{path.relative_to(src)}#{doc_i}.{ch_i}", "text": piece}
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    stats["chunks"] += 1
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Documents folder -> chunks.jsonl")
    ap.add_argument("--src", default="data/raw")
    ap.add_argument("--out", default="data/interim/chunks.jsonl")
    ap.add_argument("--chunk-chars", type=int, default=2000)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--min-chars", type=int, default=200)
    a = ap.parse_args(argv)
    print(json.dumps(ingest(Path(a.src), Path(a.out), a.chunk_chars, a.overlap, a.min_chars)))


if __name__ == "__main__":
    main()
