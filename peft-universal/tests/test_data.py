import json
import socket
import threading
import time

import pytest
from peftkit.data import format as fmt
from peftkit.data.ingest import chunk, ingest
from reportlab.pdfgen import canvas

TEXT = ("Calibration requires the pump pressure to stay below six bar. " * 30).strip()


def test_chunk_overlap_and_bounds():
    parts = chunk("A. " * 3000, size=500, overlap=50)
    assert all(len(p) <= 500 for p in parts) and len(parts) > 5
    with pytest.raises(ValueError):
        chunk("x", size=10, overlap=10)


def test_ingest_all_formats(tmp_path):
    src = tmp_path / "raw"
    (src / "sub").mkdir(parents=True)
    (src / "a.txt").write_text(TEXT)
    (src / "dup.md").write_text(TEXT)  # duplicate content -> deduped
    (src / "c.csv").write_text("id,text\n1," + TEXT.replace("pump", "valve") + "\n")  # distinct text
    (src / "d.html").write_text(f"<html><script>bad()</script><p>{TEXT} html</p></html>")
    (src / "sub" / "e.jsonl").write_text(json.dumps({"text": TEXT + " jsonl"}) + "\n")
    c = canvas.Canvas(str(src / "f.pdf"))
    c.drawString(72, 720, "PDF manual: the controller resets after holding the red button for five seconds." * 1)
    c.drawString(72, 700, "Warranty requires the serial number, purchase date and the original invoice copy.")
    c.drawString(72, 680, "Service intervals are every six months for industrial pumps under heavy load.")
    c.save()
    import docx

    d = docx.Document()
    d.add_paragraph(TEXT + " docx")
    d.save(str(src / "g.docx"))
    (src / "broken.pdf").write_text("not a pdf")
    out = tmp_path / "chunks.jsonl"
    stats = ingest(src, out, size=800, overlap=80, min_chars=100)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    sources = {r["source"].split("#")[0] for r in rows}
    assert {"a.txt", "c.csv", "d.html", "sub/e.jsonl", "f.pdf", "g.docx"} <= sources
    assert stats["dupes"] >= 1 and stats["skipped_files"] == 1
    assert all("bad()" not in r["text"] for r in rows)


def test_format_all_shapes(tmp_path, tiny_model):
    rows = [
        {"messages": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]},
        {"conversations": [{"from": "human", "value": "q2"}, {"from": "gpt", "value": "a2"}]},
        {"instruction": "q3", "input": "ctx", "output": "a3"},
        {"prompt": "q4", "completion": "a4"},
        {"question": "q5", "answer": "a5"},
        {"question": "q5", "answer": "a5"},                      # duplicate
        {"messages": [{"role": "assistant", "content": "x"}]},   # invalid start
        {"foo": "bar"},                                          # unknown shape
        {"question": "long " * 5000, "answer": "y"},             # too long
    ] + [{"question": f"extra {i}", "answer": f"ans {i}"} for i in range(20)]
    inp = tmp_path / "in.jsonl"
    inp.write_text("\n".join(json.dumps(r) for r in rows))
    stats = fmt.prepare([inp], tmp_path / "out", 0.2, 0, 512, str(tiny_model), "You are helpful.")
    assert stats["dupes"] == 1 and stats["bad_shape"] == 1 and stats["too_long"] == 1
    assert stats["invalid"] == {"must start with user": 1}
    train = [json.loads(x) for x in (tmp_path / "out/train.jsonl").read_text().splitlines()]
    assert stats["train"] + stats["eval"] == 25  # 29 read - dup - bad - invalid - too_long and len(train) == stats["train"]
    assert all(r["messages"][0]["role"] == "system" for r in train)
    assert fmt.to_messages(rows[2])[0]["content"] == "q3\n\nctx"


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_synth_against_fake_openai_server(tmp_path, monkeypatch):
    import uvicorn
    from fake_openai import app, calls

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    while not server.started:
        time.sleep(0.05)
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("\n".join(json.dumps({"id": f"c{i}", "source": "s", "text": f"passage {i}"}) for i in range(5)))
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{port}/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("SYNTH_MODEL", "fake-model")
    import asyncio

    from peftkit.data.synth import run

    out = tmp_path / "synth.jsonl"
    stats = asyncio.run(run(chunks, out, 2, 3, "sys", None))
    assert stats == {"chunks": 5, "examples": 10, "failed_chunks": 0}
    assert calls["n"] >= 6  # first call returned 429 and was retried
    stats2 = asyncio.run(run(chunks, out, 2, 3, "sys", None))  # resume: nothing left to do
    assert stats2["chunks"] == 0
    row = json.loads(out.read_text().splitlines()[0])
    assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]
    server.should_exit = True
    th.join(timeout=5)
