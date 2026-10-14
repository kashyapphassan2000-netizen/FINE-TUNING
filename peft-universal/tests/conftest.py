import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    from tiny import build_tiny

    return build_tiny(tmp_path_factory.mktemp("tiny-llama"))


@pytest.fixture(scope="session")
def sft_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("sft")
    qa = [("What is the maximum pressure?", "The maximum pressure is 6 bar."),
          ("How do I reset the controller?", "Hold the red button for five seconds."),
          ("What do warranty claims need?", "The serial number and the purchase date.")]
    rows = [{"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]} for q, a in qa] * 16
    for name, data in (("train", rows), ("eval", rows[:6])):
        with (d / f"{name}.jsonl").open("w") as f:
            for r in data:
                f.write(json.dumps(r) + "\n")
    return d


@pytest.fixture()
def llm_server(monkeypatch):
    import socket
    import threading
    import time

    import uvicorn
    from fake_llm import app

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    while not server.started:
        time.sleep(0.05)
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{port}/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("SYNTH_MODEL", "fake")
    yield
    server.should_exit = True
    th.join(timeout=5)
