"""Lab 8: the same conversation rendered with ChatML, Llama-3 and Mistral templates, plus the loss mask."""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
from tiny import build_tiny  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from peftkit.data.templates import mask_report, render  # noqa: E402

tok = AutoTokenizer.from_pretrained(build_tiny(Path(tempfile.mkdtemp())))
conv = [{"role": "system", "content": "You are a pump expert."},
        {"role": "user", "content": "Max pressure?"},
        {"role": "assistant", "content": "6 bar."},
        {"role": "user", "content": "And for testing?"},
        {"role": "assistant", "content": "7.5 bar for 10 minutes."}]
for name in ("chatml", "llama3", "mistral"):
    print(f"===== {name}")
    print(render(tok, conv, name))
    rep = mask_report(tok, conv, name)
    print(f"-- trained tokens: {rep['trained_tokens']}/{rep['tokens']} ({rep['trained_fraction']:.0%})")
    print(f"-- text the loss sees: {rep['trained_text']!r}\n")
