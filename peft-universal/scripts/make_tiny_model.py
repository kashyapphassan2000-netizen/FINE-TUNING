"""Create a tiny, offline Llama-style model for the Basics lab (runs on any laptop CPU, no download).

python scripts/make_tiny_model.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from tiny import build_tiny  # noqa: E402

out = build_tiny(ROOT / "outputs" / "tiny")
print(f"tiny model ready: {out}")
