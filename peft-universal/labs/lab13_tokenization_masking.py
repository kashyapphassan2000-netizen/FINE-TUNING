"""Lab 13: text -> input_ids -> attention_mask -> labels, token by token, with padding and truncation."""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
from tiny import build_tiny  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from peftkit.tokenize_inspect import render, table  # noqa: E402

tok = AutoTokenizer.from_pretrained(build_tiny(Path(tempfile.mkdtemp())))
text = "Hold the red button for five seconds."
ids = tok(text)["input_ids"]
print("TEXT ->", repr(text))
print("ids  ->", ids)
print("pieces ->", tok.convert_ids_to_tokens(ids))
print("back ->", repr(tok.decode(ids)))
print("A rare word is split into several pieces (this is BPE):",
      tok.convert_ids_to_tokens(tok("hydrodynamics")["input_ids"]))
print("\nspecial tokens:", {"bos": tok.bos_token, "eos": tok.eos_token, "pad": tok.pad_token,
                            "eos_id": tok.eos_token_id, "pad_id": tok.pad_token_id})

print("\nPADDING a batch of two different lengths (attention_mask says which are real):")
batch = tok(["Reset the controller.", "What is the maximum pressure during calibration?"], padding=True)
for i in range(2):
    print(f"  ids{i}       ", batch["input_ids"][i])
    print(f"  attention{i} ", batch["attention_mask"][i])

print("\nTRUNCATION cuts from the END, which deletes the ANSWER if max_length is too small:")
long = tok("Warranty claims require the serial number and the purchase date of the pump.",
           truncation=True, max_length=8)["input_ids"]
print("  max_length=8 ->", repr(tok.decode(long)))

print("\nCHAT TEMPLATE + LABELS (-100 = not trained on):")
msgs = [{"role": "system", "content": "You are a pump expert."},
        {"role": "user", "content": "What is the maximum pressure?"},
        {"role": "assistant", "content": "6 bar."}]
print(render(table(tok, msgs), limit=None))
