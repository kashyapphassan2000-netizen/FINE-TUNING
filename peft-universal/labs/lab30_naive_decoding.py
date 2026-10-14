"""Lab 30: the naive decoding problem. Without a cache, generating token n redoes the work for all
previous tokens. Measured on a real (tiny) Llama, with and without use_cache."""
import sys, tempfile, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
import torch  # noqa: E402
from tiny import build_tiny  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

torch.manual_seed(0)
d = build_tiny(Path(tempfile.mkdtemp()))
model = AutoModelForCausalLM.from_pretrained(d).eval()
tok = AutoTokenizer.from_pretrained(d)
prompt = tok("Reset the controller", return_tensors="pt")["input_ids"]

print("Generating 64 tokens from a 4-token prompt, same model, same greedy result.\n")
res = {}
for use_cache in (False, True):
    torch.manual_seed(0)
    ids = prompt.clone()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=64, do_sample=False, use_cache=use_cache,
                             pad_token_id=tok.pad_token_id)
    res[use_cache] = (time.perf_counter() - t0, out)
    print(f"use_cache={str(use_cache):<5} {res[use_cache][0]:.3f} s")
print("same tokens produced:", torch.equal(res[False][1], res[True][1]))
print(f"speed-up on this tiny model: {res[False][0] / res[True][0]:.2f}x "
      "(on a real model the gap is far larger: the cost per layer is much higher)")

print("\nWhy: attention for step n needs K and V of every earlier token.")
print(f"{'tokens generated':>17}{'no cache: K/V recomputed':>26}{'with cache: new K/V':>22}{'saved':>9}")
total_naive = total_cache = 0
for n in (128, 1024, 8192, 131072):
    naive = n * (n + 1) // 2          # sum 1..n  -> quadratic
    cached = n                         # one new K/V per step -> linear
    total_naive, total_cache = naive, cached
    print(f"{n:>17,}{naive:>26,}{cached:>22,}{naive / cached:>8.0f}x")
print("\nThe cache turns O(n^2) recomputation into O(n) work plus O(n) memory. The memory is the new")
print("problem, and every technique in this book (GQA, MQA, MLA, paging, quantised KV, sliding windows)")
print("exists to make that memory smaller or cheaper to manage.")
