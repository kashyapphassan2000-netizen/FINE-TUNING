"""Lab 31: KV cache memory math, checked against the bytes PyTorch actually allocates."""
import sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
import torch  # noqa: E402
from tiny import build_tiny  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

d = build_tiny(Path(tempfile.mkdtemp()))
model = AutoModelForCausalLM.from_pretrained(d).eval()
tok = AutoTokenizer.from_pretrained(d)
cfg = model.config
L, KV, HD = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim or cfg.hidden_size // cfg.num_attention_heads
ids = tok("Reset the controller by holding the red button", return_tensors="pt")["input_ids"]
T = ids.shape[1]
with torch.no_grad():
    out = model(ids, use_cache=True)
cache = out.past_key_values
measured = sum(t.numel() * t.element_size() for layer in cache for t in (layer if isinstance(layer, tuple) else (layer,)) if torch.is_tensor(t))
if measured == 0:  # transformers v5 Cache object
    measured = sum(t.numel() * t.element_size() for t in [*cache.layers[0].keys.new_empty(0)] ) if False else \
        sum(x.numel() * x.element_size() for lay in cache.layers for x in (lay.keys, lay.values))
formula = 2 * L * KV * HD * T * 4        # 2 (K and V) x layers x kv_heads x head_dim x tokens x bytes(fp32)
print(f"tiny model: layers={L} kv_heads={KV} head_dim={HD} tokens={T} dtype=float32")
print(f"formula  2*L*KV*HD*T*bytes = 2*{L}*{KV}*{HD}*{T}*4 = {formula:,} bytes")
print(f"measured allocation                              = {measured:,} bytes")
print("formula matches reality:", formula == measured)

print("\nPer-token cost (bytes) = 2 * layers * kv_heads * head_dim * bytes_per_element")
print(f"\n{'model (bf16 KV)':<26}{'L':>4}{'kv_h':>6}{'hd':>5}{'KV/token':>12}{'1 seq @128k':>14}{'256 seqs @8k':>15}")
rows = [("Llama-3.1-8B (GQA)", 32, 8, 128), ("Llama-3.1-70B (GQA)", 80, 8, 128),
        ("Llama-3.1-405B (GQA)", 126, 8, 128), ("hypothetical MHA 405B", 126, 128, 128),
        ("Qwen3-235B-A22B (GQA)", 94, 4, 128)]
for name, l, kv, hd in rows:
    per = 2 * l * kv * hd * 2
    print(f"{name:<26}{l:>4}{kv:>6}{hd:>5}{per / 1024:>10.1f} KB{per * 131072 / 1024**3:>12.1f} GB"
          f"{per * 8192 * 256 / 1024**3:>13.1f} GB")
print("\nDeepSeek-V3 / Kimi-K2 use MLA: cache = layers * (kv_lora_rank + qk_rope_head_dim) * bytes")
mla = 61 * (512 + 64) * 2
print(f"  61 * (512 + 64) * 2 = {mla:,} bytes/token = {mla / 1024:.1f} KB  "
      f"-> {mla * 131072 / 1024**3:.1f} GB for one 128k sequence")
print(f"  a same-depth GQA model with 8 kv heads would cost {2 * 61 * 8 * 128 * 2 / 1024:.1f} KB/token"
      f" ({2 * 61 * 8 * 128 * 2 / mla:.1f}x more)")
print("\nRules of thumb:")
print("  KV cache grows LINEARLY with: tokens in flight, layers, kv_heads, head_dim, bytes per element.")
print("  It does NOT depend on the MLP size, so a MoE model can have huge weights and a small cache.")
print("  fp8 KV halves it. Doubling context doubles it. Doubling users doubles it.")
