"""Lab 60: ONE methodology, TWELVE architecture families. No PEFT, no model-specific code paths.

For every model: classify from evidence -> discover targets from computation (not names) -> inject
custom LoRA -> gradient check -> single-batch overfit smoke test -> merge -> prove the merge is exact.
All models are built from configs with random weights: this tests the METHOD, not model quality.
"""
import math, sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import logging; logging.disable(logging.WARNING)  # noqa: E702
import torch  # noqa: E402
from torch import nn  # noqa: E402
import transformers  # noqa: E402
transformers.logging.set_verbosity_error()
from uadapt import lora as L  # noqa: E402
from uadapt.adapter import UniversalModelAdapter  # noqa: E402
from uadapt.loop import train  # noqa: E402

torch.manual_seed(0)
T = transformers


# ---------- custom models (no Hugging Face, no config) ----------
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(3, 16, 3, padding=1), nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32, 5)

    def forward(self, x):
        x = torch.relu(self.c1(x)); x = torch.max_pool2d(torch.relu(self.c2(x)), 2)  # noqa: E702
        return self.fc(x.mean((2, 3)))


class TinyLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp, self.rnn, self.out = nn.Linear(4, 32), nn.LSTM(32, 32, batch_first=True), nn.Linear(32, 1)

    def forward(self, x):
        h, _ = self.rnn(torch.tanh(self.inp(x)))
        return self.out(h[:, -1])


class TinyDenoiser(nn.Module):
    """epsilon-prediction diffusion model: forward(x_t, timestep) -> predicted noise."""
    def __init__(self, d=16):
        super().__init__()
        self.t_emb = nn.Linear(1, 32)
        self.fc1, self.fc2, self.fc3 = nn.Linear(d + 32, 64), nn.Linear(64, 64), nn.Linear(64, d)

    def forward(self, x, timestep):
        h = torch.cat([x, torch.sin(self.t_emb(timestep[:, None].float()))], -1)
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(h)))))


class TinyVLM(nn.Module):
    """vision encoder -> projector -> small transformer LM (uses nn.MultiheadAttention on purpose)."""
    def __init__(self, vocab=50, d=32):
        super().__init__()
        self.vision = nn.Sequential(nn.Conv2d(3, 16, 4, stride=4), nn.GELU(), nn.Conv2d(16, 32, 2, stride=2))
        self.projector = nn.Sequential(nn.Linear(32, d), nn.GELU(), nn.Linear(d, d))
        self.embed = nn.Embedding(vocab, d)
        layer = nn.TransformerEncoderLayer(d, 2, 64, batch_first=True, dropout=0.0)
        self.lm = nn.TransformerEncoder(layer, 2)
        self.lm_head = nn.Linear(d, vocab)

    def forward(self, pixel_values, input_ids):
        v = self.vision(pixel_values).flatten(2).transpose(1, 2)
        x = torch.cat([self.projector(v), self.embed(input_ids)], 1)
        mask = torch.triu(torch.full((x.shape[1], x.shape[1]), float("-inf")), 1)
        return self.lm_head(self.lm(x, mask=mask))[:, v.shape[1]:]


ids = lambda n=10, v=100: torch.randint(0, v, (2, n))  # noqa: E731
CASES = []


def case(name, build, batch, loss_fn):
    CASES.append((name, build, batch, loss_fn))


out_loss = lambda m, b: m(**b).loss  # noqa: E731
llama_ids = ids(10, 100)
case("decoder-only (Llama)", lambda: T.LlamaForCausalLM(T.LlamaConfig(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)),
     {"input_ids": llama_ids, "labels": llama_ids}, out_loss)
case("encoder-only (BERT cls)", lambda: T.BertForSequenceClassification(T.BertConfig(vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64, num_labels=3)),
     {"input_ids": ids(), "labels": torch.tensor([0, 2])}, out_loss)
case("encoder-decoder (T5)", lambda: T.T5ForConditionalGeneration(T.T5Config(vocab_size=100, d_model=32, d_kv=8, d_ff=64, num_layers=2, num_heads=2, decoder_start_token_id=0)),
     {"input_ids": ids(), "labels": ids(6)}, out_loss)
case("vision (ViT cls)", lambda: T.ViTForImageClassification(T.ViTConfig(image_size=32, patch_size=8, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64, num_labels=4)),
     {"pixel_values": torch.randn(2, 3, 32, 32), "labels": torch.tensor([1, 3])}, out_loss)
x_img, y_img = torch.randn(4, 3, 16, 16), torch.tensor([0, 1, 2, 3])
case("vision (custom CNN)", TinyCNN, (x_img,), lambda m, b: nn.functional.cross_entropy(m(x_img), y_img))
case("audio CTC (wav2vec2)", lambda: T.Wav2Vec2ForCTC(T.Wav2Vec2Config(vocab_size=20, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64, conv_dim=(16, 16), conv_stride=(5, 2), conv_kernel=(10, 3), num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2, pad_token_id=0)),
     {"input_values": torch.randn(2, 4000), "labels": torch.randint(1, 20, (2, 5))}, out_loss)
case("audio seq2seq (Whisper)", lambda: T.WhisperForConditionalGeneration(T.WhisperConfig(vocab_size=100, d_model=32, encoder_layers=1, decoder_layers=1, encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=64, decoder_ffn_dim=64, num_mel_bins=16, max_source_positions=16, max_target_positions=32, pad_token_id=0, bos_token_id=1, eos_token_id=2, decoder_start_token_id=1)),
     {"input_features": torch.randn(2, 16, 32), "labels": torch.randint(3, 100, (2, 6))}, out_loss)
case("state-space (Mamba)", lambda: T.MambaForCausalLM(T.MambaConfig(vocab_size=100, hidden_size=32, num_hidden_layers=2, state_size=8)),
     {"input_ids": llama_ids, "labels": llama_ids}, out_loss)
case("MoE (Qwen3-MoE)", lambda: T.Qwen3MoeForCausalLM(T.Qwen3MoeConfig(vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16, intermediate_size=64, head_dim=8)),
     {"input_ids": llama_ids, "labels": llama_ids}, out_loss)
x_seq, y_seq = torch.randn(4, 7, 4), torch.randn(4, 1)
case("recurrent (custom LSTM)", TinyLSTM, (x_seq,), lambda m, b: nn.functional.mse_loss(m(x_seq), y_seq))
x0 = torch.randn(8, 16)
tt = torch.randint(0, 100, (8,))
alpha_bar = torch.cos(tt.float() / 100 * math.pi / 2)[:, None] ** 2
noise = torch.randn_like(x0)
xt = alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * noise
case("diffusion (custom denoiser)", TinyDenoiser, (xt, tt), lambda m, b: nn.functional.mse_loss(m(xt, tt), noise))
pv, vl_ids = torch.randn(2, 3, 16, 16), torch.randint(0, 50, (2, 6))
case("multimodal (custom VLM)", TinyVLM, {"pixel_values": pv, "input_ids": vl_ids},
     lambda m, b: nn.functional.cross_entropy(m(**b)[:, :-1].reshape(-1, 50), vl_ids[:, 1:].reshape(-1)))

print(f"{'family':<28}{'classified as':<34}{'targets':>8}{'train%':>8}{'loss 0->40':>18}{'grads':>7}{'merge diff':>12}")
rows = []
for name, build, batch, loss_fn in CASES:
    torch.manual_seed(0)
    model = build().eval()
    # trace WITH labels: encoder-decoders (T5, Whisper) need decoder inputs, which they derive from labels
    batch_no_labels = batch
    ad = UniversalModelAdapter(model, batch, getattr(model, "config", None))
    cls = ad.classify()["labels"]
    disc = ad.discover_trainable_modules()
    targets = disc["lora_targets"] + disc["conv_candidates"]
    head = disc["train_fully_or_freeze"] if any(k in name for k in ("cls", "CNN", "LSTM")) else []
    inj = L.inject(model, targets, r=4, alpha=8, train_also=head)
    called = L.verify_called(model, batch)          # which wrappers run via forward() vs are only read
    inj['bypassed'] = called['not_called']
    grads = ad.validate_training(loss_fn, batch)
    before = float(loss_fn(model, batch))
    res = train(model, [batch], loss_fn, lr=1e-2, max_steps=40, log_every=40, warmup=0.05)
    after = float(loss_fn(model, batch))
    model.eval()
    with torch.no_grad():
        ref = model(**batch_no_labels) if isinstance(batch_no_labels, dict) else model(*batch_no_labels)
        ref = ref.logits if hasattr(ref, "logits") else ref
        L.merge_all(model)
        got = model(**batch_no_labels) if isinstance(batch_no_labels, dict) else model(*batch_no_labels)
        got = got.logits if hasattr(got, "logits") else got
    diff = float((ref - got).abs().max())
    rows.append((name, cls, disc, inj, grads, before, after, diff))
    print(f"{name:<28}{','.join(cls)[:33]:<34}{len(inj['wrapped']):>8}{inj['trainable_pct']:>8}"
          f"{before:>9.3f}->{after:<8.3f}{'OK' if grads['ok'] else 'FAIL':>7}{diff:>12.1e}")

print("\nWHAT THE DISCOVERY FOUND, BY COMPUTATION (leaf names shown only so you can recognise them):")
for name, cls, disc, inj, *_ in rows:
    roles = {}
    for n, r in disc["roles"].items():
        if n in inj["wrapped"] or r in ("head", "router", "small_proj"):
            roles.setdefault(r, set()).add(n.split(".")[-1])
    print(f"  {name:<28}" + "  ".join(f"{r}={sorted(v)}" for r, v in sorted(roles.items())))
    if inj["skipped"]:
        print(f"  {'':<28}skipped: {inj['skipped'][:2]}")
    if inj.get("bypassed"):
        print(f"  {'':<28}not called via forward() (read functionally; gradient-checked): {inj['bypassed']}")

print("""
Read the table honestly:
  - every family trained through the SAME code path, and every merge was exact (float noise only);
  - the discovery never used a name list: BERT 'query/key/value', T5 'q/k/v/o/wi/wo', Mamba 'in_proj/out_proj'
    were found because they receive the same input or expand/contract the width;
  - MoE: fused 3-D expert tensors are not nn.Linear, so only attention (and routers-excluded) are wrapped.
  - custom VLM, a silent failure found while building this lab: PyTorch's TransformerEncoderLayer in eval
    mode can read linear1.weight / linear2.weight FUNCTIONALLY instead of calling the layer. The first LoRA
    wrapper returned the BASE weight there, so the adapter was silently ignored at inference and the merge
    looked "inexact" (3.5). Fix, now in uadapt.lora: .weight returns the EFFECTIVE weight W + s*B@A.
    Lesson: a wrapper must be correct both when it is CALLED and when its .weight is READ.
  - nn.MultiheadAttention packs q/k/v into ONE Parameter (in_proj_weight), not Linear layers, so they are not
    discoverable or wrappable as Linears: FRAMEWORK LIMITATION. To adapt them, re-implement the attention with
    separate Linear layers: CUSTOM ENGINEERING REQUIRED.
  - T5 cross-attention: q reads the decoder state while k/v read the encoder output, so role LABELS differ
    from self-attention (k/v "parallel", q "contracting"). The labels are heuristics; the TARGET SET is right.
  - wav2vec2 CTC loss is large because its config sums (not averages) the CTC loss; it still fell.
  - a falling loss on ONE batch proves gradients flow and the objective is wired correctly. It proves
    nothing about generalisation: that is evaluation's job.""")
