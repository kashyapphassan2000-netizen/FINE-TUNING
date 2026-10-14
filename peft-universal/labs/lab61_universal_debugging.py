"""Lab 61: plant the classic silent and loud failures on DIFFERENT architectures and show how the
universal checks catch each one. Every message below is the real exception or report."""
import sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import torch  # noqa: E402
from torch import nn  # noqa: E402
import transformers as T  # noqa: E402
T.logging.set_verbosity_error()
from uadapt import lora as L  # noqa: E402
from uadapt.loop import train  # noqa: E402
from uadapt.trace import gradient_report, trace_forward  # noqa: E402

torch.manual_seed(0)
bert = lambda: T.BertForSequenceClassification(T.BertConfig(vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64, num_labels=2))  # noqa: E731
t5 = lambda: T.T5ForConditionalGeneration(T.T5Config(vocab_size=100, d_model=32, d_kv=8, d_ff=64, num_layers=1, num_heads=2, decoder_start_token_id=0))  # noqa: E731
w2v = lambda: T.Wav2Vec2ForCTC(T.Wav2Vec2Config(vocab_size=20, hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64, conv_dim=(16, 16), conv_stride=(5, 2), conv_kernel=(10, 3), num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2, pad_token_id=0, ctc_zero_infinity=False))  # noqa: E731
ids = torch.randint(0, 100, (2, 8))


def show(n, title, symptom, cause, command, fix):
    print(f"\n--- {n}. {title}\n  SYMPTOM : {symptom}\n  CAUSE   : {cause}\n  COMMAND : {command}\n  FIX     : {fix}")


# 1 names copied from another architecture
try:
    L.inject(bert(), ["bert.encoder.layer.0.attention.self.q_proj"])
    err = "none"
except AttributeError as e:
    err = f"{type(e).__name__}: {str(e)[:90]}"
show(1, "LoRA targets not found", err, "q_proj is a Llama name; BERT calls it 'query'",
     "uadapt.targets.discover(model, batch)", "use the discovered full paths, never a copied name list")

# 2 nothing trainable
m = bert()
for p in m.parameters():
    p.requires_grad_(False)
try:
    train(m, [{"input_ids": ids, "labels": torch.tensor([0, 1])}], lambda mm, b: mm(**b).loss, max_steps=1)
    err = "none"
except RuntimeError as e:
    err = str(e)[:95]
show(2, "no trainable parameters", err, "everything frozen, no adapter injected", "gradient_report / param_counts",
     "inject adapters or unfreeze a head before building the optimizer")

# 3 encoder-decoder called without labels or decoder inputs
try:
    t5()(input_ids=ids)
    err = "none"
except ValueError as e:
    err = f"ValueError: {str(e)[:80]}"
show(3, "encoder-decoder forward fails", err, "seq2seq decoders need decoder_input_ids (built from labels by shifting)",
     "inspect.signature(model.forward)", "pass labels (the model shifts them) or decoder_input_ids explicitly")

# 4 loss not wired
m = bert()
out = m(input_ids=ids)
show(4, "model returns no loss", f"outputs.loss = {out.loss}", "no labels passed, so the model computed only logits",
     "UniversalModelAdapter.identify_loss()", "pass labels, or write loss_fn(model, batch) yourself")

# 5 wrong labels for CTC: target longer than the output frames
m = w2v()
frames = trace_forward(m, {"input_values": torch.randn(1, 4000)}).calls[-1].out_shape[1]
bad = m(input_values=torch.randn(1, 4000), labels=torch.randint(1, 20, (1, frames + 50))).loss
show(5, "CTC loss is inf", f"loss = {float(bad)} (output frames {frames}, label length {frames + 50})",
     "CTC needs label length <= output frames after the conv downsampling", "trace_forward -> last out_shape[1]",
     "shorter labels / less downsampling / longer audio; ctc_zero_infinity hides it, it does not fix it")

# 6 non-finite loss: a real cause - labels outside the class range produce a garbage objective; here we
#   reproduce the NaN itself with fp32 overflow in the logits (weights scaled to infinity), which is what an
#   exploding update looks like one step later
m = bert()
L.inject(m, ["bert.encoder.layer.0.attention.self.query"], r=4)
with torch.no_grad():
    m.classifier.weight.mul_(float("inf"))
try:
    train(m, [{"input_ids": ids, "labels": torch.tensor([0, 1])}], lambda mm, b: mm(**b).loss, max_steps=3)
    err = "none"
except FloatingPointError as e:
    err = str(e)[:95]
show(6, "NaN / inf loss", err, "exploded weights or logits (after a too-high lr, fp16 overflow, or bad data)",
     "loop.train stops on the first non-finite loss", "lower lr, clip=1.0, bf16 not fp16, inspect the last batch")

# 7 adapter injected but its output is never used (a module that the forward path skips)
class Skips(nn.Module):
    def __init__(self):
        super().__init__()
        self.used, self.unused, self.head = nn.Linear(8, 8), nn.Linear(8, 8), nn.Linear(8, 2)

    def forward(self, x):
        return self.head(torch.relu(self.used(x)))
m = Skips()
L.inject(m, ["used", "unused"], r=2)
nn.functional.cross_entropy(m(torch.randn(4, 8)), torch.tensor([0, 1, 0, 1])).backward()
rep = gradient_report(m)
show(7, "silent: adapter trains nothing", f"{rep['n_no_or_zero_grad']} trainable tensors with no gradient: {rep['no_or_zero_grad']}",
     "LoRA placed on a layer the forward pass never uses", "trace.gradient_report(model) after one backward",
     "adapt layers that appear in trace_forward; remove the dead adapter")

print("\nMETHOD, for every failure: symptom -> likely causes -> one diagnostic command -> expected result -> fix -> verify.")
print("Change ONE thing, re-run the same command, and only then continue.")
