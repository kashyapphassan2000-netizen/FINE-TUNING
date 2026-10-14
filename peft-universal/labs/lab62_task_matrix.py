"""Lab 62: the TASK registry, executed. One engine; each task brings its own schema, loss and metric.

Proves the missing abstraction from the review: MODALITY x ARCHITECTURE x TASK x METHOD.
Every row: resolve the task -> discover targets -> inject custom LoRA -> train with the TASK's loss
-> score with the TASK's metric. Random-weight models: this tests the machinery, not quality.
"""
import sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import logging; logging.disable(logging.WARNING)  # noqa: E702
import torch  # noqa: E402
from torch import nn  # noqa: E402
import transformers as T  # noqa: E402
T.logging.set_verbosity_error()
from uadapt import lora as L, metrics as M  # noqa: E402
from uadapt.adapter import UniversalModelAdapter  # noqa: E402
from uadapt.loop import train  # noqa: E402
from uadapt.tasks import TASKS, contrastive_loss, hf_loss, plan, regression_loss, reward_model_loss, diffusion_loss  # noqa: E402

torch.manual_seed(0)
BC = lambda **kw: T.BertConfig(vocab_size=120, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64, **kw)  # noqa: E731
ids = lambda n=8, b=4: torch.randint(0, 120, (b, n))  # noqa: E731
rows = []


def run(label, family, task_name, build, batch, loss_fn, metric_fn, train_also=(), steps=40, lr=5e-3,
        trace_batch=None):
    # paired tasks (embedding, reward model) feed the model TWO inputs per record, so the tracer gets
    # one plain forward batch while the loss keeps the pair structure
    torch.manual_seed(0)
    model = build()
    p = plan(family, task_name)
    ad = UniversalModelAdapter(model, trace_batch or batch)
    d = ad.discover_trainable_modules()
    heads = list(train_also) or d["train_fully_or_freeze"]
    inj = L.inject(model, d["lora_targets"] + d["conv_candidates"], r=4, alpha=8, train_also=heads)
    ok = ad.validate_training(loss_fn, batch)["ok"]
    before = float(loss_fn(model, batch))
    train(model, [batch], loss_fn, lr=lr, max_steps=steps, log_every=steps, warmup=0.1)
    after = float(loss_fn(model, batch))
    model.eval()
    with torch.no_grad():
        score = metric_fn(model)
    rows.append((label, task_name, len(inj["wrapped"]), ok, before, after, score, p["metrics"][0]))
    print(f"{label:<26}{task_name:<24}{len(inj['wrapped']):>5}{'OK' if ok else 'FAIL':>6}"
          f"{before:>9.3f}->{after:<8.3f}{str(score):>22}  {p['metrics'][0]}")


print(f"{'model':<26}{'task':<24}{'LoRA':>5}{'grads':>6}{'loss 0->N':>18}{'metric value':>22}  primary metric of the task")

# 1 sequence classification (encoder-only) -> macro-F1
y = torch.tensor([0, 2, 1, 0])
b = {"input_ids": ids(), "labels": y}
run("BERT-style encoder", ["encoder-only", "text"], "sequence_classification",
    lambda: T.BertForSequenceClassification(BC(num_labels=3)), b, hf_loss,
    lambda m: M.prf(m(input_ids=b["input_ids"]).logits.argmax(-1).tolist(), y.tolist())["macro"]["f1"])

# 2 token classification -> entity-level macro F1 (token level here)
tags = torch.randint(0, 4, (4, 8))
b2 = {"input_ids": ids(), "labels": tags}
run("BERT-style encoder", ["encoder-only"], "token_classification",
    lambda: T.BertForTokenClassification(BC(num_labels=4)), b2, hf_loss,
    lambda m: M.prf(m(input_ids=b2["input_ids"]).logits.argmax(-1).flatten().tolist(),
                    tags.flatten().tolist())["macro"]["f1"])

# 3 regression -> MAE
tgt = torch.tensor([1.5, -0.5, 3.0, 0.25])
b3 = {"input_ids": ids(), "labels": tgt}
run("BERT-style encoder", ["encoder-only"], "regression",
    lambda: T.BertForSequenceClassification(BC(num_labels=1, problem_type="regression")), b3, regression_loss,
    lambda m: round(float((m(input_ids=b3["input_ids"]).logits.squeeze(-1) - tgt).abs().mean()), 4))

# 4 embedding / retrieval -> Recall@1 and MRR with in-batch negatives
anchor, positive = ids(6), ids(6)
b4 = {"anchor": {"input_ids": anchor}, "positive": {"input_ids": positive}}
def retrieval_metric(m):
    with torch.no_grad():
        emb = lambda x: nn.functional.normalize(m(input_ids=x).last_hidden_state.mean(1), dim=-1)  # noqa: E731
        sim = emb(anchor) @ emb(positive).T
    ranked = [[int(j) for j in row] for row in sim.argsort(dim=-1, descending=True)]
    gold = list(range(len(ranked)))
    return {"recall@1": M.recall_at_k(ranked, gold, 1), "mrr": M.mrr(ranked, gold),
            "ndcg@3": M.ndcg_at_k(ranked, [{g: 1.0} for g in gold], 3)}
run("BERT-style encoder", ["encoder-only"], "embedding", lambda: T.BertModel(BC()), b4, contrastive_loss,
    retrieval_metric, train_also=[], trace_batch={"input_ids": anchor})

# 5 reward model -> pairwise accuracy (chosen scored above rejected)
chosen, rejected = ids(6), ids(6)
b5 = {"chosen": {"input_ids": chosen}, "rejected": {"input_ids": rejected}}
def pairwise_acc(m):
    with torch.no_grad():
        sc = lambda x: m(input_ids=x).logits[:, 0]  # noqa: E731
        return round(float((sc(chosen) > sc(rejected)).float().mean()), 4)
run("BERT + scalar value head", ["encoder-only"], "reward_model",
    lambda: T.BertForSequenceClassification(BC(num_labels=1)), b5, reward_model_loss, pairwise_acc,
    train_also=["classifier"], trace_batch={"input_ids": chosen})

# 6 masked LM -> accuracy on the masked positions
mlm_ids = ids(8)
labels = torch.full_like(mlm_ids, -100)
labels[:, 2] = mlm_ids[:, 2]
masked = mlm_ids.clone()
masked[:, 2] = 103
b6 = {"input_ids": masked, "labels": labels}
run("BERT-style encoder", ["encoder-only"], "masked_lm", lambda: T.BertForMaskedLM(BC()), b6, hf_loss,
    lambda m: round(float((m(input_ids=masked).logits[:, 2].argmax(-1) == mlm_ids[:, 2]).float().mean()), 4))

# 7 seq2seq -> exact match on greedy generation
src, tgt_ids = ids(6, 2), torch.randint(3, 120, (2, 4))
b7 = {"input_ids": src, "labels": tgt_ids}
run("T5-style enc-dec", ["encoder-decoder"], "seq2seq",
    lambda: T.T5ForConditionalGeneration(T.T5Config(vocab_size=120, d_model=32, d_kv=8, d_ff=64, num_layers=2, num_heads=2, decoder_start_token_id=0)),
    b7, hf_loss, lambda m: M.accuracy([tuple(r) for r in m.generate(src, max_new_tokens=4)[:, 1:5].tolist()],
                                      [tuple(r) for r in tgt_ids.tolist()]), steps=60)

# 8 CTC ASR -> CER of the greedy decode
audio = torch.randn(2, 4000)
ctc_labels = torch.randint(1, 20, (2, 4))
b8 = {"input_values": audio, "labels": ctc_labels}
def cer_metric(m):
    with torch.no_grad():
        pred = m(input_values=audio).logits.argmax(-1)
    hyps = ["".join(chr(65 + int(t)) for k, t in enumerate(row) if t != 0 and (k == 0 or t != row[k - 1])) for row in pred]
    refs = ["".join(chr(65 + int(t)) for t in row) for row in ctc_labels]
    return M.cer(hyps, refs)
run("wav2vec2-style CTC", ["audio"], "asr_ctc",
    lambda: T.Wav2Vec2ForCTC(T.Wav2Vec2Config(vocab_size=20, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64, conv_dim=(16, 16), conv_stride=(5, 2), conv_kernel=(10, 3), num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2, pad_token_id=0)),
    b8, hf_loss, cer_metric)

# 9 image classification -> accuracy
imgs, img_y = torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 2, 3])
b9 = {"pixel_values": imgs, "labels": img_y}
run("ViT-style vision", ["vision"], "image_classification",
    lambda: T.ViTForImageClassification(T.ViTConfig(image_size=32, patch_size=8, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64, num_labels=4)),
    b9, hf_loss, lambda m: M.accuracy(m(pixel_values=imgs).logits.argmax(-1).tolist(), img_y.tolist()))

# 10 diffusion -> MSE of predicted noise (the task's own metric is sampling quality; MSE is the proxy)
class Denoiser(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.t_emb = nn.Linear(1, 32)
        self.fc1, self.fc2, self.fc3 = nn.Linear(d + 32, 64), nn.Linear(64, 64), nn.Linear(64, d)
    def forward(self, x, timestep):
        h = torch.cat([x, torch.sin(self.t_emb(timestep[:, None].float()))], -1)
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(h)))))
x0 = torch.randn(8, 16)
b10 = {"x0": x0, "t": torch.randint(0, 1000, (8,)), "noise": torch.randn(8, 16)}
run("custom denoiser", ["diffusion", "custom"], "image_generation", Denoiser,
    b10, diffusion_loss, lambda m: round(float(diffusion_loss(m, b10)), 4),
    trace_batch=(x0, b10["t"]))      # the denoiser's forward is (x, timestep), not the training record

print("""
What this proves: the ENGINE is one code path, and the TASK supplies the schema, the loss and the metric.
Note what changes per row and what does not:
  same: discovery, LoRA injection, gradient check, training loop, merge
  different: the record shape, the labels, the loss function, and above all THE METRIC
A model can serve several tasks (BERT appears as classification, token classification, regression,
embedding, reward model and masked LM) - which is exactly why "architecture" alone never determined
the pipeline. Tasks with no shippable generic loss (TTS, video) are registered with their recipe and
marked as needing your own loss_fn - stated, not hidden.""")
