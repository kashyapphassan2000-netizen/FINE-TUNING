import math

import pytest
import torch
import transformers as T
from torch import nn
from uadapt import inspect_model as im
from uadapt import lora as L
from uadapt import strategy, targets
from uadapt.adapter import UniversalModelAdapter
from uadapt.loop import train
from uadapt.report import report
from uadapt.trace import gradient_report, trace_forward

T.logging.set_verbosity_error()


def bert():
    torch.manual_seed(0)
    return T.BertForSequenceClassification(T.BertConfig(vocab_size=100, hidden_size=32, num_hidden_layers=2,
                                                        num_attention_heads=2, intermediate_size=64, num_labels=3))


IDS = torch.randint(0, 100, (2, 8))


def test_classification_uses_evidence():
    c = im.classify(bert())
    assert "encoder-only" in c["labels"] and "text" in c["labels"] and c["evidence"]["encoder-only"]
    llama = T.LlamaForCausalLM(T.LlamaConfig(vocab_size=50, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                             num_attention_heads=2, num_key_value_heads=1))
    assert "decoder-only" in im.classify(llama)["labels"]

    class Rnn(nn.Module):
        def __init__(self):
            super().__init__()
            self.r = nn.GRU(4, 4)

        def forward(self, x):
            return self.r(x)[0]
    assert {"custom", "recurrent"} <= set(im.classify(Rnn())["labels"])


def test_discovery_finds_attention_without_names():
    d = targets.discover(bert(), {"input_ids": IDS})
    roles = {n.split(".")[-1]: r for n, r in d["roles"].items() if "layer.0." in n}
    assert roles["query"] == roles["key"] == roles["value"] == "attention_proj"     # found by shared input
    assert any(n.endswith("classifier") for n in d["train_fully_or_freeze"])
    assert not any("embeddings" in n and "LayerNorm" not in n for n in d["lora_targets"])
    assert len(d["lora_targets"]) >= 10


def test_linear_and_conv_lora_merge_exactly_and_weight_is_effective():
    torch.manual_seed(0)
    lin = nn.Linear(6, 5)
    w = L.LoRALinear(lin, r=2)
    with torch.no_grad():
        w.lora_B.normal_()
    x = torch.randn(3, 6)
    y = w(x)
    assert torch.allclose(y, nn.functional.linear(x, w.weight, w.bias), atol=1e-6)     # functional read = correct
    assert w.lora_A.requires_grad and not w.base.weight.requires_grad
    nn.functional.linear(x, w.weight, w.bias).sum().backward()                         # grads flow through .weight
    assert w.lora_B.grad is not None
    merged = w.merged()
    assert torch.allclose(merged(x), y, atol=1e-5)
    conv = nn.Conv2d(3, 4, 3, padding=1, stride=2)
    c = L.LoRAConv(conv, r=2)
    with torch.no_grad():
        c.lora_B.weight.normal_()
    img = torch.randn(1, 3, 9, 9)
    yc = c(img)
    assert torch.allclose(yc, nn.functional.conv2d(img, c.weight, c.bias, 2, 1), atol=1e-5)
    assert torch.allclose(c.merged()(img), yc, atol=1e-5)
    with pytest.raises(ValueError):
        L.LoRAConv(nn.Conv2d(4, 4, 3, groups=4))                                        # depthwise refused


def test_inject_train_merge_on_bert_end_to_end():
    m = bert()
    batch = {"input_ids": IDS, "labels": torch.tensor([0, 2])}
    ad = UniversalModelAdapter(m, batch)
    ad.discover_trainable_modules()
    inj = ad.build_training_pipeline(r=4, train_also=["classifier"])
    assert inj["wrapped"] and 0 < inj["trainable_pct"] < 50
    assert ad.validate_training(lambda mm, b: mm(**b).loss, batch)["ok"]
    before = float(m(**batch).loss)
    res = train(m, [batch], lambda mm, b: mm(**b).loss, lr=1e-2, max_steps=30, log_every=30)
    assert float(m(**batch).loss) < before and res["first_step_gradients"]["ok"]
    with torch.no_grad():
        ref = m(**batch).logits
        L.merge_all(m)
        assert torch.allclose(m(**batch).logits, ref, atol=1e-4)
    assert not any(isinstance(x, L.LoRALinear) for x in m.modules())


def test_silent_failure_detectors():
    class Skips(nn.Module):
        def __init__(self):
            super().__init__()
            self.used, self.unused, self.head = nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 2)

        def forward(self, x):
            return self.head(self.used(x))
    m = Skips()
    L.inject(m, ["used", "unused"], r=2)
    assert L.verify_called(m, (torch.randn(2, 4),))["not_called"] == ["unused"]
    nn.functional.cross_entropy(m(torch.randn(2, 4)), torch.tensor([0, 1])).backward()
    rep = gradient_report(m)
    assert any("unused" in x for x in rep["no_or_zero_grad"])
    frozen = Skips()
    for p in frozen.parameters():
        p.requires_grad_(False)
    with pytest.raises(RuntimeError, match="no trainable"):
        train(frozen, [(torch.randn(2, 4),)], lambda mm, b: mm(*b).sum(), max_steps=1)
    exploding = Skips()
    L.inject(exploding, ["used"], r=2)
    with torch.no_grad():
        exploding.head.weight.mul_(float("inf"))
    with pytest.raises(FloatingPointError):
        train(exploding, [(torch.randn(2, 4),)], lambda mm, b: mm(*b).sum(), max_steps=1)


def test_trace_records_shapes_and_shared_inputs():
    tr = trace_forward(bert(), {"input_ids": IDS})
    q = next(c for c in tr.calls if c.name.endswith("layer.0.attention.self.query"))
    k = next(c for c in tr.calls if c.name.endswith("layer.0.attention.self.key"))
    assert q.input_id == k.input_id and q.out_shape == (2, 8, 32)
    assert "module" in tr.table(5)


def test_strategy_is_constraint_driven():
    small = strategy.recommend(10_000_000, vram_gb=24, family=["text", "decoder-only"], task="sft",
                               data_rows=100_000, lora_targets=10)
    assert small["ranked"][0]["strategy"] in ("full fine-tuning", "LoRA")
    huge = strategy.recommend(70_000_000_000, vram_gb=80, family=["text", "decoder-only"], task="sft", lora_targets=10)
    assert any(o["strategy"] == "QLoRA" for o in huge["ranked"])
    assert not any(o["strategy"] == "full fine-tuning" for o in huge["ranked"])
    moe = strategy.recommend(1_000_000_000, 80, ["text", "moe"], "sft", lora_targets=4)
    assert any("attention only" in o["strategy"] for o in moe["ranked"])


def test_report_on_a_custom_model():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv2d(3, 8, 3)
            self.fc = nn.Linear(8, 4)

        def forward(self, x):
            return self.fc(self.c(x).mean((2, 3)))
    r = report(Net(), None, (torch.randn(1, 3, 8, 8),))
    assert "custom" in r["Classification"] and "vision" in r["Classification"]
    assert r["Parameter Count"] == sum(p.numel() for p in Net().parameters())
    assert any("loss_fn" in x for x in r["Risks"]) and r["Conv candidates"]
    assert "CUSTOM ENGINEERING REQUIRED" in r["Recommended serving"]
