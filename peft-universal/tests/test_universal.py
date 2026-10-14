"""The universal layer the second review asked for: processor, collator, loss derivation, pipeline."""
import pytest
import torch
import transformers as T
from torch import nn
from uadapt import processors
from uadapt.collate import collate, make_loader
from uadapt.derive import derive_loss, input_contract, output_contract
from uadapt.pipeline import UniversalPipeline

T.logging.set_verbosity_error()
IDS = torch.randint(0, 80, (4, 8))


def bert(num_labels=3, cls=T.BertForSequenceClassification):
    torch.manual_seed(0)
    return cls(T.BertConfig(vocab_size=80, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                            intermediate_size=64, num_labels=num_labels))


def test_universal_collator_handles_every_shape():
    rows = [{"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3]),
             "pixel_values": torch.randn(3, 4, 4), "score": 0.5, "name": "a"},
            {"input_ids": torch.tensor([1, 2]), "labels": torch.tensor([1, 2]),
             "pixel_values": torch.randn(3, 4, 4), "score": 1.5, "name": "b"}]
    b = collate(rows, pad_id=7)
    assert b["input_ids"].shape == (2, 3) and b["input_ids"][1, 2] == 7      # ids padded with pad_id
    assert b["labels"][1, 2] == -100                                          # labels padded with -100
    assert b["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]             # mask built automatically
    assert b["pixel_values"].shape == (2, 3, 4, 4) and torch.is_tensor(b["score"])
    assert b["name"] == ["a", "b"]
    paired = collate([{"anchor": {"input_ids": torch.tensor([1, 2])}, "positive": {"input_ids": torch.tensor([3])}},
                      {"anchor": {"input_ids": torch.tensor([4])}, "positive": {"input_ids": torch.tensor([5, 6])}}])
    assert paired["anchor"]["input_ids"].shape == (2, 2)                      # nested pairs collate recursively


def test_processor_resolution_and_custom_contract():
    p = processors.resolve(model=bert())
    assert p.kind in ("custom", "none") and "input_ids" in p.produces
    with pytest.raises(NotImplementedError, match="no processor"):
        p({"text": "hi"})
    custom = processors.Processor(kind="custom", produces=("x",),
                                  encode=lambda rec: {"x": torch.tensor(rec["x"])})
    assert custom({"x": [1.0, 2.0]})["x"].tolist() == [1.0, 2.0]
    loader = make_loader([{"x": [1.0, 2.0]}, {"x": [3.0, 4.0]}], custom, batch_size=2, shuffle=False)
    assert next(iter(loader))["x"].shape == (2, 2)


def test_loss_is_derived_from_contracts_not_names():
    b = {"input_ids": IDS, "labels": torch.tensor([0, 1, 2, 0])}
    d = derive_loss(bert(), b)
    assert d["confidence"] == "high" and d["loss_fn"] is not None and "returned a loss" in d["why"]

    # no labels -> the contract decides: 2-D logits whose width is num_labels => class cross-entropy
    d2 = derive_loss(bert(), {"input_ids": IDS})
    assert "cross-entropy over classes" in d2["name"]

    # single continuous output => regression
    d3 = derive_loss(bert(num_labels=1), {"input_ids": IDS})
    assert "MSE" in d3["name"] and d3["confidence"] == "medium"

    # 3-D vocabulary logits => token cross-entropy
    lm = T.BertForMaskedLM(T.BertConfig(vocab_size=80, hidden_size=32, num_hidden_layers=1,
                                        num_attention_heads=2, intermediate_size=32))
    assert "token cross-entropy" in derive_loss(lm, {"input_ids": IDS})["name"]

    # embeddings without pairs => undetermined, and it says so instead of guessing
    d4 = derive_loss(T.BertModel(T.BertConfig(vocab_size=80, hidden_size=32, num_hidden_layers=1,
                                              num_attention_heads=2, intermediate_size=32)), {"input_ids": IDS})
    assert d4["loss_fn"] is None and d4["confidence"] == "low"

    # pairs + scalar => reward model; pairs + embeddings => contrastive
    assert "Bradley-Terry" in derive_loss(bert(num_labels=1), {"input_ids": IDS}, pairs=True)["name"]

    # a timestep in the signature => diffusion
    class Denoiser(nn.Module):
        def forward(self, x, timestep):
            return x * 1.0
    assert "diffusion" in derive_loss(Denoiser(), (torch.randn(2, 4), torch.randint(0, 9, (2,))))["name"]

    # a model whose output means nothing standard => honest refusal, not a guess
    class Odd(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(4, 7)
        def forward(self, x):
            return self.l(x)
    d5 = derive_loss(Odd(), (torch.randn(2, 4),))
    assert d5["loss_fn"] is None and "CUSTOM ENGINEERING REQUIRED" in d5["name"]


def test_contracts_report_what_is_missing():
    ic = input_contract(bert(), {"input_ids": IDS})
    assert "input_ids" in ic["accepts"] and ic["provided"]["input_ids"]["shape"] == (4, 8)
    oc = output_contract(bert(), {"input_ids": IDS})
    assert oc["last_dim_means"] == "classes" and oc["rank"] == 2 and not oc["has_loss"]


def test_pipeline_end_to_end_on_a_model_it_has_never_seen():
    class Invented(nn.Module):                      # two inputs, no config, no head, novel objective
        def __init__(self):
            super().__init__()
            self.a = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.GELU())
            self.b = nn.Linear(5, 8)
            self.head = nn.Linear(16, 4)
        def forward(self, grid, vec):
            return self.head(torch.cat([self.a(grid).mean((2, 3)), self.b(vec)], -1))

    grid, vec, y = torch.randn(6, 3, 8, 8), torch.randn(6, 5), torch.rand(6, 4)
    def quantile(model, batch):
        err = batch["y"] - model(batch["grid"], batch["vec"])
        return torch.maximum(0.5 * err, -0.5 * err).mean()

    p = UniversalPipeline(Invented(), (grid, vec), loss_fn=quantile)
    plan = p.plan()
    assert "custom" in plan["1_modality_architecture"]
    assert plan["4_loss"]["name"] == "CUSTOM ENGINEERING REQUIRED"     # honest: no objective is implied
    assert "loss_fn" in plan["missing"] and "processor.encode" in plan["missing"]
    info = p.adapt()
    assert info["wrapped"] and info["trainable_pct"] < 90
    s = p.smoke_test({"grid": grid, "vec": vec, "y": y}, steps=25)
    assert s["gradients_ok"] and s["unexplained_zero_grads"] == 0 and s["overfits_one_batch"]
    e = p.export("/tmp/uptest")
    assert e["merged_layers"] > 0 and e["merge_max_abs_diff"] < 1e-4
    assert "CUSTOM ENGINEERING REQUIRED" in p.serving_plan()
    assert p.report()["task"] == "custom"


def test_pipeline_degrades_when_lora_cannot_reach_the_weights():
    class OnlyPacked(nn.Module):                     # an LSTM cell: packed parameters, nothing to wrap
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(4, 4, batch_first=True)
        def forward(self, x):
            return self.rnn(x)[0]

    p = UniversalPipeline(OnlyPacked(), (torch.randn(2, 3, 4),), loss_fn=lambda m, b: m(b[0]).pow(2).mean())
    p.plan()
    info = p.adapt()
    assert info["trainable"] > 0 and "selective unfreezing" in info.get("degraded_to", "")
