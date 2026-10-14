"""The task abstraction: registry, generic losses, task metrics, and dataset lineage."""
import json

import pytest
import torch
import transformers as T
from peftkit.data import lineage
from torch import nn
from uadapt import metrics as M
from uadapt import tasks as K

T.logging.set_verbosity_error()


def test_registry_covers_the_matrix():
    assert len(K.TASKS) >= 18
    mods = {t.modality for t in K.TASKS.values()}
    assert {"text", "image", "audio", "image+text", "video", "text->audio", "any"} <= mods
    for name in ("causal_lm", "masked_lm", "seq2seq", "sequence_classification", "token_classification",
                 "regression", "embedding", "reranking", "reward_model", "preference", "multimodal_sft",
                 "asr_ctc", "asr_seq2seq", "audio_classification", "image_classification",
                 "image_generation", "tts", "video_classification", "custom"):
        t = K.TASKS[name]
        assert t.data_schema and t.labels and t.loss and t.metrics, name
        assert "TASK " in t.card()


def test_resolution_depends_on_architecture_not_habit():
    enc = [t.name for t in K.resolve(["encoder-only", "text"])]
    dec = [t.name for t in K.resolve(["decoder-only", "text"])]
    assert "sequence_classification" in enc and "causal_lm" not in enc
    assert dec[0] == "causal_lm" and "reward_model" in dec
    assert [t.name for t in K.resolve(["diffusion"])] == ["image_generation"]
    assert [t.name for t in K.resolve(["nonsense"])] == ["custom"]
    p = K.plan(["audio"], "asr_ctc")
    assert p["chosen_task"] == "asr_ctc" and p["loss_fn_available"]
    assert "WER" in p["metrics"] and "frames" in p["card"]
    tts = K.plan(["encoder-decoder"], "tts")                 # honestly marked as needing your own loss
    assert not tts["loss_fn_available"] and tts["custom_code_needed"]


def test_generic_losses_train_their_objective():
    torch.manual_seed(0)
    cfg = T.BertConfig(vocab_size=60, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                       intermediate_size=32, num_labels=1)
    ids = torch.randint(0, 60, (4, 6))
    # contrastive: identical anchor/positive must score better than random pairing
    enc = T.BertModel(cfg)
    pair = {"anchor": {"input_ids": ids}, "positive": {"input_ids": ids}}
    before_c = float(K.contrastive_loss(enc, pair))
    opt_c = torch.optim.AdamW(enc.parameters(), lr=5e-3)
    for _ in range(20):
        lc = K.contrastive_loss(enc, pair)
        lc.backward(); opt_c.step(); opt_c.zero_grad()
    assert float(K.contrastive_loss(enc, pair)) < before_c        # in-batch negatives are learnable
    # reward model: loss falls when chosen is pushed above rejected
    rm = T.BertForSequenceClassification(cfg)
    b = {"chosen": {"input_ids": ids}, "rejected": {"input_ids": torch.randint(0, 60, (4, 6))}}
    before = float(K.reward_model_loss(rm, b))
    opt = torch.optim.AdamW(rm.parameters(), lr=1e-2)
    for _ in range(20):
        loss = K.reward_model_loss(rm, b)
        loss.backward(); opt.step(); opt.zero_grad()
    assert float(K.reward_model_loss(rm, b)) < before
    # diffusion: predicting the noise is learnable
    net = nn.Sequential(nn.Linear(8, 8))
    model = lambda x, t: net(x)                                     # noqa: E731
    batch = {"x0": torch.randn(16, 8), "t": torch.randint(0, 1000, (16,)), "noise": torch.randn(16, 8)}
    d0 = float(K.diffusion_loss(model, batch))
    assert d0 > 0 and torch.isfinite(torch.tensor(d0))
    with pytest.raises(RuntimeError, match="no loss"):
        K.hf_loss(T.BertModel(cfg), {"input_ids": ids})


def test_task_metrics():
    assert M.accuracy([1, 2, 3], [1, 2, 4]) == pytest.approx(2 / 3)
    r = M.prf([1, 1, 0, 0], [1, 0, 0, 0])
    assert r["per_class"][1]["precision"] == 0.5 and r["per_class"][1]["recall"] == 1.0
    assert M.auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert M.auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert M.ece([1.0, 1.0], [1, 0]) == 0.5                          # perfectly confident, half wrong
    ranked = [[3, 1, 2], [1, 2, 3]]
    assert M.recall_at_k(ranked, [3, 2], 1) == 0.5 and M.mrr(ranked, [3, 2]) == pytest.approx(0.75)
    assert M.ndcg_at_k([[1, 2]], [{1: 1.0, 2: 1.0}], 2) == 1.0
    assert M.wer(["the pump leaks"], ["the pump works"]) == pytest.approx(1 / 3, abs=1e-3)
    assert M.cer(["abc"], ["abd"]) == pytest.approx(1 / 3, abs=1e-3)
    assert M.format_compliance(['{"a":1}', "nope"], lambda s: s.startswith("{")) == 0.5
    assert set(M.TASK_METRICS) >= set(K.TASKS)


def test_lineage_and_license_controls(tmp_path):
    rows = lineage.stamp([{"messages": [{"role": "user", "content": "hi"}]}], source_id="manual-1",
                         source_type="human", license="Apache-2.0", dataset_version="1.0")
    assert lineage.verify(rows)["ok"] and rows[0]["meta"]["record_hash"]
    assert lineage.verify([{"messages": []}])["missing_counts"]["license"] == 1
    mixed = rows + lineage.stamp([{"messages": []}], source_id="alpaca", source_type="synthetic",
                                 license="CC-BY-NC-4.0", dataset_version="1.0", generator_model="some-llm")
    chk = lineage.check_licenses(mixed, intended_use="commercial")
    assert chk["decision"] == "BLOCK" and any("not usable commercially" in b for b in chk["blockers"])
    assert any("GENERATOR" in w or "generator" in w for w in chk["warnings"])
    assert lineage.check_licenses(mixed, intended_use="research")["decision"] == "REVIEW"
    sa = lineage.stamp([{"x": 1}], source_id="osm", source_type="scraped", license="ODbL", dataset_version="1")
    assert any("share-alike" in w for w in lineage.check_licenses(sa)["warnings"])
    assert lineage.summarise(mixed)["synthetic_share"] == 0.5
    assert "not legal advice" in lineage.check_licenses(rows)["disclaimer"]
