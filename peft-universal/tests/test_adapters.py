import asyncio
import json

import httpx
import pytest
import torch
import yaml
from fastapi import FastAPI
from peft import LoraConfig, get_peft_model
from peftkit.adapters import Registry, load_many, route, runtime_load, runtime_unload, validate, vllm_args
from transformers import AutoModelForCausalLM, AutoTokenizer


def _make_adapter(base_dir, out, seed, r=4):
    torch.manual_seed(seed)
    m = get_peft_model(AutoModelForCausalLM.from_pretrained(base_dir),
                       LoraConfig(r=r, target_modules=["q_proj", "v_proj"], init_lora_weights=False))
    m.save_pretrained(out)
    return out


@pytest.fixture(scope="module")
def two_adapters(tiny_model, tmp_path_factory):
    d = tmp_path_factory.mktemp("adapters")
    return {"finance": _make_adapter(tiny_model, d / "finance", 1), "legal": _make_adapter(tiny_model, d / "legal", 2)}


def _logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids).logits


def test_multi_adapter_switch_mixed_batch_and_weighted(tiny_model, two_adapters):
    tok = AutoTokenizer.from_pretrained(tiny_model)
    ids = tok("What is the maximum pressure?", return_tensors="pt")["input_ids"]
    base = AutoModelForCausalLM.from_pretrained(tiny_model).eval()
    ref_base = _logits(base, ids)
    from peft import PeftModel

    refs = {n: _logits(PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(tiny_model), p).eval(), ids)
            for n, p in two_adapters.items()}
    model = load_many(AutoModelForCausalLM.from_pretrained(tiny_model), two_adapters).eval()
    for name in ("finance", "legal"):  # switching = identical to loading that adapter alone
        model.set_adapter(name)
        assert torch.allclose(_logits(model, ids), refs[name], atol=1e-5)
    assert not torch.allclose(refs["finance"], refs["legal"])
    with model.disable_adapter():      # base model through the same object
        assert torch.allclose(_logits(model, ids), ref_base, atol=1e-5)
    batch = ids.repeat(2, 1)           # one batch, different adapter per row
    with torch.no_grad():
        mixed = model(input_ids=batch, adapter_names=["finance", "legal"]).logits
    assert torch.allclose(mixed[0], refs["finance"][0], atol=1e-5)
    assert torch.allclose(mixed[1], refs["legal"][0], atol=1e-5)
    model.add_weighted_adapter(["finance", "legal"], [0.5, 0.5], "fin_legal", combination_type="cat")
    model.set_adapter("fin_legal")
    assert _logits(model, ids).shape == ref_base.shape
    model.delete_adapter("legal")
    assert "legal" not in model.peft_config


def _registry(tmp_path, adapters, base):
    reg = {"base": {"name": base, "revision": None}, "adapters": adapters}
    p = tmp_path / "reg.yaml"
    p.write_text(yaml.safe_dump(reg))
    return Registry.load(p)


def test_registry_validate_args_route(tmp_path, tiny_model, two_adapters):
    good = [{"name": "finance", "source": str(two_adapters["finance"]), "rank": 4, "keywords": ["invoice", "tax"]},
            {"name": "legal", "source": str(two_adapters["legal"]), "rank": 4, "keywords": ["contract"]},
            {"name": "old", "source": "x/y", "rank": 64, "status": "retired"}]
    reg = _registry(tmp_path, good, str(tiny_model))
    assert validate(reg) == []
    args = vllm_args(reg)
    assert args[args.index("--max-lora-rank") + 1] == "8"      # smallest allowed vLLM rank >= 4
    assert f"finance={two_adapters['finance']}" in args and not any(a.startswith("old=") for a in args)
    assert route(reg, "Please check this invoice and the tax line") == "finance"
    assert route(reg, "review the contract") == "legal"
    assert route(reg, "hello there") == "base"
    assert route(reg, "anything", requested="legal") == "legal"
    with pytest.raises(KeyError):
        route(reg, "x", requested="old")
    bad = good + [{"name": "Bad Name", "source": "a", "method": "dora", "rank": 1024},
                  {"name": "finance", "source": str(two_adapters["finance"])}]  # dup name, default rank 16
    errs = validate(_registry(tmp_path, bad, "some/other-base"))
    text = "\n".join(errs)
    for needle in ("name must match", "cannot be hot-swapped", "rank 1024", "duplicate name", "!= registry base",
                   "registry rank 16 != adapter r 4"):
        assert needle in text, needle


def test_runtime_load_unload_against_fake_vllm():
    app = FastAPI()
    calls = []

    @app.post("/v1/load_lora_adapter")
    async def load(body: dict):
        calls.append(("load", body))
        return {"ok": True}

    @app.post("/v1/unload_lora_adapter")
    async def unload(body: dict):
        calls.append(("unload", body))
        return {"ok": True}

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://v") as c:
            a = await runtime_load("http://v", "k", "finance", "user/fin", client=c)
            b = await runtime_unload("http://v", "k", "finance", client=c)
        return a, b

    a, b = asyncio.run(go())
    assert a["status"] == b["status"] == 200
    assert calls == [("load", {"lora_name": "finance", "lora_path": "user/fin"}), ("unload", {"lora_name": "finance"})]
    assert json.dumps(calls)
