"""Shared machinery for the SFT labs (50-54): a tiny model, a tiny tokenizer, three toy tasks,
and a robust evaluator that scores candidate answers by likelihood instead of free generation.

Why likelihood scoring: a 200k-parameter randomly initialised model generates noise, so free-form
generation would measure nothing. Scoring the fixed candidate answers isolates what SFT actually
changed. Everything here runs on CPU in seconds.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import (AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM,
                          PreTrainedTokenizerFast)

CHATML = (
    "{%- for m in messages %}{{- '<|im_start|>' + m['role'] + '\n' }}"
    "{%- if m['role'] == 'assistant' %}{% generation %}{{- m['content'] + '<|im_end|>' }}{% endgeneration %}{{- '\n' }}"
    "{%- else %}{{- m['content'] + '<|im_end|>\n' }}{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}"
)

WORDS = ("pump valve sensor motor seal filter bearing impeller gauge pipe "
         "works fails leaks jams overheats runs smoothly badly loudly quietly "
         "good bad fine broken ok yes no positive negative one two three four five six seven eight nine "
         "answer is the a an of for with and or not step then so because sum total "
         "report ticket status severity refund warranty service repair replace check")


def build_model(path: Path, corpus: list[str], hidden: int = 96, layers: int = 3) -> Path:
    """A tiny Llama plus a tokenizer trained on THIS corpus (so the words are single tokens)."""
    path.mkdir(parents=True, exist_ok=True)
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    specials = ["<unk>", "<pad>", "<|im_start|>", "<|im_end|>"]
    tk.train_from_iterator(list(corpus) + [WORDS] * 20,
                           trainers.BpeTrainer(vocab_size=900, special_tokens=specials,
                                               initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>",
                                  bos_token="<|im_start|>", eos_token="<|im_end|>")
    tok.chat_template = CHATML
    tok.save_pretrained(path)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=hidden, intermediate_size=hidden * 2,
                      num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=512, pad_token_id=tok.pad_token_id,
                      bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
                      tie_word_embeddings=False)
    torch.manual_seed(0)
    LlamaForCausalLM(cfg).save_pretrained(path)
    return path


def load(path) -> tuple:
    model = AutoModelForCausalLM.from_pretrained(path).eval()
    return model, AutoTokenizer.from_pretrained(path)


def load_adapter(base, adapter) -> tuple:
    from peft import PeftModel

    model, tok = load(base)
    return PeftModel.from_pretrained(model, adapter).eval(), tok


def answer_nll(model, tok, messages: list[dict], answer: str) -> float:
    """Mean negative log-likelihood of `answer` as the assistant turn. Lower = the model prefers it."""
    full = messages + [{"role": "assistant", "content": answer}]
    enc = tok.apply_chat_template(full, tokenize=True, return_dict=True, return_tensors="pt",
                                  return_assistant_tokens_mask=True)
    ids = enc["input_ids"]
    mask = enc["assistant_masks"][0, 1:].bool()
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, :-1]
    return float(torch.nn.functional.cross_entropy(logits[mask], ids[0, 1:][mask]))


def choose(model, tok, messages: list[dict], candidates: list[str]) -> str:
    return min(candidates, key=lambda c: answer_nll(model, tok, messages, c))


def accuracy(model, tok, items: list[dict]) -> float:
    """items: {"messages": [...], "answer": str, "candidates": [...]}"""
    hits = sum(choose(model, tok, it["messages"], it["candidates"]) == it["answer"] for it in items)
    return round(hits / max(1, len(items)), 4)


def write_jsonl(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def sft(model_dir, train_file, out_dir, epochs=12, lr=3e-3, r=16, eval_file=None, seed=0,
        assistant_only=True, bs=16):
    """One SFT run with the repository's own trainer (LoRA, assistant-only loss)."""
    import logging

    from peftkit.config import load_config
    from peftkit.train import run

    logging.disable(logging.WARNING)
    ROOT = Path(__file__).resolve().parents[1]
    cfg = load_config(str(ROOT / "configs/base.yaml"), str(ROOT / "configs/methods/lora.yaml"), overrides=dict(
        model={"name_or_path": str(model_dir), "max_seq_length": 192},
        method={"r": r, "target_modules": "all-linear"},
        data={"train_file": str(train_file), "eval_file": str(eval_file) if eval_file else None,
              "assistant_only_loss": assistant_only},
        train={"output_dir": str(out_dir), "epochs": epochs, "per_device_batch_size": bs, "grad_accum": 1,
               "lr": lr, "logging_steps": 50, "eval_steps": 10000, "save_steps": 100000, "seed": seed}))
    meta = run(cfg)
    return Path(out_dir) / "adapter", meta
