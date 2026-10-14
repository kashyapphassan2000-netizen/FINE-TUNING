"""Builds a tiny random Llama + tokenizer on disk. No network needed."""
from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from peftkit.modeling import CHATML_TEMPLATE

CORPUS = [
    "The pump pressure must stay below 6 bar during calibration.",
    "Reset the controller by holding the red button for five seconds.",
    "Warranty claims require the serial number and the purchase date.",
    "What is the maximum pressure? The maximum pressure is 6 bar.",
    "How do I reset the controller? Hold the red button for five seconds.",
] * 20


def build_tiny(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    specials = ["<unk>", "<pad>", "<|im_start|>", "<|im_end|>"]
    tk.train_from_iterator(CORPUS, trainers.BpeTrainer(vocab_size=400, special_tokens=specials,
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>",
                                  bos_token="<|im_start|>", eos_token="<|im_end|>")
    tok.chat_template = CHATML_TEMPLATE
    tok.save_pretrained(path)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
                      tie_word_embeddings=False)
    LlamaForCausalLM(cfg).save_pretrained(path)
    return path
