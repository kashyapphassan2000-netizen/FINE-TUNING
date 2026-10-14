# 40. Dataset formats: instruction pairs, chat, preference

## 40.1 Formats you will meet

```json Alpaca-style instruction pair
{"instruction": "Summarise the warranty rules.", "input": "<policy text>", "output": "Claims need ..."}
```

```json Chat / messages (the format this template trains on)
{"messages": [{"role": "system", "content": "You are a pump support assistant."},
              {"role": "user", "content": "Max pressure?"},
              {"role": "assistant", "content": "6 bar."}]}
```

```json ShareGPT (common on the Hub)
{"conversations": [{"from": "human", "value": "Max pressure?"}, {"from": "gpt", "value": "6 bar."}]}
```

```json Prompt-completion
{"prompt": "Max pressure?", "completion": "6 bar."}
```

```json Preference (TRL conversational format, for DPO/ORPO)
{"prompt": [{"role": "user", "content": "Max pressure?"}],
 "chosen": [{"role": "assistant", "content": "6 bar."}],
 "rejected": [{"role": "assistant", "content": "About 20 bar, probably."}]}
```

:::why Why normalise everything to messages
One format means one code path through templating, masking, filtering and training, and multi-turn works for free. `format.py` converts all the shapes above automatically, row by row.
:::

## 40.2 The formatter

@file src/peftkit/data/format.py

Validation rules (each failure is counted, not silently dropped): known roles only; optional system first; then user, alternating; must end with assistant; no empty content. Length is counted with the real tokenizer and chat template when `--tokenizer` is given.

:::warn Bug found while testing (fixed)
In transformers 5.x, `tokenizer.apply_chat_template(..., tokenize=True)` returns a `BatchEncoding` (a dict), so `len()` returned 2 and the length filter never removed anything. The code now asks for `return_dict=True` and counts `input_ids`. Old tutorials have this bug too.
:::

```any Run it
python -m peftkit.data.format --inputs data/interim/a.jsonl data/interim/b.jsonl \
   --out-dir data/processed --eval-ratio 0.05 --max-tokens 4096 \
   --tokenizer meta-llama/Llama-3.1-8B-Instruct --system-prompt "You are a pump support assistant."
```

```out
{"read": 12000, "bad_shape": 3, "invalid": {"must end with assistant": 12}, "dupes": 410, "too_long": 57, "train": 10939, "eval": 575}
```

(Numbers are illustrative of the shape of the report, not a measurement.)

:::check
- My data is in `data/processed/train.jsonl` and `eval.jsonl`, messages format.
- I read the report and understand every non-zero counter.
:::
