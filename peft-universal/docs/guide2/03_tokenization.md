# 3. Tokenization: text → input_ids → attention_mask → labels

:::why In one line
The model never sees your text. It sees integers. Almost every silent fine-tuning failure is a mistake in this conversion.
:::

## 3.1 The pipeline

```text
"Hold the red button."  ──tokenizer──>  input_ids  [12, 883, 41, ...]
                                        attention_mask [1, 1, 1, ...]   1 = real token, 0 = padding
                                        labels     [-100, -100, 41, ...] -100 = do not train on this token
```

- **BPE / SentencePiece / Unigram** are algorithms that learn a vocabulary of frequent pieces. Common words become one token; rare words split into several. This is why "hydrodynamics" costs more tokens than "pump", and why token counts (and therefore cost) differ per language.
- **Special tokens**: BOS (beginning), EOS (end of text / end of turn), PAD (filler so a batch is rectangular), plus chat-role markers.
- **Padding** makes all rows in a batch the same length; `attention_mask = 0` tells the model to ignore those positions.
- **Truncation** cuts sequences longer than `max_length` — **from the end**, which is where your answer lives.

## 3.2 See it for yourself

@file labs/lab13_tokenization_masking.py
@lab lab13_tokenization_masking

Three things to notice in that output:

1. A rare word is split into several pieces. Your "average tokens per example" is not "words × 1.3" in every domain; measure it (Volume 1, ch 35.1).
2. In the padded batch, the shorter row is filled with the pad id and its `attention_mask` is 0 there.
3. With `max_length=8` the sentence is cut mid-way. If that had been a training example, the model would have been taught to stop mid-sentence.

## 3.3 The label column is the whole game

In the chat-template section of the lab, every token carries a label:

- `-100` means "compute no loss here". PyTorch's cross-entropy ignores it.
- A real token id means "predicting this token is the model's job".

For SFT you want labels only on the assistant's answer and its end-of-turn token. That is loss masking (Volume 1, ch 13, and chapter 4 here).

:::warn The three tokenizer traps that cost people days
- **No pad token.** Many models set `pad = eos`. If the collator then masks pad positions, it can mask your real EOS too, and the model never learns to stop. Verify with `tokenize_inspect` that EOS is in the trained span.
- **Double BOS.** `apply_chat_template` already adds BOS; tokenizing the rendered string again with `add_special_tokens=True` adds a second one. Always tokenize through the template.
- **Tokenizer/model mismatch.** Loading tokenizer A with model B produces fluent nonsense. Load both from the same repo and revision.
:::

```any Run it on your own data before training
python -m peftkit.tokenize_inspect --tokenizer <your model> --file data/processed/train.jsonl --index 0 --max-length 2048
```

@file src/peftkit/tokenize_inspect.py

:::check
- I ran `tokenize_inspect` on my own dataset.
- Trained fraction is sensible (not 0%, not 100%), and EOS is trained.
- p95 token length ≤ `model.max_seq_length`.
:::
