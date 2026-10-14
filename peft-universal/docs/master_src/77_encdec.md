# 77. Encoder-decoder models (T5, FLAN-T5, BART, Whisper-style)

:::warn NOT ESTABLISHED BY THE PROVIDED PDF
Derived from the architecture; exercised in lab 60 (T5 and Whisper rows) and lab 61 (case 3).
:::

## 77.1 The flow

```text
encoder input (text ids / audio features)
   -> ENCODER (bidirectional)        -> encoder hidden states
                                            │
decoder input = labels shifted right  ->  DECODER (causal self-attention + CROSS-ATTENTION to the encoder)
   -> LM head -> logits -> CrossEntropy against labels
```

- **labels**: the target sequence. Pad positions set to -100.
- **decoder_input_ids**: the labels shifted right with a start token. Most Hugging Face seq2seq models build them from `labels` automatically — which is why calling the model with neither fails (lab 61 case 3: `ValueError: You must specify exactly one of input_ids or inputs_embeds`).
- **teacher forcing**: during training the decoder always sees the true previous token, not its own prediction.
- **cross-attention**: the decoder's queries attend to the encoder's keys/values — an important LoRA target for adapting how the output uses the input.

## 77.2 Code

```py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType
tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=16, lora_alpha=32,
                                         target_modules=["q", "k", "v", "o", "wi", "wo"]))
enc = tok(["summarise: the pump leaked twice and was replaced"], return_tensors="pt")
labels = tok(["pump replaced after two leaks"], return_tensors="pt").input_ids
labels[labels == tok.pad_token_id] = -100
loss = model(**enc, labels=labels).loss
```

Lab 60: T5 loss 5.286 → 1.806, Whisper 4.673 → 4.091, both merges exact; discovery found T5's `q/k/v/o/wi/wo` by computation.

## 77.3 Traps

- Forgetting `-100` on label padding: the model learns to emit padding.
- Evaluating with teacher-forced loss only: generate with `model.generate` and score the output.
- Different max lengths for source and target: set both.
- Serving: generation with an encoder pass per request; check your engine supports the architecture before assuming vLLM (chapter 90).
