# 43. Chat templates and loss masking, in depth

chapter 41 and 13 cover the templates (ChatML, Llama-3, Mistral) and prove with a measured ablation that assistant-only loss learns the answers better. This chapter is the mechanical detail underneath.

## 43.1 From messages to labels, step by step

```text
messages = [system, user, assistant]
   │  apply_chat_template  (the model's OWN template, from its tokenizer)
   ▼
rendered text:  <|im_start|>system ... <|im_end|> <|im_start|>user ... <|im_end|> <|im_start|>assistant 6 bar.<|im_end|>
   │  tokenize
   ▼
input_ids:      [ ... ids ... ]
   │  {% generation %} markers mark the assistant span
   ▼
assistant_masks:[0,0,0,...,1,1,1,1]       ->      labels = id where mask==1 else -100
```

The loss is then

$$ L = − Σ_{t} m_{t} · log p(x_{t} | x_{<t}) / Σ_{t} m_{t}

Masked tokens are still **read** as context. They simply produce no gradient.

## 43.2 What each failure mode looks like

|Mask|Result|Symptom you will see|
|---|---|---|
|all ones (no masking)|trains on system + user + answer|model continues by writing the next user turn; repeats the system prompt; wastes capacity on the prompt|
|all `-100`|no gradient at all|loss is `nan` or flat; "training" finishes and nothing changed (ch 54, case 3)|
|answer only, EOS excluded|never learns to stop|generation runs until `max_new_tokens`, then cuts mid-sentence|
|answer only, EOS included|correct|loss falls, answers end cleanly|

## 43.3 Multi-turn conversations

With a `{% generation %}` template, **every** assistant turn is trained and every user turn is masked, in one example. That is why multi-turn data needs no special handling here, and why a template without markers forces you back to prompt/completion pairs with `completion_only_loss`.

## 43.4 Verify, do not assume

```any
python -m peftkit.tokenize_inspect --tokenizer <model> --file data/processed/train.jsonl --index 0
python -m peftkit.diagnostics --config configs/base.yaml configs/methods/lora.yaml configs/models/<m>.yaml
```

The doctor (chapter 54) fails the run if the template lacks markers while `assistant_only_loss: true`, if the trained fraction is 0% or ~100%, or if EOS is not inside the trained span.

:::check
- I can explain -100 to someone else.
- My template's markers are present, or I use completion-only loss deliberately.
- EOS trained: verified, not assumed.
:::
