# 12. Chat templates: ChatML, Llama-3, Mistral

:::why In one line
A model only understands conversations laid out exactly as in its own training. The chat template is that layout. Train and serve with the same one.
:::

## 12.1 Where the real template lives

Every instruct model ships its template in the tokenizer (`tokenizer_config.json` or `chat_template.jinja`). `tok.apply_chat_template(messages)` uses it; so do TRL during training and vLLM during serving. **Rule: never hand-write prompts for an instruct model; always apply its template.** Print it once: `print(tok.chat_template)`.

## 12.2 The three families

```text Turn markup, end-of-turn token, where you see it
ChatML    <|im_start|>role\n ... <|im_end|>                       end: <|im_end|>   Qwen, SmolLM, many others
Llama-3   <|start_header_id|>role<|end_header_id|>\n\n ... <|eot_id|>   end: <|eot_id|>   Meta Llama 3.x
Mistral   <s>[INST] user [/INST] answer</s>                     end: </s>         Mistral/Mixtral instruct (v0.x style)
          Newer Mistral releases add system-prompt tokens and differ: always check the tokenizer.
```

`src/peftkit/data/templates.py` holds reference versions of these three with `{% generation %}` markers (explained in chapter 13). They are for learning, debugging and base models without a template; they are reproduced from the public formats, so compare them with the real tokenizer of your model before relying on them.

@file src/peftkit/data/templates.py

## 12.3 The same conversation, three layouts

@lab lab8_chat_templates

Note: the tiny test tokenizer does not know Llama-3's special tokens, so it splits them into many pieces (297 tokens vs 91 for ChatML). A real Llama-3 tokenizer encodes each as one token.

## 12.4 Traps

- **Base models have no template.** `modeling.py` installs a ChatML fallback and logs a warning. Prefer `*-Instruct` checkpoints; if you use a base model, the special tokens of the fallback are plain text to it, so it must learn them (and vLLM must be told the same template).
- **No pad token** (Llama-3): the template sets pad = eos with a warning. If your tokenizer has a reserved pad token, prefer it. Whatever you choose, check with `mask_report` that the end-of-turn token is inside the trained text; if it is masked away, the model never learns to stop.
- **Double BOS**: some templates already add `<|begin_of_text|>`/`<s>`; tokenising the rendered text again with `add_special_tokens=True` adds a second one. `apply_chat_template(tokenize=True)` avoids this.
- **System prompt not supported** (older Mistral): it is merged into the first user turn.
- **Train/serve mismatch**: if you trained with a custom template, pass the same one to vLLM (`--chat-template path.jinja`) or the merged tokenizer must contain it (our merge saves the tokenizer, including `chat_template.jinja`).

:::check
- I printed my model's `tok.chat_template`.
- I rendered one training row with it and it looks right.
:::
