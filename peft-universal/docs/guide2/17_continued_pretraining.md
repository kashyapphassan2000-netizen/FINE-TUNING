# 17. Continued pretraining (CPT)

:::why In one line
Keep doing plain next-token prediction on raw domain text. No chat template, no masking, every token counts. It teaches vocabulary and domain patterns, not instruction following.
:::

## 17.1 CPT vs SFT

|-|CPT|SFT|
|---|---|---|
|data|raw text (manuals, papers, code, tickets)|question → answer pairs|
|loss on|every token|assistant tokens only|
|typical size|10M-10B+ tokens|1k-100k examples|
|teaches|vocabulary, style of the field, facts as patterns|following instructions, output format, behaviour|
|risk|forgetting (high), expensive|overfitting on small data|
|after it|you still need SFT to get a usable assistant|-|

## 17.2 Measured

@file labs/lab22_continued_pretraining.py
@lab lab22_continued_pretraining

Loss on a held-out domain sentence drops after CPT on related raw text. That is what CPT does: it makes the domain's language less surprising to the model.

## 17.3 When it is worth it

**Use CPT when**: the domain has vocabulary the model has clearly never seen (specialised medical/legal/industrial terms, an internal codebase, a low-resource language), and you have millions of tokens of raw text. **Skip CPT when**: you have a few thousand QA pairs and want a behaviour change — that is SFT, and CPT on small text will mostly cause forgetting for no gain.

## 17.4 Practical notes

- With PEFT, CPT usually needs a **higher rank** than SFT (64-128 is common) because you are changing more of the model's behaviour; full fine-tuning is used when budget allows.
- Keep 5-30% general text mixed in (chapter 10) and measure a general suite before and after.
- Data prep is different: no chat template, no masking, long sequences, aggressive dedup (raw corpora are full of duplicates).
- Sequence packing matters more here than in SFT: raw text has no natural example boundaries.
- Evaluate CPT with perplexity on **held-out domain text**, not with task scores; task evaluation comes after SFT.

```yaml Sketch: CPT with this template's building blocks
# raw text -> jsonl of {"messages":[{"role":"user","content":""},{"role":"assistant","content":"<chunk>"}]}
# is a hack. For real CPT use a plain LM collator: TRL SFTTrainer with a `text` field and packing,
# assistant_only_loss: false, and a higher LoRA rank.
data: {packing: true, assistant_only_loss: false}
method: {name: lora, r: 64, alpha: 128, target_modules: all-linear}
train: {lr: 1.0e-4, epochs: 1}
```

:::check
- I have millions of tokens of raw domain text (otherwise: skip CPT).
- I measured held-out domain perplexity before and after.
- I measured a general suite for forgetting.
- SFT is planned after CPT; CPT alone does not produce an assistant.
:::
