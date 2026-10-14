# 4. Chain-of-thought fine-tuning

:::why In one line
Put the reasoning steps in the assistant target. The model learns to produce them, and conditions its final answer on its own steps.
:::

## 4.1 The data shape

```json answer only
{"messages": [{"role": "user", "content": "sum two three four"},
              {"role": "assistant", "content": "answer nine"}]}
```

```json chain of thought
{"messages": [{"role": "user", "content": "sum two three four"},
              {"role": "assistant", "content": "step two and three is five then five and four is nine answer nine"}]}
```

Nothing else changes: same trainer, same masking, same loss. The difference is entirely in what you put in the target.

## 4.2 The mechanics and the cost, measured

@file labs/lab53_cot_finetuning.py
@lab lab53_cot_finetuning

Read the honesty note in that output. The lab demonstrates the **mechanics** (how CoT targets are built and scored) and the **cost** (CoT answers are several times longer, so training tokens and generation cost scale with them). It is a 3-layer random model on 18 examples: it is not evidence about whether CoT improves quality. Reproduce that claim on a real pretrained model before repeating it.

## 4.3 What is true regardless of scale

|Fact|Consequence|
|---|---|
|CoT targets are 3-5× longer|3-5× training tokens, 3-5× output tokens per request, higher latency|
|the traces must be correct|training on wrong reasoning that lands on the right answer teaches the model to fake reasoning|
|traces can leak|answer keys, internal policies and PII end up in visible output unless you filter|
|not every task needs it|classification and extraction rarely benefit; multi-step arithmetic, planning and multi-hop questions may|

## 4.4 How teams actually build CoT data

1. **Distil**: ask a stronger model for step-by-step solutions to your tasks.
2. **Verify, then filter**: keep only traces whose final answer is correct (rejection sampling). This is the single most important step — unverified traces are a liability.
3. **Deduplicate** reasoning patterns so the model does not learn one template.
4. **Decide what the user sees**: train with traces but serve a summary, or train a short "reasoning budget" style, or keep traces internal.
5. **Measure both**: final-answer accuracy AND cost per answer. A 2-point accuracy gain for 4× the tokens is a business decision, not an obvious win.

:::warn Licences again
Distilling reasoning traces from a commercial model may be restricted by that provider's terms. Check before you build a product on it (Volume 4, chapter 5).
:::

:::check
- I know whether my task is multi-step enough to need CoT.
- My traces are verified against the correct final answer.
- I measured tokens per answer before and after.
:::
