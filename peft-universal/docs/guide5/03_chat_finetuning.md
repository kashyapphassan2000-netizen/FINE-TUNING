# 3. Chat and conversational fine-tuning

:::why In one line
A chat model must use the whole conversation, keep a persona from the system prompt, and end its turn cleanly. Single-turn data teaches none of those.
:::

## 3.1 How multi-turn training works

One example is one **conversation**. Every assistant turn is a training target; every system and user turn is masked. With a `{% generation %}` template, `assistant_only_loss` handles this automatically for any number of turns.

@file labs/lab52_chat_finetuning.py
@lab lab52_chat_finetuning

The measurement: turn 2 ("and next week?") needs both the history and an escalation rule that single-turn data never shows. Trained on single-turn data the model scores **0.33** on turn 2 (chance 0.25) while scoring 1.0 on turn 1. Trained with multi-turn examples it scores **1.0** on both. The lab also prints the token mask for a two-turn example, so you can see exactly which tokens carry loss.

## 3.2 What to put in chat training data

|Include|Why|
|---|---|
|conversations of varying length (2, 4, 8 turns)|if you train on 2 and serve 10, later turns are out of distribution|
|varied system prompts|otherwise the model only behaves with the exact string you trained on|
|topic switches mid-conversation|real users do it|
|clarifying questions from the assistant|otherwise it guesses instead of asking|
|refusals and "not in the documents"|otherwise it invents|
|corrections ("no, I meant the valve")|teaches recovery|
|the end-of-turn token, trained|otherwise it never stops (Volume 2, chapter 3)|

## 3.3 Traps specific to chat SFT

- **Template drift**: training with one chat template and serving with another silently degrades everything. Serve the tokenizer you trained with.
- **System prompt baked in**: training every example with the same system prompt makes the model dependent on it. Vary it; include the empty case.
- **Only final turns trained**: some pipelines train only the last assistant turn and mask earlier ones. That wastes most of the signal in a long conversation; train all assistant turns unless you have a reason.
- **Long conversations truncated from the left**: check what your `max_seq_length` cuts. Losing the system prompt at turn 9 changes behaviour.
- **Persona leakage**: if the data contains the assistant referring to its own instructions, the model will repeat them to users.

:::check
- My data contains conversations, not just pairs.
- System prompts vary; some examples have none.
- Every assistant turn is trained; the end-of-turn token is inside the trained span.
:::
