# 13. Loss masking: why it matters, what breaks without it

:::why In one line
Compute the loss only on the assistant's tokens. Otherwise the model spends its capacity learning to predict the system prompt and the user's questions, which it will never have to generate.
:::

## 13.1 The math

Cross-entropy over tokens t with a mask m_{t} ∈ {0, 1}:

$$ L = − Σ_{t} m_{t} · log p(x_{t} | x_{<t}) / Σ_{t} m_{t}

Without masking, m_{t} = 1 everywhere. With assistant-only loss, m_{t} = 1 only on assistant tokens (and their end-of-turn token). Masked tokens still go through the model as context; they just do not create gradient.

## 13.2 How the mask is built

The template wraps assistant text in `{% generation %} ... {% endgeneration %}`. `apply_chat_template(..., return_assistant_tokens_mask=True)` then returns `assistant_masks`. In TRL 1.13, `SFTConfig(assistant_only_loss=True)` uses it; TRL raises an error if the template has no generation markers (read from `sft_trainer.py`). In this template: `data.assistant_only_loss: true`.

TRL also has `completion_only_loss` for prompt-completion datasets.

## 13.3 What breaks without it (measured)

Same tiny model, same 48 rows (long fixed system prompt, short answers), same 12 steps, only the mask differs:

@file labs/lab9_loss_masking.py
@lab lab9_loss_masking

The masked run learned the answers much better (answer NLL 5.05 vs 5.80, untrained 5.99). The unmasked run spent most of its gradient on the repeated system prompt. Also note the two "train loss" values: they are averages over different tokens, so they cannot be compared.

Other failures you will see in real runs without masking:

- the model continues by writing the next *user* turn (it learned to produce user text);
- it repeats the system prompt;
- with long prompts and short answers, answer quality barely improves;
- long retrieved contexts get memorised (privacy risk).

## 13.4 When not to mask

- Continued pretraining on raw text: every token matters.
- Very small, answer-heavy datasets: masking changes little.
- If your template has no generation markers and you cannot add them, use prompt-completion data with `completion_only_loss`.

:::check
- My model's template has generation markers, or I added them.
- `mask_report` shows only assistant text + end-of-turn token as trained.
- `data.assistant_only_loss: true` in my run YAML (if the markers exist).
:::
