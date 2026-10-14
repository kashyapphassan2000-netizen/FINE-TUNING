"""Chat templates and loss masks.

The OFFICIAL template of a model is the `chat_template` inside its tokenizer (tokenizer_config.json or
chat_template.jinja). Always train with that. The three templates below are reference renderings for
learning, debugging, and as a fallback for base models; they add {% generation %} markers so
`return_assistant_tokens_mask=True` can build an assistant-only loss mask.
Compare against the real tokenizer before relying on them: `print(tok.chat_template)`.
"""
from __future__ import annotations

import re

_GEN = re.compile(r"\{%-?\s*generation\s*-?%\}")


def has_generation_markers(template: str | None) -> bool:
    """True only for real {% generation %} blocks. A plain substring test is WRONG: every official template contains
    'add_generation_prompt', so `"generation" in template` is always true and assistant masks come back all zeros."""
    return bool(template and _GEN.search(template))

CHATML = (
    "{%- for m in messages %}{{- '<|im_start|>' + m['role'] + '\\n' }}"
    "{%- if m['role'] == 'assistant' %}{% generation %}{{- m['content'] + '<|im_end|>' }}{% endgeneration %}{{- '\\n' }}"
    "{%- else %}{{- m['content'] + '<|im_end|>\\n' }}{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)

LLAMA3 = (
    "{{- '<|begin_of_text|>' }}"
    "{%- for m in messages %}{{- '<|start_header_id|>' + m['role'] + '<|end_header_id|>\\n\\n' }}"
    "{%- if m['role'] == 'assistant' %}{% generation %}{{- m['content'] | trim + '<|eot_id|>' }}{% endgeneration %}"
    "{%- else %}{{- m['content'] | trim + '<|eot_id|>' }}{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{%- endif %}"
)

# Mistral-instruct style: no system role; system text is prepended to the first user turn.
MISTRAL = (
    "{{- '<s>' }}"
    "{%- set sys = messages[0]['content'] + '\\n\\n' if messages[0]['role'] == 'system' else '' %}"
    "{%- set loop_msgs = messages[1:] if messages[0]['role'] == 'system' else messages %}"
    "{%- for m in loop_msgs %}"
    "{%- if m['role'] == 'user' %}{{- '[INST] ' + (sys if loop.first else '') + m['content'] + ' [/INST]' }}"
    "{%- else %}{% generation %}{{- ' ' + m['content'] + '</s>' }}{% endgeneration %}{%- endif %}"
    "{%- endfor %}"
)

TEMPLATES = {"chatml": CHATML, "llama3": LLAMA3, "mistral": MISTRAL}


def render(tok, messages: list[dict], template: str | None = None) -> str:
    return tok.apply_chat_template(messages, tokenize=False, chat_template=TEMPLATES.get(template, template))


def mask_report(tok, messages: list[dict], template: str | None = None) -> dict:
    """Which tokens would be trained on (mask=1) under assistant-only loss."""
    enc = tok.apply_chat_template(messages, tokenize=True, return_dict=True, return_assistant_tokens_mask=True,
                                  chat_template=TEMPLATES.get(template, template))
    ids, mask = enc["input_ids"], enc["assistant_masks"]
    trained = tok.decode([i for i, m in zip(ids, mask) if m])
    return {"tokens": len(ids), "trained_tokens": int(sum(mask)), "trained_fraction": round(sum(mask) / len(ids), 3),
            "trained_text": trained}
