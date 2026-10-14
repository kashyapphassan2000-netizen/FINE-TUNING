# 49. Tools: distilabel, DataDreamer, Argilla

The template's own modules are small and fully tested so you understand every step. For large or team projects these tools add caching, scaling, and human review.

|Tool|Version checked|What it gives you|Env|
|---|---|---|---|
|distilabel|1.5.3|pipelines of steps (generation, rating, dedup, formatting) with caching and batching; built-in tasks such as `SelfInstruct`, `EvolInstruct`, `Magpie`, `UltraFeedback`, `PrometheusEval`, `MinHashDedup`|loose pins; own env recommended|
|DataDreamer|0.46.0|reproducible prompt-based data steps with caching (`DataFromPrompt`, `JudgePairsWithPrompt`, `FilterWithPrompt`, ...)|**must be separate**: pins datasets<4, hub<1.0, numpy<2|
|Argilla|2.8.0|web UI for human review and labelling; Python SDK|server + SDK|

:::honest What was checked
Class names, constructor arguments and step input/output columns were read from the three wheels. The example scripts were compiled but not run (they need live LLM endpoints and an Argilla server). Verify arguments in each tool's docs before running.
:::

## 49.1 distilabel pipeline (SFT + DPO)

@file examples/distilabel_pipeline.py

Verified from source: `SelfInstruct` outputs `instructions` (a list), so `ExpandColumns` turns it into rows; `TextGeneration` outputs `generation`; `UltraFeedback` needs `instruction` + `generations` and outputs `ratings`/`rationales`; `FormatTextGenerationDPO` needs `instruction`, `generations`, `ratings`.

```any Install in its own env
uv venv .venv-data --python 3.11
uv pip install --python .venv-data/bin/python "distilabel[openai]==1.5.3" argilla==2.8.0
```

## 49.2 DataDreamer

@file examples/datadreamer_example.py

## 49.3 Argilla: humans in the loop

Run an Argilla server (their docs offer a Hugging Face Space template and Docker), create an API key in its UI, then push rows for review:

@file src/peftkit/data/argilla_export.py

```any
ARGILLA_API_URL=https://<your-space>.hf.space ARGILLA_API_KEY=... \
python -m peftkit.data.argilla_export --input data/interim/judged.jsonl --dataset pump-sft-review
```

Use the reviewed labels to (1) measure judge agreement, (2) build a clean eval set that never enters training.
