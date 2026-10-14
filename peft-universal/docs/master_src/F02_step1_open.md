# 4. Step 1 — Is the model open, and may I use it?

:::why First question, always
Everything downstream is wasted if the weights are not downloadable or the licence forbids your use.
:::

@file src/peftkit/openness.py

```any
python -m peftkit.openness meta-llama/Llama-3.1-8B-Instruct --intended-use commercial
```

@lab lab64_flow_tools

## 4.1 The three separate questions

|Question|Why it matters|Where the answer is|
|---|---|---|
|Are the **weights** downloadable?|no weights = fine-tuning is IMPOSSIBLE (use the vendor's tuning service, or another model)|the repo's `*.safetensors` files|
|What does the **licence** allow?|commercial use, derivatives, redistribution, and duties that travel to your fine-tuned model|`LICENSE` + the model card|
|Is it **gated**?|you must accept terms with the account behind your `HF_TOKEN`, or downloads 401|the model page|

The tool classifies the licence into families and returns **PROCEED** or **STOP** with the blocker. From the lab output: an Apache-2.0 model proceeds; a community-licensed model proceeds **with duties** (acceptable-use policy, attribution, a monthly-active-user threshold above which you must ask the licensor); a CC-BY-NC model is a **STOP** for commercial use but fine for research; an API-only model is a STOP with "not open-weights".

## 4.2 Duties that travel with the model

- **Use-based restrictions** (community licences, OpenRAIL) bind your fine-tuned derivative too, and must be passed to your users.
- **Share-alike** licences can force your derivative under the same terms.
- **Naming/attribution** rules sometimes require the base model's name in yours.
- Record the licence and the exact **revision** in your lineage (chapter on provenance) — "we used Llama" is not an audit answer.

:::warn Engineering check, not legal advice
This tool reads metadata and applies rules. For anything you sell, a human must read the licence text.
:::

:::check
- Decision recorded: PROCEED or STOP, with the licence family and revision.
- If gated: token accepted the terms, and the download works.
:::
