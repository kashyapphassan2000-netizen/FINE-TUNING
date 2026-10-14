# 10. Why data is the leverage point + the taxonomy of synthetic data

:::why In one line
With PEFT, the method choice moves quality by a little; the data moves it by a lot. The adapter can only learn what the examples show, including their mistakes.
:::

## 10.1 Why data is the leverage point

- **The model already has the knowledge and skills; data selects the behaviour.** LIMA (Zhou et al., 2023) fine-tuned a 65B model on 1,000 carefully curated examples and reported strong results, arguing that alignment mostly teaches format and style. Quality beat quantity there.
- **Errors are learned faithfully.** A wrong answer repeated 50 times in the data becomes a confident wrong answer.
- **Coverage defines competence.** If no example shows "I don't know" for out-of-scope questions, the model will not say it.
- **Cheap to iterate.** A new dataset version costs API calls; a new base model costs weeks.

Rule of thumb used in this template: spend more time on chapters 11-20 than on hyperparameters.

## 10.2 The taxonomy of synthetic data

|Axis|Options|Template module|
|---|---|---|
|Source|grounded in your documents / from seed tasks / from scratch (model knowledge)|`synth.py` / `self_instruct.py` / prompts|
|Target|instruction (SFT) / preference (DPO) / evaluation sets / pretraining text|`format.py` / `preference.py`|
|Generation style|Self-Instruct, Alpaca, Evol-Instruct (make tasks harder), persona-driven, Magpie (let an aligned model write the user turn)|`self_instruct.py`, distilabel `EvolInstruct`, `Magpie`|
|Teacher|stronger model (distillation) / same model (self-improvement) / several models (diversity)|`SYNTH_MODEL`, `--weak-model`|
|Labeler|LLM-as-judge / reward model / human / rules|`judge.py`, Argilla|
|Turns|single / multi-turn|`messages` format supports both|

:::warn Licences and terms
Some API providers restrict using their outputs to train competing models. Read your provider's terms (Euron or others) and the base model's licence (e.g., Llama models have their own community licence) before you publish a model or dataset.
:::

:::check
- I can name the three sources of synthetic data and which module handles each.
:::
