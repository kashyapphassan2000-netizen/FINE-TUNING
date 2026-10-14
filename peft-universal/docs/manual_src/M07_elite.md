# LAB MANUAL — PART 7: ELITE — How top practitioners and researchers do it

:::why Concept (what, why, when)
The elite level is not a secret library. It is **discipline**: search hyperparameters without fooling yourself, align behaviour with preference data after SFT, and adapt architectures nobody has packaged — all measured against baselines, with every claim reproducible. **When:** once Parts 4-6 run without surprises.
:::

## 7.1 Who and what defines the top level today

|Idea|Where it comes from|What to take from it|
|---|---|---|
|LoRA|Hu et al., Microsoft, 2021 [24]|low-rank updates on frozen weights; the basis of everything here|
|QLoRA|Dettmers et al., University of Washington, 2023 [25]|4-bit NF4 frozen base + LoRA: large models on one GPU|
|DoRA|Liu et al., NVIDIA, 2024 [26]|decompose magnitude and direction; merge before serving|
|DPO|Rafailov et al., Stanford, 2023 [27]|preference optimisation without a reward model|
|PagedAttention / vLLM|Kwon et al., UC Berkeley, 2023 [28]|paged KV cache and continuous batching; why serving is cheap|
|FLAN instruction tuning|Chung et al., Google, 2022 [29]|task and template diversity drive generalisation|
|Self-Instruct|Wang et al., 2022 [30]|bootstrapping instruction data from seed tasks|
|Hugging Face TRL / PEFT|github.com/huggingface/trl, /peft [8][9]|the reference implementations used in this book|
|Unsloth, Axolotl, LLaMA-Factory, torchtune|[11][12][13][14]|production fine-tuning frameworks (Part 8 compares them)|
|EleutherAI lm-evaluation-harness|[15]|the standard public-benchmark runner|

The hands-on below implements three of these practices with this project's tools.

## SECTION 1 — Hyperparameter search that does not fool you

**WHERE:** GPU VM (or the laptop with `configs/lesson.yaml` to practise first)

**GOAL:** the best configuration, with proof its gain is larger than seed noise

Step 1. Type and press Enter:

```any
python -m peftkit.hpo --config configs/base.yaml configs/methods/lora.yaml configs/models/my-model.yaml --space "{\"train.lr\": [5e-5, 1e-4, 2e-4, 5e-4], \"method.r\": [8, 16, 32]}" --budget 12 --rungs 20 60 180 --out outputs/hpo --seeds
```

**You should see:** a table of candidates (the weak ones stopped after the first rung), the best configuration, and a seed-noise line.

**Real output from the build (tiny model, 6 candidates):** the best configuration scored 5.2328; three seeds of it scored 5.003 / 5.026 / 5.082 — a **spread of 0.079**.

Step 2. Write the rule down: *an improvement smaller than the seed spread is noise.*

:::warn Watch out: PowerShell quoting
In PowerShell the JSON needs the escaped quotes shown above. On the GPU VM (bash) use single quotes instead: `--space '{"train.lr": [5e-5, 1e-4], "method.r": [8, 16]}'`.
:::

:::check
- You have a winning configuration AND its seed spread, and the winner beats the runner-up by more than that spread.
:::

## SECTION 2 — Preference optimisation (DPO) after SFT

**WHERE:** GPU VM, `.venv-train`

**GOAL:** an SFT model further aligned with chosen-vs-rejected pairs

Step 1. Build preference pairs (Reference chapter 47: generate several answers, judge them, keep pairs with a clear margin) into `data/processed/pref.jsonl`, one record per line:

```json one line of pref.jsonl
{"prompt": [{"role": "user", "content": "What is the maximum pressure?"}],
 "chosen": [{"role": "assistant", "content": "The maximum pressure is 6 bar."}],
 "rejected": [{"role": "assistant", "content": "Probably around twenty bar."}]}
```

Step 2. Practise the mechanics on the laptop first:

```laptop
python labs/lab23_dpo.py
```

**You should see (real output):** reward accuracy rising to 1.0 and the reward margin growing across steps.

Step 3. On the GPU, run DPO starting from your SFT adapter with TRL's `DPOTrainer` (the lab's code is the template: swap the tiny model for `outputs/RUN/merged`, the data file for `pref.jsonl`).

Step 4. Re-run Part 5 Section 9 and Part 6 Section 5 on the DPO model.

:::warn Watch out
Always SFT first. Watch the average answer length before and after: a judge that preferred long answers makes DPO produce verbose models.
:::

:::check
- Reward accuracy and margin rose, AND the domain/general suites did not fall after DPO.
:::

## SECTION 3 — Adapt an architecture nobody has packaged

**WHERE:** laptop

**GOAL:** see the universal method derive and run a pipeline for a model invented today

Step 1. Type and press Enter:

```laptop
python labs/lab63_universal_pipeline.py
```

**You should see (real output, last block):** for an invented "seismic fusion" model with no config and no processor, the engine prints `CUSTOM ENGINEERING REQUIRED` for the loss, names the two missing pieces (`loss_fn`, `processor.encode`), then — with a 4-line loss supplied — adapts it (22.1% trainable), verifies gradients, trains, and merges with a difference of 1.8e-07.

Step 2. Open `labs/lab63_universal_pipeline.py` and read the class `SeismicFusionNet` and the function `quantile_loss`.

Step 3. Change `quantile_loss` to use the 0.9 quantile, save, and run Step 1 again.

**You should see:** the same pipeline, a different loss curve — the engine did not change.

Step 4. Apply the same object to your own model:

```py the elite pattern for any model
from uadapt.pipeline import UniversalPipeline
p = UniversalPipeline(model, sample_batch, loss_fn=my_loss)   # loss_fn only if plan() says it is missing
print(p.plan())            # modality, contracts, derived loss, targets, processor, task, what is MISSING
p.adapt(); print(p.smoke_test(training_batch)); p.fit(loader, lr=2e-4, max_steps=1000)
print(p.export("export/"))  # merge proof included
```

:::check
- You can explain why the engine refused to guess a loss for the invented model, and what you supplied.
:::

## 7.4 Common mistake at this level

**Declaring a win inside the noise.** A new method, a new rank, a new data mix: "+0.4 points". If two seeds of the same configuration differ by 0.8, that +0.4 means nothing. **How to spot it:** any result table without a seed-noise line or confidence interval.
