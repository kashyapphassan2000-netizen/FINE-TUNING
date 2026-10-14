# 6. Step 3: generation — five methods, one interface

@file src/synthkit/generate.py

## 6.1 Which generator for which data

|Method|Use for|Strength|Weakness|
|---|---|---|---|
|statistical (copula)|tabular business data|marginals + correlations + conditionals, instant, free|no deep causal logic, no sequences|
|simulator / rules|physics, queues, supply chains, fraud rings, machine failure|encodes real mechanics; unlimited rare events|you must know the mechanics|
|ML generative (GAN/VAE/diffusion)|complex tabular, images|can capture complex joints|needs data and tuning; memorisation risk|
|LLM|text, tickets, notes, dialogue, code|language realism, few-shot control|style homogeneity, invented facts, cost|
|hybrid|most real projects|numbers from the copula/simulator, language from the LLM|two things to validate|

## 6.2 The copula generator, and the bug this book fixed

It samples the categorical **combination** from its observed joint distribution, draws correlated uniforms through a Gaussian copula, and maps them through the empirical quantiles **of that group**.

:::warn Two measured findings
1. An earlier version nudged numbers by a per-category mean shift. It looked reasonable and destroyed the marginals: PSI 2.2 in the tests (anything above 0.25 is a large shift). Conditional quantile mapping replaced it.
2. With `smoothing=0`, every generated number is a **real observed value**. The combination is new, the values are not: distance-to-closest-record ratio measured **0.17**, which is a privacy problem (chapter 9). `CopulaGenerator(..., smoothing=0.25)` jitters each draw by a fraction of the group's spread; the tests assert DCR rises above 0.5 while PSI stays below 0.25.
:::

## 6.3 Simulators: when you know the mechanics, use them

```py a fraud simulator in ten lines (this is lab 42's generator)
def txn(rng, fraud=None):
    is_fraud = rng.random() < 0.004 if fraud is None else fraud
    if is_fraud:
        amount, hour, device_age = rng.lognormal(7.4, .8), rng.choice([1,2,3,4,23]), rng.exponential(2)
    else:
        amount, hour, device_age = rng.lognormal(4.2, 1.), np.clip(rng.normal(14,4),0,23), rng.exponential(400)+5
    return {"amount": float(amount), "hour": int(hour), "device_age_days": float(device_age), "fraud": bool(is_fraud)}
```

A simulator gives you unlimited rare events, exact ground truth, counterfactuals for free, and no privacy risk. Its weakness is that it encodes **your beliefs** about the world — which is why it still has to pass the real-data comparison in chapters 8 and 11.

## 6.4 LLM generation, done properly

`generate.llm_records` sends the schema, the constraints, a few **real** examples and asks for strict JSON. That grounding is what separates useful records from generic filler. Everything after it — validation, dedup, judging, privacy scan — exists because LLM output cannot be trusted on arrival (lab 44 produces a templated record and a record with PII and fake personal data on purpose; both are caught).

## 6.5 Enterprise record schema: keep the metadata

```json one record, with the provenance you will need in six months
{"task_id": "...", "domain": "support", "skill": "refund_policy", "difficulty": 3,
 "instruction": "...", "context": "...", "constraints": ["no PII"], "ideal_response": "...",
 "failure_modes": ["hallucinated refund window"], "source": "synthetic",
 "generator_model": "...", "generator_version": "1.4", "prompt_version": "p7",
 "contract_version": "1.0", "judge_scores": {"correctness": 4}, "created_at": "..."}
```

Without provenance you cannot answer "which batch poisoned the model?", and you will be asked.

:::check
- My generator matches the data type (tabular → copula/simulator, text → LLM, both → hybrid).
- Smoothing or another privacy control is on if I resample real values.
- Every record carries generator, prompt and contract versions.
:::
