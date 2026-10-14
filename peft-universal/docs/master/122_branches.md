# 122. The non-generative branches: classification, embeddings, rerankers, reward models

:::why Why this chapter exists
"Fine-tuning" does not mean "SFT → generate text". These four branches are where most business value sits, and every one has a different loss and a different metric.
:::

## 122.1 Classification, token classification, regression, ranking

|Branch|Record|Loss|Head|Primary metric|Trap|
|---|---|---|---|---|---|
|sequence classification|`{text, label}`|CE (class weights if imbalanced)|new linear over classes|macro-F1, per-class recall, **AUROC**, **ECE**|accuracy hides the minority class|
|multi-label|`{text, labels[]}`|BCEWithLogits|one logit per label|per-label F1, mAP|thresholds must be tuned per label|
|token classification|`{tokens[], tags[]}`|CE per token|token head|entity-level P/R/F1|align labels to sub-words; continuations = -100|
|regression|`{inputs, target}`|MSE / Huber|scalar head|MAE, RMSE vs a naive baseline|normalise targets with TRAIN statistics only|
|pairwise ranking / reranking|`{query, document, label}`|BCE or pairwise|cross-encoder head|**nDCG@10**, MRR, latency|score only the top-K from retrieval — it is expensive|

Lab 62 trains classification (macro-F1 0.600), token classification (F1 0.812) and regression (MAE 0.884) through the same engine.

## 122.2 Embedding models (retrieval)

```text
query ─┐                            ┌─ same encoder ─→ vector ─┐
       ├→ encoder → pooled vector   │                          ├→ cosine similarity → contrastive loss
doc ───┘                            └────────────────────────────┘
```

- **Loss**: multiple-negatives ranking — every other document in the batch is a negative. Shipped as `uadapt.tasks.contrastive_loss` with temperature scaling.
- **Batch size matters more than epochs**: more in-batch negatives = a harder, more useful objective.
- **Hard negatives** beat more data: mine them with the current model (documents it ranks highly but that are wrong), then re-train.
- **Pooling is part of the contract**: mean, CLS or last-token — training and serving must use the same one.
- **Metrics**: Recall@K, MRR, nDCG@K (all shipped). Lab 62: after 40 steps on one batch, Recall@1 1.0 / MRR 1.0 / nDCG@3 1.0 — which proves the loss and metrics are wired, nothing more.
- **Serving**: embed once, store in a vector index; serve nearest-neighbour search. The model is a feature extractor, not a chat endpoint.

## 122.3 Reward models vs preference optimization

```text
preference data {prompt, chosen, rejected}
        ├─────────────────────────────┐
        ▼                             ▼
   REWARD MODEL                     DPO / IPO / KTO / ORPO
   scores an answer                 trains the policy directly
   loss: -log σ(s_chosen - s_rejected)   loss: log-ratio vs a frozen reference
   used for: best-of-n, RL (PPO/GRPO), filtering
```

|Method|Needs a reward model?|Needs a reference model?|Data|Note|
|---|---|---|---|---|
|reward modelling|it IS one|no|pairs|scalar head on a backbone; shipped as `reward_model_loss`|
|DPO|no|yes|pairs|chapter 64, executed|
|IPO|no|yes|pairs|changes DPO's objective to reduce over-fitting to the preference|
|KTO|no|yes|**labels, not pairs** (good/bad)|cheaper data collection|
|ORPO|no|**no**|pairs|merges SFT and preference in one stage|
|PPO / GRPO|yes|yes|prompts + reward|full RL loop: **not implemented in this guide**|

Lab 62 trains a reward model to pairwise accuracy 1.0 on one batch (loss 0.692 → 0.004), which is the Bradley-Terry objective doing exactly what it should.

:::warn The trap
A reward model's absolute score is meaningless; only differences are trained. Never expose it as "quality = 0.87" to users, and re-calibrate it whenever the policy changes.
:::
