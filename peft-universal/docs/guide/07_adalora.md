# 7. AdaLoRA: adaptive rank allocation

:::why In one line
Not every layer needs the same rank. Start every layer at a higher rank, measure which singular directions matter, and prune the unimportant ones so the total rank budget flows to the layers that need it. Zhang et al., 2023 (ICLR, arXiv 2303.10512).
:::

## 7.1 The math

The update is written in SVD form:

$$ ΔW = P Λ Q ,     P ∈ R^{d_out × r}, Λ = diag(λ_{1..r}), Q ∈ R^{r × d_in}

Pruning a direction means setting λ_{i} = 0, which is cheap and reversible. To keep P and Q close to orthogonal (so λ really behaves like singular values) a regulariser is added:

$$ R(P, Q) = ‖PᵀP − I‖_{F} + ‖QQᵀ − I‖_{F}     (paper: squared norms; PEFT code: Frobenius norm)

**Importance.** For a parameter w with gradient g: sensitivity I = |w · g|. It is noisy, so AdaLoRA keeps an exponential moving average Ī (β_{1}) and an uncertainty Ū = EMA of |I − Ī| (β_{2}), and scores s = Ī · Ū. The paper's triplet score combines the score of λ_{i} with the average scores of the i-th column of P and row of Q.

**Budget schedule.** For `tinit` steps nothing is pruned. Then the total budget falls from `init_r × modules` to `target_r × modules` along a cubic curve. The last `tfinal` steps keep the ranks fixed. Every `deltaT` steps the lowest-scoring λ's are masked.

## 7.2 Worked example

A simplified version (score of λ only) on 4 layers whose true ranks are [6, 1, 3, 2]:

@file labs/lab5_adalora.py
@lab lab5_adalora

With the same total budget (12 = 4 × 3), AdaLoRA recovered exactly the true per-layer ranks.

## 7.3 In the template

@file configs/methods/adalora.yaml
@file src/peftkit/callbacks.py

:::warn Two traps found in the real source
- **The Hugging Face Trainer never calls `update_and_allocate`.** Without a callback, AdaLoRA silently trains as a fixed-rank adapter with no pruning. `AdaLoraCallback` calls it in `on_optimizer_step`, which the Trainer runs after `optimizer.step()` and before `zero_grad()` (read from transformers 5.17 `trainer.py`), exactly when AdaLoRA needs the gradients.
- **AdaLoRA adds the orthogonality penalty to `outputs.loss`, in training AND evaluation.** Its eval loss is not comparable to other methods. `train.py` sets the weight to 0 for the final evaluation it writes to `run_meta.json`.
:::

:::warn Schedules need steps
PEFT requires `total_step` and `tinit < total_step − tfinal`. The template computes total steps from the dataset size and sets tinit/tfinal as ratios. In a 30-step test AdaLoRA scored worse than LoRA; over 180 steps it reached 93.9% token accuracy vs 98% for LoRA on the same toy. Use AdaLoRA for runs of thousands of steps, not tiny ones. With `packing: true` you must set `train.max_steps` yourself.
:::

## 7.4 Serving

The saved adapter has `peft_type = ADALORA` and per-layer masks, and a layer may end with rank 0 (PEFT then warns about zero-sized tensors; that warning is harmless here, and a save → reload check matched the trained model to 1.2e-7). vLLM loads only LoRA-format adapters, so merge first. Our merge was exact (2.4e-7).

:::check
- I can explain P Λ Q, the importance score and the budget schedule.
- I know why the callback is required.
:::
