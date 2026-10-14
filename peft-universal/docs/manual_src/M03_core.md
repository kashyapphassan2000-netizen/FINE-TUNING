# LAB MANUAL — PART 3: The Core Idea

## 3.1 The analogy

A pretrained model is a **brilliant new graduate**: they speak the language and know a lot, but not your company's way of doing things. **Fine-tuning** is on-the-job training with a handbook of worked examples. **LoRA** is giving them a small notebook of company-specific shortcuts instead of re-educating them from scratch — the notebook is tiny, and you can swap notebooks per department. **Evaluation** is the probation review: the same test before and after, on cases they have never seen. **Serving** is putting them at the front desk, where many customers arrive at once.

## 3.2 The 20% that explains 80%

1. **The model predicts the next token.** Training makes the right next token more likely. The number that measures "how wrong" is the **loss**.
2. **Only the answer is trained.** In chat data the prompt is context; only the assistant's reply (and its end-of-turn token) carries loss. Get this wrong and the model learns to repeat questions or never stop.
3. **LoRA trains ~0.1-2% of the parameters.** The original weights stay frozen; a small add-on learns the change. That is why a big model fits on one GPU.
4. **QLoRA stores the frozen weights in 4 bits.** Roughly a quarter of the memory, a bit slower, usually similar quality.
5. **Data decides quality.** 500 correct, varied examples beat 50,000 sloppy ones. Diversity of phrasings and entities decides whether it generalises.
6. **Evaluate against a baseline, on unseen data.** "The loss went down" proves nothing; "the score on 100 held-out questions went from 0.42 to 0.71 and the general suite did not drop" proves something.
7. **Memory = weights + training state + activations.** For inference, add the KV cache per user. These sums decide the GPU.
8. **Serving is a different machine and a different problem.** Latency, concurrency and cost per request; measured with a load test, scaled with replicas.
9. **Every gate must pass before the next stage:** licence → data → doctor → evaluation → served parity → load test → release.

The Reference explains each in depth: fundamentals (chapters 19-21), LoRA family (23-29), memory (35-37), data (38-51), evaluation (58-62), serving (65-73).
