# 72. Step 3 — Find the training objective

:::why In one line
Never assume cross-entropy on next tokens. Find the loss in the source, confirm it with a forward pass that returns `loss`, and write your own when the model returns none.
:::

## 72.1 How to find it

```py
import inspect, re
src = inspect.getsource(type(model).forward)
print(re.findall(r"(CrossEntropyLoss|cross_entropy|CTCLoss|ctc_loss|MSELoss|mse_loss|BCEWithLogitsLoss|"
                 r"SmoothL1Loss|HuberLoss|TripletMarginLoss|KLDivLoss|clip_loss)", src))
out = model(**batch_with_labels)
print(type(out).__name__, getattr(out, "loss", None))
```

`UniversalModelAdapter.identify_loss()` does both and reports whether a custom `loss_fn` is required.

## 72.2 The objectives, per architecture

|Architecture / task|Input|Target|Prediction|Loss|Gradient signal|
|---|---|---|---|---|---|
|decoder LM (SFT)|token ids|next token ids, prompt masked with -100|logits over vocab|CrossEntropy (masked mean)|only answer tokens|
|encoder classification|token ids|class id|logits over classes|CrossEntropy|head + adapted layers|
|multi-label|token ids / image|0/1 per label|one logit per label|BCEWithLogits|per-label|
|regression|any|real value|scalar|MSE / Huber|scale-sensitive: normalise targets|
|masked LM (encoder CPT)|ids with [MASK]|original ids at masked positions|logits|CrossEntropy on masked positions|15%-style masking|
|seq2seq (T5, Whisper)|encoder ids / audio|target token ids|decoder logits|CrossEntropy, teacher forcing|decoder + cross-attention|
|CTC ASR (wav2vec2)|waveform|character/token ids (no alignment)|per-frame logits|CTC|requires label length ≤ output frames|
|contrastive (CLIP, retrieval)|pairs|which pairs match|similarity matrix|symmetric CE / InfoNCE|batch size matters (negatives)|
|triplet / metric learning|anchor, positive, negative|margin|embeddings|TripletMargin|hard negatives drive learning|
|diffusion (ε-prediction)|noised sample x_t, timestep t|the noise ε|predicted ε|MSE|sampled over t|
|preference (DPO)|prompt, chosen, rejected|preference|log-prob ratio vs reference|DPO loss (chapter 50)|relative, needs a reference model|
|ranking|query, items|relevance order|scores|pairwise/listwise|ordering, not values|
|multi-task|any|several targets|several heads|weighted sum|weights are a design decision|

:::warn NOT ESTABLISHED BY THE PROVIDED PDF
Chapters 5-65 only establish masked next-token cross-entropy and DPO. The CTC, contrastive, diffusion and multi-task rows above are derived from the architectures and exercised in lab 60 (CTC, seq2seq, classification, regression, diffusion) — not from the earlier chapters.
:::

## 72.3 Writing your own loss

```py the contract used by uadapt.loop: loss_fn(model, batch) -> scalar tensor
import torch.nn.functional as F

def loss_fn(model, batch):                      # classification with a custom model
    return F.cross_entropy(model(batch["x"]), batch["y"])

def diffusion_loss(model, batch):               # epsilon-prediction
    x0 = batch["x"]; t = torch.randint(0, 1000, (x0.shape[0],))
    a = alpha_bar[t].view(-1, *[1] * (x0.dim() - 1)); eps = torch.randn_like(x0)
    return F.mse_loss(model(a.sqrt() * x0 + (1 - a).sqrt() * eps, t), eps)
```

:::check
- I found the loss in the source, or I wrote one and know why it is right for the task.
- I checked the label format against the loss's expectations (lengths for CTC, -100 for masking, float for BCE).
:::
