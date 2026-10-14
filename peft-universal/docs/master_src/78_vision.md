# 78. Vision models (CNN, ViT, Swin, CLIP, detection, segmentation, vision + LLM)

:::warn NOT ESTABLISHED BY THE PROVIDED PDF
Derived from the architecture; exercised in lab 60 (ViT, custom CNN, custom VLM rows).
:::

## 78.1 The flow

```text
image -> PROCESSOR (resize, crop, normalise with the model's mean/std) -> pixel_values [B, C, H, W]
      -> PATCH EMBEDDING (Conv2d stride = patch) or CNN STEM
      -> VISION ENCODER (ViT blocks / conv stages) -> features
      -> [PROJECTOR -> LANGUAGE MODEL]  or  HEAD (classes / boxes / masks)
```

## 78.2 What changes, per task

|Task|Labels|Loss|Head|Evaluation|
|---|---|---|---|---|
|classification|class id|CrossEntropy|linear over classes (new)|accuracy, per-class recall, confusion matrix|
|multi-label|0/1 per label|BCEWithLogits|one logit per label|mAP, F1 per label|
|detection|boxes + classes per image|matching + box + class losses (model-specific)|detection head|mAP@IoU|
|segmentation|mask per pixel|per-pixel CE / Dice|mask decoder|mIoU|
|captioning / VLM|target text|CE on text tokens|language model|generation metrics + human review|
|CLIP-style retrieval|image-text pairs|symmetric contrastive|projection heads|recall@k|

## 78.3 Adaptation targets and memory

- ViT: attention q/k/v/o and MLP fc1/fc2 (found by discovery in lab 60), patch-embedding conv optionally with conv-LoRA; the classifier head trains fully.
- CNN: conv-LoRA on the deeper stages (lab 60 custom CNN: `c1`, `c2`), or selective unfreezing of the last stages; the head trains fully.
- Memory grows with **image resolution²** and patch count, not with a context length: activation memory ≈ batch × patches × hidden × layers. Halving the image side quarters it.

```py
from transformers import AutoImageProcessor, AutoModelForImageClassification
proc = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=5,
                                                        ignore_mismatched_sizes=True)   # new head
batch = proc(images=[pil_image], return_tensors="pt"); batch["labels"] = torch.tensor([3])
```

## 78.4 Traps

- Using a different normalisation than the processor's: the model sees shifted colours and underperforms silently.
- Augmentation on the evaluation set.
- `ignore_mismatched_sizes=True` means the head is new: the base model has no meaningful baseline for your labels.
- Serving: plain forward pass (batching images), ONNX/TensorRT export for latency; vLLM only for vision-LANGUAGE models it explicitly supports.
