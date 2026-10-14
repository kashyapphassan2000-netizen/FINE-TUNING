# 123. Image generation in depth

Chapter 97 covered the objective and ran a toy denoiser. This chapter is the real pipeline the review asked for.

```text
image ─→ VAE encoder ─→ LATENT z0 (much smaller than pixels)
caption ─→ text encoder ─→ conditioning c
z_t = sqrt(ᾱ_t)·z0 + sqrt(1-ᾱ_t)·ε      (noise schedule)
denoiser (U-Net or DiT), conditioned on (t, c) ─→ predicts ε (or v)
loss = MSE(prediction, target), optionally SNR-weighted
sampling: start from noise, denoise N steps with a sampler ─→ VAE decoder ─→ image
```

## 123.1 The pieces you must get right

|Piece|What it is|What goes wrong|
|---|---|---|
|VAE|compresses pixels to latents (and back)|training in pixel space by mistake; mismatched VAE at inference|
|noise schedule (ᾱ)|how much noise at each timestep|using a schedule the checkpoint was not trained with|
|prediction target|ε-prediction or v-prediction|mismatched target = the model learns noise|
|SNR weighting|re-weights timesteps so mid-noise steps matter more|unweighted training can under-train the useful range|
|resolution bucketing|group images by aspect ratio instead of square-cropping|cropping destroys composition; mixed sizes break batching|
|text conditioning|the frozen text encoder's embeddings|tokenizer/encoder mismatch silently degrades adherence|
|sampler + steps|how images are generated|comparing checkpoints at different step counts|

## 123.2 The training variants

|Variant|What trains|Data|Use|
|---|---|---|---|
|LoRA on the denoiser|attention (especially **cross-attention**)|10-1,000 images|style or concept adaptation; the standard choice|
|DreamBooth|denoiser (often + LoRA), with a rare token for the subject and a class-preservation set|3-20 images of one subject|"this specific person/product"|
|textual inversion|**only a new embedding vector** — the model is untouched|3-10 images|a concept token; tiny file, weaker effect|
|full fine-tuning|everything|large curated sets|a new base model; expensive|
|ControlNet-style|a new conditioning branch|paired condition→image data|pose/depth/edge control; a separate module to train and serve|

## 123.3 Evaluation (a loss number is not evidence)

Generate on a **fixed prompt set with fixed seeds** before and after, then measure: human preference (A/B), prompt adherence (e.g. CLIP-style image-text similarity), subject/style consistency across prompts, artifact rate, and a distribution metric (FID-style) on a held-out set. Always check **regressions**: a style LoRA that breaks hands or text is a failure even if the style is right.

## 123.4 Serving

Latency = steps × step time, so requests are jobs, not chats: a queue, progress reporting, and per-request caps. Batch the text encoder; keep the VAE decode on the same device; cache by (prompt, seed, settings) when you can.

:::honest Not executed
Only the toy denoiser of lab 60 and the registry's `diffusion_loss` (lab 62) ran here. The variants above are standard practice in the diffusion ecosystem and are described, not benchmarked, in this book.
:::
