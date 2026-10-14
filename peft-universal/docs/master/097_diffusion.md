# 97. Diffusion models

:::warn NOT ESTABLISHED BY THE PROVIDED PDF
Derived from the architecture; exercised in lab 60 (custom denoiser row).
:::

## 97.1 The objective

Training: take a clean sample x₀, pick a timestep t, add noise ε with the schedule ᾱ_t, and teach the model to predict the noise:

$$ x_{t} = sqrt(ᾱ_{t}) · x_{0} + sqrt(1 − ᾱ_{t}) · ε ,     L = ‖ ε − ε_{θ}(x_{t}, t) ‖²

Inference: start from pure noise and denoise step by step with a sampler. The forward signature takes the noisy sample **and** the timestep (and usually a text/condition embedding) — that is how chapter 84 detects it.

## 97.2 Adaptation

- Targets: attention (self and cross) and linear layers of the denoiser (U-Net or transformer). Cross-attention LoRA is the standard way to teach a new style or subject.
- The text encoder and the VAE are normally frozen.
- Lab 60: a custom denoiser trained with the ε-MSE loss through `uadapt.loop`; loss 0.919 → 0.305, merge exact. The timestep embedding linear was classified "small" and skipped.

## 97.3 Evaluation and serving

- Loss is weakly related to sample quality. Evaluate by generating: human preference review, prompt-adherence checks, and distribution metrics (FID-style) on a held-out set.
- Serving: many denoising steps per request, so latency is steps × step time; batch and cache the text encoder. Diffusion serving is a different stack from LLM serving (chapter 104).
