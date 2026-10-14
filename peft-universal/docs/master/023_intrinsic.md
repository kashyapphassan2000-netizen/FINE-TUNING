# 23. The intrinsic dimensionality insight

:::why In one line
A pretrained model already knows almost everything. Adapting it to a task only needs a change that lives in a very small subspace, so we only need to train a very small number of parameters.
:::

## 23.1 Intuition

Think of a city map with millions of streets. You live in the city already (the pretrained model). To get to a new shop (your task) you do not need to redraw the map; you need a few turns. Fine-tuning all weights is redrawing the map. PEFT is learning the few turns.

## 23.2 The problem it solves

Full fine-tuning of an 8B model in bf16 with Adam needs, per parameter: 2 bytes weight + 2 bytes gradient + 12 bytes of fp32 master weights and Adam moments. That is roughly 16 bytes × 8 billion ≈ 120 GB before activations. It also produces an 16 GB checkpoint per task. PEFT trains a few million parameters, needs a single GPU, and produces checkpoints of tens of megabytes.

## 23.3 The math

Take all D parameters as one vector θ. Freeze the starting point θ_{0}. Pick a random matrix P of shape D × d with d ≪ D. Train only the small vector z:

$$ θ = θ_{0} + P z,      z ∈ R^{d} , d ≪ D

The **intrinsic dimension** d_{90} is the smallest d for which this subspace training reaches 90% of the full fine-tuning score (Li et al., 2018, "Measuring the Intrinsic Dimension of Objective Landscapes").

Aghajanyan, Zettlemoyer and Gupta (2020, "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning") applied this to pretrained language models. Two findings matter for us: pretraining lowers the intrinsic dimension of downstream tasks, and larger pretrained models tend to have lower intrinsic dimension. Their abstract reports that about 200 parameters, randomly projected back to the full space, reach 90% of full performance for RoBERTa on MRPC.

LoRA (next chapter) turns this into a practical rule: instead of one random projection of the whole model, give each weight matrix a trainable **low-rank** update.

## 23.4 Worked numeric example

Our toy network has D = 1,474 parameters. A random subspace of size d = 200 (13.6% of D) reached 97.8% accuracy against 100% for full training, so d_{90} ≤ 200 for this toy. For a toy MLP trained from random init the ratio is large; for pretrained LLMs it is tiny, which is the point of the papers above.

@lab lab1_intrinsic_dimension

:::warn Do not over-read the toy
The toy starts from random weights, so it needs a much bigger subspace than a pretrained LLM would. It shows the mechanism (training only z), not the size of the effect.
:::

## 23.5 Where it is used

Every PEFT method in this guide relies on it. It also explains why r = 8 or 16 is often enough, and why a bigger base model usually needs a smaller relative adapter.

:::check
- I can explain θ = θ_{0} + P z and which part is trained.
- I can say what d_{90} means.
- I ran `python labs/lab1_intrinsic_dimension.py` and got similar numbers.
:::
