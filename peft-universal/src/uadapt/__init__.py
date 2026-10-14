"""uadapt: model-agnostic adaptation toolkit.

Works on ANY torch.nn.Module (Hugging Face or your own class). It does not rely on layer NAMES:
it reads the module tree, runs a traced forward pass, and classifies layers by what they compute.

Universal methodology != universal code. This package automates the inspection and the mechanics
(tracing, gradient checks, LoRA injection, merging, a generic training loop). Choosing the objective,
the data and the evaluation still needs an engineer.
"""

__version__ = "1.0.0"
