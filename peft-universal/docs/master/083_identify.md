# 83. Step 0 — Identify the model

:::why In one line
Before touching data or GPUs, find out what the model is from its own files: config, code, module tree, forward signature, parameters.
:::

## 83.1 Read the repository before loading it

```text a typical unfamiliar repository
model/
├── config.json            <- hyper-parameters: sizes, vocab, model_type, architectures, quantization
├── modeling_x.py          <- THE truth: forward(), where the loss is computed, custom layers
├── configuration_x.py     <- which config fields exist and their defaults
├── processing_x.py        <- how raw inputs (text/image/audio) become tensors
├── tokenizer*/            <- vocabulary, special tokens, chat template
├── *.safetensors          <- weights (read the index for parameter names and shapes, without loading)
└── preprocessor_config.json / feature_extractor.json   <- image size, sampling rate, normalisation
```

```any read weights' names and shapes WITHOUT loading them into memory
python - <<'PY'
from safetensors import safe_open
with safe_open("model/model.safetensors", "pt") as f:
    for k in list(f.keys())[:40]:
        print(k, f.get_slice(k).get_shape(), f.get_slice(k).get_dtype())
PY
```

## 83.2 The core inspection commands

```py run these, in order, on ANY model
import inspect, torch
print(model)                                          # module tree: every layer and its type
print(getattr(model, "config", None))                 # hyper-parameters (Hugging Face models)
print(inspect.signature(model.forward))               # what forward() accepts
print(inspect.getsource(type(model).forward))         # where the loss is computed, what is returned
for n, m in model.named_modules():                    # every submodule with its full path
    print(n, type(m).__name__)
for n, p in model.named_parameters():                 # every tensor: shape, dtype, trainable?
    print(n, tuple(p.shape), p.dtype, p.requires_grad)
print(sum(p.numel() for p in model.parameters()))     # total parameters
print(sum(p.numel() for p in model.parameters() if p.requires_grad))   # trainable parameters
```

## 83.3 Inspect a huge model without the memory

```py build on the meta device: exact structure and counts, zero bytes allocated
import torch
from transformers import AutoConfig, AutoModelForCausalLM
cfg = AutoConfig.from_pretrained("org/model")          # downloads config.json only
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(cfg)
print(sum(p.numel() for p in model.parameters()) / 1e9, "B params")
```

This is how chapter 36 counted DeepSeek-V3 (671.0B) and a Kimi-K2-like config (1026.4B) on a laptop.

## 83.4 The same, automated

@file src/uadapt/inspect_model.py

```py
from uadapt.adapter import UniversalModelAdapter
ad = UniversalModelAdapter(model, sample_batch)
ad.inspect()          # parameter counts by module type, forward signature, layer inventory
ad.classify()         # family labels WITH the evidence for each (chapter 84)
ad.identify_inputs()  # forward() arguments and defaults
ad.identify_outputs() # output keys and shapes for the sample batch
ad.identify_loss()    # does it return a loss? which loss functions appear in its source?
```

## 83.5 Advanced inspection

|Tool|Use|Caveat|
|---|---|---|
|`torchinfo.summary(model, input_data=batch)`|per-layer output shapes and parameter counts in one table|optional package; must run a forward|
|`torch.fx.symbolic_trace(model)`|the full computation graph as code|fails on data-dependent control flow (most Hugging Face models); use hooks instead (chapter 85)|
|`torch.export.export(model, args)`|a traceable graph for export/compilation|stricter still; good test of deployability|
|forward/backward hooks|shapes, activations, gradients at every layer|always works; this guide's default (chapter 85)|
|reading `modeling_x.py`|the only complete truth|slow, necessary for custom architectures|

:::check
- I can list a model's inputs, outputs, parameter count and module types without documentation.
- I know where its loss is computed, or that it computes none.
:::
