# 117. The universal processor layer

:::why In one line
The processor is whatever turns a human record into the tensors `forward()` accepts. This layer finds the model's own processor, states what it produces, and — when none exists — defines the tiny contract you implement so the rest of the engine never changes.
:::

@file src/uadapt/processors.py

## 117.1 Resolution order (and why)

```py
from uadapt import processors
p = processors.resolve(model_id="org/model", model=model)
print(processors.describe(p))
batch_piece = p({"text": "the pump leaks"})      # one record -> tensors
```

1. `AutoProcessor` — multimodal wrapper (it owns the tokenizer *and* the image/audio processor, and inserts modality placeholder tokens).
2. `AutoTokenizer` — text.
3. `AutoImageProcessor` — images (resize, crop, normalise with the model's own mean/std).
4. `AutoFeatureExtractor` — audio (sampling rate, spectrogram).
5. None found → a **custom contract**, printed with the exact input names taken from `forward()`.

## 117.2 The record → tensor rules

|Record contains|What happens|
|---|---|
|`messages`|`apply_chat_template(..., return_dict=True)` — the model's own template|
|`image` + `text`|multimodal processor call (placeholders included)|
|`image`|image processor|
|`audio` (+ `sampling_rate`)|feature extractor **at the model's rate** — the commonest silent audio bug|
|`text`|tokenizer with truncation|

## 117.3 A modality nobody has built yet

```py three lines is the whole contract
from uadapt.processors import Processor
proc = Processor(kind="custom", produces=("grid", "geology"),
                 encode=lambda rec: {"grid": torch.tensor(rec["grid"]),
                                     "geology": torch.tensor(rec["geology"])})
```

That is the only modality-specific code in the entire pipeline for a new sensor type (chapter 121 trains such a model). If you do not provide it, the engine raises a message telling you exactly which input names `forward()` expects — it never guesses.

:::warn Processor versioning
Pin the processor with the model revision. A changed image normalisation or a changed chat template silently shifts every input, and your evaluation will blame the model.
:::
