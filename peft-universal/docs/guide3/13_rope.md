# 13. RoPE (Rotary Position Embedding)

:::why In one line
Rotate each pair of dimensions in Q and K by an angle proportional to the token's position. The dot product between a query at position i and a key at position j then depends only on `i − j` — absolute encoding that behaves relatively.
:::

## 13.1 The maths

Split the head dimension into `d/2` pairs. Pair m has frequency

$$ θ_{m} = base^{−2m/d}   (base is `rope_theta`, commonly 10,000 … 1,000,000)

Position p rotates pair m by angle `p·θ_m`:

$$ RoPE(x, p)_{m} = ( x_{2m} cos(pθ_{m}) − x_{2m+1} sin(pθ_{m}) ,  x_{2m} sin(pθ_{m}) + x_{2m+1} cos(pθ_{m}) )

Then `⟨RoPE(q, i), RoPE(k, j)⟩` is a function of `(i − j)` only. That is the property that makes it work.

## 13.2 Verified, with frequencies you can see

@file labs/lab36_rope.py
@lab lab36_rope

The lab shows the relative-distance property holding numerically, the fastest pair rotating ~1 radian per token and the slowest with a period of hundreds of thousands of tokens — high-frequency pairs carry local order, low-frequency pairs carry long-range position.

## 13.3 Why long context needs more than "set max_position_embeddings higher"

Past the trained length, the slow pairs reach angles the model never saw. The standard fixes, all present in `transformers` 5.17 (`rope_type` values read from the library: `default`, `linear`, `dynamic`, `yarn`, `llama3`, `longrope`, `proportional`):

|Method|What it does|
|---|---|
|linear (position interpolation)|divide positions by a factor: every frequency is squeezed|
|dynamic NTK|raise the base as the sequence grows|
|YaRN|interpolate low frequencies, keep high ones, plus an attention temperature|
|llama3|piecewise: keep high frequencies, interpolate low ones, smooth between|
|longrope|per-dimension learned scale factors|

In transformers 5.17 these live under the config key **`rope_parameters`** (e.g. Llama's default is `{'rope_theta': 10000.0, 'rope_type': 'default'}`). Older tutorials use `rope_scaling` and `rope_theta` at the top level; that moved.

## 13.4 Interaction with everything else in this book

- **KV cache**: RoPE is applied to K *before* caching. Cached keys are position-stamped, so you cannot re-use a cache under different rope settings, and you cannot shift a sequence's positions after the fact.
- **MLA**: needs a separate decoupled rotary dimension because rotation does not commute with low-rank absorption (chapter 10).
- **Sliding windows**: Gemma-3 uses a different base on local and global layers (chapter 12).
- **Fine-tuning**: extending context is a config change **plus** training on long examples, then long-context evaluation. It is not a serving flag.

## 13.5 Common mistakes

- Setting a scaling factor at inference on a model that was never trained with it and reporting the perplexity as fine.
- Mixing two rope configurations behind one endpoint (cache corruption, silent quality loss).
- Forgetting that the rotation is applied per head dimension pair — a custom attention implementation that reshapes heads wrongly produces plausible-looking but broken positions.

:::check
- I can state the relative-distance property and why it matters.
- I know my model's `rope_parameters` and whether any scaling is active.
- I know that cached keys already carry their positions.
:::
