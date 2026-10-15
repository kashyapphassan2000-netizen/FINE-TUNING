# Transformers 101 — Zero to Elite Lab Manual

A hands-on guide to the 10 core Transformer topics: embeddings, the attention mechanism, self-attention, multi-head attention, masked multi-head attention, positional encoding, encoder–decoder, encoder-only, decoder-only and cross-attention. It goes from first principles up to how 2026-era LLMs are built (RoPE, RMSNorm, SwiGLU, GQA, KV cache, induction heads).

## What's here

| Path | What it is |
|---|---|
| `output/Transformers_101_Zero_to_Elite.pdf` | **The guide.** Diagrams, step-by-step labs, real outputs, cheat sheet, self-test |
| `output/Transformers_101_Zero_to_Elite.docx` | Same content as Word, for importing into Google Docs |
| `output/Transformers_101_Zero_to_Elite.md` (+ `figs/`, `diagrams/`) | Same content as Markdown |
| `notebooks/T101_Part4..7_*.ipynb` | Ready-to-run Colab notebooks, one per hands-on Part |
| `labs/*.py` | Lab source code; each `# %% ID` block is one notebook cell |
| `guide/*.md` | Guide text with `{{code:ID}}`, `{{out:ID}}` and `{{fig:ID}}` placeholders |
| `diagrams/make_diagrams.py` | Generates every SVG diagram |
| `build/run_labs.py`, `build/build_guide.py` | Runs every lab cell and builds the PDF, DOCX, MD and notebooks |

## Open a notebook in Colab

1. Go to `colab.research.google.com`.
2. Click **File > Open notebook > GitHub**.
3. Paste this repository's URL, tick **Include private repos** if it is private, and pick a notebook under `transformers-101/notebooks/`.
4. Click **Runtime > Change runtime type > T4 GPU > Save**, then run the cells in order.

## Put the guide into Google Docs

- Upload `output/Transformers_101_Zero_to_Elite.docx` to Google Drive, right-click it, choose **Open with > Google Docs**, then copy it into your doc's tab. Images and tables come across.
- Or, in Google Docs, turn on **Tools > Preferences > Enable Markdown** and use right-click **Paste from Markdown** with the `.md` file's text (images will not come across this way).

## Rebuild

Every output in the guide comes from actually running the lab code (PyTorch 2.14.1, Transformers 5.20.0, CPU build machine; the long training loops were shortened for the build run only, and the guide says so where that applies).

```bash
python -m venv venv && source venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install transformers matplotlib numpy markdown pygments pypdf pillow
python diagrams/make_diagrams.py
cd build
T101_OVERRIDES='{"6.3-D": [["MAX_ITERS = 3000", "MAX_ITERS = 600"], ["EVAL_EVERY = 300", "EVAL_EVERY = 100"], ["EVAL_ITERS = 50", "EVAL_ITERS = 10"]], "7.1-C": [["MAX_ITERS = 3000", "MAX_ITERS = 600"], ["EVAL_EVERY = 300", "EVAL_EVERY = 100"], ["EVAL_ITERS = 50", "EVAL_ITERS = 10"]]}' \
  python run_labs.py part4_basics part5_intermediate part6_advanced part7_elite
python build_guide.py   # needs Chromium (set CHROME in build_guide.py) and pandoc
```
