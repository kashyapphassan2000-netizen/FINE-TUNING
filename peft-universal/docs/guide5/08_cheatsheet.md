# 9. Checklist and one-page cheat sheet

```text SFT CHEAT SHEET
POSITION      pretraining -> [CPT] -> SFT -> preference optimization -> evaluation
TRAINS ON     assistant tokens only (+ end-of-turn token). Prompt is context, not a target.
TEACHES       format, style, task behaviour, refusals, tool-call shape
DOES NOT      add facts (use retrieval), fix a base model that cannot do the task at all

DATA SHAPES
  alpaca   {"instruction","input","output"}          -> normalise to messages
  flan     task + many templates, zero/few-shot      -> diversity is the product
  chat     {"messages":[system,user,assistant,...]}  -> one example = one conversation
  cot      reasoning steps INSIDE the assistant target

WHAT THE LANDMARKS TAUGHT  (checked 19 Sep 2026)
  FLAN 2022  1,836 tasks, Apache-2.0 : held-out quality rises log-linearly with task count
  Alpaca     52K from 175 seeds via text-davinci-003, CC BY-NC 4.0 : cheap distillation, licence trap
  OpenHermes-style compilations      : inherit the most restrictive upstream licence; check the card

MEASURED IN THIS BOOK (toy scale, direction not magnitude)
  6 examples          -> 1.00 on trained prompts, 0.00 on unseen ones (memorisation)
  1 template vs 5     -> 0.31 vs 0.50 on unseen phrasing + unseen entities (chance 0.25)
  single vs multi-turn-> 0.33 vs 1.00 on turn 2 (chance 0.25)
  CoT targets         -> several times longer: train + serve cost scales with them

DEFAULTS FOR A REAL MODEL
  LoRA r=16 alpha=32, lr 2e-4, cosine, warmup 3%, 1-3 epochs, effective batch 16-64
  assistant_only_loss: true, max_seq_length = p95 of your token lengths
  eval every 20-50 steps, keep the best checkpoint, not the last

CHECKLIST BEFORE TRAINING
  [ ] eval set written FIRST, real, holds out entities and phrasings
  [ ] 3+ phrasings per intent, varied word order; refusals included
  [ ] every dataset's licence read (commercial use?)
  [ ] chat template = the model's own; generation markers present
  [ ] tokenize_inspect: trained fraction sane, end-of-turn trained, no truncation
CHECKLIST AFTER TRAINING
  [ ] domain score up on unseen entities
  [ ] general suite not down (replay data if it is)
  [ ] format compliance measured, not assumed
  [ ] per-segment scores reviewed
  [ ] best checkpoint merged, serve_check passed
```

## 9.1 Verification of this book

```any
python labs/lab50_sft_stage1.py     # ... through lab54_domain_sft.py
python -m pytest -q                 # the whole repository
```
