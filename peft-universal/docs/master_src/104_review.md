# 104. PART M — The external review, point by point

An experienced reviewer read this guide and wrote a detailed critique. This chapter answers every point: what was **valid**, what was **already covered** (some criticisms describe an earlier edition without Part J), and what is **new in Part M**. Nothing is argued away.

## 104.1 The verdict I accept

> "This is a very strong LLM/decoder-only fine-tuning engineering manual, but it is NOT yet a truly universal 'fine-tune any model' manual."

Correct for the edition reviewed. The single most important criticism — **there is no TASK abstraction** — was right, and it is now implemented as code (chapter 107), not prose.

## 104.2 Point by point

|#|Review point|Verdict|Where|
|---|---|---|---|
|1|Lifecycle coverage is the strength|agreed|Parts C-I|
|2|"Universal" wording is too broad|**valid**|scope statement rewritten, chapter 112|
|3|**Missing TASK abstraction** (modality × architecture × task × method)|**valid, the central point**|chapters 106-107, `uadapt/tasks.py`, lab 62|
|4|Need a training-objective registry|**valid**|chapter 107: 19 tasks registered with schema, loss, metrics|
|5|PEFT ≠ architecture ≠ objective ≠ modality|**valid**|chapter 106, stated as a first principle|
|6|Need Level 1-5 support boundaries|**valid**|chapter 105|
|7|Multimodal needs a real module|partly covered (ch 80), **deepened**|chapter 111|
|8|Audio is a serious gap|covered in ch 79 after the review's edition; **TTS was genuinely missing**|ch 79, chapter 110|
|9|Image generation is a giant hole|shallow in ch 84; **deepened**|chapter 109|
|10|Need a classification branch|**valid**|chapter 108 + registry|
|11|Embedding models deserve their own pipeline|**valid**|chapter 108, contrastive loss + Recall@K/MRR/nDCG shipped|
|12|Reward modelling ≠ DPO|**valid**|chapter 108: Bradley-Terry loss shipped, DPO/IPO/KTO/ORPO compared|
|13|Data provenance and lineage|**valid**|chapter 113, `peftkit/data/lineage.py`|
|14|Licensing/legal engineering controls|**valid**|chapter 113: licence buckets, conflict detection, BLOCK/REVIEW|
|15|Task-specific evaluation across modalities|partly (ch 89); **now code**|chapter 107 metrics registry, `uadapt/metrics.py`|
|16|"works" ≠ "works well"|already central|chapters 44-47, 54-56|
|17|Push the unseen-model challenge harder|agreed|chapter 115: 16-point determination|
|18|Evolve into an engine (registry → analyzer → adapters → collator → loss → PEFT → trainer → evaluator)|**valid**|chapter 114 with the implemented mapping|
|19|Rename: "model-agnostic framework for adapting modern language models"|**valid**|chapter 112|
|20|Final assessment: not yet "any AI model"|accepted, with the corrected claim|chapter 112|

## 104.3 Where the review described an older edition

Vision, audio (ASR), multimodal, MoE, SSM/RNN and diffusion **branches already exist** in Part J (chapters 78-83), each exercised in lab 60 across 12 architecture families with exact merges, plus custom-LoRA support where PEFT does not reach. The review's "missing" verdict on those is out of date; its verdict on **tasks, embeddings, reward models, TTS, video, lineage and licensing** was right, and those are what Part M adds.

## 104.4 What is still NOT claimed

- **TTS and video**: registered with their recipes and metrics, but **no generic loss is shipped** and nothing was executed. Chapter 110.
- **Diffusion**: the objective, LoRA targets and evaluation are covered and a toy denoiser is trained end to end; **DreamBooth/ControlNet-scale pipelines were not run** (no GPU, no dataset). Chapter 109.
- **RLHF with PPO/GRPO**: reward modelling is shipped; the RL loop is not. Chapter 108.
- Nothing in Parts J-M ran on a real pretrained model.
