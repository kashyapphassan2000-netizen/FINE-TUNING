# 118. PART M — The external review, point by point

An experienced reviewer read this guide and wrote a detailed critique. This chapter answers every point: what was **valid**, what was **already covered** (some criticisms describe an earlier edition without Part J), and what is **new in Part M**. Nothing is argued away.

## 118.1 The verdict I accept

> "This is a very strong LLM/decoder-only fine-tuning engineering manual, but it is NOT yet a truly universal 'fine-tune any model' manual."

Correct for the edition reviewed. The single most important criticism — **there is no TASK abstraction** — was right, and it is now implemented as code (chapter 121), not prose.

## 118.2 Point by point

|#|Review point|Verdict|Where|
|---|---|---|---|
|1|Lifecycle coverage is the strength|agreed|Parts C-I|
|2|"Universal" wording is too broad|**valid**|scope statement rewritten, chapter 126|
|17|**Missing TASK abstraction** (modality × architecture × task × method)|**valid, the central point**|chapters 120-121, `uadapt/tasks.py`, lab 62|
|18|Need a training-objective registry|**valid**|chapter 121: 19 tasks registered with schema, loss, metrics|
|19|PEFT ≠ architecture ≠ objective ≠ modality|**valid**|chapter 120, stated as a first principle|
|20|Need Level 1-5 support boundaries|**valid**|chapter 119|
|21|Multimodal needs a real module|partly covered (ch 94), **deepened**|chapter 125|
|22|Audio is a serious gap|covered in ch 93 after the review's edition; **TTS was genuinely missing**|ch 93, chapter 124|
|23|Image generation is a giant hole|shallow in ch 98; **deepened**|chapter 123|
|24|Need a classification branch|**valid**|chapter 122 + registry|
|25|Embedding models deserve their own pipeline|**valid**|chapter 122, contrastive loss + Recall@K/MRR/nDCG shipped|
|26|Reward modelling ≠ DPO|**valid**|chapter 122: Bradley-Terry loss shipped, DPO/IPO/KTO/ORPO compared|
|27|Data provenance and lineage|**valid**|chapter 127, `peftkit/data/lineage.py`|
|28|Licensing/legal engineering controls|**valid**|chapter 127: licence buckets, conflict detection, BLOCK/REVIEW|
|29|Task-specific evaluation across modalities|partly (ch 103); **now code**|chapter 121 metrics registry, `uadapt/metrics.py`|
|30|"works" ≠ "works well"|already central|chapters 58-61, 54-56|
|31|Push the unseen-model challenge harder|agreed|chapter 129: 16-point determination|
|32|Evolve into an engine (registry → analyzer → adapters → collator → loss → PEFT → trainer → evaluator)|**valid**|chapter 128 with the implemented mapping|
|33|Rename: "model-agnostic framework for adapting modern language models"|**valid**|chapter 126|
|34|Final assessment: not yet "any AI model"|accepted, with the corrected claim|chapter 126|

## 118.3 Where the review described an older edition

Vision, audio (ASR), multimodal, MoE, SSM/RNN and diffusion **branches already exist** in Part J (chapters 92-97), each exercised in lab 60 across 12 architecture families with exact merges, plus custom-LoRA support where PEFT does not reach. The review's "missing" verdict on those is out of date; its verdict on **tasks, embeddings, reward models, TTS, video, lineage and licensing** was right, and those are what Part M adds.

## 118.4 What is still NOT claimed

- **TTS and video**: registered with their recipes and metrics, but **no generic loss is shipped** and nothing was executed. Chapter 124.
- **Diffusion**: the objective, LoRA targets and evaluation are covered and a toy denoiser is trained end to end; **DreamBooth/ControlNet-scale pipelines were not run** (no GPU, no dataset). Chapter 123.
- **RLHF with PPO/GRPO**: reward modelling is shipped; the RL loop is not. Chapter 122.
- Nothing in Parts J-M ran on a real pretrained model.
