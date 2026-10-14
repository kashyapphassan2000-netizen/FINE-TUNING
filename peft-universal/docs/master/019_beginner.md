# 19. Plain-English primer (start here if you are new)

:::why Who this chapter is for
Someone who has never fine-tuned anything. No maths needed here. Read it once, then the rest of the book makes sense. Everything is explained again with maths in the earlier chapters.
:::

## 19.1 The whole thing in one picture

```text What you are actually doing
 1. Pick a model somebody already trained    (the "base model", free on Hugging Face)
 2. Write down examples of the behaviour you want   (the "dataset": question -> good answer)
 3. Rent a computer with a GPU for a few hours      (a GPU is a chip that does maths very fast)
 4. Train a SMALL extra piece that sits next to the model  (the "adapter", a few MB)
 5. Test it, save it, put it online so apps can call it    ("serving")
```

You never change the big model. You train a small clip-on part. That is what LoRA and its relatives are.

## 19.2 Words you will keep seeing

|Word|Plain meaning|
|---|---|
|model|a big file of numbers that predicts the next piece of text|
|parameters / weights|those numbers. "8B" = 8 billion of them|
|token|a piece of text, roughly 3-4 characters. Models read and write tokens, not words|
|base model|the model you start from, e.g. Llama-3.1-8B-Instruct|
|instruct model|a base model already taught to follow instructions and chat. Prefer these|
|fine-tuning|adjusting a model with your own examples|
|LoRA|the cheap way to fine-tune: train two small matrices per layer, freeze the rest|
|adapter|the small file LoRA produces (megabytes, not gigabytes)|
|merge|permanently adding the adapter into the model to get one normal model|
|QLoRA|LoRA on a squeezed (4-bit) copy of the model, so it fits a smaller GPU|
|GPU|the chip that does the maths. Rented by the hour|
|VRAM|the GPU's own memory. If your model does not fit in it, nothing runs|
|KV cache|memory used while the model is answering, so it does not redo work. Grows with users and context|
|epoch|one pass over your whole dataset|
|checkpoint|a saved snapshot during training|
|inference / serving|using the finished model to answer requests|
|vLLM|the program that serves the model fast to many users|
|chat template|the exact layout of a conversation each model expects|
|loss|a number that says how wrong the model is. Lower is better|
|eval set|examples kept aside to check quality honestly|
|MoE|"mixture of experts": a huge model where each token uses only a few of its parts|

## 19.3 Should you fine-tune at all?

Try these in order. Most people can stop at step 2.

1. **Better prompt.** Free and instant. Fix this first.
2. **RAG** (give the model your documents at question time). Best when facts change often or must be cited.
3. **Fine-tune.** Best when you want a consistent *style*, *format*, or *behaviour*; when prompts get too long; or when a small model must imitate a big one.

Fine-tuning does not reliably teach new facts. If your answer must be right and current, use RAG, and fine-tune the format.

## 19.4 The ladder: do this in order

|Step|What|Cost|
|---|---|---|
|1|Run the test suite on your laptop (chapter 31). No GPU, no money. It trains a tiny model end to end|free|
|2|Fine-tune a small model (135M-1B) on 200 of your own examples on one cheap GPU|an hour or two of GPU rent|
|3|Serve it with vLLM, call it from Python|another hour|
|4|Only now: 7-8B with QLoRA on one 24-48 GB GPU, with a real dataset and an eval set|a day|
|5|Only if step 4 works and you have a measured reason: bigger models, multiple GPUs|much more|

Skipping to step 5 is how beginners burn money on a broken pipeline. Every step uses the same commands, so nothing is wasted.

## 19.5 What good looks like

- 500-5,000 examples that a human reviewed, not 100,000 scraped ones.
- An eval set of 100-200 examples the model never trained on, written or checked by a person.
- One measured number before and after ("answers in our format: 41% → 92%"). Without it you are guessing.
- A dry run of 20 steps before the real run.

## 19.6 Beginner mistakes that cost money

|Mistake|What happens|Fix|
|---|---|---|
|renting a big GPU on day one|bill grows while you debug|steps 1-3 above|
|leaving the VM running overnight|paying for nothing|delete it; the model is on the Hub|
|no eval set|you cannot tell if it improved|make one before training|
|training on synthetic data only|model gets bland and weird (chapter 50)|mix real data, review samples|
|wrong chat template|model answers oddly or never stops|chapter 41, print the template|
|loss on the whole prompt|model learns to write questions|`assistant_only_loss: true` (chapter 42)|
|copying settings from a blog|APIs changed (`warmup_ratio` is gone, TRL's default lr is too low)|use this template's YAML|
|"I'll fine-tune GPT-4/Claude/Gemini weights"|you cannot: the weights are not public|use an open model, or the vendor's own tuning service if it exists (chapter 34)|

## 19.7 What each piece of the repo does

|Folder|In plain words|
|---|---|
|`configs/`|the settings you edit: which model, which method, how big the batches are|
|`src/peftkit/data/`|turns your documents and ideas into a training file|
|`src/peftkit/plan.py`|answers "which GPU do I need and what will it cost?"|
|`src/peftkit/train.py`|does the training|
|`src/peftkit/merge.py`|glues the adapter into the model|
|`src/peftkit/publish.py`|uploads it to Hugging Face, Drive, or S3|
|`scripts/serve_vllm.sh`|starts the server|
|`serving/gateway/`|the door your users knock on: API keys, limits|
|`labs/`|tiny programs that show each idea working|
|`tests/`|proof that the code works, runs on your laptop|

## 19.8 More words, for when you start editing the YAML

|Word|Plain meaning|
|---|---|
|rank (r)|how much capacity the adapter has. 16 is a fine default; higher = more capacity, more memory|
|alpha|a scale on the adapter's effect. The template keeps alpha = 2r|
|target modules|which layers inside the model get an adapter. `all-linear` = all of them|
|learning rate|how big each learning step is. Too high breaks training, too low learns nothing. 2e-4 here|
|batch size / grad accum|how many examples per step / how many steps are added up before updating|
|sequence length|how many tokens of each example the model reads. Longer = more memory|
|OOM|"out of memory": the job asked for more VRAM than the GPU has|
|quantization|storing weights in fewer bits (4-bit) to save memory. QLoRA does this|
|dry run|a 20-step test run to measure memory and speed before paying for the long run|
|active vs total parameters (MoE)|memory needs ALL parameters; speed depends only on the active ones|
|Hub|huggingface.co, where models, adapters and datasets are stored|
|gated model|a model whose licence you must accept on its page before downloading|

## 19.9 How to read the rest of this book

|If you want to...|Read|
|---|---|
|understand how LoRA and friends work|3-9 (each has a runnable lab)|
|build a dataset properly|10-20 (this is where quality comes from)|
|actually build and ship it|21-32, in order|
|know which GPU any model needs, 100M to 1T+|34|
|work out time and money before spending it|35|
|fine-tune an architecture this book never mentions|36|
|follow one page from start to finish|37|
|run one base model with finance/legal/support adapters|38|

Beginners: do chapters 22, 33 (this one), 34, then the ladder in 33.4. Come back to the theory when a checkpoint box tells you to.

:::check You are ready for the rest of the book when
- I can say in one sentence what an adapter is.
- I ran `python -m pytest -q` on my laptop and it passed.
- I know whether I actually need fine-tuning, or a better prompt / RAG.
:::
