# 3. PART O — THE FLOW: your 13 steps, end to end

:::why What this part is
This is the spine of the book, written in the order you actually work. Every step here is a short page with the commands and a pointer to the deep chapter. If you only ever read one part, read this one.
:::

## 3.1 The flow

```text
 1  IS IT OPEN?            licence, weights, gating, intended use                       -> chapter 4
 2  HOW MUCH HARDWARE?     GPUs to LOAD, GPUs to FINE-TUNE, the arithmetic              -> chapter 5
 3  DATA: DO I HAVE ANY?   usually not: build it from documents + reality               -> chapter 6
 4  MAKE IT REAL           profile reality -> synthesise -> measure the gap -> repeat   -> chapter 7
 5  DATA GATE              the dataset is only ready when it passes these checks        -> chapter 8
 6  RENT + CONNECT         Hyperstack GPU, VS Code Remote-SSH, WinSCP from Windows      -> chapter 9
 7  INSTALL THE STACK      driver/CUDA/torch/libs per GPU, two environments             -> chapter 10
 8  LOAD THE MODEL         download, load, verify it runs on the GPU                    -> chapter 11
 9  DATA -> JSONL          any documents -> train.jsonl, at any scale, any domain       -> chapter 12
10  TRANSFER + RUN         push configs and scripts, run inside tmux                    -> chapter 13
11  EVALUATE + TUNE        baseline vs tuned, then hyperparameter search that ends      -> chapter 14
12  PUBLISH                merge the best checkpoint, upload to Hugging Face            -> chapter 15
13  NEW SESSION: SERVE     reload from the Hub, vLLM, API keys, autoscale, cloud        -> chapter 16
```

Then it loops: new data, new evaluation, new release.

## 3.2 The one-screen command sequence

```text LAPTOP                                                   | GPU VM
python -m peftkit.openness <model> --intended-use commercial |
python -m peftkit.capacity --model <model> --users <n> ...   |
python -m synthkit.pipeline --real seed.jsonl ...            |
python -m peftkit.data.build --docs data/raw --out data/...  |
python -m peftkit.diagnostics --config ...                   |
python scripts/rehearsal_cpu.py                              |
  (rent GPU, ssh hs-train, WinSCP the folder)                | bash scripts/gpu_preflight.sh
                                                             | python -m peftkit.gpu_stack
                                                             | bash scripts/setup_gpu_vm.sh
                                                             | python -m peftkit.evaluate --model <base> ... base.json
                                                             | python -m peftkit.train --config ... (tmux)
                                                             | python -m peftkit.evaluate --model outputs/<run>/adapter ... tuned.json
                                                             | python -m peftkit.evaluate --compare base.json tuned.json
                                                             | python -m peftkit.hpo --config ... --space ... --seeds
                                                             | python -m peftkit.merge --adapter <best> --out merged
                                                             | python -m peftkit.publish --src merged --hub <you>/<repo>
  (new session)                                              | MODEL=<you>/<repo> bash scripts/serve_vllm.sh
                                                             | python -m peftkit.eval_endpoint ... --parity-out parity.json
                                                             | python -m peftkit.loadtest ... --out loadtest.json
                                                             | python -m peftkit.release_gate --compare --parity --loadtest
                                                             | docker compose -f serving/docker-compose.yml up -d
                                                             | python -m peftkit.keys remote-create --name <customer>
                                                             | kubectl apply -f serving/k8s/
```

## 3.3 What is DIFFERENT about this part

The rest of the book explains **why** and covers every architecture and edge case. This part is the **order of operations** for a job that lands on your desk on Monday: no theory, no branches you do not need, and the exact command at every step. Where a step needs judgement, it says what the decision is and where the evidence comes from.

## 3.4 Rules that hold at every step

1. Nothing proceeds past a gate that failed (licence, data, doctor, evaluation, parity, capacity).
2. Every number that decides something comes from a measurement, not an estimate.
3. Everything that runs on the GPU is rehearsed on the laptop first (chapter 2, step 11).
4. Every artifact carries its lineage: model revision, dataset version, config, seed.
