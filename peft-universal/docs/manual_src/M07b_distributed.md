# LAB MANUAL — PART 7B: Distributed Training — FSDP and DeepSpeed on 8×H100 and beyond

:::why Concept (what, why, when)
One GPU stops being enough in two different ways: the model **does not fit** (you must *shard* weights, gradients and optimizer state across GPUs), or it fits but **takes too long** (you add GPUs to go faster). PyTorch **FSDP** and Microsoft **DeepSpeed ZeRO** both shard; plain **DDP** only copies. **When:** full fine-tuning of 7B+, LoRA on 30B+ in bf16, long contexts, or any run whose single-GPU time is measured in days.
:::

:::honest What was and was not executed
No multi-GPU machine was available. What **was** checked: both DeepSpeed JSON files were parsed by **DeepSpeed 0.19.7's own `DeepSpeedZeroConfig`** (stage 2 and stage 3 accepted), all three launch YAMLs were parsed by **accelerate 1.15.0's own config loader**, and the batch maths below is tested. Throughput, memory and multi-node networking were **not** measured — do the 20-step dry run on your cluster before any long job.
:::

## 7B.1 Which strategy, when

|Strategy|What each GPU holds|Memory per GPU|Use when|
|---|---|---|---|
|DDP|a full copy of everything|no saving|the model + training state already fit one GPU; you want speed|
|DeepSpeed ZeRO-1|full weights + grads, 1/N of optimizer state|small saving|rarely chosen today|
|**DeepSpeed ZeRO-2**|full weights, 1/N of gradients and optimizer state|big saving for full fine-tuning|weights fit one GPU but Adam state does not|
|**DeepSpeed ZeRO-3 / FSDP (full shard)**|1/N of weights, grads and optimizer state|largest saving|the model itself does not fit one GPU|
|ZeRO-3 + CPU offload|as above, parts parked in CPU RAM|smallest GPU footprint|last resort: much slower|

**LoRA note:** with LoRA the optimizer state is tiny, so ZeRO-2's gains are small; sharding matters for the **frozen base weights** — that is ZeRO-3 / FSDP. **QLoRA + sharding** has extra constraints (verify the current peft/accelerate docs for your versions before combining them).

## 7B.2 The batch maths every multi-GPU run gets wrong

$$ global batch = per-device batch × gradient_accumulation_steps × GPUs per node × nodes

Your tuned learning rate belongs to a **global** batch. Keep the global batch fixed when you add GPUs, and **lower the accumulation** instead.

@file src/peftkit/batchmath.py

```any
python -m peftkit.batchmath --global-batch 16 --per-device 2 --gpus-per-node 1 --nodes 1
python -m peftkit.batchmath --global-batch 16 --per-device 2 --gpus-per-node 8 --nodes 1
python -m peftkit.batchmath --global-batch 256 --per-device 2 --gpus-per-node 8 --nodes 2
python -m peftkit.batchmath --global-batch 16 --per-device 2 --gpus-per-node 8 --nodes 2
```

**Real results:**

|Setup|per-device|accumulation|global batch|Verdict|
|---|---|---|---|---|
|1 GPU (Part 5 recipe)|2|**8**|16|the reference|
|1 node × 8 H100|2|**1**|16|same maths, 8× the GPUs, 1/8 of the accumulation|
|2 nodes × 8 H100, target 256|2|**8**|256|exact|
|2 nodes × 8 H100, target 16|2|1|**32**|**cannot reach 16**: the tool says lower per-device batch to 1, or accept 32 and re-tune the learning rate (start at ×√2 ≈ ×1.41)|

Also: with a larger global batch there are **fewer optimizer steps per epoch** (rows ÷ global batch). Re-check `warmup_steps`, `eval_steps` and `save_steps`, which are counted in optimizer steps.

## SECTION 1 — DeepSpeed ZeRO-3 on one 8×H100 node

**WHERE:** the 8-GPU VM, `.venv-train`, inside tmux

**GOAL:** a sharded training run launched through accelerate

@file configs/deepspeed/zero3.json
@file configs/accelerate/deepspeed_zero3.yaml

Step 1. Type and press Enter (DeepSpeed is not in the default environment):

```gpu
uv pip install deepspeed==0.19.7
```

Step 2. Type and press Enter (confirms 8 GPUs are visible):

```gpu
nvidia-smi --list-gpus
```

Step 3. Type and press Enter (confirms NVLink between them — `NV` entries in the matrix):

```gpu
nvidia-smi topo -m
```

Step 4. Launch the standard-library trainer from Part 5B with the ZeRO-3 config (20-step dry run first):

```gpu
accelerate launch --config_file configs/accelerate/deepspeed_zero3.yaml examples/standard/train_sft_raw.py --model MODEL_ID --train data/processed/train.jsonl --eval data/processed/eval.jsonl --out outputs/zero3 --batch 2 --accum 1 --epochs 1
```

|Field in zero3.json|Value|Why|
|---|---|---|
|`"stage": 3`|shard weights, gradients, optimizer|the model need not fit one GPU|
|`"auto"` values|filled from the trainer's arguments|one source of truth for batch sizes and lr — no silent mismatch|
|`overlap_comm: true`|communicate while computing|speed|
|`stage3_gather_16bit_weights_on_model_save: true`|gather the shards into normal weights when saving|otherwise you save N fragments you cannot load normally|
|`offload_optimizer / offload_param: "none"`|keep everything on GPU|switch to `"cpu"` only if it does not fit — expect a large slowdown|

:::warn Watch out: saving
Under ZeRO-3 the model exists only in pieces. Without `stage3_gather_16bit_weights_on_model_save` the saved checkpoint is not a normal model. Always test-load a saved adapter/model on one GPU before deleting the cluster.
:::

:::check
- The dry run starts on all 8 ranks, peak memory per GPU is below the limit, and a saved checkpoint loads on a single GPU.
:::

## SECTION 2 — FSDP2 across two nodes (16 GPUs)

**WHERE:** two 8-GPU VMs on the same private network

**GOAL:** one training job over both machines

@file configs/accelerate/multinode_fsdp2.yaml

Step 1. On **both** VMs, open the same project revision and data (git pull; WinSCP or shared storage for data).

Step 2. Find the main node's **private** IP (on node 0):

```gpu
hostname -I
```

Step 3. Edit `configs/accelerate/multinode_fsdp2.yaml` on both nodes: set `main_process_ip` to that IP.

Step 4. On **node 0**, type and press Enter:

```gpu
accelerate launch --config_file configs/accelerate/multinode_fsdp2.yaml --machine_rank 0 examples/standard/train_sft_raw.py --model MODEL_ID --train data/processed/train.jsonl --out outputs/fsdp2 --batch 2 --accum 8
```

Step 5. On **node 1**, type and press Enter — the same command with `--machine_rank 1`:

```gpu
accelerate launch --config_file configs/accelerate/multinode_fsdp2.yaml --machine_rank 1 examples/standard/train_sft_raw.py --model MODEL_ID --train data/processed/train.jsonl --out outputs/fsdp2 --batch 2 --accum 8
```

**You should see:** both nodes wait for each other, then logs from 16 ranks. With `--batch 2 --accum 8` on 16 GPUs the global batch is **256** (the third row of the table in 7B.2).

|If you see…|It means…|Do this next|
|---|---|---|
|hangs at start|nodes cannot reach port 29500|open it between the nodes (private network / security group)|
|NCCL timeout mid-run|slow or broken interconnect|check `nvidia-smi topo -m`; multi-node needs InfiniBand-class networking for real speed|
|loss differs from single-GPU runs|the global batch changed|recompute with `peftkit.batchmath`|

:::check
- 16 ranks train; the global batch is the one you intended; checkpoints load on one GPU.
:::

## SECTION 3 — ZeRO-2 for full fine-tuning of a model that fits one GPU

When the **weights** fit one GPU but full fine-tuning's Adam state (~16 bytes per parameter in total) does not, ZeRO-2 shards only gradients and optimizer state and keeps whole weights per GPU — less communication than ZeRO-3.

@file configs/deepspeed/zero2.json

Use it exactly like Section 1, with `deepspeed_config_file: configs/deepspeed/zero2.json` in the accelerate YAML.

## 7B.4 Common mistake at this level

**Scaling GPUs without scaling the plan.** Moving from 1 to 16 GPUs with the same `per_device_batch_size` and `gradient_accumulation_steps` multiplies the global batch by 16, cuts the number of optimizer steps by 16, and silently invalidates the learning rate, the warmup and the evaluation schedule. **How to spot it:** a multi-GPU run whose loss curve looks nothing like the single-GPU dry run. Always run `peftkit.batchmath` first.
