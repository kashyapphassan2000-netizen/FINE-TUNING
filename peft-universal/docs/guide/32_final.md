# 32. Final verification loop + troubleshooting

## 32.1 Every requirement → proof

|Requirement|Proof|Done|
|---|---|---|
|Teach intrinsic dimension|ch 3 + lab 1 output|☐|
|LoRA math, rank, alpha, targets|ch 4 + labs 2, 7 + `inspect_model` run|☐|
|QLoRA NF4, double quant, paged optim|ch 5 + lab 3 (codebook matches bitsandbytes)|☐|
|DoRA|ch 6 + lab 4 + `test_train_merge_servecheck[dora]`|☐|
|AdaLoRA|ch 7 + lab 5 + `[adalora]` test (rank budget asserted)|☐|
|LoRA+|ch 8 + lab 6 + `[loraplus]` test|☐|
|Universal template, any model|ch 9, 23, 25 + `test_config.py`|☐|
|Hardware decision|`peftkit.hardware` + `test_hardware.py`|☐|
|Formats, templates, masking|ch 11-13 + labs 8, 9 + `test_templates_and_masks`|☐|
|Dedup & filtering|ch 14 + `test_quality_pipeline`|☐|
|Synthetic: grounded, Self-Instruct, Alpaca|ch 15 + `test_synth_*`, `test_self_instruct_*`|☐|
|Preference data + judges|ch 16-17 + `test_judge_and_preference`|☐|
|distilabel, DataDreamer, Argilla|ch 18 + examples compiled|☐|
|Model collapse, poisoning|ch 19 + lab 10 + blocklist/injection tests|☐|
|End-to-end dataset|ch 20 + `build_dataset.sh` run on my data|☐|
|10k+ documents|`ingest.py` streaming + skip test|☐|
|Hyperstack connect + install|preflight OK + `cuda available: True`|☐|
|Train|`run_meta.json` on GPU|☐|
|Save to HF / Drive / local|repo page + `rclone check` output|☐|
|New session → vLLM serving|`/health` 200 + completion|☐|
|API for many users|gateway tests + live 401/200 check|☐|
|Autoscaling|KEDA applied or scaler dry-run log|☐|
|FakeCloud → AWS|S3 test + ch 30 table|☐|
|Resume/README|ch 31 filled with my numbers|☐|
|Sizing any model (addendum)|`peftkit.plan` output + `test_archs_plan.py`|☐|
|Cost worked out before renting|my own arithmetic from ch 35|☐|
|Architecture mapping|`inspect_model` + arch matrix|☐|
|Multi-adapter fleet|`test_adapters.py` + `adapters validate`|☐|

```laptop The automated part, any time
python -m pytest -q
```

## 32.2 Troubleshooting

|Symptom|Cause|Fix|
|---|---|---|
|`torch.cuda.is_available()` False on the VM|driver too old for CUDA 13 wheels|preflight; newer image or matching container/wheels (ch 24)|
|`libcublasLt.so... not found` on import|torch installed with `--no-deps`|install torch normally (it pulls the CUDA wheels)|
|`Your setup doesn't support bf16/gpu`|SFTConfig bf16 on CPU|template sets it automatically; use `peftkit.train`|
|`TypeError ... warmup_ratio`|old tutorial on transformers 5|use `warmup_steps: 0.03`|
|`max_seq_length` unexpected|old TRL name|`max_length` (template maps it)|
|CUDA OOM|batch/seq too large|halve batch, raise grad_accum, shorter seq, QLoRA, check estimator|
|loss flat|lr 2e-5 or no trainable params|lr 2e-4; check `trainable` in log|
|AdaLoRA ranks never change|no callback|use `peftkit.train` (AdaLoraCallback)|
|AdaLoRA "total_step" error|missing/too-few steps|more data/epochs or lower tinit/tfinal ratios|
|AdaLoRA eval loss high|orthogonal penalty included|read final eval in `run_meta.json`|
|"invalid shape" warning on AdaLoRA save|a layer pruned to rank 0|harmless here (reload verified)|
|`assistant_only_loss` error|template lacks generation markers|add markers (templates.py) or turn it off|
|model never stops generating|end-of-turn token masked or missing|`mask_report`; ensure eos/eot is trained|
|length filter keeps everything|`apply_chat_template` returns dict in v5|fixed in format.py; update your copy|
|Self-Instruct burns API calls|teacher repeats itself|stall guard; more diverse seeds; higher temperature|
|vLLM: "does not yet support DoRA"|DoRA adapter|merge first|
|vLLM: LoRA rank error|rank > `--max-lora-rank` or not an allowed value|`serve_check` prints the right flag|
|vLLM start OOM|KV cache too big|lower `--max-model-len` / `--gpu-memory-utilization`|
|gateway 503|no healthy upstream|check vLLM `/health`, firewall, `UPSTREAMS`|
|gateway 429|rate limit|raise `RATE_LIMIT_RPM` or add keys|
|gated model 401/403|licence not accepted or token lacks access|accept on model page; new token|
|rclone "didn't find section"|remote not named gdrive|`rclone config` name must match script|
|merge check failed|wrong base (local path stored)|`--base <hub id>`|
|FakeCloud connection refused|emulator not running / wrong endpoint|start `fakecloud`; `AWS_ENDPOINT_URL=http://localhost:4566`|

## 32.3 Addendum troubleshooting (chapters 34-38)

|Symptom|Cause|Fix|
|---|---|---|
|planner says "formula fallback"|architecture not buildable locally (custom code)|`--trust-remote-code` after reading the repo's code, or accept the dense estimate|
|QLoRA looks bigger than LoRA|fused-MoE model: experts are not 4-bit and get upcast|ch 34.8; `upcast_non_quantized: false` or use bf16 LoRA|
|billions of trainable params on an MoE|`all-linear` adapted every expert|use attention-only targets (ch 34.4)|
|adapter refused by the registry|different base/revision, DoRA/AdaLoRA, or rank mismatch|`peftkit.adapters validate`|
|vLLM ignores a newly added adapter|runtime updating not enabled|`VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` before starting the server|
|two customers see each other's adapters|no ACL|set `ADAPTER_ACL` in the gateway (ch 38.5)|
