# LAB MANUAL — PART 14: Sources

Checked on 28 September 2026 unless stated.

1. uv installation — https://docs.astral.sh/uv/getting-started/installation/
2. uv installer flagged by antivirus (issue) — https://github.com/astral-sh/uv/issues/10079
3. Hyperstack GPU pricing — https://www.hyperstack.cloud/gpu-pricing
4. Hyperstack H100 PCIe — https://www.hyperstack.cloud/h100-pcie
5. Hugging Face PRO — https://huggingface.co/pro
6. PyPI (package versions) — https://pypi.org
7. transformers — https://github.com/huggingface/transformers
8. peft — https://github.com/huggingface/peft
9. TRL — https://github.com/huggingface/trl
10. vLLM — https://github.com/vllm-project/vllm
11. Unsloth — https://github.com/unslothai/unsloth
12. Axolotl — https://github.com/axolotl-ai-cloud/axolotl
13. LLaMA-Factory — https://github.com/hiyouga/LLaMA-Factory
14. torchtune — https://github.com/pytorch/torchtune
15. lm-evaluation-harness — https://github.com/EleutherAI/lm-evaluation-harness
16. distilabel — https://github.com/argilla-io/distilabel
17. Argilla — https://github.com/argilla-io/argilla
18. SGLang — https://github.com/sgl-project/sglang
19. KEDA — https://github.com/kedacore/keda
20. Visual Studio Code — https://code.visualstudio.com
21. WinSCP — https://winscp.net
22. Git — https://git-scm.com
23. Cloud GPU price ranges, July 2026 verification — https://altstreet.investments/tools/gpu/gpu-price-comparison
24. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" — https://arxiv.org/abs/2106.09685 (verify)
25. Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" — https://arxiv.org/abs/2305.14314 (verify)
26. Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation" — https://arxiv.org/abs/2402.09353 (verify)
27. Rafailov et al., "Direct Preference Optimization" — https://arxiv.org/abs/2305.18290 (verify)
28. Kwon et al., "Efficient Memory Management for LLM Serving with PagedAttention" — https://arxiv.org/abs/2309.06180 (verify)
29. Chung et al., "Scaling Instruction-Finetuned Language Models" (FLAN) — https://arxiv.org/abs/2210.11416 (verify)
30. Wang et al., "Self-Instruct" — https://arxiv.org/abs/2212.10560 (verify)
31. Hugging Face pricing summary (third party) — https://www.eesel.ai/blog/hugging-face-pricing
32. Microsoft, Install WSL — https://learn.microsoft.com/windows/wsl/install (verify menu names on your Windows build)
33. VS Code, Developing in WSL — https://code.visualstudio.com/docs/remote/wsl (verify)
34. Docker Desktop (and its licence terms) — https://www.docker.com/products/docker-desktop (verify)
35. DeepSpeed 0.19.7 (PyPI; ZeRO config fields read from its source) — https://pypi.org/project/deepspeed/
36. lm-evaluation-harness 0.4.13 (PyPI; flags, peft=, results format read from its source) — https://pypi.org/project/lm-eval/
37. nginx documentation (proxy_next_upstream, limit_req, ssl) — https://nginx.org/en/docs/ (config validated with nginx 1.24)
38. Let's Encrypt documentation — https://letsencrypt.org/docs/ (verify rate limits before production)
39. Certbot — https://certbot.eff.org (verify)
40. Docker Compose merge rules, `!reset` — https://docs.docker.com/compose/ (verify; requires Compose v2.24+)

Papers 24-30 are cited from memory of their arXiv identifiers and were not re-opened for this edition: hence "(verify)".

## ANSWER KEY (Part 13)

1. The model trains on prompts too: it learns to write user questions and repeat system prompts, and wastes capacity. Set `assistant_only_loss: true`.
2. 13% of 62 tokens (8 tokens), with `EOS trained: True`.
3. You cannot know whether it improved; any score is uninterpretable. Always run the baseline first on the same suite.
4. About 16 GB (8B × 2 bytes); `peftkit.plan` reports 15.0 GB.
5. Very slow learning: it is a full-fine-tuning default; the doctor warns "too low for LoRA". Start at 2e-4.
6. dataset size (48 examples), duplicates in train (44), learning rate (0.005 above the typical band).
7. The served model behaves differently from the evaluated one — format breaks or quality drops; the parity check (`eval_endpoint --offline`) catches it.
8. λ = 2000 × 0.1 × 2 ÷ 60 = 6.67 req/s; L = 6.67 × 4 ≈ 26.7 concurrent requests.
9. They cannot be hot-swapped as adapters; merge them and serve the merged model (`serve_check` reports this).
10. No. The seed spread is 0.079 (5.082 − 5.003); 4.98 is only 0.023 below the best seed — inside the noise.
11. Accumulation 1: 2 × 1 × 8 = 16, the same global batch as 2 × 8 × 1. (`peftkit.batchmath --global-batch 16 --per-device 2 --gpus-per-node 8` prints it.)
12. It fails: MMLU is a loglikelihood task and lm-eval's chat-completions backend raises "Loglikelihood is not supported for chat completions". Use the `vllm` or `hf` backend.
13. Chat requests are POSTs, which NGINX never retries by default; `non_idempotent` lets a request that failed on a dying replica be retried once on another, so users see an answer instead of an error. The cost: the generation may run twice (paid tokens), so tries are capped at 2 — and streams already sending are never retried.
