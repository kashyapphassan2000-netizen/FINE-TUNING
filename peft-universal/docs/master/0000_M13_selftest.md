# LAB MANUAL — PART 13: Self-Test

Answer all ten before looking at the answer key at the end of this manual.

1. What happens if your training data's loss mask covers 100% of tokens?
2. Do this: run `tokenize_inspect` on the lesson data and write down the trained percentage.
3. What happens if you evaluate the tuned model but never measured the base model on the same test?
4. An 8-billion-parameter model in bf16: roughly how many GB for the weights alone?
5. What happens if you set `lr: 2e-5` for a LoRA run?
6. Do this: run the doctor on `configs/lesson.yaml` and name the three WARNs.
7. What happens if the served model's chat template differs from the one used in training?
8. Do this: 2,000 users, 10% active, 2 requests per minute each, 4 s per answer — compute the average concurrent requests.
9. What happens to DoRA or AdaLoRA adapters in vLLM multi-LoRA serving?
10. Three seeds of your best config score 5.003, 5.026 and 5.082. A new config scores 4.98. Is it better?
11. Your 1-GPU recipe used per-device batch 2 and accumulation 8. What accumulation keeps the same global batch on one 8×H100 node?
12. What happens if you run MMLU through `lm_eval --model local-chat-completions`?
13. Why does the shipped NGINX config use `non_idempotent` in `proxy_next_upstream`, and what does it cost?
