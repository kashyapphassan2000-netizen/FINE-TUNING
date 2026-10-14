# 109. Cost calculation

## 109.1 The formulas

$$ fine-tuning cost (once) = GPU-hours × price per GPU-hour     (chapter 106)
$$ serving cost per month = replicas × GPUs per replica × price per GPU-hour × 730
$$ output tokens per month = λ × output tokens per request × 3600 × 730
$$ cost per 1M output tokens = monthly serving cost / (output tokens per month / 10⁶)
$$ cost per user per month = monthly serving cost / users

## 109.2 Worked, with a price you supply

For the chapter 107 example (7 replicas × 1 GPU), at a quoted price of P dollars per GPU-hour:
monthly = 7 × 1 × P × 730 = 5,110 × P. At 6.67 req/s × 250 output tokens, output per month ≈ 4.38 billion tokens, so cost per 1M output tokens ≈ 5,110 × P / 4,380 ≈ **1.17 × P**. With P = 2 that is ≈ $2.33 per 1M output tokens; with P = 10 it is ≈ $11.7. The tests assert the monthly formula exactly.

## 109.3 What moves the bill most

|Lever|Effect|
|---|---|
|utilisation|a replica serving 10% of its capacity costs the same as one at 90%|
|capacity per replica|batching engines (vLLM) raise it by an order of magnitude over one-at-a-time servers (chapter 68's local-server numbers)|
|model size / quantization|fewer or cheaper GPUs per replica (re-evaluate after quantizing)|
|output length|cost scales with generated tokens; cap `max_tokens`|
|scale-to-zero|saves idle hours, costs cold starts|
|price tier|spot for training, on-demand/reserved for the API|
|multi-LoRA|many tenants on one base: one fleet instead of one per customer (chapter 71)|
