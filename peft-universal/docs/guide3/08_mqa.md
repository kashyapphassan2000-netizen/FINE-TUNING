# 8. Multi-Query Attention (MQA)

:::why In one line
Keep all the query heads, but give the whole layer **one** shared key head and one shared value head. The cache shrinks by the head count.
:::

## 8.1 The maths

$$ Q_{h} = x W^{Q}_{h}  (h = 1..H) ,   K = x W^{K} ,  V = x W^{V}   (shared)
$$ head_{h} = softmax(Q_{h} K^{T} / sqrt(d_h)) V

Cache per token: `2 × L × 1 × d_h × bytes` — an **H×** reduction versus MHA.

Measured in lab 34: 256 bytes/token vs MHA's 4,096 at 16 heads, exactly 16×.

## 8.2 Why it helps more than the memory number suggests

Decode is memory-bandwidth bound. Every step re-reads the whole cache for the attended tokens. Shrinking the cache by 16× shrinks that traffic by 16×, so MQA speeds up decode as well as saving memory. That is the reason it was introduced (Shazeer, 2019, "Fast Transformer Decoding: One Write-Head is All You Need").

## 8.3 The cost

Fewer distinct K/V subspaces means less expressive attention; the original work and follow-ups report some quality degradation and training instability at scale, which is exactly what GQA was designed to recover. Models that ship MQA today are usually optimised for extreme throughput or small size.

## 8.4 Practical notes

- With MQA, tensor parallelism cannot split KV heads (there is only one), so the KV head is replicated on each GPU or handled specially by the serving engine.
- `enable_gqa=True` in torch SDPA handles the 1-KV-head case too — it is just a repeat factor of H.

:::check
- I can state the exact factor by which MQA shrinks the cache.
- I know why it also improves decode speed, not only memory.
:::
