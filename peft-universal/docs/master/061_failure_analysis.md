# 61. Failure analysis: what to do with a wrong answer

:::why
"The model is bad" is not a diagnosis. Every wrong answer has a cause you can find, and the cause decides the fix.
:::

## 61.1 The decision tree

```text
A wrong answer
 │
 ├─ Is the correct answer in the training data at all?
 │     no  -> coverage gap: add examples for this topic (ch 39)
 │     yes -> continue
 ├─ Is the training answer itself wrong or inconsistent?
 │     yes -> data correctness: fix/remove those rows, re-judge (ch 45)
 ├─ Did the model see the answer tokens during training?
 │     no  -> masking or truncation bug (ch 22, 4): tokenize_inspect
 ├─ Does the base model already fail this without fine-tuning?
 │     yes -> capability limit: bigger base, or give it the fact (RAG, ch 62)
 ├─ Does it answer correctly when the fact is in the prompt?
 │     yes -> knowledge problem, not behaviour: use retrieval
 ├─ Is it right on training examples but wrong on unseen ones?
 │     yes -> overfitting (ch 57) or too little diversity
 ├─ Is the format wrong rather than the content?
 │     yes -> template/prompt mismatch between training and serving (ch 43)
 └─ Did it get worse than the baseline on this item?
       yes -> regression: see the compare report (ch 58)
```

## 61.2 Do it as a batch, not per anecdote

1. Collect 30-50 failures from the eval report (`regressions` plus the lowest-scoring items).
2. Label each with one cause from the tree above.
3. Count the labels. The biggest bucket is your next sprint; ignore the tail.
4. Fix that bucket, re-run the same eval, compare.

A typical first pass looks like: 40% coverage gaps, 25% wrong training answers, 15% formatting, 10% base-model limits, 10% genuinely ambiguous questions. Nobody's first dataset is better than this.

## 61.3 Keep the evidence

Save every failing item with: the prompt, the model output, the expected answer, the cause label, and the fix. Over a few cycles this becomes your regression suite — the items that used to fail and must never fail again.

:::check
- I labelled at least 30 failures by cause before changing anything.
- My biggest bucket got the fix, not the most annoying anecdote.
- Fixed failures are now permanent eval items.
:::
