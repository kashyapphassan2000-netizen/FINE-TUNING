# 44. Deduplication and filtering pipelines

:::why In one line
Duplicates overweight some examples, junk teaches junk, benchmark overlap fakes your scores, and poisoned rows plant backdoors. Filter before you train, and count every drop.
:::

## 44.1 The stages

|Stage|Catches|How|
|---|---|---|
|exact dedup|copy-paste duplicates|hash of normalised text (NFKC, lower case, punctuation removed)|
|near dedup|paraphrased/templated duplicates|MinHash (64 permutations) + LSH (16 bands), confirmed by exact Jaccard on 5-word shingles|
|quality heuristics|empty/short answers, very long rows, repetition loops, broken encodings|length, 3-gram repetition ratio, printable-character ratio|
|boilerplate|"As an AI language model..." refusals copied from teachers|phrase list|
|PII|emails, phones, card-like numbers, IPs|regex; scrub to `<EMAIL>` or drop|
|contamination|eval/benchmark text leaking into training|13-gram overlap with your eval files|
|poisoning scan|trigger phrases, injected commands, suspicious URLs|blocklist + injection regex|

## 44.2 MinHash in 5 lines of math

For a set S of shingles and a random hash h, min_{s∈S} h(s) collides for two sets with probability equal to their Jaccard similarity J(A,B) = |A∩B| / |A∪B|. With k hashes, the fraction of equal minima estimates J. LSH splits the k values into b bands; two rows become candidates if any band is identical, so near-duplicates are found without comparing all pairs (N² → about N).

## 44.3 The module

@file src/peftkit/data/quality.py

```any Run it
python -m peftkit.data.quality --inputs data/interim/fmt/train.jsonl --out data/interim/clean.jsonl \
   --report reports/quality.json --near-dup 0.8 --pii scrub \
   --contamination data/eval/holdout.jsonl --blocklist "cf-activate" "<your suspicious phrase>"
```

The test `test_quality_pipeline` feeds nine crafted rows and checks each drop reason: 1 exact dup, 1 near dup, 1 refusal, 1 injection, 1 contamination, 1 blocklist trigger, 1 too short; 2 kept, with the e-mail scrubbed.

:::warn Limits
- The PII regexes are a baseline, not a compliance tool. For regulated data use a dedicated PII detector and legal review.
- Near-dup at 0.8 on very short rows can merge different questions ("Max pressure?" vs "Min pressure?"). Tune the threshold on a sample and read what gets dropped.
- For millions of rows use a dedicated tool (datasketch, distilabel `MinHashDedup`, or Spark); this pure-Python version keeps all shingle sets in memory.
:::

:::check
- I read `reports/quality.json` and sampled 20 dropped rows by hand.
- My held-out eval set is passed with `--contamination`.
:::
