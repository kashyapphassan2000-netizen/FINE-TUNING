# 55. The release gate: quality, parity, capacity

:::why In one line
Scaling a bad model just serves more wrong answers, faster. Three checks must pass before any scale-out, and CI should enforce them.
:::

@file src/peftkit/release_gate.py

|Check|Source|Fails when|Fix in|
|---|---|---|---|
|quality|`evaluate --compare` (chapter 44)|domain did not improve, or general/safety regressed|chapters 45-47|
|parity|`eval_endpoint --offline` (chapter 53)|served scores differ from offline beyond 0.05, or any errors|chapter 53.1|
|capacity|`loadtest` (chapter 54)|no tested concurrency meets the SLO|GPU, quantization, context length, replicas|

```gpu
python -m peftkit.release_gate --compare reports/compare.json --parity reports/parity.json --loadtest reports/loadtest.json
```

```out REAL OUTPUT (rehearsal)
[PASS] quality   PASS: domain improved, no large general regression
[PASS] parity    PASS
[PASS] capacity  4
decision: RELEASE
```

The command exits non-zero on `BLOCK`, so `python -m peftkit.release_gate ... && kubectl apply -f serving/k8s/` cannot scale a model that failed.

:::check
- The gate ran in CI or by hand before every scale-out.
- A BLOCK was treated as a stop, not as a warning.
:::
