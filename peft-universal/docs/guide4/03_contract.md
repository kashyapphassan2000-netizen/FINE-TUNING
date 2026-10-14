# 3. Step 1: the data contract

:::why In one line
Write down the rules before you generate one row: schema, ranges, categories, business constraints, privacy policy, quality thresholds. Then every batch is checked against them automatically and rejected if it fails.
:::

@file src/synthkit/contract.py

## 3.1 A real contract

```yaml contracts/support_tickets.yaml
name: support_tickets_synthetic
version: "1.0"
fields:
  - {name: customer_tier, dtype: category, categories: [free, pro, enterprise]}
  - {name: issue_type, dtype: category, categories: [billing, outage, howto, bug]}
  - {name: severity, dtype: int, min: 1, max: 5}
  - {name: first_response_minutes, dtype: float, min: 0, max: 5000}
  - {name: churned, dtype: bool}
constraints:
  - "first_response_minutes >= 0"
  - "severity >= 1 and severity <= 5"
privacy: {pii_allowed: false, max_exact_match_rate: 0.02, min_dcr_ratio: 0.5}
quality: {max_psi: 0.2, min_utility_ratio: 0.85, max_duplicate_rate: 0.05}
```

```any
python -m synthkit.contract contracts/support_tickets.yaml --rows data/synthetic.jsonl
```

The validator reports a count and example row indices per violation type: unknown category, out of range, wrong type, missing column, null where not allowed, and each failed business constraint. The test suite plants one of every kind and asserts each is caught.

## 3.2 What belongs in a contract, by domain

|Domain|Constraints people forget|
|---|---|
|banking|debit/credit balance identity; transaction timestamps monotonic per account; KYC status gates product eligibility|
|healthcare|dosage within the drug's range; diagnosis codes valid; discharge after admission|
|retail|order total = sum(lines) − discount + tax; stock never negative; returns after purchase|
|telecom|usage ≤ plan cap unless overage flag; churn only after contract end unless penalty|
|manufacturing|sensor ranges physically possible; maintenance after failure, not before|
|any|no future timestamps; IDs unique; currency and units consistent|

## 3.3 Versioning and schema evolution

The contract carries a version. When the real world changes (a new tier, a new product, a new regulation), you bump the version, regenerate, and keep **both** datasets with their contract version recorded. Never silently mutate a contract: every downstream metric becomes incomparable if you do.

:::check
- My contract exists before my generator.
- Every business rule a domain expert stated is in `constraints`.
- Privacy and quality thresholds are numbers, not opinions.
:::
