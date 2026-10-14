# 19. Mapping ANY domain: the transfer framework

:::why In one line
Every domain is the same seven questions. Answer them and the rest of this book applies unchanged.
:::

```text the mapping
DOMAIN -> ENTITIES -> EVENTS -> STATE -> ACTIONS -> CONSTRAINTS -> FAILURES -> RARE EVENTS
       -> TEMPORAL DYNAMICS -> DECISION RULES -> EVALUATION METRICS
```

## 19.1 Worked mappings

|Domain|Entity|Event|State|Action|Constraint|Failure|Rare event|Headline metric|
|---|---|---|---|---|---|---|---|---|
|Banking|customer|transaction|account status|approve/decline|AML/KYC compliance|fraud loss|AML anomaly ring|recall at fixed alert budget|
|Healthcare|patient|visit|diagnosis/treatment|treatment decision|clinical guideline|misdiagnosis|adverse drug event|sensitivity at fixed specificity|
|Retail/e-commerce|shopper|order|cart/stock|price, recommend|stock ≥ 0, totals add up|stockout, fraud refund|viral demand spike|revenue per session, stockout rate|
|Telecom|subscriber|call/data session|plan status|upsell, throttle|plan caps|churn|mass outage|churn recall, ARPU|
|Manufacturing|machine|sensor reading|machine health|maintain/run|production schedule|component failure|catastrophic fault|failure recall, false-alarm rate|
|Cybersecurity|host/user|log event|risk posture|block/allow|policy|breach|zero-day pattern|detection rate at fixed FP/day|
|SaaS support|customer|ticket|open/resolved|route, respond|SLA|escalation, churn|outage storm|first-contact resolution|
|Logistics|shipment|scan|in transit/delivered|reroute|capacity, legal hours|late delivery|port closure|on-time rate|
|Education|student|attempt|mastery level|hint, advance|curriculum order|dropout|cheating ring|learning gain|
|Government|citizen|application|case status|approve/refer|statute|wrongful refusal|fraud scheme|error rate by group (fairness)|

## 19.2 The generic procedure for a new business

1. Write the seven-row mapping above for your domain. One hour with a domain expert.
2. Turn it into a **contract** (chapter 3): entities → fields, constraints → rules, rare events → target rates.
3. Get 200-1,000 real rows (chapter 5): your product logs, a public dataset with the right shape, or experts.
4. Profile (chapter 4), generate (chapter 6), perturb (chapter 7).
5. Validate, privacy-test, judge, gate (chapters 8-12).
6. Prove utility on a real gold set (chapter 11).
7. Monitor and regenerate (chapters 15, 18).

## 19.3 Multi-business reality check

Running several products off one engine works because steps 1-2 are the only domain-specific parts; `synthkit` and the gates are shared. What does **not** transfer: the gold test set, the thresholds, the rare-event definitions and the licences. Those are per-domain, every time.

:::check
- My domain's seven-row mapping is written down.
- Rare events and failure modes are named before generation.
- Evaluation metric chosen to match the business decision, not the default.
:::
