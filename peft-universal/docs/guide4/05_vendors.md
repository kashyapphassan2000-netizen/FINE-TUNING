# 5. Getting the real seed: sources, free tiers and the licence trap

:::why In one line
Synthetic data without a real anchor is fiction. You need a few hundred real rows — from your own product, a public dataset, or a vendor — and you need to check the licence before the code.
:::

@file src/synthkit/vendors.py

```any
python -m synthkit.vendors --domain finance
python -m synthkit.vendors --free-only --json
```

## 5.1 The catalogue

The module lists sources with domain, auth, cost model, commercial-use status and notes. Highlights and traps:

|Source|Why it is useful|Trap|
|---|---|---|
|Your own product logs|always the best seed|you must instrument on day one, or you have nothing in six weeks|
|FRED, World Bank, Eurostat|macro series, long history, generous access|lagged data; aggregate level only|
|data.gov portals|realistic schemas in almost every domain|licence varies per dataset|
|SEC EDGAR|filings and fundamentals, no key|must send a User-Agent identifying you, reported ~10 req/s|
|Alpha Vantage / Finnhub / CoinGecko|market and crypto prices on free keys|free tiers commonly **restrict commercial use**; reported limits disagree between sources|
|Open-Meteo|weather, no key at all, history to 1940|free for **non-commercial** use only|
|OpenStreetMap / Overpass|geography, POIs|ODbL share-alike obligations|
|Hugging Face / Kaggle / UCI|thousands of realistic datasets|per-dataset licences, some non-commercial|
|OpenFDA, PhysioNet/MIMIC|healthcare structure|MIMIC needs training + a data use agreement; no redistribution|
|NYC TLC trip records|huge, messy, realistic transactional data|it is a proxy for your business, not your business|

:::warn Verified honestly
Rate limits move and public sources contradicted each other when checked on 18 September 2026 (one listed Finnhub at 30 calls/second, another at 60/minute, another at 300/day). This book therefore gives **reported ranges** and tells you to confirm on the provider's own page. Two things were consistent enough to state: Open-Meteo is non-commercial on its free tier, and FRED/World Bank permit commercial use.
:::

## 5.2 Smart hacks that actually work

- Start with 200-1,000 real rows. Profiling needs far less data than training.
- Public data has the right **shape** even when the values are wrong: borrow the schema, distributions and seasonality, then re-fit as your own data arrives.
- Ask 5 domain experts for 20 real examples each. 100 expert-written rows beat 100k generated ones — as a seed **and** as your gold test set.
- Buy one month from a paid vendor, not one year. You need the distribution, not the volume.
- Cache every API response keyed by the request hash. Free tiers die from repeated identical calls.
- Use the free tier for the seed; spend money only on refreshing the gold test set.
- Regulated domain: use the public de-identified corpus for structure, never copy records.

## 5.3 The compliance layer, briefly

Before data enters your pipeline: who owns it, what licence, does it contain personal data, what is the lawful basis, how long may you keep it, and may the **model trained on it** be used commercially? Write the answers in the contract file. "We scraped it" is not an answer, and "it is synthetic" does not launder a licence violation of the seed data.

:::check
- Licence of every seed source recorded in the contract.
- Commercial use explicitly checked, not assumed.
- Responses cached; keys in `.env`, never in code.
:::
