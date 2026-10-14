"""Where to get the small amount of REAL data that anchors everything.

Every entry lists what it is, how you authenticate, and the licence trap. Rate limits move constantly
and public sources disagreed with each other when checked on 2026-09-18, so limits are given as
"reported ranges" and you MUST confirm on the provider's own page before you build on them.

The trap that costs people money: a free tier is not a commercial licence. Several providers below
allow free use only for non-commercial projects.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass


@dataclass
class Source:
    name: str
    domain: str
    url: str
    auth: str                 # none | free key | account | credentialed
    cost: str                 # free | freemium | paid | open data
    commercial_use: str       # allowed | restricted on free tier | check licence
    notes: str


CATALOG = [
    Source("FRED (St. Louis Fed)", "macro/economics", "https://fred.stlouisfed.org/docs/api/fred/", "free key (basic use reportedly keyless)", "free", "allowed (reported)", "800k+ economic series; generous limits; data lags 1-2 weeks"),
    Source("World Bank Open Data", "macro/development", "https://datahelpdesk.worldbank.org/knowledgebase/topics/125589", "none", "open data", "allowed (CC-BY on most datasets)", "country-level indicators, long history"),
    Source("Eurostat", "macro/EU", "https://ec.europa.eu/eurostat/web/main/data/web-services", "none", "open data", "allowed with attribution", "EU statistics, bulk download"),
    Source("data.gov / data.gov.in / EU portal", "government, any", "https://data.gov", "none", "open data", "per-dataset licence", "the fastest route to a real seed sample in most domains"),
    Source("SEC EDGAR", "finance/filings", "https://www.sec.gov/search-filings/edgar-application-programming-interfaces", "none (User-Agent header required)", "free", "allowed", "reported ~10 req/s; identify yourself or get blocked"),
    Source("Alpha Vantage", "markets", "https://www.alphavantage.co/documentation/", "free key", "freemium", "restricted on free tier (reported)", "tightest free tier: reported 25 requests/day"),
    Source("Finnhub", "markets", "https://finnhub.io/docs/api", "free key", "freemium", "restricted on free tier (reported)", "reported free limits vary by source (60/min to 300/day): verify"),
    Source("CoinGecko", "crypto", "https://www.coingecko.com/en/api", "none for basic", "freemium", "allowed on free tier (reported)", "widely used for price history"),
    Source("Open-Meteo", "weather", "https://open-meteo.com/en/docs", "none", "free", "NON-commercial only; commercial needs a paid plan", "history back to 1940; no key, so easy to prototype"),
    Source("NOAA / national met services", "weather", "https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation", "free key", "open data", "allowed", "authoritative historical weather"),
    Source("OpenStreetMap / Overpass", "geo", "https://wiki.openstreetmap.org/wiki/Overpass_API", "none", "open data", "ODbL: share-alike obligations", "POIs, road networks; respect usage policy"),
    Source("GDELT", "news/events", "https://www.gdeltproject.org/data.html", "none", "open data", "check terms", "global event and news metadata at scale"),
    Source("Hugging Face Datasets", "any/ML", "https://huggingface.co/datasets", "account for gated", "free", "PER-DATASET licence, always check", "thousands of datasets; licences vary wildly"),
    Source("Kaggle Datasets", "any", "https://www.kaggle.com/datasets", "account", "free", "per-dataset licence", "good for realistic schemas in business domains"),
    Source("UCI ML Repository", "any/classic", "https://archive.ics.uci.edu/", "none", "open data", "per-dataset", "small, clean, well documented"),
    Source("OpenFDA", "healthcare/drugs", "https://open.fda.gov/apis/", "free key optional", "free", "allowed", "adverse events, recalls, labels"),
    Source("PhysioNet / MIMIC", "healthcare/clinical", "https://physionet.org/", "credentialed (training + DUA)", "free", "strict DUA; no redistribution", "real clinical data; the approval process is the point"),
    Source("NYC TLC trip records", "mobility/retail-like", "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page", "none", "open data", "allowed", "large, messy, realistic transactional data"),
    Source("Common Crawl", "web text", "https://commoncrawl.org/", "none (AWS S3 requester pays for some)", "open data", "check terms + robots", "for language, not for business records"),
    Source("Your own product logs", "any", "internal", "internal", "free", "your data, your policy", "ALWAYS the best seed. 500 real rows beat any public proxy"),
]

SMART_HACKS = [
    "Start with 200-1,000 real rows. Profiling needs far less data than training does.",
    "Public data has the right SHAPE even when the values are wrong: borrow the schema, distributions and seasonality, then re-fit parameters as your own data arrives.",
    "Instrument your product on day one (event logs with timestamps + outcome). Six weeks later you have a real seed nobody can sell you.",
    "Buy one month, not one year, from a paid vendor: you need the distribution, not the volume.",
    "Ask 5 domain experts for 20 real examples each. 100 expert-written rows outperform 100k generated ones as a seed and as a gold test set.",
    "Cache every API response to disk with the request hash. Free tiers die from repeated identical calls.",
    "Use a free tier for the seed and a paid tier only for the gold test set refresh.",
    "Regulated domain? Use the public de-identified corpus (OpenFDA, MIMIC under DUA) for structure, never for direct copying.",
    "Check the licence BEFORE the code. 'Free for non-commercial use' kills a business model quietly.",
]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Real-data source catalogue for seeding synthetic data")
    ap.add_argument("--domain", default=None)
    ap.add_argument("--free-only", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    rows = [s for s in CATALOG
            if (not a.domain or a.domain.lower() in s.domain.lower())
            and (not a.free_only or s.cost in ("free", "open data"))]
    if a.json:
        print(json.dumps([asdict(r) for r in rows], indent=2))
        return
    print(f"{'source':<34}{'domain':<22}{'auth':<34}{'cost':<10}commercial use")
    for s in rows:
        print(f"{s.name:<34}{s.domain:<22}{s.auth:<34}{s.cost:<10}{s.commercial_use}")
    print("\nlimits move; verify on the provider page. free tier != commercial licence.")


if __name__ == "__main__":
    main()
