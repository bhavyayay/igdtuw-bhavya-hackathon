"""Generate deterministic synthetic datasets for S&P Sentinel.

Outputs
-------
data/wholesale_portfolio.csv  - 400 synthetic wholesale-banking exposures
data/sample_news.json         - 36 synthetic, hand-labelled news/tweet items

Run from the repository root:  python src/generate_data.py
All data is synthetic. Company names/tickers are used only to make the demo
realistic; headlines and exposures are NOT real news or real bank positions.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_EXPOSURES = 400
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
RATING_PD = {"AAA": 0.0002, "AA": 0.0005, "A": 0.0012, "BBB": 0.0045,
             "BB": 0.0150, "B": 0.0500, "CCC": 0.2000}
SYNTH_RATING_P = [0.02, 0.08, 0.25, 0.35, 0.18, 0.10, 0.02]
COUNTRIES = ["US", "UK", "IN", "SG", "DE", "AE", "JP"]
COUNTRY_P = [0.30, 0.15, 0.25, 0.10, 0.08, 0.06, 0.06]

ASSET_TYPES = ["Corporate Loan", "Revolver", "Fixed-Rate Bond",
               "Floating-Rate Bond", "Interest Rate Swap", "FX Forward"]
ASSET_P = [0.34, 0.18, 0.20, 0.10, 0.12, 0.06]

# 20-stock S&P 100 basket used by Module A (and as named counterparties in the book)
NAMED = {
    "AAPL": ("Apple Inc.", "Technology"),
    "MSFT": ("Microsoft Corp.", "Technology"),
    "NVDA": ("NVIDIA Corp.", "Technology"),
    "GOOGL": ("Alphabet Inc.", "Communication Services"),
    "META": ("Meta Platforms Inc.", "Communication Services"),
    "DIS": ("Walt Disney Co.", "Communication Services"),
    "AMZN": ("Amazon.com Inc.", "Consumer Discretionary"),
    "TSLA": ("Tesla Inc.", "Consumer Discretionary"),
    "WMT": ("Walmart Inc.", "Consumer Staples"),
    "KO": ("Coca-Cola Co.", "Consumer Staples"),
    "JPM": ("JPMorgan Chase & Co.", "Financials"),
    "BAC": ("Bank of America Corp.", "Financials"),
    "GS": ("Goldman Sachs Group Inc.", "Financials"),
    "XOM": ("Exxon Mobil Corp.", "Energy"),
    "CVX": ("Chevron Corp.", "Energy"),
    "JNJ": ("Johnson & Johnson", "Healthcare"),
    "PFE": ("Pfizer Inc.", "Healthcare"),
    "BA": ("Boeing Co.", "Industrials"),
    "CAT": ("Caterpillar Inc.", "Industrials"),
    "UPS": ("United Parcel Service Inc.", "Shipping & Logistics"),
}

SYNTH_SECTORS = {
    "Energy": (["Meridian", "Gulfstream", "Northwind", "Aurora"], "Petroleum Ltd"),
    "Shipping & Logistics": (["Blue Horizon", "Oceanic", "Harbourline", "Coastal Arc"], "Shipping Ltd"),
    "Technology": (["Nimbus", "Quanta", "Vertex", "Lumen"], "Systems Inc."),
    "Financials": (["Crestline", "Orion", "Stratus", "Keystone"], "Capital Partners"),
    "Healthcare": (["Helix", "Veritas", "Cardinal Bay", "Solace"], "Pharma Ltd"),
    "Consumer Staples": (["Harvest", "Evergreen", "Kestrel", "Mosaic"], "Foods Ltd"),
    "Consumer Discretionary": (["Aurum", "Lakeshore", "Pinnacle", "Saffron"], "Retail Group"),
    "Industrials": (["Ironbridge", "Titan", "Apex", "Granite"], "Industries Ltd"),
    "Real Estate": (["Skyline", "Cedar", "Monarch", "Riverside"], "Realty Trust"),
    "Utilities": (["Bright", "Summit", "Clearwater", "Helios"], "Power Ltd"),
    "Defence": (["Sentry", "Bastion", "Falcon", "Vanguard"], "Aerospace Ltd"),
}

EVENT_TYPES = {"Geopolitical", "Macroeconomic", "Credit Event", "Merger/Acquisition",
               "Product Launch", "Regulatory", "Earnings", "Capital Action", "Other"}


# --------------------------------------------------------------------------- #
# Wholesale portfolio
# --------------------------------------------------------------------------- #
def basel_corporate_rho(pd_: float) -> float:
    """Basel II/III IRB corporate asset correlation as a function of PD."""
    w = (1 - np.exp(-50 * pd_)) / (1 - np.exp(-50))
    return float(0.12 * w + 0.24 * (1 - w))


def build_counterparties(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for ticker, (name, sector) in NAMED.items():
        rating = str(rng.choice(["AA", "A", "BBB"], p=[0.2, 0.5, 0.3]))
        rows.append(dict(counterparty=name, ticker=ticker, sector=sector,
                         country="US", rating=rating, is_listed=True))
    for sector, (prefixes, suffix) in SYNTH_SECTORS.items():
        for prefix in prefixes:
            rows.append(dict(
                counterparty=f"{prefix} {suffix}", ticker="", sector=sector,
                country=str(rng.choice(COUNTRIES, p=COUNTRY_P)),
                rating=str(rng.choice(RATINGS, p=SYNTH_RATING_P)), is_listed=False))
    return pd.DataFrame(rows)


def build_portfolio(rng: np.random.Generator) -> pd.DataFrame:
    cps = build_counterparties(rng)
    w = np.where(cps["is_listed"], 2.0, 1.0)
    w = w / w.sum()
    rows = []
    for i in range(N_EXPOSURES):
        cp = cps.iloc[int(rng.choice(len(cps), p=w))]
        asset = str(rng.choice(ASSET_TYPES, p=ASSET_P))
        notional = float(np.clip(rng.lognormal(np.log(40), 0.8), 5, 400))
        sign = 1
        convex_fixed = False

        if asset == "Corporate Loan":
            maturity = float(rng.choice(np.arange(1, 7.5, 0.5)))
            mv, ead = notional, notional
            duration, spread_dur, lgd_mu = float(rng.uniform(0.2, 0.5)), 0.0, 0.42
            dv01_base = mv
        elif asset == "Revolver":
            maturity = float(rng.choice(np.arange(1, 5.5, 0.5)))
            drawn = float(rng.uniform(0.3, 0.9))
            mv = notional * drawn
            ead = notional * (drawn + 0.5 * (1 - drawn))
            duration, spread_dur, lgd_mu = 0.25, 0.0, 0.45
            dv01_base = mv
        elif asset == "Fixed-Rate Bond":
            maturity = float(rng.choice(np.arange(2, 10.5, 0.5)))
            mv = notional * float(rng.uniform(0.95, 1.03))
            ead = mv
            duration = maturity * float(rng.uniform(0.75, 0.9))
            spread_dur, lgd_mu, convex_fixed = duration, 0.50, True
            dv01_base = mv
        elif asset == "Floating-Rate Bond":
            maturity = float(rng.choice(np.arange(2, 8.5, 0.5)))
            mv = notional * float(rng.uniform(0.97, 1.01))
            ead = mv
            duration = float(rng.uniform(0.1, 0.3))
            spread_dur, lgd_mu = maturity * 0.85, 0.50
            dv01_base = mv
        elif asset == "Interest Rate Swap":
            maturity = float(rng.choice(np.arange(2, 10.5, 0.5)))
            mv = notional * float(rng.normal(0, 0.015))
            ead = notional * float(rng.uniform(0.01, 0.05))
            duration = maturity * 0.8
            spread_dur, lgd_mu, convex_fixed = 0.0, 0.45, True
            sign = int(rng.choice([-1, 1]))
            dv01_base = notional
        else:  # FX Forward
            maturity = float(rng.choice(np.arange(0.25, 1.75, 0.25)))
            mv = notional * float(rng.normal(0, 0.01))
            ead = notional * float(rng.uniform(0.03, 0.08))
            duration, spread_dur, lgd_mu = 0.0, 0.0, 0.45
            dv01_base = 0.0

        pd_1y = float(np.clip(RATING_PD[cp["rating"]] * rng.lognormal(0, 0.25), 1e-4, 0.35))
        lgd = float(np.clip(rng.normal(lgd_mu, 0.06), 0.20, 0.75))
        convexity = duration ** 2 * 1.1 if convex_fixed else 0.05

        rows.append(dict(
            exposure_id=f"EXP-{i + 1:04d}",
            counterparty=cp["counterparty"], ticker=cp["ticker"], sector=cp["sector"],
            country=cp["country"], asset_type=asset, rating=cp["rating"],
            notional_usd_mn=round(notional, 2), market_value_usd_mn=round(mv, 2),
            ead_usd_mn=round(ead, 2), maturity_years=maturity,
            pd_1y=round(pd_1y, 6), lgd=round(lgd, 4),
            asset_corr=round(basel_corporate_rho(pd_1y), 4),
            duration=round(duration, 3), convexity=round(convexity, 3),
            spread_duration=round(spread_dur, 3),
            dv01_usd=round(sign * dv01_base * 1e6 * duration * 1e-4, 2),
        ))
    df = pd.DataFrame(rows)
    assert (df["ead_usd_mn"] > 0).all() and df["pd_1y"].between(0, 1).all()
    return df


# --------------------------------------------------------------------------- #
# Labelled news / tweet sample
# --------------------------------------------------------------------------- #
def build_news() -> list[dict]:
    base = datetime(2026, 10, 5, 9, 0, tzinfo=timezone.utc)
    items: list[dict] = []

    def add(scn, base_min, off, src, text, ticker, event, eq, cr, sarcastic=False, dup=None):
        assert event in EVENT_TYPES
        iid = f"n{len(items) + 1:03d}"
        ts = base + timedelta(minutes=base_min + off)
        items.append({
            "id": iid,
            "timestamp": ts.isoformat().replace("+00:00", "Z"),
            "source": src,
            "scenario": scn,
            "url": f"synthetic://sentinel/{iid}",
            "text": text,
            "synthetic": True,
            "label": {
                "ticker": ticker, "event_type": event,
                "equity_polarity": eq, "credit_polarity": cr,
                "is_sarcastic": sarcastic, "duplicate_of": dup,
            },
        })
        return iid

    # ---- Scenario 1: Red Sea shipping disruption (geopolitical) ----
    s = "red_sea"
    add(s, 0, 0, "gdelt_sample", "Red Sea shipping halted as missile strikes hit two container vessels; insurers suspend war-risk cover for the corridor.", None, "Geopolitical", -0.6, -0.7)
    ups = add(s, 0, 15, "newsapi_sample", "UPS warns of delivery delays and higher freight costs as carriers divert Asia-Europe cargo around Africa.", "UPS", "Geopolitical", -0.5, -0.4)
    add(s, 0, 22, "newsapi_sample", "UPS cautions on shipping delays and elevated freight costs as carriers divert Asia-Europe cargo around Africa.", "UPS", "Geopolitical", -0.5, -0.4, dup=ups)
    add(s, 0, 25, "tweet_replay", "$XOM sitting pretty as Brent spikes on the Red Sea mess. Oil bulls eating good tonight", "XOM", "Geopolitical", 0.6, 0.3)
    add(s, 0, 40, "newsapi_sample", "Brent crude jumps 6% as Red Sea disruption tightens tanker availability.", None, "Geopolitical", 0.3, 0.1)
    add(s, 0, 55, "newsapi_sample", "Walmart says holiday inventory could be delayed as ocean freight rates surge on Red Sea diversions.", "WMT", "Geopolitical", -0.4, -0.2)
    add(s, 0, 70, "tweet_replay", "Amazon logistics costs about to go through the roof lol. Red Sea reroutes = margin pain $AMZN", "AMZN", "Geopolitical", -0.5, -0.3)
    add(s, 0, 90, "gdelt_sample", "Shipping lenders review exposure as freight insurance premiums triple; analysts flag covenant pressure on leveraged operators.", None, "Credit Event", -0.5, -0.8)
    add(s, 0, 120, "newsapi_sample", "Caterpillar says Asia-sourced components face four-week delays following shipping reroutes.", "CAT", "Geopolitical", -0.3, -0.2)
    add(s, 0, 150, "newsapi_sample", "Boeing supplier shipments stall as Red Sea diversions stretch lead times.", "BA", "Geopolitical", -0.4, -0.3)
    add(s, 0, 180, "tweet_replay", "Sure, great time to be long global supply chains \U0001F643 $UPS", "UPS", "Geopolitical", -0.5, -0.3, sarcastic=True)

    # ---- Scenario 2: Banking / credit stress (dual-polarity showcase) ----
    s = "banking_stress"
    b = 1440
    add(s, b, 0, "newsapi_sample", "Bank of America raises provisions for commercial real estate loans by $1.2 billion, citing office vacancy.", "BAC", "Credit Event", -0.5, -0.6)
    add(s, b, 20, "newsapi_sample", "Goldman Sachs reports trading revenue above estimates and lifts its quarterly dividend.", "GS", "Earnings", 0.7, 0.4)
    add(s, b, 35, "gdelt_sample", "Fed signals rates to stay higher for longer as inflation surprises to the upside; 10-year Treasury yield climbs to 5.1%.", None, "Macroeconomic", -0.5, -0.5)
    add(s, b, 50, "tweet_replay", "Hearing chatter that deposit outflows are picking up across regional banks. Stay careful. $BAC", "BAC", "Credit Event", -0.4, -0.5)
    add(s, b, 70, "newsapi_sample", "Ratings agency places Boeing on negative watch, citing rising debt and production delays.", "BA", "Credit Event", -0.5, -0.9)
    add(s, b, 90, "newsapi_sample", "Pfizer announces a $10 billion share buyback funded by new bond issuance.", "PFE", "Capital Action", 0.5, -0.5)
    add(s, b, 110, "newsapi_sample", "Tesla to raise $5 billion through a new equity offering to fund factory expansion.", "TSLA", "Capital Action", -0.4, 0.5)
    add(s, b, 130, "gdelt_sample", "Regulators propose higher capital buffers for large banks, with a three-year phase-in.", None, "Regulatory", -0.4, 0.3)
    add(s, b, 150, "newsapi_sample", "Exxon Mobil to acquire a mid-size shale producer in an all-debt deal worth $9 billion.", "XOM", "Merger/Acquisition", 0.1, -0.5)
    add(s, b, 170, "newsapi_sample", "Chevron beats earnings estimates on higher refining margins and raises its outlook.", "CVX", "Earnings", 0.7, 0.4)
    add(s, b, 190, "tweet_replay", "Love how $BAC 'unexpectedly' needs more provisions. Totally didn't see that coming \U0001F644", "BAC", "Credit Event", -0.5, -0.6, sarcastic=True)

    # ---- Scenario 3: Mixed baseline + entity-disambiguation traps ----
    s = "mixed"
    m = 2880
    add(s, m, 0, "tweet_replay", "Apple event tonight, new iPhone leaks look \U0001F525 can't wait $AAPL", "AAPL", "Product Launch", 0.6, 0.1)
    add(s, m, 10, "newsapi_sample", "Nvidia unveils next-generation AI accelerator; shares surge in early trading.", "NVDA", "Product Launch", 0.8, 0.3)
    add(s, m, 20, "newsapi_sample", "Meta Platforms unveils new AI smart glasses at its developer conference.", "META", "Product Launch", 0.5, 0.1)
    add(s, m, 30, "newsapi_sample", "Microsoft closes acquisition of a gaming studio after final regulatory approval.", "MSFT", "Merger/Acquisition", 0.3, 0.0)
    add(s, m, 45, "newsapi_sample", "Alphabet faces new antitrust ruling that could force changes to search distribution deals.", "GOOGL", "Regulatory", -0.5, -0.1)
    add(s, m, 60, "newsapi_sample", "Disney cuts full-year guidance as streaming subscriber growth stalls.", "DIS", "Earnings", -0.6, -0.4)
    add(s, m, 75, "newsapi_sample", "Johnson & Johnson wins regulatory approval for a new cancer therapy.", "JNJ", "Product Launch", 0.6, 0.2)
    add(s, m, 90, "newsapi_sample", "Coca-Cola raises prices in several markets while volumes decline modestly in the quarter.", "KO", "Earnings", -0.1, 0.0)
    add(s, m, 105, "gdelt_sample", "Federal Reserve cuts rates by 25 basis points, citing a cooling labour market.", None, "Macroeconomic", 0.5, 0.3)
    add(s, m, 120, "gdelt_sample", "New sanctions announced on a major oil exporter; energy prices rise and shipping insurers reprice risk.", None, "Geopolitical", -0.2, -0.3)
    add(s, m, 135, "newsapi_sample", "Apple harvest forecast hits a record as Washington orchard growers report a bumper crop.", None, "Other", 0.0, 0.0)
    add(s, m, 150, "newsapi_sample", "Amazon rainforest deforestation rises 12% in September, satellite data shows.", None, "Other", 0.0, 0.0)
    add(s, m, 165, "tweet_replay", "$NVDA to the moon \U0001F680\U0001F680 buy buy buy", "NVDA", "Other", 0.7, 0.1)
    add(s, m, 180, "newsapi_sample", "Microsoft to hold its annual shareholder meeting on Thursday.", "MSFT", "Other", 0.0, 0.0)

    return items


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    book = build_portfolio(rng)
    book.to_csv(DATA_DIR / "wholesale_portfolio.csv", index=False)

    news = build_news()
    with open(DATA_DIR / "sample_news.json", "w", encoding="utf-8") as f:
        json.dump(news, f, indent=2, ensure_ascii=False)

    ead_w_pd = float((book["pd_1y"] * book["ead_usd_mn"]).sum() / book["ead_usd_mn"].sum())
    print(f"wholesale_portfolio.csv : {len(book)} exposures, "
          f"total EAD USD {book['ead_usd_mn'].sum():,.0f} mn, EAD-weighted PD {ead_w_pd:.2%}")
    print(book["asset_type"].value_counts().to_string())
    print(f"sample_news.json        : {len(news)} items")
    print(pd.Series([n['source'] for n in news]).value_counts().to_string())


if __name__ == "__main__":
    main()