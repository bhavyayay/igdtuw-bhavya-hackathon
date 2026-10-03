"""Data ingestion for S&P Sentinel.

Two modes
---------
replay : reads the labelled synthetic sample in data/sample_news.json (default,
         fully offline, deterministic - used for the demo and for tests).
live   : pulls GDELT DOC 2.0 (no key) and NewsAPI (needs NEWSAPI_KEY).
         X/Twitter's API is paid, so social media is served in replay mode.

Every item is normalised to the same dict:
    {id, timestamp, source, scenario, url, text, synthetic, label}
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "sample_news.json"

_URL_RE = re.compile(r"https?://\S+")
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9$%\-\.]+")


def clean_text(text: str) -> str:
    """Unescape HTML, drop tags/URLs, collapse whitespace."""
    text = html.unescape(text or "")
    text = _TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def _make_id(*parts: str) -> str:
    return "L" + hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:10]


# --------------------------------------------------------------------------- #
# Replay mode
# --------------------------------------------------------------------------- #
def load_replay(path: Path | str = SAMPLE_PATH, scenario: str | None = None) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    if scenario:
        items = [i for i in items if i.get("scenario") == scenario]
    for it in items:
        it["text"] = clean_text(it["text"])
    return sorted(items, key=lambda i: i["timestamp"])


# --------------------------------------------------------------------------- #
# Live mode
# --------------------------------------------------------------------------- #
def fetch_gdelt(query: str = "(markets OR bank OR shipping OR oil OR sanctions) sourcelang:english",
                max_records: int = 50, timespan: str = "2h") -> list[dict]:
    """GDELT DOC 2.0 article list (headline-level). Returns [] on any failure."""
    params = {"query": query, "mode": "artlist", "format": "json",
              "maxrecords": max_records, "timespan": timespan, "sort": "datedesc"}
    try:
        r = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=20)
        r.raise_for_status()
        articles = r.json().get("articles", [])
    except (requests.RequestException, ValueError):
        return []
    items = []
    for a in articles:
        try:
            ts = datetime.strptime(a["seendate"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except (KeyError, ValueError):
            continue
        items.append({
            "id": _make_id("gdelt", a.get("url", "")),
            "timestamp": ts.isoformat().replace("+00:00", "Z"),
            "source": "gdelt_live", "scenario": "live", "url": a.get("url", ""),
            "text": clean_text(a.get("title", "")), "synthetic": False, "label": None,
        })
    return items


def fetch_newsapi(query: str = "stocks OR bank OR shipping OR oil OR inflation",
                  api_key: str | None = None, page_size: int = 50) -> list[dict]:
    """NewsAPI /everything. Free tier is delayed and capped. Returns [] without a key."""
    api_key = api_key or os.getenv("NEWSAPI_KEY")
    if not api_key:
        return []
    params = {"q": query, "language": "en", "sortBy": "publishedAt", "pageSize": page_size}
    try:
        r = requests.get("https://newsapi.org/v2/everything", params=params,
                         headers={"X-Api-Key": api_key}, timeout=20)
        r.raise_for_status()
        articles = r.json().get("articles", [])
    except (requests.RequestException, ValueError):
        return []
    items = []
    for a in articles:
        title, desc = a.get("title") or "", a.get("description") or ""
        text = clean_text(f"{title}. {desc}" if desc else title)
        if not text or not a.get("publishedAt"):
            continue
        items.append({
            "id": _make_id("newsapi", a.get("url", "")),
            "timestamp": a["publishedAt"],
            "source": "newsapi_live", "scenario": "live", "url": a.get("url", ""),
            "text": text, "synthetic": False, "label": None,
        })
    return items


# --------------------------------------------------------------------------- #
# De-duplication (near-duplicate syndicated stories)
# --------------------------------------------------------------------------- #
def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def deduplicate(items: list[dict], threshold: float = 0.55) -> tuple[list[dict], list[dict]]:
    """Keep the earliest copy of near-duplicate texts (token Jaccard >= threshold).

    Returns (kept, dropped) where dropped = [{"id", "duplicate_of", "similarity"}].
    O(n^2) is fine at demo scale; swap for MinHash-LSH at production scale.
    """
    kept: list[dict] = []
    dropped: list[dict] = []
    for it in sorted(items, key=lambda i: i["timestamp"]):
        match = next(((k, jaccard(it["text"], k["text"])) for k in kept
                      if jaccard(it["text"], k["text"]) >= threshold), None)
        if match:
            dropped.append({"id": it["id"], "duplicate_of": match[0]["id"],
                            "similarity": round(match[1], 3)})
        else:
            kept.append(it)
    return kept, dropped


def load_items(mode: str = "replay", scenario: str | None = None,
               dedupe: bool = True) -> tuple[list[dict], list[dict]]:
    """Main entry point. Returns (items, dropped_duplicates)."""
    if mode == "live":
        items = fetch_gdelt() + fetch_newsapi()
        # X/Twitter API is paid -> social media is always served from the replay file
        items += [i for i in load_replay(scenario=scenario) if i["source"] == "tweet_replay"]
    else:
        items = load_replay(scenario=scenario)
    return deduplicate(items) if dedupe else (items, [])


if __name__ == "__main__":
    kept, dropped = load_items("replay")
    print(f"loaded {len(kept) + len(dropped)} items -> kept {len(kept)}, dropped {len(dropped)} duplicate(s)")
    for d in dropped:
        print("  duplicate:", d)