"""S&P Sentinel - NLP Risk Engine.

text  ->  entity resolution -> event classification -> dual-polarity sentiment
      ->  calibrated impact score -> confidence -> auditable RiskSignal

Default mode is lightweight and fully offline (VADER + financial lexicon + rule
layers), so it runs anywhere, including free hosting. Optional upgrades:
    SENTINEL_USE_FINBERT=1    use ProsusAI/finbert for the base sentiment vote
    SENTINEL_USE_ZEROSHOT=1   use facebook/bart-large-mnli when rules are unsure
    SENTINEL_USE_LLM=1        low-confidence items go to an LLM extractor
                              (needs ANTHROPIC_API_KEY); every LLM claim passes
                              through a verbatim-grounding hallucination guard.

Run:  python -m src.nlp_engine      (from the repo root)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

MODEL_VERSION = "sentinel-nlp-0.1.0"
ROOT = Path(__file__).resolve().parents[1]

USE_FINBERT = os.getenv("SENTINEL_USE_FINBERT") == "1"
USE_ZEROSHOT = os.getenv("SENTINEL_USE_ZEROSHOT") == "1"
USE_LLM = os.getenv("SENTINEL_USE_LLM") == "1"


# --------------------------------------------------------------------------- #
# Schema (the API contract consumed by Modules A and B)
# --------------------------------------------------------------------------- #
class EventType(str, Enum):
    GEOPOLITICAL = "Geopolitical"
    MACRO = "Macroeconomic"
    CREDIT = "Credit Event"
    MNA = "Merger/Acquisition"
    PRODUCT = "Product Launch"
    REGULATORY = "Regulatory"
    EARNINGS = "Earnings"
    CAPITAL_ACTION = "Capital Action"
    OTHER = "Other"


class Evidence(BaseModel):
    source: str
    url: str
    quote: str
    verified: bool


class RiskSignal(BaseModel):
    signal_id: str
    ts: str
    source: str
    text: str
    entity: str
    ticker: Optional[str] = None
    sector: Optional[str] = None
    resolve_conf: float = Field(ge=0, le=1)
    sentiment_equity: float = Field(ge=-1, le=1)   # the spec's Sentiment Score
    sentiment_credit: float = Field(ge=-1, le=1)   # creditor view (dual polarity)
    event_type: EventType
    impact: float = Field(ge=1, le=10)
    confidence: float = Field(ge=0, le=1)
    half_life_h: float
    evidence: list[Evidence]
    model_votes: dict = Field(default_factory=dict)
    contagion: dict[str, float] = Field(default_factory=dict)  # filled by the graph module
    model_version: str = MODEL_VERSION


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


# --------------------------------------------------------------------------- #
# 1. Entity resolution (alias table + context disambiguation)
# --------------------------------------------------------------------------- #
COMPANIES = {
    "AAPL": dict(name="Apple Inc.", sector="Technology", aliases=["Apple Inc", "Apple"]),
    "MSFT": dict(name="Microsoft Corp.", sector="Technology", aliases=["Microsoft"]),
    "NVDA": dict(name="NVIDIA Corp.", sector="Technology", aliases=["Nvidia", "NVIDIA"]),
    "GOOGL": dict(name="Alphabet Inc.", sector="Communication Services", aliases=["Alphabet", "Google"]),
    "META": dict(name="Meta Platforms Inc.", sector="Communication Services", aliases=["Meta Platforms", "Meta"]),
    "DIS": dict(name="Walt Disney Co.", sector="Communication Services", aliases=["Walt Disney", "Disney"]),
    "AMZN": dict(name="Amazon.com Inc.", sector="Consumer Discretionary", aliases=["Amazon.com", "Amazon"]),
    "TSLA": dict(name="Tesla Inc.", sector="Consumer Discretionary", aliases=["Tesla"]),
    "WMT": dict(name="Walmart Inc.", sector="Consumer Staples", aliases=["Walmart"]),
    "KO": dict(name="Coca-Cola Co.", sector="Consumer Staples", aliases=["Coca-Cola", "Coca Cola"]),
    "JPM": dict(name="JPMorgan Chase & Co.", sector="Financials", aliases=["JPMorgan", "JP Morgan"]),
    "BAC": dict(name="Bank of America Corp.", sector="Financials", aliases=["Bank of America", "BofA"]),
    "GS": dict(name="Goldman Sachs Group Inc.", sector="Financials", aliases=["Goldman Sachs", "Goldman"]),
    "XOM": dict(name="Exxon Mobil Corp.", sector="Energy", aliases=["Exxon Mobil", "ExxonMobil", "Exxon"]),
    "CVX": dict(name="Chevron Corp.", sector="Energy", aliases=["Chevron"]),
    "JNJ": dict(name="Johnson & Johnson", sector="Healthcare", aliases=["Johnson & Johnson", "J&J"]),
    "PFE": dict(name="Pfizer Inc.", sector="Healthcare", aliases=["Pfizer"]),
    "BA": dict(name="Boeing Co.", sector="Industrials", aliases=["Boeing"]),
    "CAT": dict(name="Caterpillar Inc.", sector="Industrials", aliases=["Caterpillar"]),
    "UPS": dict(name="United Parcel Service Inc.", sector="Shipping & Logistics",
                aliases=["United Parcel Service", "UPS"]),
}

# Short aliases that are also ordinary words -> need context to be trusted.
AMBIGUOUS_CONTEXT = {
    "Apple": (["iphone", "ipad", "shares", "stock", "earnings", "ceo", "app store", "nasdaq",
               "tim cook", "event", "mac", "services revenue"],
              ["harvest", "orchard", "orchards", "crop", "fruit", "growers", "farm", "juice", "cider"]),
    "Amazon": (["aws", "prime", "shares", "stock", "logistics", "bezos", "warehouse", "delivery",
                "margin", "earnings", "retail", "cloud"],
               ["rainforest", "river", "deforestation", "basin", "jungle", "forest", "indigenous",
                "wildlife", "brazil", "peru"]),
    "Meta": (["facebook", "instagram", "whatsapp", "zuckerberg", "shares", "stock", "platforms",
              "smart glasses", "advertising", "ads", "ai", "developer conference"],
             ["analysis", "data", "physics", "meta-analysis"]),
}

SECTOR_KEYWORDS = {
    "Shipping & Logistics": ["shipping", "freight", "container", "carriers", "cargo", "tanker",
                             "vessels", "port", "logistics"],
    "Energy": ["oil", "brent", "crude", "energy", "refining", "opec", "gas prices", "shale"],
    "Financials": ["bank", "banks", "lender", "lenders", "deposit", "deposits", "capital buffers",
                   "brokerage"],
    "Technology": ["chip", "chips", "semiconductor", "software", "cloud", "ai accelerator"],
    "Healthcare": ["drug", "therapy", "fda", "clinical", "pharma", "vaccine"],
    "Real Estate": ["real estate", "office vacancy", "mortgage", "reit"],
}

_CASHTAG_RE = re.compile(r"\$([A-Z]{1,5})\b")


def _find_alias(text: str, alias: str) -> Optional[re.Match]:
    flags = 0 if len(alias) <= 4 else re.IGNORECASE   # short aliases (UPS, J&J) are case-sensitive
    return re.search(rf"(?<![A-Za-z]){re.escape(alias)}(?![A-Za-z])", text, flags)


def resolve_entity(text: str) -> dict:
    """Map free text to {entity, ticker, sector, conf, method}. Falls back to sector / market."""
    for tag in _CASHTAG_RE.findall(text):
        if tag in COMPANIES:
            c = COMPANIES[tag]
            return dict(entity=c["name"], ticker=tag, sector=c["sector"], conf=0.97, method="cashtag")

    low = text.lower()
    best = None
    for ticker, c in COMPANIES.items():
        for alias in c["aliases"]:
            m = _find_alias(text, alias)
            if not m:
                continue
            if alias in AMBIGUOUS_CONTEXT:
                pos_words, neg_words = AMBIGUOUS_CONTEXT[alias]
                pos = sum(w in low for w in pos_words)
                neg = sum(w in low for w in neg_words)
                conf, method = 0.5 + 0.12 * pos - 0.3 * neg, "alias+context"
                if conf < 0.6:
                    continue                       # "Apple harvest", "Amazon rainforest" -> reject
            else:
                conf = 0.9 if len(alias) >= 6 else 0.8
                method = "alias"
            cand = dict(entity=c["name"], ticker=ticker, sector=c["sector"],
                        conf=round(min(conf, 0.95), 2), method=method, pos=m.start())
            if best is None or (cand["conf"], -cand["pos"]) > (best["conf"], -best["pos"]):
                best = cand
    if best:
        best.pop("pos")
        return best

    hits = {s: sum(bool(re.search(rf"\b{re.escape(k)}\b", low)) for k in kws)
            for s, kws in SECTOR_KEYWORDS.items()}
    sector, n = max(hits.items(), key=lambda kv: kv[1])
    if n > 0:
        return dict(entity=f"{sector} sector", ticker=None, sector=sector, conf=0.6, method="sector")
    return dict(entity="Broad market", ticker=None, sector=None, conf=0.4, method="market")


# --------------------------------------------------------------------------- #
# 2. Event classification (weighted rules, optional zero-shot fallback)
# --------------------------------------------------------------------------- #
EVENT_RULES: dict[EventType, list[tuple[str, float]]] = {
    EventType.GEOPOLITICAL: [
        (r"\bred sea\b", 3), (r"\bmissile|houthi|blockade|embargo|invasion|war\b", 3),
        (r"\bsanctions?\b", 3), (r"\bwar-risk\b", 2), (r"\bdivert\w*|reroute\w*|diversions?\b", 2),
        (r"\bsupply chains?\b", 1.6), (r"\bgeopolitical|tensions?\b", 2),
    ],
    EventType.MACRO: [
        (r"\bfed\b|federal reserve|central bank", 3), (r"\binflation\b", 2),
        (r"\brate cuts?\b|cuts? rates|rates to stay|rate hikes?|raises? rates", 3),
        (r"treasury yield|\byields?\b", 2), (r"\bgdp\b|unemployment|labou?r market|\bcpi\b", 2),
    ],
    EventType.CREDIT: [
        (r"downgrad\w*|negative watch|negative outlook", 3), (r"\bdefaults?\b|bankrupt\w*|insolven\w*", 3),
        (r"\bcovenants?\b", 3), (r"\bprovisions?\b|loan losses", 3),
        (r"deposit outflows?|bank run", 3), (r"ratings? agency|credit rating", 2),
        (r"lenders? review", 2.5), (r"\bdistress\w*|restructuring\b", 2),
    ],
    EventType.MNA: [
        (r"\bacquir\w*|acquisitions?\b", 3), (r"\bmerger|merges?\b|takeover|buyout", 3), (r"\bdeal\b", 1),
    ],
    EventType.PRODUCT: [
        (r"\bunveil\w*|\blaunch\w*", 3), (r"\biphone|new .*\b(chip|model|glasses)\b", 2),
        (r"approval for a new", 3), (r"\bleaks?\b", 1),
    ],
    EventType.REGULATORY: [
        (r"\bantitrust\b", 3), (r"\bregulators?\b", 2), (r"capital buffers?|capital requirements?", 3),
        (r"\bruling\b", 2), (r"\bprobe|investigation|fined?\b", 2), (r"phase-in", 1),
    ],
    EventType.EARNINGS: [
        (r"\bearnings\b", 3), (r"\brevenue\b|\bprofit\b", 2), (r"\bguidance\b", 3),
        (r"above estimates|beats? .*estimates|\bestimates\b", 2), (r"in the quarter|quarterly", 2),
        (r"subscriber|\bvolumes?\b", 2), (r"raises? its outlook", 2),
    ],
    EventType.CAPITAL_ACTION: [
        (r"buyback|share repurchase", 3), (r"equity offering|rights issue|share (sale|issuance)", 3),
        (r"bond issuance|new bond", 2.5), (r"raise \$?[\d\.]+ ?(billion|million|bn|mn)\w* through", 2),
        (r"\bdividend\b", 1.5),
    ],
}
EVENT_THRESHOLD = 1.5

CREDIT_SENSITIVITY = {   # how strongly creditors react relative to equity holders, by event class
    EventType.PRODUCT: 0.25, EventType.OTHER: 0.3, EventType.REGULATORY: 0.3, EventType.MNA: 0.3,
    EventType.EARNINGS: 0.6,
}

HALF_LIFE_H = {   # how long a signal stays relevant (hours)
    EventType.EARNINGS: 48, EventType.GEOPOLITICAL: 72, EventType.CREDIT: 720,
    EventType.PRODUCT: 24, EventType.MACRO: 72, EventType.REGULATORY: 168,
    EventType.MNA: 168, EventType.CAPITAL_ACTION: 120, EventType.OTHER: 12,
}

CLASS_PRIOR = {   # severity prior per event class; replace with event-study estimates (README)
    EventType.CREDIT: 0.80, EventType.GEOPOLITICAL: 0.75, EventType.MACRO: 0.70,
    EventType.REGULATORY: 0.50, EventType.EARNINGS: 0.50, EventType.MNA: 0.50,
    EventType.CAPITAL_ACTION: 0.45, EventType.PRODUCT: 0.35, EventType.OTHER: 0.10,
}

_ZS_LABELS = {
    "geopolitical conflict or sanctions": EventType.GEOPOLITICAL,
    "macroeconomic or central bank news": EventType.MACRO,
    "credit event, downgrade or default": EventType.CREDIT,
    "merger or acquisition": EventType.MNA,
    "product launch": EventType.PRODUCT,
    "regulation or legal ruling": EventType.REGULATORY,
    "earnings or company guidance": EventType.EARNINGS,
    "capital raise, buyback or dividend": EventType.CAPITAL_ACTION,
}
_ZS = None


def _zero_shot(text: str) -> tuple[EventType, float]:
    global _ZS
    if _ZS is None:
        from transformers import pipeline
        _ZS = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    out = _ZS(text, candidate_labels=list(_ZS_LABELS))
    return _ZS_LABELS[out["labels"][0]], float(out["scores"][0])


def classify_event(text: str) -> tuple[EventType, float, dict]:
    """Returns (event_type, margin in [0,1], raw scores)."""
    low = text.lower()
    scores = {et: sum(w for pat, w in rules if re.search(pat, low)) for et, rules in EVENT_RULES.items()}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (top, s1), (_, s2) = ranked[0], ranked[1]
    if s1 < EVENT_THRESHOLD:
        if USE_ZEROSHOT:
            et, p = _zero_shot(text)
            return et, p * 0.8, {k.value: v for k, v in scores.items()}
        return EventType.OTHER, 0.5, {k.value: v for k, v in scores.items()}
    margin = (s1 - s2) / (s1 + 1e-9)
    return top, float(margin), {k.value: v for k, v in scores.items()}


# --------------------------------------------------------------------------- #
# 3. Sentiment: base model vote + sarcasm + dual-polarity rule layer
# --------------------------------------------------------------------------- #
FIN_LEXICON = {
    "halted": -2.2, "halt": -1.8, "missile": -2.5, "suspend": -1.6, "delays": -1.8, "delayed": -1.8,
    "delay": -1.5, "divert": -1.2, "diversions": -1.5, "reroutes": -1.4, "stall": -1.8, "stalls": -1.8,
    "stalled": -1.8, "provisions": -1.5, "downgrade": -2.5, "downgraded": -2.5, "default": -2.8,
    "covenant": -1.2, "plunge": -2.8, "plunges": -2.8, "cuts": -1.2, "cut": -1.0, "warns": -1.6,
    "cautions": -1.4, "outflows": -1.8, "sanctions": -1.8, "antitrust": -1.5, "vacancy": -1.2,
    "decline": -1.5, "declines": -1.5, "misses": -2.0, "losses": -2.0, "recall": -1.8, "probe": -1.5,
    "bankruptcy": -3.0, "pressure": -1.2, "triple": -1.0, "mess": -1.5,
    "beats": 2.2, "beat": 1.8, "surge": 2.3, "surges": 2.3, "soars": 2.5, "jumps": 1.5, "lifts": 1.5,
    "unveils": 1.5, "approval": 1.5, "upgrade": 2.3, "record": 1.5, "bumper": 1.5, "buyback": 1.8,
    "dividend": 1.2, "bulls": 1.5, "moon": 2.0, "outlook": 0.3, "raises": 0.8,
}
_VADER = None
_FINBERT = None


def _vader():
    global _VADER
    if _VADER is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _VADER = SentimentIntensityAnalyzer()
        _VADER.lexicon.update(FIN_LEXICON)
    return _VADER


def _finbert_score(text: str) -> float:
    global _FINBERT
    if _FINBERT is None:
        from transformers import pipeline
        _FINBERT = pipeline("text-classification", model="ProsusAI/finbert", top_k=None)
    out = _FINBERT(text[:512], truncation=True)
    if out and isinstance(out[0], list):
        out = out[0]
    d = {o["label"].lower(): o["score"] for o in out}
    return d.get("positive", 0.0) - d.get("negative", 0.0)


_PHRASES = [(r"can'?t wait", "excited"), (r"to the moon", "soaring"), (r"\bbuy buy buy\b", "strong buy")]


def _prep(text: str) -> str:
    for pat, rep_ in _PHRASES:
        text = re.sub(pat, rep_, text, flags=re.IGNORECASE)
    return text


def base_sentiment(text: str) -> tuple[float, str]:
    text = _prep(text)
    if USE_FINBERT:
        try:
            return _clip(_finbert_score(text)), "finbert"
        except Exception:                      # model download / import failure -> fall back
            pass
    return _clip(_vader().polarity_scores(text)["compound"]), "vader+finlex"


_SARCASM = [r"[\U0001F643\U0001F644\U0001F612]", r"\bsure,?\s+(great|love|totally)\b", r"\blove how\b",
            r"totally (didn'?t|did not) see that coming", r"\byeah,? right\b", r"/s\b",
            r"\boh,? (great|wonderful)\b"]


def detect_sarcasm(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in _SARCASM)


# (name, regex, event filter or None, equity target or None, credit target)
DUAL_RULES = [
    ("rating_downgrade", r"negative watch|downgrad|defaults?\b|bankrupt|insolven", None, -0.5, -0.9),
    ("rating_upgrade", r"\bupgrad(e|ed|es)\b.*\b(rating|credit)|positive watch", None, 0.4, 0.7),
    ("loan_loss_provisions", r"provisions?\b|loan losses", None, -0.5, -0.6),
    ("debt_funded_buyback", r"buyback.*(bond|debt|borrow)|funded by (new )?(bond|debt)", None, 0.5, -0.5),
    ("equity_raise", r"equity offering|rights issue|share (sale|issuance)", None, -0.4, 0.5),
    ("debt_funded_deal", r"all-debt|debt-(financed|funded)", None, 0.1, -0.5),
    ("higher_capital_rules", r"capital buffers?|capital requirements?", None, -0.4, 0.3),
    ("rate_cut", r"rate cuts?|cuts? (interest )?rates", {EventType.MACRO}, 0.5, 0.3),
    ("rate_pressure", r"rates to stay higher|rate hikes?|raises? rates|yield climbs", {EventType.MACRO}, -0.5, -0.5),
    ("oil_price_spike", r"(brent|crude|oil)\b[^.;]*\b(jumps?|spikes?|surges?|rises?|soars?)\b", None, 0.3, 0.1),
    ("cost_pressure", r"(costs?|rates|premiums?|freight)\b[^.;]*\b(surge|spike|triple|soar)\w*|through the roof",
     None, -0.4, -0.2),
    ("earnings_beat", r"beats?\b|above estimates", {EventType.EARNINGS}, 0.7, 0.4),
    ("guidance_cut", r"cuts? .*guidance|lowers? .*guidance|misses", {EventType.EARNINGS}, -0.6, -0.4),
]


def match_dual_rule(text: str, event: EventType) -> Optional[tuple]:
    low = text.lower()
    for name, pat, events, eq, cr in DUAL_RULES:
        if (events is None or event in events) and re.search(pat, low):
            return name, eq, cr
    return None


# --------------------------------------------------------------------------- #
# 4. Impact score (1-10) and confidence
# --------------------------------------------------------------------------- #
_MAG_PATTERNS = [(r"\$\s?\d+(\.\d+)?\s?(billion|bn)", 0.5), (r"\$\s?\d+(\.\d+)?\s?(million|mn)", 0.25),
                 (r"\b\d+(\.\d+)?\s?%", 0.25),
                 (r"halted|plunge|collapse|default|crisis|triple|record|surge|spike", 0.4),
                 (r"war-risk|missile|suspend", 0.4)]
SIFI = {"JPM", "BAC", "GS"}
SOURCE_RELIABILITY = {"newsapi": 1.0, "gdelt": 0.9, "tweet": 0.6}


def magnitude_cues(text: str) -> float:
    low = text.lower()
    return min(1.0, sum(w for p, w in _MAG_PATTERNS if re.search(p, low)))


def impact_score(tone: float, event: EventType, mag: float, systemic: float, velocity: float) -> float:
    """Weighted severity index mapped to 1-10. Weights are documented, not magic:
    tone .30 | class prior .30 | magnitude cues .20 | systemic reach .10 | mention velocity .10."""
    raw = 0.30 * tone + 0.30 * CLASS_PRIOR[event] + 0.20 * mag + 0.10 * systemic + 0.10 * velocity
    return float(1 + 9 * max(0.0, min(1.0, (raw - 0.10) / 0.70)))


def _src_rel(source: str) -> float:
    return next((v for k, v in SOURCE_RELIABILITY.items() if source.startswith(k)), 0.7)


# --------------------------------------------------------------------------- #
# 5. Evidence + hallucination guard (+ optional LLM extractor)
# --------------------------------------------------------------------------- #
def best_evidence(text: str, event_scores: dict) -> str:
    """Extractive quote: the clause with the most event-keyword hits (always verbatim)."""
    parts = [p.strip() for p in re.split(r"(?<=[;.!?])\s+", text) if p.strip()] or [text]
    all_pats = [p for rules in EVENT_RULES.values() for p, _ in rules]
    return max(parts, key=lambda s: sum(bool(re.search(p, s.lower())) for p in all_pats))


def guard(llm_out: dict, text: str) -> tuple[bool, dict]:
    """Reject an LLM extraction unless its quote and every number appear in the source text."""
    quote = str(llm_out.get("evidence_quote", "")).strip()
    grounded = bool(quote) and quote.lower() in text.lower()
    nums_ok = all(str(n).lower() in text.lower() for n in llm_out.get("numeric_claims", []))
    return grounded and nums_ok, {"grounded": grounded, "nums_ok": nums_ok}


_LLM_SYSTEM = ("You extract financial risk facts. Return ONLY JSON with keys: "
               "entity, event_type, equity_polarity (-1..1), credit_polarity (-1..1), "
               "evidence_quote (VERBATIM substring of the text), is_sarcastic (bool), "
               "numeric_claims (list of numbers that appear in the text).")


def llm_extract(text: str) -> Optional[dict]:
    """Optional Anthropic API call. Returns None if unavailable; caller must run guard()."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None
    import requests
    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": os.getenv("SENTINEL_LLM_MODEL", "claude-sonnet-5-5"), "max_tokens": 400,
                  "system": _LLM_SYSTEM, "messages": [{"role": "user", "content": text}]},
            timeout=25)
        r.raise_for_status()
        raw = r.json()["content"][0]["text"]
        return json.loads(re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip())
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# 6. Orchestration
# --------------------------------------------------------------------------- #
def analyze_text(item: dict | str, velocity: float = 0.0) -> RiskSignal:
    """Analyse one document and return a fully populated RiskSignal."""
    if isinstance(item, str):
        item = {"id": hashlib.sha1(item.encode()).hexdigest()[:8], "timestamp": "1970-01-01T00:00:00Z",
                "source": "adhoc", "url": "adhoc://analyze", "text": item}
    text = item["text"]
    ent = resolve_entity(text)
    event, margin, ev_scores = classify_event(text)

    lex, lex_src = base_sentiment(text)
    sarcastic = detect_sarcasm(text)
    if sarcastic:                              # positive-sounding sarcasm in finance is almost always negative
        lex = -max(abs(lex), 0.3)

    rule = match_dual_rule(text, event)
    if rule:
        name, r_eq, r_cr = rule
        eq, cr = _clip(0.25 * lex + 0.75 * r_eq), _clip(r_cr)
    else:
        name = None
        eq = lex
        cr = _clip(CREDIT_SENSITIVITY.get(event, 0.7) * lex + (-0.2 if event == EventType.CREDIT and lex < 0 else 0.0))
    if event == EventType.OTHER and ent["ticker"] is None:   # nothing market-relevant -> damp to ~0
        eq, cr = eq * 0.2, cr * 0.2

    votes = {"base_model": {"name": lex_src, "score": round(lex, 3)}, "sarcasm": sarcastic,
             "dual_rule": name, "event_scores": {k: v for k, v in ev_scores.items() if v}}

    evidence_quote = best_evidence(text, ev_scores)
    verified = evidence_quote in text

    base_conf = 0.35 * ent["conf"] + 0.25 * margin + 0.25 * (1.0 if (lex * eq >= 0) else 0.3) \
        + 0.15 * _src_rel(item["source"])
    if sarcastic:
        base_conf *= 0.85

    # Optional LLM second opinion for low-confidence items (guarded).
    if USE_LLM and base_conf < 0.6:
        out = llm_extract(text)
        if out:
            ok, why = guard(out, text)
            votes["llm"] = {"accepted": ok, **why}
            if ok:
                eq = _clip(0.5 * eq + 0.5 * float(out.get("equity_polarity", eq)))
                cr = _clip(0.5 * cr + 0.5 * float(out.get("credit_polarity", cr)))
                evidence_quote, base_conf = str(out["evidence_quote"]).strip(), min(1.0, base_conf + 0.1)
            else:
                votes["llm"]["rejected_reason"] = "claim not grounded in source text"

    systemic = 1.0 if ent["method"] == "market" else 0.8 if ent["method"] == "sector" \
        else 0.7 if ent["ticker"] in SIFI else 0.4
    impact = impact_score(max(abs(eq), abs(cr)), event, magnitude_cues(text), systemic, velocity)

    uid = hashlib.sha1(f'{item["id"]}|{text}'.encode()).hexdigest()[:12]
    return RiskSignal(
        signal_id=uid, ts=item["timestamp"], source=item["source"], text=text,
        entity=ent["entity"], ticker=ent["ticker"], sector=ent["sector"], resolve_conf=ent["conf"],
        sentiment_equity=round(eq, 3), sentiment_credit=round(cr, 3), event_type=event,
        impact=round(impact, 2), confidence=round(_clip(base_conf, 0, 1), 3),
        half_life_h=HALF_LIFE_H[event],
        evidence=[Evidence(source=item["source"], url=item.get("url", ""),
                           quote=evidence_quote, verified=verified)],
        model_votes=votes)


def analyze_batch(items: list[dict]) -> list[RiskSignal]:
    """Chronological batch run. Mention velocity = recent same-class, non-trivial signals (3h window)."""
    from datetime import datetime
    def parse(ts): return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    signals: list[RiskSignal] = []
    for it in sorted(items, key=lambda i: i["timestamp"]):
        t = parse(it["timestamp"])
        probe = analyze_text(it, velocity=0.0)
        n = sum(1 for s in signals
                if s.event_type == probe.event_type and probe.event_type != EventType.OTHER
                and 0 <= (t - parse(s.ts)).total_seconds() <= 3 * 3600
                and max(abs(s.sentiment_equity), abs(s.sentiment_credit)) > 0.3)
        vel = min(1.0, math.log1p(n) / math.log1p(8))
        signals.append(analyze_text(it, velocity=vel) if vel > 0 else probe)
    return signals


def write_signals(signals: list[RiskSignal], path: Path | str = ROOT / "data" / "signals_sample.json") -> Path:
    path = Path(path)
    path.write_text(json.dumps([s.model_dump(mode="json") for s in signals], indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# 7. Evaluation against the hand-labelled sample
# --------------------------------------------------------------------------- #
def _sign(x: float, eps: float = 0.20) -> int:
    return 0 if abs(x) < eps else (1 if x > 0 else -1)


def _mean(xs) -> float:
    xs = list(xs)
    return float(np.mean(xs)) if xs else float("nan")


def evaluate(items: list[dict], signals: list[RiskSignal]) -> dict:
    from sklearn.metrics import f1_score
    # signal_id is hashed, so pair items and signals by chronological order
    labelled = [(i, s) for i, s in zip(sorted(items, key=lambda x: x["timestamp"]), signals) if i.get("label")]
    y_true = [i["label"]["event_type"] for i, _ in labelled]
    y_pred = [s.event_type.value for _, s in labelled]
    ent_ok = _mean([i["label"]["ticker"] == s.ticker for i, s in labelled])
    eq_ok = _mean([_sign(i["label"]["equity_polarity"]) == _sign(s.sentiment_equity) for i, s in labelled])
    cr_ok = _mean([_sign(i["label"]["credit_polarity"]) == _sign(s.sentiment_credit) for i, s in labelled])
    div = [(i, s) for i, s in labelled
           if _sign(i["label"]["equity_polarity"]) != _sign(i["label"]["credit_polarity"])]
    cr_div = _mean([_sign(i["label"]["credit_polarity"]) == _sign(s.sentiment_credit) for i, s in div])
    single_div = _mean([_sign(i["label"]["credit_polarity"]) == _sign(s.sentiment_equity) for i, s in div])
    sarc = [(i, s) for i, s in labelled if i["label"]["is_sarcastic"]]
    sarc_ok = _mean([_sign(i["label"]["equity_polarity"]) == _sign(s.sentiment_equity) for i, s in sarc])
    return {
        "n": len(labelled),
        "entity_accuracy": ent_ok,
        "event_accuracy": _mean([a == b for a, b in zip(y_true, y_pred)]),
        "event_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "equity_sign_accuracy": eq_ok,
        "credit_sign_accuracy": cr_ok,
        "n_divergent_items": len(div),
        "credit_acc_on_divergent_dual": cr_div,
        "credit_acc_on_divergent_single_polarity_baseline": single_div,
        "sarcasm_items": len(sarc), "sarcasm_equity_sign_accuracy": sarc_ok,
    }

if __name__ == "__main__":
    try:
        from src.ingestion import load_items
    except ImportError:                       # running as `python src/nlp_engine.py`
        from ingestion import load_items

    items, dropped = load_items("replay")
    sigs = analyze_batch(items)
    print(f"{len(items)} items analysed, {len(dropped)} duplicate(s) removed\n")
    print(f"{'id':5} {'ticker/entity':24} {'event':19} {'eq':>6} {'cr':>6} {'imp':>5} {'conf':>5}")
    for it, s in zip(sorted(items, key=lambda x: x['timestamp']), sigs):
        who = (s.ticker or s.entity)[:23]
        print(f"{it['id']:5} {who:24} {s.event_type.value:19} {s.sentiment_equity:6.2f} "
              f"{s.sentiment_credit:6.2f} {s.impact:5.1f} {s.confidence:5.2f}")
    print("\nEVALUATION (dev set)")
    for k, v in evaluate(items, sigs).items():
        print(f"  {k:48} {v:.3f}" if isinstance(v, float) else f"  {k:48} {v}")
    print("\nwrote", write_signals(sigs))