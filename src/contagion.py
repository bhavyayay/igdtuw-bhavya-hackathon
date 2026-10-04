"""Contagion graph: spreads a risk signal from the entity it names to related names.

Every signal gets a propagation-weight map {ticker: w in [0,1]} (seed = 1). Downstream modules
multiply the signal's equity / credit sentiment by w, so a "Shipping sector" shock reaches
UPS directly and AMZN / WMT / CAT / BA through supply-chain links.
Edges are illustrative (same-sector links + a small supply-chain seed list), not licensed data.
"""
from __future__ import annotations

import numpy as np

SECTOR = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "GOOGL": "Communication Services", "META": "Communication Services", "DIS": "Communication Services",
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "WMT": "Consumer Staples", "KO": "Consumer Staples",
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
    "XOM": "Energy", "CVX": "Energy", "JNJ": "Healthcare", "PFE": "Healthcare",
    "BA": "Industrials", "CAT": "Industrials", "UPS": "Shipping & Logistics",
}
TICKERS = list(SECTOR)
_SUPPLY_CHAIN = [("UPS", "AMZN", 0.6), ("UPS", "WMT", 0.5), ("AMZN", "WMT", 0.3), ("UPS", "CAT", 0.3),
                 ("UPS", "BA", 0.3), ("BA", "CAT", 0.4), ("XOM", "UPS", 0.2), ("CVX", "UPS", 0.2),
                 ("NVDA", "MSFT", 0.4), ("GS", "JPM", 0.6)]


def build_adjacency() -> np.ndarray:
    idx = {t: i for i, t in enumerate(TICKERS)}
    A = np.zeros((len(TICKERS), len(TICKERS)))
    for a in TICKERS:
        for b in TICKERS:
            if a != b and SECTOR[a] == SECTOR[b]:
                A[idx[a], idx[b]] = 0.6 if SECTOR[a] == "Financials" else 0.5
    for a, b, w in _SUPPLY_CHAIN:
        A[idx[a], idx[b]] = A[idx[b], idx[a]] = max(A[idx[a], idx[b]], w)
    return A


def propagate(A: np.ndarray, seed: np.ndarray, alpha: float = 0.5, hops: int = 2) -> np.ndarray:
    """Decayed multi-hop spread: out = seed + a*P'seed + a^2*(P')^2 seed (P row-normalised), clipped to 1."""
    P = A / (A.sum(axis=1, keepdims=True) + 1e-9)
    out, cur = seed.copy(), seed.copy()
    for _ in range(hops):
        cur = alpha * P.T @ cur
        out += cur
    return np.clip(out, 0.0, 1.0)


def seed_vector(ticker, sector) -> np.ndarray:
    seed = np.zeros(len(TICKERS))
    if ticker in TICKERS:
        seed[TICKERS.index(ticker)] = 1.0
    elif sector in set(SECTOR.values()):
        seed[[i for i, t in enumerate(TICKERS) if SECTOR[t] == sector]] = 1.0
    else:                                   # market-wide news: small uniform exposure, no propagation
        seed[:] = 0.3
    return seed


def apply_contagion(signals: list) -> list:
    """Fill signal.contagion for every RiskSignal in place and return the list."""
    A = build_adjacency()
    for s in signals:
        seed = seed_vector(s.ticker, s.sector)
        w = seed if seed.max() == 0.3 and s.ticker is None and s.sector not in set(SECTOR.values()) \
            else propagate(A, seed)
        s.contagion = {t: round(float(x), 3) for t, x in zip(TICKERS, w) if x >= 0.05}
    return signals