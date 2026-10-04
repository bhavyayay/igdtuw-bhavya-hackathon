"""Module A - Tactical index rebalancer.

20-stock S&P 100 basket, equal-weight benchmark. Every `step_min` minutes:
  1. decayed sentiment state   S_i(t) = sum_k eq_k * w_ki * conf_k * src_k * exp(-ln2 (t-t_k)/halflife_k)
  2. velocity                   V_i = EWMA(dS_i)   (deteriorating names are penalised faster)
  3. expected-return tilt       mu_i = a * clip(S_i + beta*V_i)
  4. optimiser                  max mu'(w-wb) - 0.5*gamma*|L'(w-wb)|^2 - tau*|w-w_prev|_1
                                s.t. sum w = 1, 0 <= w <= cap, tracking error <= te_max
  5. hysteresis                 trades smaller than 25 bp are skipped
Module B feeds back through `risk_budget_fn(t) -> utilisation`: gamma goes up, te_max shrinks.

Prices: data/prices.csv. Fetch real data with `python -m src.module_a_rebalancer --fetch-prices`;
without it a clearly-labelled synthetic series is generated so the demo always runs.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable, Optional

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

try:
    from src.contagion import TICKERS, SECTOR
except ImportError:
    from contagion import TICKERS, SECTOR

ROOT = Path(__file__).resolve().parents[1]
PRICES_PATH = ROOT / "data" / "prices.csv"

SRC_WEIGHT = {"newsapi": 1.0, "gdelt": 0.9, "tweet": 0.5}
PARAMS = dict(a=0.0025, beta=0.5, gamma0=60.0, te_annual=0.04, cap=0.15, tau=0.0005,
              hysteresis=0.0025, ewma=0.5, step_min=30, horizon_h=6, min_conf=0.3)


# --------------------------------------------------------------------------- #
# Prices
# --------------------------------------------------------------------------- #
def fetch_prices(period: str = "2y") -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(TICKERS, period=period, auto_adjust=True, progress=False)["Close"]
    df = df[TICKERS].dropna(how="all").ffill().dropna()
    df.to_csv(PRICES_PATH)
    return df


def synthetic_prices(days: int = 504, seed: int = 7) -> pd.DataFrame:
    """Factor-model prices (market + sector + idiosyncratic). Used only when no real file exists."""
    rng = np.random.default_rng(seed)
    sectors = sorted(set(SECTOR.values()))
    mkt = rng.normal(0.0004, 0.009, days)
    sec = {s: rng.normal(0, 0.006, days) for s in sectors}
    rets = np.column_stack([mkt * rng.uniform(0.8, 1.3) + sec[SECTOR[t]] + rng.normal(0, 0.009, days)
                            for t in TICKERS])
    idx = pd.date_range(end=pd.Timestamp("2026-10-02"), periods=days, freq="B")
    return pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=TICKERS)


def load_prices() -> tuple[pd.DataFrame, str]:
    if PRICES_PATH.exists():
        df = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
        return df[TICKERS], "file"
    return synthetic_prices(), "synthetic"


# --------------------------------------------------------------------------- #
# Sentiment state
# --------------------------------------------------------------------------- #
def _src_w(source: str) -> float:
    return next((v for k, v in SRC_WEIGHT.items() if source.startswith(k)), 0.7)


def sentiment_state(signals: list, t: pd.Timestamp, tickers: list[str] = TICKERS,
                    min_conf: float = PARAMS["min_conf"]) -> np.ndarray:
    S = np.zeros(len(tickers))
    for s in signals:
        ts = pd.Timestamp(s.ts)
        if ts > t or s.confidence < min_conf:
            continue
        lam = math.log(2) / s.half_life_h
        decay = math.exp(-lam * (t - ts).total_seconds() / 3600.0)
        w = s.contagion or ({s.ticker: 1.0} if s.ticker else {})
        for i, tk in enumerate(tickers):
            if tk in w:
                S[i] += s.sentiment_equity * w[tk] * s.confidence * _src_w(s.source) * decay
    return S


# --------------------------------------------------------------------------- #
# Optimiser
# --------------------------------------------------------------------------- #
def optimise(mu: np.ndarray, Sigma: np.ndarray, w_prev: np.ndarray, w_bench: np.ndarray,
             gamma: float, te_daily: float, p: dict = PARAMS) -> np.ndarray:
    n = len(mu)
    L = np.linalg.cholesky(Sigma + 1e-10 * np.eye(n))
    w = cp.Variable(n)
    active = L.T @ (w - w_bench)
    obj = cp.Maximize(mu @ (w - w_bench) - 0.5 * gamma * cp.sum_squares(active)
                      - p["tau"] * cp.norm1(w - w_prev))
    prob = cp.Problem(obj, [cp.sum(w) == 1, w >= 0, w <= p["cap"], cp.norm(active, 2) <= te_daily])
    try:
        prob.solve()
    except cp.error.SolverError:
        return w_prev
    return w_prev if w.value is None else np.clip(w.value, 0, p["cap"])


def apply_hysteresis(w_new: np.ndarray, w_prev: np.ndarray, thr: float, cap: float) -> np.ndarray:
    """Skip trades below `thr`; push the residual onto names that did trade, then re-check the cap."""
    delta = w_new - w_prev
    keep = np.abs(delta) >= thr
    out = np.where(keep, w_new, w_prev)
    resid = 1.0 - out.sum()
    if keep.any() and abs(resid) > 1e-12:
        share = np.abs(delta) * keep
        out = out + resid * share / share.sum()
    out = np.clip(out, 0, cap)
    return out / out.sum()


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def run_module_a(signals: list, prices: pd.DataFrame, risk_budget_fn: Optional[Callable] = None,
                 params: Optional[dict] = None, tickers: list[str] = TICKERS) -> dict:
    p = {**PARAMS, **(params or {})}
    rets = np.log(prices[tickers]).diff().dropna().tail(252)
    Sigma = LedoitWolf().fit(rets.values).covariance_
    n = len(tickers)
    w_bench = np.full(n, 1.0 / n)

    times = sorted(pd.Timestamp(s.ts) for s in signals)
    grid = pd.date_range(times[0].floor("30min"), times[-1] + pd.Timedelta(hours=p["horizon_h"]),
                         freq=f'{p["step_min"]}min', tz="UTC")
    w_prev, S_prev, V = w_bench.copy(), np.zeros(n), np.zeros(n)
    W, SS, TO, G, TE = [], [], [], [], []
    for t in grid:
        S = sentiment_state(signals, t, tickers, p["min_conf"])
        V = p["ewma"] * V + (1 - p["ewma"]) * (S - S_prev)
        mu = p["a"] * np.clip(S + p["beta"] * V, -1.5, 1.5)
        util = float(risk_budget_fn(t)) if risk_budget_fn else 0.0
        gamma = p["gamma0"] * (1 + 1.5 * util)                       # Module B -> more risk-averse
        te_daily = p["te_annual"] * max(0.25, 1 - 0.75 * util) / math.sqrt(252)   # ... and tighter TE
        w = optimise(mu, Sigma, w_prev, w_bench, gamma, te_daily, p)
        w = apply_hysteresis(w, w_prev, p["hysteresis"], p["cap"])
        W.append(w); SS.append(S); TO.append(0.5 * np.abs(w - w_prev).sum()); G.append(gamma)
        TE.append(float(np.sqrt((w - w_bench) @ Sigma @ (w - w_bench)) * math.sqrt(252)))
        w_prev, S_prev = w, S
    idx = pd.DatetimeIndex(grid)
    return {"weights": pd.DataFrame(W, index=idx, columns=tickers),
            "sentiment": pd.DataFrame(SS, index=idx, columns=tickers),
            "turnover": pd.Series(TO, index=idx, name="turnover"),
            "gamma": pd.Series(G, index=idx, name="gamma"),
            "tracking_error": pd.Series(TE, index=idx, name="te_annual"),
            "benchmark": pd.Series(w_bench, index=tickers)}


if __name__ == "__main__":
    if "--fetch-prices" in sys.argv:
        df = fetch_prices()
        print(f"saved {PRICES_PATH} ({df.shape[0]} days x {df.shape[1]} tickers)")
        sys.exit(0)
    try:
        from src.ingestion import load_items
        from src.nlp_engine import analyze_batch
        from src.contagion import apply_contagion
    except ImportError:
        from ingestion import load_items
        from nlp_engine import analyze_batch
        from contagion import apply_contagion
    items, _ = load_items("replay", scenario="red_sea")
    sigs = apply_contagion(analyze_batch(items))
    prices, src = load_prices()
    res = run_module_a(sigs, prices)
    W = res["weights"]
    print(f"prices: {src} | steps: {len(W)} | max turnover/step: {res['turnover'].max():.2%} "
          f"| max TE: {res['tracking_error'].max():.2%}")
    chg = (W.iloc[-1] - res["benchmark"]).sort_values()
    print("biggest underweights:", {k: f"{v:+.2%}" for k, v in chg.head(3).items()})
    print("biggest overweights :", {k: f"{v:+.2%}" for k, v in chg.tail(3).items()})
    print("weights sum:", round(W.iloc[-1].sum(), 6), "| max weight:", f"{W.iloc[-1].max():.2%}")