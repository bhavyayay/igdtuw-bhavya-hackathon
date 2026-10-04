"""Module B - Strategic event-driven stress test for a synthetic wholesale book.

NLP event + impact  ->  systematic factor shock Z (Vasicek)  ->  stressed PD, downturn LGD
                    ->  rate / spread repricing  ->  loss, RWA (IRB), CET1 headroom, rating migration.

Shock sizes are ILLUSTRATIVE scenario templates scaled by the NLP impact score (production would
calibrate to CCAR / EBA-style scenarios). The IRB formula is the simplified corporate one.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
BOOK_PATH = ROOT / "data" / "wholesale_portfolio.csv"

RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
RATING_PD = np.array([0.0002, 0.0005, 0.0012, 0.0045, 0.0150, 0.0500, 0.2000])
INVESTMENT_GRADE = {"AAA", "AA", "A", "BBB"}

# event -> template at impact 10:  Zmax, rate shock (bp), spread shock IG/HY (bp), LGD add-on, sector emphasis
TEMPLATES = {
    "Geopolitical":  dict(zmax=0.8, rate_bp=-50, sp_ig=60, sp_hy=150, lgd_add=0.03,
                          beta={"Energy": 1.5, "Shipping & Logistics": 1.8, "Industrials": 1.3, "Defence": 1.5}),
    "Macroeconomic": dict(zmax=0.6, rate_bp=150, sp_ig=30, sp_hy=80, lgd_add=0.02,
                          beta={"Real Estate": 1.6, "Utilities": 1.3, "Financials": 1.2}),
    "Credit Event":  dict(zmax=1.0, rate_bp=0, sp_ig=80, sp_hy=250, lgd_add=0.08,
                          beta={"Financials": 1.3}),
    "Regulatory":    dict(zmax=0.5, rate_bp=0, sp_ig=20, sp_hy=50, lgd_add=0.01,
                          beta={"Financials": 1.5}),
}
DEFAULT_TEMPLATE = dict(zmax=0.15, rate_bp=0, sp_ig=10, sp_hy=25, lgd_add=0.0, beta={})
Z_FLOOR = -3.5
RWA_PASSTHROUGH = 0.35   # IRB PDs are through-the-cycle: only this share of the stressed PD (in log space) reaches RWA
CET1_RATIO, MIN_RATIO = 0.14, 0.105     # assumed available CET1 / regulatory minimum + buffers (of RWA)


# --------------------------------------------------------------------------- #
# Risk maths
# --------------------------------------------------------------------------- #
def stressed_pd(pd_, rho, z):
    """Vasicek PD after a systematic-factor shift z (z < 0 = adverse; z = 0 returns the input PD).

    Derivation: PD(z) = N((G(PD_ttc) - sqrt(rho) z) / sqrt(1-rho)). Treating today's PD as the starting
    state gives the probit-shift form  PD_s = N( G(PD0) - sqrt(rho/(1-rho)) * dz ).
    """
    pd_ = np.clip(pd_, 1e-6, 0.999)
    return norm.cdf(norm.ppf(pd_) - np.sqrt(rho / (1 - rho)) * z)


def stressed_lgd(lgd, pd_base, pd_stress, add_on):
    """Downturn LGD: base + event add-on + a linear PD-LGD correlation term (Frye-Jacobs style proxy)."""
    return np.minimum(0.95, lgd + add_on + 0.3 * (pd_stress - pd_base))


def irb_capital(pd_, lgd, maturity, rho):
    """Simplified Basel IRB corporate capital requirement K (fraction of EAD)."""
    pd_ = np.clip(pd_, 0.0003, 0.999)                      # Basel PD floor
    m = np.clip(maturity, 1.0, 5.0)
    b = (0.11852 - 0.05478 * np.log(pd_)) ** 2
    k = lgd * (norm.cdf((norm.ppf(pd_) + np.sqrt(rho) * norm.ppf(0.999)) / np.sqrt(1 - rho)) - pd_)
    return k * (1 + (m - 2.5) * b) / (1 - 1.5 * b)


def _bucket(pd_):
    """Nearest rating bucket (in log-PD) -> index 0..6."""
    return np.abs(np.log(np.clip(pd_, 1e-6, 1))[:, None] - np.log(RATING_PD)[None, :]).argmin(axis=1)


# --------------------------------------------------------------------------- #
# Stress engine
# --------------------------------------------------------------------------- #
def load_book(path: Path | str = BOOK_PATH) -> pd.DataFrame:
    return pd.read_csv(path).fillna({"ticker": ""})


def run_stress(book: pd.DataFrame, event_type: str, impact: float, confidence: float = 1.0,
               signal_sector: Optional[str] = None, contagion: Optional[dict] = None) -> dict:
    """Apply an event of given type / impact to the book. Returns summary + detail tables."""
    tpl = TEMPLATES.get(event_type, DEFAULT_TEMPLATE)
    s = float(np.clip(impact / 10.0, 0, 1.5))
    contagion = contagion or {}
    df = book.copy()

    beta = df["sector"].map(tpl["beta"]).fillna(1.0).to_numpy(dtype=float)
    if signal_sector:
        beta = beta + 0.5 * (df["sector"] == signal_sector).to_numpy()
    beta = beta + 0.8 * df["ticker"].map(contagion).fillna(0.0).to_numpy()
    z = np.clip(-tpl["zmax"] * s * beta * confidence, Z_FLOOR, 0.0)

    pd0, lgd0, rho = df["pd_1y"].to_numpy(), df["lgd"].to_numpy(), df["asset_corr"].to_numpy()
    ead, mv, mat = df["ead_usd_mn"].to_numpy(), df["market_value_usd_mn"].to_numpy(), df["maturity_years"].to_numpy()
    pd1 = stressed_pd(pd0, rho, z)
    lgd1 = stressed_lgd(lgd0, pd0, pd1, tpl["lgd_add"] * s)

    # --- valuation by instrument ---
    dy = tpl["rate_bp"] * s / 1e4
    is_ig = df["rating"].isin(INVESTMENT_GRADE).to_numpy()
    ds = np.where(is_ig, tpl["sp_ig"], tpl["sp_hy"]) * s * np.clip(beta, 0.5, 3.0) / 1e4
    at = df["asset_type"].to_numpy()
    bond = np.isin(at, ["Fixed-Rate Bond", "Floating-Rate Bond"])
    swap = at == "Interest Rate Swap"
    dP = -df["duration"].to_numpy() * dy + 0.5 * df["convexity"].to_numpy() * dy ** 2 \
         - df["spread_duration"].to_numpy() * ds
    mv1 = mv.copy()
    mv1[bond] = mv[bond] * (1 + dP[bond])                                        # bonds: duration + spread duration
    mv1[swap] = mv[swap] - df["dv01_usd"].to_numpy()[swap] * (tpl["rate_bp"] * s) / 1e6   # swaps: DV01
    credit0 = np.where(bond, 0.0, ead * pd0 * lgd0)       # bonds are marked via spreads -> no extra EL (avoids double count)
    credit1 = np.where(bond, 0.0, ead * pd1 * lgd1)       # loans: EL ; derivatives: CVA proxy (EPE ~ EAD)
    val0, val1 = mv - credit0, mv1 - credit1

    # --- capital ---
    rwa0 = irb_capital(pd0, lgd0, mat, rho) * 12.5 * ead
    pd_rwa = pd0 ** (1 - RWA_PASSTHROUGH) * pd1 ** RWA_PASSTHROUGH
    rwa1 = irb_capital(pd_rwa, lgd1, mat, rho) * 12.5 * ead
    cap0 = CET1_RATIO * rwa0.sum()
    loss = float(val0.sum() - val1.sum())
    cap1 = cap0 - loss
    excess0 = cap0 - MIN_RATIO * rwa0.sum()
    headroom = cap1 - MIN_RATIO * rwa1.sum()
    mig = _bucket(pd_rwa) - _bucket(pd0)          # through-the-cycle rating migration

    df["z"], df["pd_stress"], df["lgd_stress"] = z, pd1, lgd1
    df["value_before"], df["value_after"] = val0, val1
    df["loss"] = val0 - val1
    df["d_el"] = ead * pd1 * lgd1 - ead * pd0 * lgd0
    df["notch_change"] = mig
    sector = df.groupby("sector").agg(ead=("ead_usd_mn", "sum"), loss=("loss", "sum"), d_el=("d_el", "sum"),
                                      avg_z=("z", "mean")).sort_values("loss", ascending=False)
    top = df.groupby("counterparty")["loss"].sum().sort_values(ascending=False).head(10)

    summary = dict(
        event_type=event_type, impact=float(impact), confidence=float(confidence),
        value_before=float(val0.sum()), value_after=float(val1.sum()), loss=loss,
        loss_pct=loss / float(val0.sum()), delta_el=float((ead * pd1 * lgd1).sum() - (ead * pd0 * lgd0).sum()),
        rwa_before=float(rwa0.sum()), rwa_after=float(rwa1.sum()), delta_rwa=float(rwa1.sum() - rwa0.sum()),
        cet1_before=CET1_RATIO, cet1_after=float(cap1 / rwa1.sum()), capital_headroom=float(headroom),
        breach=bool(headroom < 0), risk_budget_util=float(loss / excess0) if excess0 > 0 else 9.9,
        pct_ead_downgraded=float(ead[mig >= 1].sum() / ead.sum()), pct_names_downgraded=float((mig >= 1).mean()),
        mean_z=float(z.mean()), rate_shock_bp=float(tpl["rate_bp"] * s),
    )
    return {"summary": summary, "exposures": df, "sectors": sector, "top_losses": top}


# --------------------------------------------------------------------------- #
# Trigger, reverse stress test, bridge to Module A
# --------------------------------------------------------------------------- #
def should_trigger(sig, min_impact: float = 7.0, min_conf: float = 0.6) -> bool:
    return sig.impact >= min_impact and sig.confidence >= min_conf


def stress_from_signal(book: pd.DataFrame, sig) -> dict:
    return run_stress(book, sig.event_type.value if hasattr(sig.event_type, "value") else sig.event_type,
                      sig.impact, sig.confidence, sig.sector, sig.contagion)


def reverse_stress(book: pd.DataFrame, event_type: str, tol: float = 0.05) -> Optional[float]:
    """Smallest impact score (0.5-10) at which capital headroom hits zero. None if the book survives."""
    f = lambda x: run_stress(book, event_type, x)["summary"]["capital_headroom"]
    lo, hi = 0.5, 10.0
    if f(hi) > 0:
        return None
    if f(lo) <= 0:
        return lo
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return round(hi, 2)


def build_stress_events(book: pd.DataFrame, signals: list, min_impact: float = 7.0, min_conf: float = 0.6,
                        cooldown_h: float = 3.0) -> list[dict]:
    """Run a stress for every triggering signal (with a cooldown to avoid re-firing on the same story)."""
    events, last_ts = [], None
    for sig in sorted(signals, key=lambda s: s.ts):
        if not should_trigger(sig, min_impact, min_conf):
            continue
        t = pd.Timestamp(sig.ts)
        if last_ts is not None and (t - last_ts).total_seconds() < cooldown_h * 3600:
            continue
        res = stress_from_signal(book, sig)
        events.append({"ts": t, "signal": sig, "result": res})
        last_ts = t
    return events


def make_risk_budget_fn(events: list[dict], half_life_h: float = 24.0):
    """Module B -> Module A bridge: utilisation u(t) = max_e util_e * 0.5^((t - t_e)/half_life)."""
    def fn(t):
        t = pd.Timestamp(t)
        vals = [min(1.5, e["result"]["summary"]["risk_budget_util"]) *
                0.5 ** ((t - e["ts"]).total_seconds() / 3600 / half_life_h)
                for e in events if e["ts"] <= t]
        return max(vals) if vals else 0.0
    return fn


if __name__ == "__main__":
    try:
        from src.ingestion import load_items
        from src.nlp_engine import analyze_batch
        from src.contagion import apply_contagion
    except ImportError:
        from ingestion import load_items
        from nlp_engine import analyze_batch
        from contagion import apply_contagion
    book = load_book()
    items, _ = load_items("replay")
    sigs = apply_contagion(analyze_batch(items))
    events = build_stress_events(book, sigs)
    print(f"{len(events)} stress test(s) triggered (impact>=7, confidence>=0.6)\n")
    for e in events:
        r = e["result"]["summary"]
        print(f"{e['ts']:%d-%b %H:%M} {e['signal'].event_type.value:14} imp {e['signal'].impact:4.1f} | "
              f"loss ${r['loss']:7.1f}mn ({r['loss_pct']:.1%}) | dEL ${r['delta_el']:6.1f}mn | "
              f"dRWA ${r['delta_rwa']:7.0f}mn | CET1 {r['cet1_before']:.1%}->{r['cet1_after']:.2%} | "
              f"util {r['risk_budget_util']:.2f} | EAD downgraded {r['pct_ead_downgraded']:.0%}")
    print()
    for et in TEMPLATES:
        print(f"reverse stress [{et:13}] -> breaking impact:", reverse_stress(book, et))