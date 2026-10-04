"""Unit tests for the maths that judges will probe. Run: pytest -q"""
import numpy as np
import pytest

from src.nlp_engine import analyze_text, guard, EventType
from src.ingestion import deduplicate
from src.contagion import TICKERS, build_adjacency, propagate
from src.module_a_rebalancer import apply_hysteresis, optimise, PARAMS
from src.module_b_stresstest import (stressed_pd, stressed_lgd, irb_capital, load_book,
                                     run_stress, reverse_stress)


def test_stressed_pd_monotonic_in_z():
    z = np.array([0.0, -0.5, -1.0, -2.0, -3.0])
    pd_s = stressed_pd(0.01, 0.2, z)
    assert np.all(np.diff(pd_s) > 0) and pd_s[0] == pytest.approx(0.01, rel=1e-6)


def test_lgd_rises_with_pd_and_is_capped():
    assert stressed_lgd(0.45, 0.01, 0.05, 0.03) > 0.45
    assert stressed_lgd(0.9, 0.01, 0.9, 0.2) <= 0.95


def test_irb_capital_increases_with_pd():
    k = irb_capital(np.array([0.001, 0.01, 0.05]), 0.45, 2.5, 0.15)
    assert np.all(np.diff(k) > 0)


def test_stress_loss_increases_with_impact():
    book = load_book()
    losses = [run_stress(book, "Credit Event", i)["summary"]["loss"] for i in (3, 6, 9)]
    assert losses[0] < losses[1] < losses[2]


def test_reverse_stress_hits_zero_headroom():
    book = load_book()
    x = reverse_stress(book, "Credit Event")
    assert x is not None
    assert run_stress(book, "Credit Event", x + 0.1)["summary"]["capital_headroom"] <= 0


def test_guard_rejects_fabricated_quote_and_numbers():
    text = "Brent crude jumps 6% as Red Sea disruption tightens tanker availability."
    assert guard({"evidence_quote": "Brent crude jumps 6%", "numeric_claims": [6]}, text)[0]
    assert not guard({"evidence_quote": "Brent crude jumps 60%", "numeric_claims": [60]}, text)[0]


def test_dual_polarity_buyback_is_bullish_equity_bearish_credit():
    s = analyze_text("Pfizer announces a $10 billion share buyback funded by new bond issuance.")
    assert s.sentiment_equity > 0 > s.sentiment_credit and s.event_type == EventType.CAPITAL_ACTION


def test_entity_disambiguation_rejects_fruit_and_rainforest():
    assert analyze_text("Apple harvest forecast hits a record as orchard growers report a bumper crop.").ticker is None
    assert analyze_text("Amazon rainforest deforestation rises 12% in September.").ticker is None


def test_dedupe_drops_syndicated_copy():
    a = {"id": "1", "timestamp": "2026-01-01T00:00:00Z", "text": "UPS warns of delivery delays as carriers divert cargo around Africa"}
    b = {"id": "2", "timestamp": "2026-01-01T00:10:00Z", "text": "UPS cautions on delivery delays as carriers divert cargo around Africa"}
    kept, dropped = deduplicate([a, b])
    assert len(kept) == 1 and dropped[0]["duplicate_of"] == "1"


def test_contagion_seed_is_max_and_decays():
    A = build_adjacency()
    seed = np.zeros(len(TICKERS)); seed[TICKERS.index("UPS")] = 1
    w = propagate(A, seed)
    assert w[TICKERS.index("UPS")] == 1.0 and 0 < w[TICKERS.index("AMZN")] < 1 and w[TICKERS.index("KO")] < 0.05


def test_optimizer_constraints():
    rng = np.random.default_rng(0)
    n = 20
    X = rng.normal(0, 0.01, (300, n)); Sigma = np.cov(X.T)
    wb = np.full(n, 1 / n); mu = rng.normal(0, 0.003, n)
    w = optimise(mu, Sigma, wb, wb, gamma=60, te_daily=0.04 / np.sqrt(252))
    assert w.sum() == pytest.approx(1, abs=1e-6) and w.min() >= -1e-9 and w.max() <= PARAMS["cap"] + 1e-6
    assert np.sqrt((w - wb) @ Sigma @ (w - wb)) <= 0.04 / np.sqrt(252) + 1e-6


def test_hysteresis_skips_small_trades_and_keeps_weights_valid():
    wp = np.full(20, 0.05); wn = wp.copy(); wn[0] += 0.001; wn[1] -= 0.001; wn[2] += 0.02; wn[3] -= 0.02
    out = apply_hysteresis(wn, wp, 0.0025, 0.15)
    assert out.sum() == pytest.approx(1) and out[0] == pytest.approx(0.05, abs=1e-3)