"""S&P Sentinel - interactive dashboard.   Run:  streamlit run src/app.py"""
from __future__ import annotations

import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.contagion import SECTOR, TICKERS, apply_contagion, build_adjacency
from src.ingestion import load_items
from src.module_a_rebalancer import PARAMS, load_prices, run_module_a
from src.module_b_stresstest import (TEMPLATES, build_stress_events, load_book, make_risk_budget_fn,
                                     reverse_stress, run_stress, should_trigger)
from src.nlp_engine import MODEL_VERSION, analyze_batch, analyze_text, evaluate, guard

st.set_page_config(page_title="S&P Sentinel", page_icon="🛰️", layout="wide")

GREEN, RED, GREY, BLUE = "#1a9850", "#d73027", "#9aa0a6", "#2c7fb8"
TEMPLATE = "plotly_white"
SAMPLE_TEXT = "Pfizer announces a $10 billion share buyback funded by new bond issuance."


# --------------------------------------------------------------------------- #
# Pipeline (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Running Sentinel pipeline: NLP -> contagion -> Module B -> Module A ...")
def run_pipeline(scenario: str, mode: str, min_impact: float, min_conf: float) -> dict | None:
    book = load_book()
    live = mode.startswith("Live")
    items, dropped = load_items("live" if live else "replay", scenario=None if (live or scenario == "All") else scenario)
    if not items:
        return None
    signals = apply_contagion(analyze_batch(items))
    events = build_stress_events(book, signals, min_impact, min_conf)
    prices, price_src = load_prices()
    fn = make_risk_budget_fn(events)
    open_loop = run_module_a(signals, prices)
    closed_loop = run_module_a(signals, prices, risk_budget_fn=fn) if events else open_loop
    util = pd.Series([fn(t) for t in open_loop["weights"].index], index=open_loop["weights"].index)
    labelled = any(i.get("label") for i in items)
    return dict(items=items, dropped=dropped, signals=signals, events=events, open=open_loop,
                closed=closed_loop, util=util, price_src=price_src,
                evaluation=evaluate(items, signals) if labelled else None)


@st.cache_data(show_spinner="Running reverse stress tests ...")
def reverse_table() -> pd.DataFrame:
    book = load_book()
    rows = []
    for et in TEMPLATES:
        x = reverse_stress(book, et)
        rows.append({"Event class": et, "Impact score that breaches the CET1 minimum":
                     "survives even at 10" if x is None else f"{x:.1f}"})
    return pd.DataFrame(rows)


@st.cache_data
def graph_layout() -> dict:
    A = build_adjacency()
    G = nx.Graph()
    G.add_nodes_from(range(len(TICKERS)))
    for i in range(len(TICKERS)):
        for j in range(i + 1, len(TICKERS)):
            if A[i, j] > 0:
                G.add_edge(i, j, weight=float(A[i, j]))
    pos = nx.spring_layout(G, seed=7, weight="weight", k=0.9)
    return {"pos": {int(k): (float(v[0]), float(v[1])) for k, v in pos.items()},
            "edges": [(int(a), int(b)) for a, b in G.edges()]}


# --------------------------------------------------------------------------- #
# Small UI helpers
# --------------------------------------------------------------------------- #
def sgn(x: float, eps: float = 0.2) -> int:
    return 0 if abs(x) < eps else (1 if x > 0 else -1)


def badge(label: str, value: float) -> str:
    color = GREEN if value > 0.2 else RED if value < -0.2 else GREY
    return (f"<span style='background:{color};color:white;padding:2px 9px;border-radius:10px;"
            f"font-size:0.85rem;margin-right:6px'>{label} {value:+.2f}</span>")


def signal_label(s) -> str:
    return f"{pd.Timestamp(s.ts):%d %b %H:%M} | {s.ticker or s.entity} | {s.event_type.value} | impact {s.impact:.1f}"


def signals_df(signals: list, min_impact: float, min_conf: float) -> pd.DataFrame:
    return pd.DataFrame([{
        "Time": pd.Timestamp(s.ts).strftime("%d %b %H:%M"), "Source": s.source.replace("_", " "),
        "Entity": s.ticker or s.entity, "Event": s.event_type.value, "Equity": s.sentiment_equity,
        "Credit": s.sentiment_credit, "Impact": s.impact, "Confidence": s.confidence,
        "Dual": "⚡" if sgn(s.sentiment_equity) * sgn(s.sentiment_credit) < 0 else "",
        "Stress": "🔴" if should_trigger(s, min_impact, min_conf) else ""} for s in signals])


def highlight(text: str, quote: str) -> str:
    t, q = html.escape(text), html.escape(quote)
    return t.replace(q, f"<mark>{q}</mark>", 1) if q and q in t else t


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def vline(fig, x):
    fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, yref="paper", line=dict(color="black", dash="dash", width=1))


def fig_weights(res: dict, t) -> go.Figure:
    W = res["weights"] * 100
    fig = go.Figure()
    for tk in W.columns:
        fig.add_trace(go.Scatter(x=W.index, y=W[tk], name=tk, mode="lines", stackgroup="w",
                                 line=dict(width=0.4, shape="hv"), hovertemplate=f"{tk}: %{{y:.2f}}%<extra></extra>"))
    vline(fig, t)
    fig.update_layout(template=TEMPLATE, height=380, yaxis_title="Portfolio weight (%)", margin=dict(t=30, b=10),
                      legend=dict(orientation="h", y=-0.2), title="Index weights over time")
    return fig


def fig_active(res: dict, t) -> go.Figure:
    a = ((res["weights"].loc[t] - res["benchmark"]) * 100).sort_values()
    fig = go.Figure(go.Bar(x=a.values, y=a.index, orientation="h", marker_color=[GREEN if v > 0 else RED for v in a.values],
                           hovertemplate="%{y}: %{x:+.2f} pp<extra></extra>"))
    fig.update_layout(template=TEMPLATE, height=420, margin=dict(t=30, b=10), xaxis_title="Active weight vs equal-weight (pp)",
                      title="Over / underweights at selected time")
    return fig


def fig_sentiment_heat(res: dict, t) -> go.Figure:
    S = res["sentiment"]
    fig = go.Figure(go.Heatmap(z=S.T.values, x=S.index, y=S.columns, colorscale="RdYlGn", zmid=0,
                               colorbar=dict(title="S(t)"), hovertemplate="%{y} %{x|%d %b %H:%M}: %{z:+.2f}<extra></extra>"))
    vline(fig, t)
    fig.update_layout(template=TEMPLATE, height=420, margin=dict(t=30, b=10), title="Decayed sentiment state S_i(t)")
    return fig


def fig_loop(pipe: dict) -> go.Figure:
    o, c, u = pipe["open"], pipe["closed"], pipe["util"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.4, 0.6], vertical_spacing=0.08,
                        subplot_titles=("Module B risk-budget utilisation", "Module A annualised tracking error"))
    fig.add_trace(go.Scatter(x=u.index, y=u.values, fill="tozeroy", name="Risk budget utilisation",
                             line=dict(color=RED, shape="hv")), row=1, col=1)
    fig.add_trace(go.Scatter(x=o["tracking_error"].index, y=o["tracking_error"] * 100, name="Open loop (no Module B)",
                             line=dict(color=GREY, shape="hv")), row=2, col=1)
    fig.add_trace(go.Scatter(x=c["tracking_error"].index, y=c["tracking_error"] * 100, name="Closed loop (A <- B)",
                             line=dict(color=BLUE, shape="hv")), row=2, col=1)
    fig.update_yaxes(title_text="utilisation", row=1, col=1)
    fig.update_yaxes(title_text="TE (%)", row=2, col=1)
    fig.update_layout(template=TEMPLATE, height=460, margin=dict(t=40, b=10), legend=dict(orientation="h", y=-0.12))
    return fig


def fig_waterfall(res: dict) -> go.Figure:
    ex, s = res["exposures"], res["summary"]
    by = ex.groupby("asset_type")["loss"].sum().sort_values(ascending=False)
    x = ["Value before"] + list(by.index) + ["Value after"]
    y = [s["value_before"]] + [-v for v in by.values] + [s["value_after"]]
    fig = go.Figure(go.Waterfall(x=x, y=y, measure=["absolute"] + ["relative"] * len(by) + ["total"],
                                 text=[f"{v:,.0f}" for v in y], textposition="outside",
                                 decreasing=dict(marker_color=RED), increasing=dict(marker_color=GREEN),
                                 totals=dict(marker_color=BLUE), connector=dict(line=dict(color=GREY))))
    lo = min(s["value_after"], s["value_before"])
    fig.update_yaxes(range=[lo * 0.97, s["value_before"] * 1.01], title="USD mn")
    fig.update_layout(template=TEMPLATE, height=400, margin=dict(t=40, b=10), title="Portfolio value: before -> after (loss by instrument)")
    return fig


def fig_sector_heat(res: dict) -> go.Figure:
    ex = res["exposures"]
    g = ex.groupby(["sector", "asset_type"]).agg(loss=("loss", "sum"), ead=("ead_usd_mn", "sum")).reset_index()
    g["pct"] = np.where(g["ead"] > 0, g["loss"] / g["ead"] * 100, 0.0)
    p = g.pivot(index="sector", columns="asset_type", values="pct").fillna(0)
    p = p.loc[res["sectors"].index]
    fig = go.Figure(go.Heatmap(z=p.values, x=p.columns, y=p.index, colorscale="Reds", zmin=0,
                               colorbar=dict(title="loss % EAD"), hovertemplate="%{y} / %{x}: %{z:.2f}% of EAD<extra></extra>"))
    fig.update_layout(template=TEMPLATE, height=400, margin=dict(t=40, b=10), title="Loss as % of EAD: sector x instrument")
    return fig


def fig_migration(res: dict) -> go.Figure:
    ex = res["exposures"]
    n = ex["notch_change"].clip(-1, 3)
    share = (ex.groupby(n)["ead_usd_mn"].sum() / ex["ead_usd_mn"].sum() * 100)
    names = {-1: "Upgrade", 0: "No change", 1: "1 notch down", 2: "2 notches down", 3: "3+ notches down"}
    fig = go.Figure(go.Bar(x=[names[int(i)] for i in share.index], y=share.values,
                           marker_color=[GREEN if i < 0 else GREY if i == 0 else RED for i in share.index],
                           text=[f"{v:.1f}%" for v in share.values], textposition="outside"))
    fig.update_layout(template=TEMPLATE, height=340, margin=dict(t=40, b=10), yaxis_title="% of EAD",
                      title="Rating migration (through-the-cycle PD buckets)")
    return fig


def fig_top_losses(res: dict) -> go.Figure:
    t = res["top_losses"].sort_values()
    fig = go.Figure(go.Bar(x=t.values, y=t.index, orientation="h", marker_color=RED))
    fig.update_layout(template=TEMPLATE, height=340, margin=dict(t=40, b=10), xaxis_title="Loss (USD mn)",
                      title="Top-10 loss contributors (counterparty)")
    return fig


def fig_contagion(sig, lens: str) -> go.Figure:
    lay = graph_layout()
    pos = lay["pos"]
    val = sig.sentiment_equity if lens.startswith("Equity") else sig.sentiment_credit
    ex, ey = [], []
    for a, b in lay["edges"]:
        ex += [pos[a][0], pos[b][0], None]
        ey += [pos[a][1], pos[b][1], None]
    w = np.array([sig.contagion.get(t, 0.0) for t in TICKERS])
    fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#d0d3d8", width=1), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[pos[i][0] for i in range(len(TICKERS))], y=[pos[i][1] for i in range(len(TICKERS))],
        mode="markers+text", text=TICKERS, textposition="middle center", textfont=dict(size=10),
        marker=dict(size=22 + 34 * w, color=w * val, colorscale="RdYlGn", cmin=-1, cmax=1, line=dict(color="white", width=1),
                    colorbar=dict(title=f"{lens.split()[0]}<br>impact")),
        customdata=np.column_stack([[SECTOR[t] for t in TICKERS], w, w * val]),
        hovertemplate="<b>%{text}</b> (%{customdata[0]})<br>propagation w = %{customdata[1]:.2f}"
                      "<br>signed impact = %{customdata[2]:+.2f}<extra></extra>"))
    fig.update_layout(template=TEMPLATE, height=520, showlegend=False, margin=dict(t=10, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("🛰️ S&P Sentinel")
st.sidebar.caption("One event, two risk lenses.")
scenario = st.sidebar.radio("Scenario replay", ["All", "red_sea", "banking_stress", "mixed"],
                            format_func=lambda s: {"All": "All scenarios", "red_sea": "Red Sea shipping disruption",
                                                   "banking_stress": "Banking / credit stress", "mixed": "Mixed + entity traps"}[s])
mode = st.sidebar.radio("Data mode", ["Replay (offline, deterministic)", "Live (GDELT + NewsAPI; tweets replayed)"])
min_impact = st.sidebar.slider("Stress trigger: min impact", 1.0, 10.0, 7.0, 0.5)
min_conf = st.sidebar.slider("Stress trigger: min confidence", 0.0, 1.0, 0.6, 0.05)
closed_loop = st.sidebar.toggle("Closed loop: Module B tightens Module A", value=True)
with st.sidebar.expander("Data and assumptions"):
    st.markdown(
        "- News, tweets and the wholesale book are **synthetic**; tickers are used only for realism.\n"
        "- X/Twitter API is paid, so social posts run in **replay mode**.\n"
        "- Shock sizes, CET1 levels and `CLASS_PRIOR` are **illustrative**, not calibrated to CCAR/EBA.\n"
        "- IRB capital uses the simplified corporate formula.\n"
        f"- Model version `{MODEL_VERSION}`.")

pipe = run_pipeline(scenario, mode, min_impact, min_conf)
st.title("S&P Sentinel: Unified Risk & Portfolio Analytics")
if pipe is None:
    st.error("No items loaded. In Live mode this means GDELT/NewsAPI returned nothing (no internet or no NEWSAPI_KEY).")
    st.stop()

signals, events = pipe["signals"], pipe["events"]
res_a = pipe["closed"] if closed_loop else pipe["open"]
book = load_book()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Signals", len(signals), help="After de-duplication")
k2.metric("Duplicates removed", len(pipe["dropped"]))
k3.metric("Stress tests triggered", len(events))
k4.metric("Peak risk-budget use", f"{pipe['util'].max():.0%}")
k5.metric("Peak tracking error", f"{res_a['tracking_error'].max():.2%}")
if pipe["price_src"] == "synthetic":
    st.warning("Module A is using SYNTHETIC prices. Run `python -m src.module_a_rebalancer --fetch-prices` for real data.")

tab_feed, tab_an, tab_a, tab_b, tab_c = st.tabs(
    ["📡 Signal feed & audit", "✍️ Analyze text", "⚖️ Module A: Rebalancer", "🏦 Module B: Stress test", "🕸️ Contagion & closed loop"])

# ---------------------------- Tab 1: feed & audit ---------------------------- #
with tab_feed:
    types = sorted({s.event_type.value for s in signals})
    pick = st.multiselect("Filter by event class", types, default=types)
    shown = [s for s in signals if s.event_type.value in pick]
    df = signals_df(shown, min_impact, min_conf)
    st.dataframe(df, hide_index=True, column_config={
        "Equity": st.column_config.NumberColumn(format="%+.2f", help="Sentiment Score from the equity holder's view"),
        "Credit": st.column_config.NumberColumn(format="%+.2f", help="Creditor view; ⚡ = polarity differs from equity"),
        "Impact": st.column_config.ProgressColumn(min_value=1, max_value=10, format="%.1f"),
        "Confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
    st.caption("⚡ equity and credit polarity diverge (e.g. debt-funded buyback)  |  🔴 triggers a Module B stress test")

    if shown:
        st.subheader("Audit trail")
        sig = st.selectbox("Signal", shown, format_func=signal_label, index=0)
        st.markdown(f"**{sig.entity}** ({sig.ticker or 'no ticker'}) | **{sig.event_type.value}** | impact **{sig.impact:.1f}**/10 | "
                    f"confidence **{sig.confidence:.2f}** | half-life **{sig.half_life_h:.0f}h**", unsafe_allow_html=True)
        st.markdown(badge("Equity", sig.sentiment_equity) + badge("Credit", sig.sentiment_credit), unsafe_allow_html=True)
        ev = sig.evidence[0]
        st.markdown(f"<div style='padding:10px;border-left:4px solid {BLUE};background:rgba(44,127,184,0.08)'>"
                    f"{highlight(sig.text, ev.quote)}</div>", unsafe_allow_html=True)
        st.caption(f"Evidence quote {'✅ verified verbatim in source' if ev.verified else '❌ NOT found in source'} | "
                   f"source: {ev.source} | url: {ev.url} | signal_id: {sig.signal_id}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Model votes**")
            st.json(sig.model_votes, expanded=False)
        with c2:
            st.markdown("**Contagion weights** (top 6)")
            top = sorted(sig.contagion.items(), key=lambda kv: kv[1], reverse=True)[:6]
            st.dataframe(pd.DataFrame(top, columns=["Ticker", "Propagation weight"]), hide_index=True)

        st.subheader("Hallucination guard demo")
        mode_g = st.radio("Simulated LLM extraction", ["Grounded output", "Hallucinated output"], horizontal=True)
        q = ev.quote if mode_g == "Grounded output" else ev.quote + " and wiped out $4.2 billion of value"
        nums = [] if mode_g == "Grounded output" else ["4.2"]
        ok, why = guard({"evidence_quote": q, "numeric_claims": nums}, sig.text)
        st.code(f'evidence_quote: "{q}"\nnumeric_claims: {nums}', language="text")
        if ok:
            st.success(f"ACCEPTED {why}")
        else:
            st.error(f"REJECTED: claim not grounded in source text {why}")

    if pipe["evaluation"]:
        with st.expander("Engine evaluation on the labelled sample (development set, rules were tuned on it)"):
            st.dataframe(pd.DataFrame(pipe["evaluation"].items(), columns=["Metric", "Value"]), hide_index=True)

# ---------------------------- Tab 2: analyze text ---------------------------- #
with tab_an:
    st.markdown("Paste any headline or tweet. This is the engine's `/analyze` path: text in, structured risk signal out.")
    txt = st.text_area("Text", SAMPLE_TEXT, height=90)
    if txt.strip():
        sig = apply_contagion([analyze_text(txt.strip())])[0]
        st.markdown(f"**{sig.entity}** | **{sig.event_type.value}** | impact **{sig.impact:.1f}** | confidence **{sig.confidence:.2f}**")
        st.markdown(badge("Equity", sig.sentiment_equity) + badge("Credit", sig.sentiment_credit), unsafe_allow_html=True)
        if should_trigger(sig, min_impact, min_conf):
            r = run_stress(book, sig.event_type.value, sig.impact, sig.confidence, sig.sector, sig.contagion)["summary"]
            st.warning(f"Would trigger a stress test: loss ${r['loss']:,.0f} mn, CET1 {r['cet1_before']:.1%} -> {r['cet1_after']:.2%}")
        st.json(sig.model_dump(mode="json"), expanded=False)

# ---------------------------- Tab 3: Module A ---------------------------- #
with tab_a:
    idx = list(res_a["weights"].index)
    labels = [f"{t:%d %b %H:%M}" for t in idx]
    sel = st.select_slider("Replay time", options=labels, value=labels[-1])
    t = idx[labels.index(sel)]
    m1, m2, m3 = st.columns(3)
    act = (res_a["weights"].loc[t] - res_a["benchmark"]) * 100
    m1.metric("Largest overweight", f"{act.idxmax()}", f"{act.max():+.2f} pp")
    m2.metric("Largest underweight", f"{act.idxmin()}", f"{act.min():+.2f} pp")
    m3.metric("Turnover this step", f"{res_a['turnover'].loc[t]:.2%}", f"gamma {res_a['gamma'].loc[t]:.0f}", delta_color="off")
    st.plotly_chart(fig_weights(res_a, t))
    ca, cb = st.columns(2)
    ca.plotly_chart(fig_active(res_a, t))
    cb.plotly_chart(fig_sentiment_heat(res_a, t))
    st.caption(f"mu = {PARAMS['a']} x (S + {PARAMS['beta']} x V) | cap {PARAMS['cap']:.0%} | TE limit {PARAMS['te_annual']:.0%} | "
               f"hysteresis {PARAMS['hysteresis'] * 1e4:.0f} bp | weights sum to 1, long-only. Equal-weight benchmark.")

# ---------------------------- Tab 4: Module B ---------------------------- #
with tab_b:
    options = [f"{e['ts']:%d %b %H:%M} | {e['signal'].event_type.value} | impact {e['signal'].impact:.1f}" for e in events] + ["Manual what-if"]
    choice = st.selectbox("Stress scenario", options)
    if choice == "Manual what-if":
        w1, w2, w3 = st.columns(3)
        et = w1.selectbox("Event class", list(TEMPLATES) + ["Other"])
        imp = w2.slider("Impact score", 1.0, 10.0, 8.0, 0.1)
        cf = w3.slider("Confidence", 0.1, 1.0, 1.0, 0.05)
        res_b = run_stress(book, et, imp, cf)
    elif events:
        e = events[options.index(choice)]
        res_b = e["result"]
        st.info(f"Triggered by: \"{e['signal'].text}\"")
    else:
        st.info("No signal crossed the trigger thresholds. Use the Manual what-if or lower the thresholds in the sidebar.")
        res_b = run_stress(book, "Credit Event", 8.0)
    s = res_b["summary"]
    c = st.columns(5)
    c[0].metric("Portfolio loss", f"${s['loss']:,.0f} mn", f"-{s['loss_pct']:.2%}", delta_color="inverse")
    c[1].metric("Change in expected loss", f"${s['delta_el']:,.0f} mn")
    c[2].metric("Change in RWA", f"${s['delta_rwa']:,.0f} mn")
    c[3].metric("CET1 ratio", f"{s['cet1_after']:.2%}", f"{(s['cet1_after'] - s['cet1_before']) * 100:+.2f} pp", delta_color="inverse")
    c[4].metric("Capital headroom", f"${s['capital_headroom']:,.0f} mn", "BREACH" if s["breach"] else "OK",
                delta_color="inverse" if s["breach"] else "normal")
    st.plotly_chart(fig_waterfall(res_b))
    d1, d2 = st.columns(2)
    d1.plotly_chart(fig_sector_heat(res_b))
    d2.plotly_chart(fig_migration(res_b))
    st.plotly_chart(fig_top_losses(res_b))
    st.markdown("**Reverse stress test**: how bad must the headline be before the book breaches its CET1 minimum?")
    st.dataframe(reverse_table(), hide_index=True)
    st.caption("Chain: impact score -> systematic factor Z -> Vasicek conditional PD -> downturn LGD -> EL, RWA (IRB), migration. "
               "Bonds: duration + spread duration. Swaps: DV01. Loans: expected loss. Derivatives: CVA proxy. All shocks illustrative.")

# ---------------------------- Tab 5: contagion & loop ---------------------------- #
with tab_c:
    cand = sorted(signals, key=lambda s: s.impact, reverse=True)
    sig_c = st.selectbox("Signal to propagate", cand, format_func=signal_label, key="contagion_sig")
    lens = st.radio("Lens", ["Equity lens", "Credit lens"], horizontal=True)
    st.plotly_chart(fig_contagion(sig_c, lens))
    st.caption("Node size = propagation weight from the named entity (same sector + supply-chain links, decayed over 2 hops). "
               "Colour = weight x sentiment. Switch lens to see equity and credit polarity diverge.")
    st.subheader("Closed loop: Module B risk budget -> Module A optimiser")
    st.plotly_chart(fig_loop(pipe))
    st.caption("When a stress consumes the risk budget, Module A raises risk aversion (gamma) and tightens its tracking-error limit.")