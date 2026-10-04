"""Render docs/architecture.png.   Run from repo root:  python docs/make_architecture.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "architecture.png"
C = {"src": "#e8eef7", "ing": "#e3f4ea", "nlp": "#fff2d9", "sig": "#f3e5f5", "mod": "#fde4e1", "ui": "#dff0f7"}
EDGE = {"src": "#4a6fa5", "ing": "#2e8b57", "nlp": "#c98a00", "sig": "#8e44ad", "mod": "#c0392b", "ui": "#1f78a8"}

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 160)
ax.set_ylim(0, 90)
ax.axis("off")


def box(x, y, w, h, text, kind, dashed=False, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", linewidth=1.6,
                                edgecolor=EDGE[kind], facecolor=C[kind], linestyle="--" if dashed else "-"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(x1, y1, x2, y2, color="#555", style="-|>", dashed=False, rad=0.0, lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14, linewidth=lw, color=color,
                                 linestyle="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}"))


def header(x, w, text, kind):
    ax.text(x + w / 2, 87, text, ha="center", fontsize=10.5, fontweight="bold", color=EDGE[kind])


ax.text(80, 91.2, "S&P Sentinel: one event, two risk lenses", ha="center", fontsize=15, fontweight="bold")

# column headers
header(1, 26, "1  DATA SOURCES", "src"); header(33, 25, "2  INGESTION", "ing"); header(64, 38, "3  NLP RISK ENGINE", "nlp")
header(108, 22, "4  SIGNAL CONTRACT", "sig"); header(136, 23, "5  DOWNSTREAM MODULES", "mod")

# 1 sources
box(1, 70, 26, 11, "GDELT DOC 2.0\n(live, no key)", "src")
box(1, 55, 26, 11, "NewsAPI\n(live, free tier)", "src")
box(1, 40, 26, 11, "Tweets: replay mode\n(X API is paid)", "src", dashed=True)
box(1, 25, 26, 11, "yfinance prices\n(public, 2y daily)\n-> Module A covariance", "src", fs=8.6)
box(1, 10, 26, 11, "Synthetic wholesale book\n(400 exposures, seeded)\n-> Module B portfolio", "src", fs=8.6)

# 2 ingestion
box(33, 55, 25, 24, "Normalise + clean\n\nNear-duplicate removal\n(token Jaccard)\n\nReplay / live switch", "ing")

# 3 NLP engine (stacked)
ys = [72, 60.5, 49, 37.5, 26]
texts = ["Entity resolution\nalias table + cashtags + context\n(\"Apple harvest\" is not AAPL)",
         "Event classification\nweighted rules (+ optional zero-shot)",
         "Dual-polarity sentiment\nVADER+fin lexicon (+ optional FinBERT)\nsarcasm flag + equity/credit rule layer",
         "Impact score 1-10 + confidence\nclass prior, magnitude, reach, velocity",
         "Evidence + hallucination guard\nverbatim quote check (+ optional LLM)"]
for y, t in zip(ys, texts):
    box(64, y, 38, 9.5, t, "nlp", fs=8.4)
for a, b in zip(ys[:-1], ys[1:]):
    arrow(83, a, 83, b + 9.5, color=EDGE["nlp"], lw=1.2)
box(64, 10, 38, 11, "Contagion graph\nsector + supply-chain links (2 hops)\n-> per-ticker propagation weights", "nlp", fs=8.4)
arrow(83, 26, 83, 21, color=EDGE["nlp"], lw=1.2)

# 4 signal contract
box(108, 55, 22, 24, "RiskSignal (pydantic)\n\nsentiment_equity\nsentiment_credit\nevent_type, impact\nconfidence, evidence\nmodel_votes, contagion\n\n-> signals_sample.json", "sig", fs=8.4)
box(108, 38, 22, 12, "FastAPI + WebSocket\n(production path,\nnot in this prototype)", "sig", dashed=True, fs=8.4)

# 5 modules
box(136, 55, 23, 24, "MODULE A\nTactical rebalancer\n\ndecayed sentiment + velocity\ncvxpy mean-variance\ncap, tracking-error limit\n25 bp hysteresis", "mod", fs=8.4)
box(136, 20, 23, 30, "MODULE B\nEvent-driven stress test\n\nZ shock -> Vasicek PD\ndownturn LGD, IRB RWA\nbond / swap / loan valuation\nrating migration, CET1\nreverse stress test", "mod", fs=8.4)
arrow(147.5, 50, 147.5, 55, color=EDGE["mod"], lw=2.2)
ax.text(149, 52.5, "risk budget\n(closed loop)", fontsize=8, color=EDGE["mod"], va="center", fontweight="bold")

# dashboard band
box(33, 1.5, 126, 5.5, "Streamlit dashboard:  feed + audit trail  |  analyze text  |  Module A  |  Module B  |  contagion + closed loop",
    "ui", fs=9.5, bold=True)

# main flow arrows
for y in (75, 60, 45):
    arrow(27, y, 33, 67 if y == 75 else 67 if y == 60 else 62, color=EDGE["src"])
arrow(58, 67, 64, 76, color=EDGE["ing"])
arrow(102, 66, 108, 66, color=EDGE["nlp"], lw=2.2)
arrow(102, 31, 108, 56, color=EDGE["nlp"], rad=-0.2)
arrow(130, 70, 136, 70, color=EDGE["sig"], lw=2.2)
arrow(130, 62, 136, 40, color=EDGE["sig"], rad=-0.15)
ax.text(83, 4.0, "", fontsize=1)
arrow(147.5, 20, 147.5, 7.2, color=EDGE["ui"])
arrow(119, 38, 119, 7.2, color=EDGE["ui"], dashed=True)
arrow(83, 10, 83, 7.2, color=EDGE["ui"])

fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
