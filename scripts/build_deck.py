"""
Builds docs/presentation.pdf (7 slides, 16:9) from the artifacts in reports/ and data/outputs/,
so every number on a slide traces back to a reproducible file.

    python -m scripts.build_deck
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib.colors import HexColor  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.utils import ImageReader  # noqa: E402
from reportlab.pdfgen import canvas  # noqa: E402
from reportlab.platypus import Paragraph  # noqa: E402

FIG = "docs/figures"
OUT = "docs/presentation.pdf"
W, H = 960, 540
NAVY, ACCENT, GREY, LIGHT = HexColor("#0B1F3A"), HexColor("#C8102E"), HexColor("#5F6B7A"), HexColor("#F2F4F7")
OOS_START = "2019-11-01"

CANDIDATE = "Arsh Agrawal"
EMAIL = "ARSH.23FE10CDS00069@muj.manipal.edu"
COLLEGE = "Manipal University Jaipur"

BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=12.5, leading=17, textColor=HexColor("#1F2933"))
SMALL = ParagraphStyle("small", parent=BODY, fontSize=11.5, leading=15.5)
TINY = ParagraphStyle("tiny", parent=BODY, fontSize=8.5, leading=11, textColor=GREY)


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pct(x, digits=1, sign=True):
    return f"{x * 100:+.{digits}f}%" if sign else f"{x * 100:.{digits}f}%"


def count_tests(path="tests"):
    n = 0
    for name in os.listdir(path):
        if name.startswith("test_") and name.endswith(".py"):
            with open(os.path.join(path, name), encoding="utf-8") as f:
                n += sum(1 for line in f if line.lstrip().startswith("def test_"))
    return n


def money(x):
    return f"-${abs(x) / 1e6:,.0f}m" if x < 0 else f"+${x / 1e6:,.0f}m"


# ---------------------------------------------------------------- figures
def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def fig_module_a():
    nav = pd.read_csv("data/outputs/module_a_nav.csv", parse_dates=["Date"], index_col="Date")
    spy = pd.read_csv("data/prices/SPY.csv", parse_dates=["Date"], index_col="Date")
    col = "Adj Close" if "Adj Close" in spy.columns else "Close"
    spy = spy[col].reindex(nav.index).ffill()
    spy = spy / spy.iloc[0]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 4.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    a1.plot(nav.index, nav["strategy"], label="Sentiment-tilted index", color="#C8102E", lw=1.4)
    a1.plot(nav.index, nav["equal_weight"], label="Equal-weight benchmark", color="#0B1F3A", lw=1.1)
    a1.plot(spy.index, spy.values, label="SPY", color="#9AA5B1", lw=0.9, ls="--")
    rel = nav["strategy"] / nav["equal_weight"] - 1
    a2.fill_between(rel.index, rel.values * 100, 0, color="#C8102E", alpha=0.35, lw=0)
    a2.set_ylabel("excess %", fontsize=8)
    for a in (a1, a2):
        a.axvspan(pd.Timestamp(OOS_START), nav.index[-1], color="#1A73E8", alpha=0.07, lw=0)
        style(a)
    a1.text(pd.Timestamp(OOS_START), a1.get_ylim()[1], " out-of-sample", fontsize=7, va="top", color="#1A73E8")
    a1.set_ylabel("NAV (start = 1)", fontsize=8)
    a1.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/module_a_nav.png", dpi=180)
    plt.close(fig)


def fig_module_a_weights():
    w = pd.read_csv("data/outputs/module_a_weights.csv", parse_dates=["Date"], index_col="Date")
    active = (w - 1 / w.shape[1]).resample("W").last() * 100
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    lim = np.nanpercentile(np.abs(active.values), 99)
    im = ax.imshow(active.T.values, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                   extent=[0, len(active), len(active.columns), 0], interpolation="nearest")
    ax.set_yticks(np.arange(len(active.columns)) + 0.5, active.columns, fontsize=6.5)
    ticks = np.linspace(0, len(active) - 1, 6).astype(int)
    ax.set_xticks(ticks + 0.5, [active.index[i].strftime("%b %y") for i in ticks], fontsize=7)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("active weight vs 1/N (pp)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_title("Index weights over time (weekly snapshot, deviation from equal weight)", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(f"{FIG}/module_a_weights.png", dpi=180)
    plt.close(fig)


def fig_module_b(stress):
    sc = stress["reference_scenarios"]
    names = [n for n in sc if abs(sc[n]["total_pnl"]) > 0]
    names = sorted(names, key=lambda n: sc[n]["total_pnl"])
    classes = ["Loan", "Bond", "Derivative"]
    colors = {"Loan": "#0B1F3A", "Bond": "#C8102E", "Derivative": "#9AA5B1"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    y = np.arange(len(names))
    left_neg, left_pos = np.zeros(len(names)), np.zeros(len(names))
    for c in classes:
        vals = np.array([sc[n]["pnl_by_asset_class"].get(c, 0) / 1e6 for n in names])
        pos, neg = np.where(vals > 0, vals, 0), np.where(vals < 0, vals, 0)
        a1.barh(y, neg, left=left_neg, color=colors[c], label=c, height=0.6)
        a1.barh(y, pos, left=left_pos, color=colors[c], height=0.6)
        left_neg, left_pos = left_neg + neg, left_pos + pos
    for i, n in enumerate(names):
        a1.text(left_neg[i] - 8, i, money(sc[n]["total_pnl"]), va="center", ha="right", fontsize=7.5)
    a1.set_yticks(y, [n.replace("/", "/\n") for n in names], fontsize=7.5)
    a1.axvline(0, color="black", lw=0.6)
    a1.set_xlabel("P&L by asset class ($m), scenario at impact 9", fontsize=8)
    a1.set_xlim(left_neg.min() * 1.45, max(left_pos.max() * 1.3, 40))
    a1.legend(fontsize=7.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    style(a1)
    before = [sc[n]["cet1_ratio_before"] * 100 for n in names]
    after = [sc[n]["cet1_ratio_after"] * 100 for n in names]
    a2.barh(y + 0.18, before, height=0.34, color="#CBD2D9", label="before")
    a2.barh(y - 0.18, after, height=0.34, color=["#C8102E" if sc[n]["breaches_buffer"] else "#0B1F3A" for n in names],
            label="after")
    a2.axvline(sc[names[0]]["cet1_minimum_with_buffer"] * 100, color="#C8102E", ls="--", lw=0.9)
    a2.set_yticks(y, [""] * len(names))
    a2.set_xlabel("CET1 ratio % (dashed = 7% buffer)", fontsize=8)
    a2.legend(fontsize=7.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    style(a2)
    fig.tight_layout()
    fig.savefig(f"{FIG}/module_b_scenarios.png", dpi=180)
    plt.close(fig)
    return names


# ---------------------------------------------------------------- slide primitives
def header(c, title, n):
    c.setFillColor(NAVY)
    c.rect(0, H - 62, W, 62, fill=1, stroke=0)
    c.setFillColor(ACCENT)
    c.rect(0, H - 66, W, 4, fill=1, stroke=0)
    c.setFillColor(HexColor("#FFFFFF"))
    c.setFont("Helvetica-Bold", 24)
    c.drawString(36, H - 42, title)
    c.setFont("Helvetica", 9)
    c.setFillColor(GREY)
    c.drawString(36, 16, f"{CANDIDATE} · {COLLEGE} · S&P Global × Crisil Campus Hackathon 2026")
    c.drawRightString(W - 36, 16, f"{n} / 7")


def para(c, html, x, y_top, width, style_=BODY):
    p = Paragraph(html, style_)
    _, h = p.wrap(width, H)
    p.drawOn(c, x, y_top - h)
    return y_top - h


def bullets(c, items, x, y_top, width, style_=BODY, gap=6):
    for it in items:
        y_top = para(c, f"<bullet>&bull;</bullet>{it}", x, y_top, width,
                     ParagraphStyle("b", parent=style_, leftIndent=14, bulletIndent=0)) - gap
    return y_top


def image(c, path, x, y, w=None, h=None):
    img = ImageReader(path)
    iw, ih = img.getSize()
    if w and not h:
        h = w * ih / iw
    elif h and not w:
        w = h * iw / ih
    elif w and h:
        s = min(w / iw, h / ih)
        w, h = iw * s, ih * s
    c.drawImage(img, x, y, w, h, mask="auto")
    return w, h


def kpi(c, x, y, value, label, color=NAVY, w=150):
    c.setFillColor(LIGHT)
    c.roundRect(x, y, w, 58, 6, fill=1, stroke=0)
    c.setFillColor(color)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(x + 10, y + 30, value)
    c.setFillColor(GREY)
    c.setFont("Helvetica", 8.5)
    c.drawString(x + 10, y + 12, label)


# ---------------------------------------------------------------- slides
def main():
    os.makedirs(FIG, exist_ok=True)
    ev = load("reports/engine_eval.json")
    a = load("reports/module_a_backtest.json")
    b = load("reports/module_b_stress.json")
    ds = pd.read_csv("data/samples/unified_risk_dataset.csv", usecols=["source", "split", "impact_scope", "has_event_label",
                                                                       "has_impact_label", "has_sentiment_label"])
    n_signals = sum(1 for _ in open("data/samples/signals.jsonl", encoding="utf-8"))
    fig_module_a()
    fig_module_a_weights()
    scen_order = fig_module_b(b)

    tr = ev["test"].get("transformer", ev["test"]["baseline"])
    bl = ev["test"]["baseline"]
    model_name = "multi-task DistilRoBERTa" if "transformer" in ev["test"] else "TF-IDF baseline"
    lat_key = next((k for k in ev["latency"] if k.startswith("transformer") and not k.endswith("cpu")), "baseline_cpu")
    lat = ev["latency"][lat_key]
    lat_cpu = ev["latency"].get("transformer_cpu", ev["latency"]["baseline_cpu"])
    oos = a["out_of_sample"]
    macro = b["reference_scenarios"]["Macroeconomic"]
    credit = b["reference_scenarios"]["Credit Event"]
    replay = b["historical_replay"]

    c = canvas.Canvas(OUT, pagesize=(W, H))
    c.setTitle("AI/NLP Financial Risk Engine — S&P Global x Crisil Hackathon 2026")
    c.setAuthor(CANDIDATE)

    # 1 · Title
    c.setFillColor(NAVY)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(ACCENT)
    c.rect(60, 300, 6, 120, fill=1, stroke=0)
    c.setFillColor(HexColor("#FFFFFF"))
    c.setFont("Helvetica-Bold", 38)
    c.drawString(84, 380, "From Headlines to Hedges")
    c.setFont("Helvetica", 20)
    c.drawString(84, 345, "An AI/NLP risk engine that turns news & social text into")
    c.drawString(84, 320, "sentiment, event and impact signals for index and credit risk")
    c.setFont("Helvetica", 13)
    c.setFillColor(HexColor("#CBD2D9"))
    c.drawString(84, 230, f"{CANDIDATE}  ·  {COLLEGE}")
    c.drawString(84, 210, EMAIL)
    c.drawString(84, 170, "S&P Global × Crisil Campus Hackathon 2026 · Phase III case study")
    c.drawString(84, 150, "Modules delivered: A (sentiment index rebalancing) and B (event-driven stress testing)")
    c.showPage()

    # 2 · Problem & approach
    header(c, "Problem & approach", 2)
    y = para(c, "<b>The problem.</b> Market-moving information arrives as unstructured text — a downgrade, a "
                "sanctions headline, a CEO tweet — long before it shows up in prices, spreads or ratings. Risk "
                "teams need it <b>quantified, classified and routed</b> to the processes that act on it.",
             36, H - 90, 430)
    y = para(c, "<b>The approach.</b>", 36, y - 16, 430)
    bullets(c, [
        "<b>Three outputs per text</b>: sentiment in [-1, 1], one of 9 event types, and an impact score 1-10 "
        "that is <i>anchored to realised market reactions</i>, not a guess.",
        "<b>Three live-capable sources</b>: financial news (Benzinga), social posts (Stock Tweets), and the "
        "GDELT global event feed for macro & geopolitical news.",
        "<b>One shared model</b>: a fine-tuned multi-task transformer, with a TF-IDF baseline as a measured "
        "yardstick and automatic fallback.",
        "<b>Two consumers</b>: an index desk (Module A) and a credit-risk / capital desk (Module B), fed "
        "through a REST API and a JSONL file.",
    ], 36, y - 8, 430, SMALL, gap=8)
    c.setFillColor(LIGHT)
    c.roundRect(490, 70, 434, 390, 8, fill=1, stroke=0)
    para(c, "<b>Design choices that matter to a risk user</b>", 506, 448, 400)
    bullets(c, [
        "<b>No look-ahead anywhere.</b> Chronological train/val/test split; weights formed on day t trade on t+1; "
        "post-16:00 ET headlines roll to the next session.",
        "<b>Impact = event study, at the right scope.</b> Company news: abnormal return over [0,+1] vs SPY "
        "÷ 60-day idiosyncratic vol. Market-wide news: the SPY move ÷ its own 60-day vol (a market model would "
        "net systemic shocks out). Train-set deciles, so a 9 means a top-decile surprise.",
        "<b>Honest labels.</b> Event classes come from BART-MNLI zero-shot reconciled with keyword rules, and a "
        "268-row hand-labelled set is the independent check.",
        "<b>Uncertainty is first-class.</b> The impact head predicts a mean and a variance, so every signal "
        "carries a confidence that downstream modules can threshold.",
        "<b>Regulatory framing.</b> Module B speaks the bank's language: IFRS 9 ECL & SICR staging, "
        "standardised RWA, CET1 vs the 7% buffer.",
    ], 506, 420, 404, SMALL, gap=5)
    c.showPage()

    # 3 · System design
    header(c, "System design", 3)
    image(c, "docs/architecture.png", 36, 34, h=440)
    bullets(c, [
        "<b>Ingestion</b>: per-source adapters normalise to a single <font face='Courier'>Document</font> "
        "schema (UTC time, stable id, entity hints).",
        "<b>Entity linking</b>: cashtags, aliases and context rules map text to a 20-stock S&amp;P 100 universe; "
        "ambiguous names (Apple, Meta) need financial context. Unlinked macro news becomes an event-level signal.",
        "<b>Inference</b>: one encoder pass yields all three outputs and their confidences.",
        "<b>Delivery</b>: FastAPI (<font face='Courier'>/analyze, /signals, /signals/{ticker}, /ingest/gdelt, "
        "/modules/*</font>) and <font face='Courier'>signals.jsonl</font>.",
        "<b>Consumers</b>: Module A subscribes to sentiment; Module B to event type + impact. A Streamlit "
        "dashboard exposes all of it live.",
    ], 470, H - 92, 454, SMALL, gap=7)
    c.showPage()

    # 4 · Implementation
    header(c, "Implementation", 4)
    src_counts = ds.groupby(["source", "impact_scope"]).size().to_dict()
    rows = [("Source", "Rows", "Supplies")]
    pretty = {("financial_phrasebank", "company"): ("FinancialPhraseBank", "human sentiment labels"),
              ("benzinga_news", "company"): ("Benzinga company news", "events + company impact"),
              ("benzinga_news", "market"): ("Benzinga market-wide news", "market impact (SPY move)"),
              ("stock_tweets", "company"): ("Stock Tweets", "social events + impact")}
    for k, (name, what) in pretty.items():
        if k in src_counts:
            rows.append((name, f"{src_counts[k]:,}", what))
    y = H - 92
    para(c, "<b>Training data</b> (public Kaggle sources; all files in <font face='Courier'>data/</font>)", 36, y, 430)
    y -= 26
    c.setFont("Helvetica-Bold", 10)
    for i, r in enumerate(rows):
        c.setFont("Helvetica-Bold" if i == 0 else "Helvetica", 10)
        c.setFillColor(NAVY if i == 0 else HexColor("#1F2933"))
        c.drawString(44, y, r[0])
        c.drawRightString(262, y, r[1])
        c.drawString(276, y, r[2])
        y -= 16
    y = para(c, f"Total {len(ds):,} rows · {int(ds.has_sentiment_label.sum()):,} sentiment · "
                f"{int(ds.has_event_label.sum()):,} event · {int(ds.has_impact_label.sum()):,} impact labels. "
                f"Test split = news published after {ev['splits']['val_end']} (includes the COVID-19 crash).",
             36, y - 4, 430, SMALL)
    bullets(c, [
        "<b>Model</b>: DistilRoBERTa encoder, mean pooling, three heads — tanh sentiment regression, 9-way "
        "event softmax, heteroscedastic impact (mean + log-variance, Gaussian NLL).",
        "<b>Masked multi-task loss</b>: each row only trains the heads it has labels for; class-balanced "
        "event weights; majority class capped at 40%.",
        "<b>Training</b>: AdamW, heads at 10× encoder LR, warm-up, early stopping on a composite validation "
        "score. Runs on an Apple M3 (MPS) in minutes.",
        f"<b>Engineering</b>: {count_tests()} pytest tests (no-look-ahead, trigger rule, bond / swap signs, API), "
        "pinned requirements, <font face='Courier'>make</font> targets for every stage.",
    ], 36, y - 14, 430, SMALL, gap=5)
    c.setFillColor(LIGHT)
    c.roundRect(490, 70, 434, 390, 8, fill=1, stroke=0)
    para(c, "<b>Downstream modules</b>", 506, 448, 400)
    bullets(c, [
        f"<b>Module A</b>: per-stock EWMA of {'impact-weighted ' if a['config'].get('impact_weighted') else ''}"
        f"sentiment (half-life {a['config']['halflife_days']:.0f}d); target "
        f"w ∝ (1/N)·exp(λ·s), λ={a['config']['tilt_lambda']:g}; bounds "
        f"[{a['config']['min_weight']:.0%}, {a['config']['max_weight']:.0%}]; "
        f"{a['config']['max_daily_turnover']:.0%} daily turnover budget; {a['config']['cost_bps']:.0f} bps costs. "
        f"λ, half-life and weighting chosen on the in-sample window only ({len(a['parameter_selection']['grid'])}-point grid).",
        f"<b>Module B</b>: synthetic wholesale book of {b['portfolio']['positions']} positions — "
        "corporate loans, IG/HY & sovereign bonds, rates swaps, equity TRS & puts, FX forwards, commodity "
        "swaps, CDS.",
        "Trigger: impact ≥ 8, a systemic event type, adverse sentiment and event confidence ≥ 0.6; one "
        "scenario per event-day. Shock vector (equity, rates, IG/HY spreads, USD, commodities, sector PD "
        "multipliers) is scaled by impact/10.",
        "Revaluation: bonds by duration + convexity; loans by IFRS 9 ECL with SICR staging (2× PD) and "
        "floating-rate PD uplift; derivatives by DV01 / delta-gamma / CS01; capital via standardised RWA → CET1.",
    ], 506, 420, 404, SMALL, gap=5)
    c.showPage()

    # 5 · Key results
    header(c, "Key results", 5)
    image(c, f"{FIG}/model_comparison.png", 30, 250, w=440, h=220)
    image(c, f"{FIG}/module_a_nav.png", 488, 210, w=440, h=262)
    kx = 36
    for v, l in [(f"{tr['sentiment']['pearson_r']:.2f}", "sentiment Pearson r (test)"),
                 (f"{tr['event_type']['macro_f1']:.2f}", "event macro-F1 (test)"),
                 (f"{tr['impact_score']['spearman_rho']:.2f}", "impact Spearman ρ (test)")]:
        kpi(c, kx, 172, v, l, w=140)
        kx += 146
    para(c, f"<b>{model_name}</b> vs TF-IDF baseline on the same held-out test split: sentiment r "
            f"{bl['sentiment']['pearson_r']:.2f} → {tr['sentiment']['pearson_r']:.2f}, event macro-F1 "
            f"{bl['event_type']['macro_f1']:.2f} → {tr['event_type']['macro_f1']:.2f}, impact ρ "
            f"{bl['impact_score']['spearman_rho']:.2f} → {tr['impact_score']['spearman_rho']:.2f}. "
            f"Latency {lat['single_p50_ms']:.0f} ms p50 per headline ({lat_key.split('_')[-1].upper()}), "
            f"{lat_cpu['single_p50_ms']:.0f} ms on CPU; {lat['batch_throughput_per_s']:.0f} headlines/s batched.",
         36, 160, 430, SMALL)
    kx = 488
    for v, l, col in [(pct(oos["active"]["excess_total_return"]), "excess return vs EW (OOS)", ACCENT),
                      (f"{oos['active']['information_ratio']:.2f}", "information ratio (OOS)", NAVY),
                      (f"{oos['sentiment_ic']['mean']:.3f}", f"sentiment IC (OOS, t={oos['sentiment_ic']['t_stat']:.1f})",
                       NAVY)]:
        kpi(c, kx, 136, v, l, color=col, w=140)
        kx += 146
    para(c, f"Module A, out-of-sample {oos['period']['start']} → {oos['period']['end']} (through the COVID "
            f"crash): Sharpe {oos['strategy']['sharpe']:.2f} vs {oos['equal_weight_benchmark']['sharpe']:.2f} "
            f"for equal weight, tracking error {pct(oos['active']['tracking_error'], sign=False)}, "
            f"daily hit rate {pct(oos['active']['daily_hit_rate'], 0, False)}, "
            f"avg turnover {pct(oos['avg_daily_turnover'], 2, False)}/day. "
            f"Signals from <i>{a['model_version']}</i>; {n_signals:,} signals in the stream.",
         488, 124, 436, SMALL)
    c.showPage()

    # 6 · Domain impact
    header(c, "Domain impact", 6)
    image(c, f"{FIG}/module_b_scenarios.png", 30, 150, w=500)
    para(c, f"Reference scenarios at impact 9 on the synthetic $"
            f"{b['reference_scenarios']['Macroeconomic']['portfolio_value_before'] / 1e9:.1f}bn book. "
            f"A Macroeconomic shock (rates +, spreads wider) costs <b>{money(macro['total_pnl'])}</b> "
            f"({pct(macro['pnl_pct_of_value'])}), moves {macro['loans_moved_to_stage2']} loans to Stage 2 and takes "
            f"CET1 from {pct(macro['cet1_ratio_before'], sign=False)} to <b>{pct(macro['cet1_ratio_after'], sign=False)}"
            f"</b> — {'below' if macro['breaches_buffer'] else 'above'} the 7% buffer. A Credit Event moves "
            f"{credit['loans_moved_to_stage2']} loans to Stage 2 (ECL {money(-credit['ecl_before'])[1:]} → "
            f"{money(-credit['ecl_after'])[1:]}). Pay-fixed swaps and CDS protection offset part of the losses.",
         36, 140, 490, SMALL)
    c.setFillColor(LIGHT)
    c.roundRect(552, 40, 372, 420, 8, fill=1, stroke=0)
    para(c, "<b>What this changes for a risk desk</b>", 568, 448, 340)
    bullets(c, [
        "<b>Minutes, not days.</b> A downgrade or sanctions headline becomes a sized, typed, scored signal "
        "and — if severe — an automatic stress run with a capital read-out.",
        "<b>Early-warning for credit.</b> Sentiment and Credit-Event signals per obligor flag SICR candidates "
        "before ratings move; ECL and Stage-2 migration are quantified immediately.",
        "<b>Capital planning.</b> The CET1 waterfall shows which desks and sectors consume the buffer — the "
        "input for hedging or limit decisions.",
        "<b>Systematic overlay for index products.</b> Module A's tilt is bounded, turnover-capped and "
        "cost-aware; weighting sentiment by predicted impact beat plain averaging in-sample.",
        f"<b>Audit trail.</b> Every signal keeps source, timestamp, model version and confidence; history "
        f"replay scanned {replay['signals_scanned']:,} signals and triggered {replay['stress_tests_triggered']} "
        "stress tests.",
    ], 568, 420, 340, SMALL, gap=6)
    c.showPage()

    # 7 · Limitations
    header(c, "Limitations & next steps", 7)
    hand = ev.get("hand_labelled", {})
    hand_txt = (f"Hand-labelled check: {hand.get('labelled_rows', 0)} rows scored."
                if hand.get("status") in ("complete", "partial") else
                "The 268-row hand-labelled event set is prepared but not yet fully annotated.")
    bullets(c, [
        "<b>Label quality.</b> Event labels are zero-shot + rules (a teacher), so test macro-F1 measures "
        f"agreement with the teacher, not ground truth. {hand_txt}",
        "<b>Impact is noisy by nature.</b> Daily returns are dominated by non-news variance: company-impact ρ is "
        f"{tr.get('impact_score_company', tr['impact_score'])['spearman_rho']:.2f} on test, and the day-level "
        f"market-wide label does not yet generalise to the COVID window (ρ "
        f"{tr.get('impact_score_market', tr['impact_score'])['spearman_rho']:.2f}). Intraday prices would sharpen both.",
        "<b>Data window.</b> Free Benzinga and Stock Tweets data end in mid-2020 and cover 20 large caps; "
        "social posts are sparse outside 2018 H2. GDELT is live but rate-limited.",
        "<b>Module B is a sensitivity model.</b> Linear / quadratic revaluation, a static balance sheet and "
        "a hand-calibrated scenario library; no full revaluation, liquidity or second-round effects.",
        f"<b>Module A is a 2-year backtest</b> on 20 names with simple costs; the out-of-sample edge is small "
        f"(IC t-stat {oos['sentiment_ic']['t_stat']:.1f}, not significant) and needs a longer, wider universe "
        "before any capital is allocated.",
    ], 36, H - 92, 440, SMALL, gap=8)
    c.setFillColor(LIGHT)
    c.roundRect(500, 70, 424, 390, 8, fill=1, stroke=0)
    para(c, "<b>Next steps</b>", 516, 448, 390)
    bullets(c, [
        "Active-learning loop: route low-confidence signals to analysts via the labelling tool; retrain weekly.",
        "Larger finance encoders (FinBERT / domain-adapted RoBERTa) and entity-level sentiment for multi-company headlines.",
        "Intraday event study and sector-relative abnormal returns for sharper impact labels.",
        "Obligor-level signal aggregation feeding PD overlays; reverse stress testing for the CET1 buffer.",
        "Production hardening: streaming ingestion (Kafka), model registry, drift monitoring on sentiment and class mix.",
    ], 516, 420, 392, SMALL, gap=7)
    para(c, "AI-assistance disclosure: code was written with an AI coding assistant under the author's direction; "
            "all design decisions, data choices and results were reviewed by the author.", 36, 60, 440, TINY)
    c.showPage()
    c.save()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
