"""
Risk Engine dashboard.

    streamlit run app/dashboard.py
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.dashboard_logic import (  # noqa: E402
    DEMO_HEADLINES, STRESS_DEFAULT_HEADLINE, STRESS_FORCED, STRESS_TRIGGERED, calibration_range,
    cet1_breach_count, fmt_money, gate_failures, hand_label_summary, model_card_rows, oos_start,
    parse_headlines, result_for_download, stream_scope, trigger_gates,
)
from src.engine.pipeline import RiskEngine  # noqa: E402
from src.engine.store import SignalStore  # noqa: E402
from src.ingestion.gdelt import GdeltAdapter  # noqa: E402
from src.modules.rebalancer import RebalancerConfig, load_close_prices, load_universe, run_backtest  # noqa: E402
from src.modules.stress import StressTestEngine  # noqa: E402

SIGNALS = "data/samples/signals.jsonl"
OOS_START = oos_start()
BACKTEST_START, BACKTEST_END = "2018-04-02", "2020-06-10"
EVENT_COLORS = {
    "Geopolitical": "#d62728", "Macroeconomic": "#ff7f0e", "Credit Event": "#8c564b",
    "Merger/Acquisition": "#9467bd", "Product Launch": "#2ca02c", "Regulatory/Legal": "#e377c2",
    "Earnings/Guidance": "#1f77b4", "Operational/Supply-chain": "#bcbd22", "Other/None": "#7f7f7f",
}
PARAM_KEYS = {"tilt_lambda": "p_lambda", "halflife_days": "p_halflife", "max_weight": "p_wmax",
              "max_daily_turnover": "p_turnover", "impact_weighted": "p_impact"}

st.set_page_config(page_title="AI/NLP Financial Risk Engine", layout="wide")


# ---- cached resources --------------------------------------------------------------------

BACKEND = os.environ.get("RISK_ENGINE_BACKEND", "auto")


@st.cache_resource(show_spinner="Loading risk engine (the first start downloads the ~300 MB model)...")
def get_engine() -> RiskEngine:
    if BACKEND != "baseline":
        # No-op when the checkpoint exists; on a fresh host (e.g. Streamlit Cloud) it pulls the
        # release asset. A failed download leaves "auto" on the committed TF-IDF baseline.
        subprocess.run([sys.executable, "-m", "scripts.fetch_model"], cwd=ROOT, check=False)
    return RiskEngine(backend=BACKEND)


@st.cache_resource
def get_stress() -> StressTestEngine:
    return StressTestEngine()


@st.cache_resource(show_spinner="Loading signal stream...")
def signal_objects(mtime: float):
    return SignalStore(SIGNALS).read()


@st.cache_data(show_spinner="Loading signal stream...")
def load_signals(mtime: float) -> pd.DataFrame:
    df = pd.DataFrame([s.model_dump() for s in signal_objects(mtime)])
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["date"] = df["timestamp"].dt.tz_convert(None).dt.normalize()
    return df


@st.cache_data
def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_csv(path: str, **kw) -> pd.DataFrame:
    return pd.read_csv(path, **kw) if os.path.exists(path) else pd.DataFrame()


@st.cache_data
def prices() -> pd.DataFrame:
    return load_close_prices(load_universe() + ["SPY"])


@st.cache_data(show_spinner="Backtesting...", max_entries=16)
def backtest(lam: float, hl: float, wmax: float, to: float, iw: bool, mtime: float):
    out = run_backtest(signal_objects(mtime), BACKTEST_START, BACKTEST_END, OOS_START,
                       RebalancerConfig(tilt_lambda=lam, halflife_days=hl, max_weight=wmax,
                                        max_daily_turnover=to, impact_weighted=iw))
    return out["report"], out["nav"], out["weights"], out["sentiment_state"]


@st.cache_data(max_entries=64)
def scenario(event_type: str, impact: int) -> dict:
    stress = get_stress()
    return stress.run(stress.shock_for(event_type, impact))


def signals_frame(signals) -> pd.DataFrame:
    return pd.DataFrame([{
        "ticker": s.ticker or "-", "sentiment": s.sentiment_score, "label": s.sentiment_label,
        "event_type": s.event_type, "event_conf": s.event_confidence, "impact": s.impact_score,
        "impact_conf": s.impact_confidence, "text": s.text_excerpt,
    } for s in signals])


SIGNAL_COLUMNS = {
    "sentiment": st.column_config.ProgressColumn("sentiment (-1 to +1)", min_value=-1, max_value=1, format="%.2f"),
    "impact": st.column_config.ProgressColumn("impact (1-10)", min_value=1, max_value=10, format="%d"),
    "event_conf": st.column_config.NumberColumn("event confidence", format="percent"),
    "impact_conf": st.column_config.NumberColumn("impact confidence", format="percent"),
}

mtime = os.path.getmtime(SIGNALS) if os.path.exists(SIGNALS) else 0.0
engine = get_engine()
stress = get_stress()
sig_df = load_signals(mtime)
ev = load_json("reports/engine_eval.json")

# ==== Header ===============================================================================
st.title("AI/NLP Financial Risk Engine")
st.markdown("Turns public financial news, social posts and a global event feed into scored risk signals "
            "(**sentiment**, **event type**, **impact 1-10**) and feeds them into two downstream decisions: "
            "tilting an equity index and stress-testing a banking book.")

event_f1 = ev.get("test", {}).get("transformer", {}).get("event_type", {}).get("macro_f1")
event_f1_base = ev.get("test", {}).get("baseline", {}).get("event_type", {}).get("macro_f1")
h1, h2, h3, h4 = st.columns(4)
h1.metric("Signals scored", f"{len(sig_df):,}", help="Every headline/post in the committed stream, scored by the transformer.")
h2.metric("Stocks in the index", f"{len(load_universe())}", help="Module A tilts an equal-weighted mock index.")
h3.metric("Banking-book positions", f"{len(stress.portfolio)}", help="Synthetic loans, bonds and derivatives (Module B).")
if event_f1 is not None:
    h4.metric("Event macro-F1 (held-out)", f"{event_f1:.2f}",
              f"{event_f1 - event_f1_base:+.2f} vs TF-IDF baseline" if event_f1_base is not None else None,
              help="Nine event classes, chronological test split never seen in training.")

with st.container(border=True):
    st.markdown("**Public news · social · GDELT** → **Multitask NLP model** (sentiment · event · impact) → "
                "**Module A**: index weights tilt toward positive sentiment → "
                "**Module B**: severe adverse events trigger a portfolio stress test")
    live = ("fine-tuned DistilRoBERTa transformer" if engine.backend == "transformer"
            else "TF-IDF baseline (fallback)")
    st.caption(f"Live scoring: **{live}** · Historical stream {BACKTEST_START} to {BACKTEST_END} plus a recent "
               f"GDELT snapshot · Banking book is **synthetic** · No client or proprietary data")

if engine.backend == "baseline":
    reason = ("is set to the lightweight TF-IDF baseline on this deployment" if BACKEND == "baseline" else
              "is using the TF-IDF baseline because the fine-tuned transformer could not be downloaded. "
              "Run `python run.py fetch-model` and restart the dashboard to use it")
    st.info(f"Live scoring {reason}. The precomputed signal stream, backtests and reports below "
            "come from the transformer either way.")

with st.expander("Evaluate this in 60 seconds"):
    st.markdown(
        "1. **Risk Engine** – press *Score headlines*: six curated headlines are scored live.\n"
        "2. **Module B** – the default headline already passes all four trigger gates; see the portfolio "
        "loss, credit-loss build and CET1 drop it causes.\n"
        "3. **Module A** – compare the sentiment-tilted index with its equal-weight parent, and read the "
        "out-of-sample caveat.\n"
        "4. **Model Performance** – where the model is strong, where it is weak, and appropriate use.")
    st.caption(f"Model `{engine.model_version}` · backend `{engine.backend}` · data coverage: "
               + "; ".join(stream_scope(sig_df)))

tab_engine, tab_a, tab_b, tab_perf = st.tabs(
    ["Risk Engine", "Module A · Index Rebalancer", "Module B · Stress Testing", "Model Performance"]
)

# ==== Tab 1: engine ========================================================================
with tab_engine:
    st.subheader("Score headlines live")
    with st.form("score_form", border=False):
        text = st.text_area("One headline or post per line", "\n".join(DEMO_HEADLINES), height=170)
        submitted = st.form_submit_button("Score headlines", type="primary")
    if submitted:
        lines = parse_headlines(text)
        if not lines:
            st.warning("Enter at least one headline to score.")
            st.session_state.pop("adhoc", None)
        else:
            with st.spinner(f"Scoring {len(lines)} headline(s)..."):
                st.session_state["adhoc"] = engine.analyze_texts(lines, source="dashboard", source_type="news")
    if st.session_state.get("adhoc"):
        adhoc = st.session_state["adhoc"]
        st.caption(f"{len(adhoc)} signal(s) — a headline naming several index companies yields one row per "
                   "company. These are model predictions, not ground truth.")
        st.dataframe(signals_frame(adhoc), hide_index=True, width="stretch", column_config=SIGNAL_COLUMNS)

    st.subheader("Live global event feed (GDELT 2.0)")
    st.write("Pull the latest macro, geopolitical and credit headlines and score them.")
    if st.button("Fetch GDELT headlines"):
        adapter = GdeltAdapter(timeout_s=10.0)
        with st.spinner("Querying GDELT (rate-limited; falls back to the saved snapshot)..."):
            try:
                docs = adapter.load_documents(max_records=40)
                live_pull = adapter.last_fetch_was_live
            except Exception:
                docs, live_pull = adapter.load_snapshot(), False
            st.session_state["gdelt"] = (engine.analyze_documents(docs) if docs else [], live_pull,
                                         max((d.published_at for d in docs), default=None))
    if "gdelt" in st.session_state:
        gsig, live_pull, latest = st.session_state["gdelt"]
        if not gsig:
            st.warning("GDELT returned no articles and no saved snapshot is available.")
        else:
            when = f"{latest:%d %b %Y %H:%M} UTC" if latest is not None else "unknown time"
            st.caption(f"Live pull, latest article {when}" if live_pull else
                       f"GDELT unavailable right now: replaying the saved snapshot (latest article {when})")
            st.dataframe(signals_frame(gsig).sort_values("impact", ascending=False), hide_index=True,
                         width="stretch", column_config=SIGNAL_COLUMNS)

    st.divider()
    st.subheader("Signal stream explorer")
    if sig_df.empty:
        st.info("No signal stream yet. Run `python -m scripts.generate_signals`.")
    else:
        st.caption("Coverage: " + " · ".join(stream_scope(sig_df)) + ". The historical part drives the Module A "
                   "backtest; the GDELT rows are a recent live snapshot.")
        f1, f2 = st.columns(2)
        sources = f1.multiselect("Source", sorted(sig_df["source"].unique()), placeholder="All sources")
        events = f2.multiselect("Event type", list(EVENT_COLORS), placeholder="All event types")
        f3, f4 = st.columns(2)
        tick = f3.selectbox("Ticker", ["All"] + sorted(sig_df["ticker"].dropna().unique()))
        min_imp = f4.slider("Minimum impact", 1, 10, 1)

        view = sig_df[sig_df["impact_score"] >= min_imp]
        if sources:
            view = view[view["source"].isin(sources)]
        if events:
            view = view[view["event_type"].isin(events)]
        if tick != "All":
            view = view[view["ticker"] == tick]
        st.markdown(f"Showing **{len(view):,}** of {len(sig_df):,} signals")

        if view.empty:
            st.info("No signals match these filters. Clear a filter or lower the minimum impact.")
        else:
            c1, c2 = st.columns(2)
            mix = view.groupby(["source", "event_type"]).size().reset_index(name="signals")
            c1.plotly_chart(px.bar(mix, x="signals", y="source", color="event_type", orientation="h",
                                   color_discrete_map=EVENT_COLORS, title="Event mix by source"), width="stretch")
            c2.plotly_chart(px.histogram(view, x="impact_score", color="source", nbins=10, barmode="group",
                                         title="Impact score distribution"), width="stretch")

            if tick != "All":
                px_all = prices()[tick]
                daily = view.groupby("date")["sentiment_score"].mean()
                daily = daily[daily.index <= px_all.index.max()]
                if not daily.empty:
                    px_t = px_all.loc[daily.index.min():daily.index.max()]
                    fig = go.Figure()
                    fig.add_bar(x=daily.index, y=daily.values, name="daily mean sentiment (filtered)",
                                yaxis="y2", opacity=0.45)
                    fig.add_scatter(x=px_t.index, y=px_t.values, name=f"{tick} close", line=dict(color="black"))
                    fig.update_layout(title=f"{tick}: price vs. engine sentiment (current filters)",
                                      yaxis2=dict(overlaying="y", side="right", range=[-1, 1], title="sentiment"))
                    st.plotly_chart(fig, width="stretch")

            cols = ["timestamp", "source", "ticker", "sentiment_score", "event_type", "event_confidence",
                    "impact_score", "impact_confidence", "text_excerpt"]
            table = view.sort_values("timestamp", ascending=False)[cols]
            st.dataframe(table.head(500), hide_index=True, width="stretch",
                         column_config={"event_confidence": st.column_config.NumberColumn(format="percent"),
                                        "impact_confidence": st.column_config.NumberColumn(format="percent")})
            st.download_button(f"Download filtered signals ({len(table):,} rows, CSV)",
                               table.to_csv(index=False).encode("utf-8"), "filtered_signals.csv", "text/csv")

# ==== Tab 2: Module A ======================================================================
with tab_a:
    st.subheader("Tactical index rebalancing from real-time sentiment")
    st.write("A 20-stock mock index starts equal-weighted. Each session the engine's company-level "
             "`sentiment_score` (weighted by each headline's predicted `impact_score`) updates an EWMA "
             "state per stock; weights tilt as w ∝ exp(λ · sentiment) within per-name bounds and a daily "
             "turnover budget, with 5 bps trading costs. Weights set after a session's news only earn the "
             f"following session's return. Backtest {BACKTEST_START} to {BACKTEST_END}; parameters were "
             f"selected on the in-sample window before {OOS_START}.")

    chosen = load_json("reports/module_a_backtest.json").get("config", {})
    selected = {"tilt_lambda": float(chosen.get("tilt_lambda", 1.0)),
                "halflife_days": float(chosen.get("halflife_days", 5.0)),
                "max_weight": float(chosen.get("max_weight", 0.12)),
                "max_daily_turnover": float(chosen.get("max_daily_turnover", 0.10)),
                "impact_weighted": bool(chosen.get("impact_weighted", True))}
    for name, key in PARAM_KEYS.items():
        st.session_state.setdefault(key, selected[name])

    def reset_params():
        for n, k in PARAM_KEYS.items():
            st.session_state[k] = selected[n]

    with st.expander("Strategy parameters", expanded=False):
        p1, p2, p3 = st.columns(3)
        lam = p1.slider("Tilt strength λ", 0.5, 8.0, step=0.5, key=PARAM_KEYS["tilt_lambda"])
        hl = p2.slider("Sentiment half-life (days)", 1.0, 20.0, step=1.0, key=PARAM_KEYS["halflife_days"])
        wmax = p3.slider("Max weight per stock", 0.06, 0.25, step=0.01, key=PARAM_KEYS["max_weight"])
        p4, p5, p6 = st.columns(3)
        to = p4.slider("Max daily turnover", 0.02, 0.30, step=0.01, key=PARAM_KEYS["max_daily_turnover"])
        iw = p5.checkbox("Impact-weighted sentiment", key=PARAM_KEYS["impact_weighted"])
        p6.button("Reset to selected parameters", on_click=reset_params)
    current = {"tilt_lambda": lam, "halflife_days": hl, "max_weight": wmax, "max_daily_turnover": to,
               "impact_weighted": iw}
    if any(not np.isclose(float(current[k]), float(v)) for k, v in selected.items()):
        st.warning("Parameters differ from the in-sample selection, so the out-of-sample figures below are no "
                   "longer a clean test (they reflect choices made after seeing that period).")

    if sig_df.empty:
        st.info("Generate the signal stream first.")
    else:
        report, nav, weights, state = backtest(lam, hl, wmax, to, iw, mtime)
        oos = report["out_of_sample"]
        excess, ic_t = oos["active"]["excess_total_return"], oos["sentiment_ic"]["t_stat"]
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Excess return (OOS)", f"{excess:+.2%}",
                  help="Strategy total return minus the equal-weight parent, after costs, out-of-sample.")
        k2.metric("Information ratio (OOS)", f"{oos['active']['information_ratio']:.2f}",
                  help="Annualised excess return per unit of tracking error.")
        k3.metric("Sharpe: strategy (OOS)", f"{oos['strategy']['sharpe']:.2f}",
                  f"{oos['strategy']['sharpe'] - oos['equal_weight_benchmark']['sharpe']:+.2f} vs equal weight")
        k4.metric("Sentiment IC (OOS)", f"{oos['sentiment_ic']['mean']:.3f}", f"t = {ic_t:.2f}", delta_color="off",
                  help="Average daily rank correlation between sentiment state and next-day return.")
        k5.metric("Avg daily turnover (OOS)", f"{oos['avg_daily_turnover']:.2%}")

        if abs(ic_t) < 2:
            st.info(f"**Honest read:** the out-of-sample uplift ({excess:+.2%}) is small and the sentiment IC "
                    f"t-statistic ({ic_t:.2f}) is below 2, so it is not statistically significant over "
                    f"{oos['period']['days']} sessions. Treat this as a demonstration of a leakage-free, "
                    "cost-aware rebalancing method, not as evidence of alpha.")

        spy = prices()["SPY"].loc[nav.index]
        fig = go.Figure()
        fig.add_scatter(x=nav.index, y=nav["strategy"], name="Sentiment-tilted index", line=dict(width=2.5))
        fig.add_scatter(x=nav.index, y=nav["equal_weight"], name="Equal-weight parent", line=dict(dash="dash"))
        fig.add_scatter(x=nav.index, y=spy / spy.iloc[0], name="SPY (reference)", line=dict(color="gray", width=1))
        fig.add_vrect(x0=pd.Timestamp(OOS_START), x1=nav.index[-1], fillcolor="LightSkyBlue", opacity=0.15, line_width=0)
        fig.add_annotation(x=pd.Timestamp(OOS_START), y=1.02, yref="paper", text="out-of-sample →",
                           showarrow=False, xanchor="left")
        fig.update_layout(title="Growth of $1", yaxis_title="NAV", hovermode="x unified")
        st.plotly_chart(fig, width="stretch")

        summary = []
        for window in ("full_period", "out_of_sample"):
            for leg, label in (("strategy", "Sentiment-tilted"), ("equal_weight_benchmark", "Equal weight")):
                m = report[window][leg]
                summary.append({"window": "Full period" if window == "full_period" else "Out-of-sample",
                                "portfolio": label, "total return": f"{m['total_return']:+.2%}",
                                "volatility": f"{m['volatility']:.1%}", "Sharpe": f"{m['sharpe']:.2f}",
                                "max drawdown": f"{m['max_drawdown']:.1%}"})
        st.dataframe(pd.DataFrame(summary), hide_index=True, width="stretch")

        w = weights.copy()
        w.index.name = "date"
        long = w.reset_index().melt(id_vars="date", var_name="ticker", value_name="weight")
        st.plotly_chart(px.area(long, x="date", y="weight", color="ticker", title="Index weights over time"),
                        width="stretch")

        c1, c2 = st.columns([1, 2])
        names = list(weights.columns)
        pick = c1.selectbox("Inspect a stock", names, index=names.index("NVDA") if "NVDA" in names else 0)
        latest = pd.DataFrame({"weight": weights.iloc[-1], "sentiment state": state.iloc[-1]}).sort_values("weight")
        c1.dataframe(latest.style.format({"weight": "{:.2%}", "sentiment state": "{:+.3f}"}), width="stretch")
        fig = go.Figure()
        fig.add_scatter(x=weights.index, y=weights[pick], name="weight")
        fig.add_scatter(x=state.index, y=state[pick], name="sentiment state", yaxis="y2", line=dict(dash="dot"))
        fig.add_hline(y=1 / len(names), line_dash="dash", line_color="gray", annotation_text="equal weight")
        fig.update_layout(title=f"{pick}: weight follows sentiment",
                          yaxis=dict(tickformat=".0%"), yaxis2=dict(overlaying="y", side="right", title="sentiment"))
        c2.plotly_chart(fig, width="stretch")

        with st.expander("Full backtest report (technical, JSON)"):
            st.json(report)

# ==== Tab 3: Module B ======================================================================


def render_stress(result: dict, label: str, trigger_signal: dict = None):
    s = result["shock"]
    if label == STRESS_TRIGGERED:
        st.success(f"**{label}.** All trigger gates passed, so the matching scenario ran automatically.")
    elif label == STRESS_FORCED:
        st.warning(f"**{label}.** Shown for exploration only; the live engine would not have run it.")
    else:
        st.info(f"**{label}.**")
    st.caption(f"**Scenario:** {s['narrative']} Severity {s['severity']:.0%} → equity {s['equity']:+.1%}, "
               f"rates {s['rates_bp']:+.0f}bp, IG {s['ig_spread_bp']:+.0f}bp, HY {s['hy_spread_bp']:+.0f}bp, "
               f"USD {s['fx_usd']:+.1%}, commodities {s['commodity']:+.1%}")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Portfolio value", fmt_money(result["portfolio_value_after"]), fmt_money(result["total_pnl"], signed=True))
    k2.metric("Stressed P&L", f"{result['pnl_pct_of_value']:.2%}", help="Loss as a share of portfolio value.")
    k3.metric("Expected credit loss", fmt_money(result["ecl_after"]),
              fmt_money(result["ecl_after"] - result["ecl_before"], signed=True), delta_color="inverse",
              help="IFRS 9 ECL on the loan book after stressed PDs and staging.")
    k4.metric("CET1 ratio", f"{result['cet1_ratio_after']:.2%}",
              f"{(result['cet1_ratio_after'] - result['cet1_ratio_before']) * 100:+.2f} pp",
              help="Common Equity Tier 1 capital / risk-weighted assets.")
    k5.metric("Stage 2 loans", f"+{result['loans_moved_to_stage2']}",
              help="Loans moved to IFRS 9 Stage 2 because their PD at least doubled (significant increase "
                   "in credit risk).")
    if result["breaches_buffer"]:
        st.error(f"CET1 falls below the {result['cet1_minimum_with_buffer']:.0%} minimum-plus-buffer threshold.")

    c1, c2 = st.columns(2)
    classes = ["Loan", "Bond", "Derivative"]
    wf = go.Figure(go.Waterfall(
        x=["Value before"] + [f"{c}s" for c in classes] + ["Value after"],
        measure=["absolute"] + ["relative"] * 3 + ["total"],
        y=[result["portfolio_value_before"]] + [result["pnl_by_asset_class"].get(c, 0) for c in classes] + [0],
        connector={"line": {"color": "gray"}},
    ))
    lo = min(result["portfolio_value_before"], result["portfolio_value_after"])
    wf.update_layout(title="Portfolio value before → after stress",
                     yaxis=dict(range=[lo * 0.97, result["portfolio_value_before"] * 1.01]))
    c1.plotly_chart(wf, width="stretch")
    sec = pd.Series(result["pnl_by_sector"]).sort_values() / 1e6
    c2.plotly_chart(px.bar(x=sec.values, y=sec.index, orientation="h", title="Stressed P&L by sector ($m)",
                           labels={"x": "P&L ($m)", "y": ""}), width="stretch")

    cap = go.Figure()
    cap.add_bar(x=["Before", "After"], y=[result["cet1_ratio_before"], result["cet1_ratio_after"]],
                marker_color=["#1f77b4", "#d62728" if result["breaches_buffer"] else "#ff7f0e"])
    cap.add_hline(y=result["cet1_minimum_with_buffer"], line_dash="dash", annotation_text="4.5% min + 2.5% buffer")
    cap.update_layout(title="CET1 capital ratio", yaxis=dict(tickformat=".1%"), height=320)
    c1, c2 = st.columns([1, 2])
    c1.plotly_chart(cap, width="stretch")
    c2.write("Largest position losses")
    top = pd.DataFrame(result["top_losses"])
    if not top.empty:
        top = top.assign(notional=top["notional"] / 1e6, pnl=top["pnl"] / 1e6)
    c2.dataframe(top, hide_index=True, width="stretch",
                 column_config={"notional": st.column_config.NumberColumn("notional ($m)", format="%.1f"),
                                "pnl": st.column_config.NumberColumn("P&L ($m)", format="%.1f")})

    payload = result_for_download(result)
    payload["run_type"] = label
    if trigger_signal:
        payload["trigger_signal"] = trigger_signal
    st.download_button("Download this stress result (JSON)", json.dumps(payload, indent=2, default=str),
                       "stress_result.json", "application/json")


def stress_for_headline(headline: str, force: bool) -> dict:
    sig = engine.analyze_texts([headline], source="dashboard", source_type="event_feed")[0]
    triggered = stress.should_trigger(sig)
    return {"signal": sig, "triggered": triggered, "forced": force and not triggered,
            "result": scenario(sig.event_type, sig.impact_score) if (triggered or force) else None}


with tab_b:
    trig_cfg = stress.trigger
    st.subheader("Event-driven stress testing of a wholesale banking portfolio")
    st.write("Subscribes to `event_type` and `impact_score`. When a signal passes all four trigger gates below, "
             "the matching scenario runs, scaled by impact / 10, across loans (expected credit loss with IFRS 9 "
             "staging), bonds (duration and convexity) and derivatives (sensitivities). The portfolio is synthetic.")

    port = stress.portfolio
    with st.expander("Synthetic portfolio", expanded=False):
        c1, c2 = st.columns([2, 3])
        c1.plotly_chart(px.sunburst(port, path=["asset_class", "sector"], values="notional",
                                    title=f"{len(port)} positions · {fmt_money(port['notional'].sum())} notional"),
                        width="stretch")
        c2.dataframe(port, hide_index=True, width="stretch", height=380)

    mode = st.radio("Trigger source", ["Headline through the engine", "Historical triggered signal", "Manual scenario"],
                    horizontal=True)
    if mode == "Headline through the engine":
        with st.form("stress_form", border=False):
            h = st.text_input("Headline", STRESS_DEFAULT_HEADLINE)
            force = st.checkbox("Run the scenario even if the trigger rule is not met (what-if)", value=False)
            run = st.form_submit_button("Run stress test", type="primary")
        if run:
            if not h.strip():
                st.warning("Enter a headline to run the stress test.")
            else:
                with st.spinner("Scoring the headline and revaluing the portfolio..."):
                    st.session_state["stress_run"] = stress_for_headline(h.strip(), force)
        elif "stress_run" not in st.session_state:
            st.session_state["stress_run"] = stress_for_headline(STRESS_DEFAULT_HEADLINE, False)

        run_state = st.session_state["stress_run"]
        sig = run_state["signal"]
        st.caption(f"Scored headline: *{sig.text_excerpt}*")
        m1, m2, m3 = st.columns(3)
        m1.metric("event_type", sig.event_type, f"confidence {sig.event_confidence:.0%}", delta_color="off")
        m2.metric("impact_score", sig.impact_score, f"confidence {sig.impact_confidence:.0%}", delta_color="off")
        m3.metric("sentiment_score", f"{sig.sentiment_score:+.2f}")
        gates = trigger_gates(sig, trig_cfg)
        st.dataframe(pd.DataFrame([{**g, "passed": "Pass" if g["passed"] else "Fail"} for g in gates]),
                     hide_index=True, width="stretch", column_order=["gate", "passed", "signal", "rule"],
                     column_config={"gate": "Trigger gate", "rule": "Required", "signal": "This signal",
                                    "passed": "Result"})
        if run_state["result"] is None:
            st.info("**Not triggered.** Failed: " + "; ".join(gate_failures(gates)) +
                    ". Tick the what-if box to run the scenario anyway.")
        else:
            render_stress(run_state["result"], STRESS_TRIGGERED if run_state["triggered"] else STRESS_FORCED,
                          {"text_excerpt": sig.text_excerpt, "event_type": sig.event_type,
                           "event_confidence": sig.event_confidence, "impact_score": sig.impact_score,
                           "sentiment_score": sig.sentiment_score})

    elif mode == "Historical triggered signal":
        trig = load_csv("data/outputs/module_b_triggers.csv", parse_dates=["timestamp"])
        if trig.empty:
            st.info("No historical triggers. Run `python -m scripts.run_module_b`.")
        else:
            minimum = stress.capital_cfg["cet1_minimum_with_buffer"]
            breaches = cet1_breach_count(trig, minimum)
            trig = trig.assign(**{"CET1 buffer": np.where(trig["cet1_ratio_after"] < minimum, "breached", "held"),
                                  "P&L ($m)": trig["total_pnl"] / 1e6})
            st.markdown(f"Replaying the historical stream, the live engine would have triggered **{len(trig)}** "
                        f"stress tests; **{breaches}** of them push CET1 below the {minimum:.0%} "
                        "minimum-plus-buffer.")
            fig = px.scatter(trig, x="timestamp", y="P&L ($m)", color="event_type", size="impact_score",
                             symbol="CET1 buffer", symbol_map={"breached": "x", "held": "circle"},
                             hover_data=["ticker", "text_excerpt", "cet1_ratio_after"], color_discrete_map=EVENT_COLORS,
                             title="Historical stress tests (x = CET1 buffer breached)")
            st.plotly_chart(fig, width="stretch")
            labels = [f"{r.timestamp:%Y-%m-%d} · {r.event_type} · impact {r.impact_score} · {r.text_excerpt[:90]}"
                      for r in trig.itertuples()]
            pick = st.selectbox("Select a triggering signal", range(len(trig)), format_func=lambda i: labels[i],
                                index=int(trig["total_pnl"].values.argmin()))
            row = trig.iloc[pick]
            render_stress(scenario(row["event_type"], int(row["impact_score"])), "Historical trigger replay",
                          {"text_excerpt": row["text_excerpt"], "event_type": row["event_type"],
                           "impact_score": int(row["impact_score"]), "timestamp": str(row["timestamp"])})
    else:
        c1, c2 = st.columns(2)
        et = c1.selectbox("Event type", list(stress.scenarios))
        imp = c2.slider("Impact score", 1, 10, 9)
        render_stress(scenario(et, imp), "Manual what-if scenario")

# ==== Tab 4: model performance =============================================================
with tab_perf:
    if not ev:
        st.info("Run `python -m scripts.evaluate_engine` to populate this tab.")
    else:
        tf = ev["test"].get("transformer", {})
        st.subheader("Model card")
        st.caption(f"Held-out chronological test split (data after {ev.get('splits', {}).get('val_end', OOS_START)}), "
                   "never used for training or model selection. Labels come from market reactions and "
                   "weak supervision, not from humans.")
        good, weak = st.columns(2)
        with good.container(border=True):
            st.markdown("**Where it is strong**")
            if tf:
                st.markdown(f"- Sentiment correlates with the label at r = **{tf['sentiment']['pearson_r']:.2f}** "
                            f"(baseline {ev['test']['baseline']['sentiment']['pearson_r']:.2f}).\n"
                            f"- Event type macro-F1 **{tf['event_type']['macro_f1']:.2f}** across nine classes "
                            f"(baseline {ev['test']['baseline']['event_type']['macro_f1']:.2f}); the baseline "
                            "misses rare classes such as Credit Event entirely.")
        with weak.container(border=True):
            st.markdown("**Where it is weak**")
            if tf:
                bullets = [f"- Impact ranking is weak: Spearman **{tf['impact_score']['spearman_rho']:.2f}** overall"
                           + (f" and **{tf['impact_score_market']['spearman_rho']:.2f}** for market-wide news."
                              if "impact_score_market" in tf else ".")]
                cal_lo_hi = calibration_range(ev.get("impact_calibration", []))
                if cal_lo_hi:
                    bullets.append(f"- Calibration is flat: realised severity only moves from {cal_lo_hi[0]:.1f} "
                                   f"to {cal_lo_hi[1]:.1f} between the lowest and highest predicted bucket.")
                bullets.append("- Some tone errors on macro news (e.g. an inflation spike can read as positive).")
                st.markdown("\n".join(bullets))
        st.info("**Appropriate use:** triaging news flow and prioritising which stress scenarios to examine. Not "
                "for automated trading, credit approval or regulatory capital figures without human review.")

        card = model_card_rows(ev)
        st.dataframe(card, hide_index=True, width="stretch",
                     column_config={"TF-IDF baseline": st.column_config.NumberColumn(format="%.3f"),
                                    "transformer": st.column_config.NumberColumn(format="%.3f"),
                                    "test rows": st.column_config.NumberColumn(format="%d")})

        if tf:
            c1, c2 = st.columns(2)
            et = tf["event_type"]
            cm = np.array(et["confusion"])
            c1.plotly_chart(px.imshow(cm, x=et["labels"], y=et["labels"], text_auto=True, color_continuous_scale="Blues",
                                      labels=dict(x="predicted", y="label"), title="Event confusion matrix (transformer)"),
                            width="stretch")
            pc = pd.DataFrame(et["per_class"]).T.reset_index().rename(columns={"index": "class"})
            c2.plotly_chart(px.bar(pc, x="class", y="f1", hover_data=["support"], title="Per-class F1 (transformer)"),
                            width="stretch")
        if ev.get("impact_calibration"):
            cal = pd.DataFrame(ev["impact_calibration"])
            fig = px.line(cal, x="predicted_bucket", y="realised_mean", markers=True,
                          title="Impact calibration: realised severity by predicted bucket (test split)")
            fig.add_scatter(x=[1, 10], y=[1, 10], mode="lines", line=dict(dash="dash", color="gray"), name="ideal")
            st.plotly_chart(fig, width="stretch")
            st.caption("A well-calibrated model would follow the dashed line. The upward slope shows the ranking "
                       "carries some information, but the score should be read as relative priority, not as an "
                       "absolute severity forecast.")

        hand = hand_label_summary(ev)
        if hand is not None:
            st.subheader("Human-labelled evaluation set")
            if ev["hand_labelled"].get("status") == "partial":
                st.caption(f"Partial: {ev['hand_labelled']['labelled_rows']} rows labelled so far.")
            st.dataframe(hand, hide_index=True, width="stretch")
        else:
            st.caption("An independent human-labelled check of event types (268 headlines) is in progress; every "
                       "metric above uses the automatically labelled held-out split.")

        if ev.get("latency"):
            st.subheader("Inference latency (single headline)")
            lat = pd.DataFrame(ev["latency"]).T.rename_axis("backend / device").rename(columns={
                "single_p50_ms": "median (ms)", "single_p95_ms": "95th percentile (ms)",
                "batch_throughput_per_s": "batch throughput (headlines/s)"})
            st.dataframe(lat, width="stretch",
                         column_config={"batch throughput (headlines/s)": st.column_config.NumberColumn(format="%.0f")})
