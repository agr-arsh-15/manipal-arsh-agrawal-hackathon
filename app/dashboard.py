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

from src.engine.pipeline import RiskEngine  # noqa: E402
from src.engine.store import SignalStore  # noqa: E402
from src.ingestion.gdelt import GdeltAdapter  # noqa: E402
from src.modules.rebalancer import RebalancerConfig, load_close_prices, load_universe, run_backtest  # noqa: E402
from src.modules.stress import StressTestEngine  # noqa: E402

SIGNALS = "data/samples/signals.jsonl"
OOS_START = "2019-11-01"
EVENT_COLORS = {
    "Geopolitical": "#d62728", "Macroeconomic": "#ff7f0e", "Credit Event": "#8c564b",
    "Merger/Acquisition": "#9467bd", "Product Launch": "#2ca02c", "Regulatory/Legal": "#e377c2",
    "Earnings/Guidance": "#1f77b4", "Operational/Supply-chain": "#bcbd22", "Other/None": "#7f7f7f",
}
EXAMPLES = (
    "Russia launches invasion as Western allies prepare sweeping sanctions\n"
    "Federal Reserve signals 75 basis point rate hike to fight surging inflation\n"
    "Moody's downgrades Boeing to junk as cash burn accelerates\n"
    "Apple shares hit record after iPhone revenue beats analyst estimates\n"
    "$TSLA recalls 360,000 vehicles over self-driving software defect"
)

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


@st.cache_data(show_spinner="Loading signal stream...")
def load_signals(mtime: float) -> pd.DataFrame:
    rows = [s.model_dump() for s in SignalStore(SIGNALS).read()]
    df = pd.DataFrame(rows)
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


def signals_frame(signals) -> pd.DataFrame:
    return pd.DataFrame([{
        "ticker": s.ticker or "-", "sentiment": s.sentiment_score, "label": s.sentiment_label,
        "event_type": s.event_type, "event_conf": s.event_confidence, "impact": s.impact_score,
        "impact_conf": s.impact_confidence, "text": s.text_excerpt,
    } for s in signals])


def money(x: float) -> str:
    return f"${x / 1e6:,.1f}m"


engine = get_engine()
sig_df = load_signals(os.path.getmtime(SIGNALS) if os.path.exists(SIGNALS) else 0.0)

st.title("AI/NLP Financial Risk Engine")
st.caption(f"Backend: **{engine.backend}** · model `{engine.model_version}` · "
           f"{len(sig_df):,} signals in stream · sources: news, social, global event feed")
if engine.backend == "baseline":
    reason = ("is set to the lightweight TF-IDF baseline on this deployment" if BACKEND == "baseline" else
              "is using the TF-IDF baseline because the fine-tuned transformer could not be downloaded. "
              "Run `python run.py fetch-model` and restart the dashboard to use it")
    st.info(f"Live scoring {reason}. The precomputed signal stream, backtests and reports below "
            "come from the transformer either way.")

tab_engine, tab_a, tab_b, tab_perf = st.tabs(
    ["Risk Engine", "Module A · Index Rebalancer", "Module B · Stress Testing", "Model Performance"]
)

# ==== Tab 1: engine ========================================================================
with tab_engine:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Analyse text")
        text = st.text_area("One headline or post per line", EXAMPLES, height=150)
        source_type = st.radio("Source type", ["news", "social", "event_feed"], horizontal=True)
        if st.button("Run engine", type="primary"):
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            st.session_state["adhoc"] = engine.analyze_texts(lines, source="dashboard", source_type=source_type)
    with right:
        st.subheader("Live global event feed")
        st.write("Pull the latest macro, geopolitical and credit headlines from GDELT 2.0.")
        if st.button("Fetch GDELT headlines"):
            with st.spinner("Querying GDELT (rate-limited to one call per 5s)..."):
                adapter = GdeltAdapter()
                docs = adapter.load_documents(max_records=40)
                st.session_state["gdelt"] = (engine.analyze_documents(docs), adapter.last_fetch_was_live)

    if "adhoc" in st.session_state:
        df = signals_frame(st.session_state["adhoc"])
        st.dataframe(
            df, hide_index=True, width="stretch",
            column_config={
                "sentiment": st.column_config.ProgressColumn("sentiment", min_value=-1, max_value=1, format="%.2f"),
                "impact": st.column_config.ProgressColumn("impact (1-10)", min_value=1, max_value=10, format="%d"),
                "event_conf": st.column_config.NumberColumn(format="%.2f"),
                "impact_conf": st.column_config.NumberColumn(format="%.2f"),
            },
        )
    if "gdelt" in st.session_state:
        gsig, live = st.session_state["gdelt"]
        if not gsig:
            st.warning("GDELT returned no articles (rate limit or offline) and no snapshot is available yet.")
        else:
            st.caption("Live pull" if live else "GDELT unavailable: replaying the last saved snapshot")
            st.dataframe(signals_frame(gsig).sort_values("impact", ascending=False), hide_index=True, width="stretch")

    st.divider()
    st.subheader("Signal stream explorer")
    if sig_df.empty:
        st.info("No signal stream yet. Run `python -m scripts.generate_signals`.")
    else:
        f1, f2, f3, f4 = st.columns(4)
        sources = f1.multiselect("Source", sorted(sig_df["source"].unique()), default=sorted(sig_df["source"].unique()))
        events = f2.multiselect("Event type", list(EVENT_COLORS), default=[e for e in EVENT_COLORS if e != "Other/None"])
        tick = f3.selectbox("Ticker", ["All"] + sorted(sig_df["ticker"].dropna().unique()))
        min_imp = f4.slider("Minimum impact", 1, 10, 1)
        view = sig_df[sig_df["source"].isin(sources) & sig_df["event_type"].isin(events) & (sig_df["impact_score"] >= min_imp)]
        if tick != "All":
            view = view[view["ticker"] == tick]

        c1, c2 = st.columns(2)
        mix = view.groupby(["source", "event_type"]).size().reset_index(name="signals")
        c1.plotly_chart(px.bar(mix, x="signals", y="source", color="event_type", orientation="h",
                               color_discrete_map=EVENT_COLORS, title="Event mix by source"), width="stretch")
        c2.plotly_chart(px.histogram(view, x="impact_score", color="source", nbins=10, barmode="group",
                                     title="Impact score distribution"), width="stretch")

        if tick != "All":
            daily = sig_df[sig_df["ticker"] == tick].groupby("date")["sentiment_score"].mean()
            px_t = prices()[tick].loc[daily.index.min():daily.index.max()]
            fig = go.Figure()
            fig.add_bar(x=daily.index, y=daily.values, name="daily mean sentiment", yaxis="y2", opacity=0.45)
            fig.add_scatter(x=px_t.index, y=px_t.values, name=f"{tick} close", line=dict(color="black"))
            fig.update_layout(title=f"{tick}: price vs. engine sentiment",
                              yaxis2=dict(overlaying="y", side="right", range=[-1, 1], title="sentiment"))
            st.plotly_chart(fig, width="stretch")

        st.dataframe(
            view.sort_values("timestamp", ascending=False)[
                ["timestamp", "source", "ticker", "sentiment_score", "event_type", "event_confidence",
                 "impact_score", "impact_confidence", "text_excerpt"]].head(500),
            hide_index=True, width="stretch",
        )

# ==== Tab 2: Module A ======================================================================
with tab_a:
    st.subheader("Tactical index rebalancing from real-time sentiment")
    st.write("A 20-stock mock index starts equal-weighted. Each session the engine's company-level "
             "`sentiment_score` (weighted by each headline's predicted `impact_score`) updates an EWMA "
             "state per stock; weights tilt as w ∝ exp(λ · sentiment) within per-name bounds and a daily "
             "turnover budget. Weights set after a session's news only earn the following session's return. "
             "Defaults are the parameters selected on the in-sample window (before 2019-11-01).")

    chosen = load_json("reports/module_a_backtest.json").get("config", {})
    with st.expander("Strategy parameters", expanded=False):
        p1, p2, p3, p4, p5 = st.columns(5)
        lam = p1.slider("Tilt strength λ", 0.5, 8.0, float(chosen.get("tilt_lambda", 3.0)), 0.5)
        hl = p2.slider("Sentiment half-life (days)", 1.0, 20.0, float(chosen.get("halflife_days", 5.0)), 1.0)
        wmax = p3.slider("Max weight per stock", 0.06, 0.25, float(chosen.get("max_weight", 0.12)), 0.01)
        to = p4.slider("Max daily turnover", 0.02, 0.30, float(chosen.get("max_daily_turnover", 0.10)), 0.01)
        iw = p5.checkbox("Impact-weighted sentiment", value=bool(chosen.get("impact_weighted", True)))

    @st.cache_data(show_spinner="Backtesting...")
    def backtest(lam, hl, wmax, to, iw, mtime):
        signals = SignalStore(SIGNALS).read()
        out = run_backtest(signals, "2018-04-02", "2020-06-10", OOS_START,
                           RebalancerConfig(tilt_lambda=lam, halflife_days=hl, max_weight=wmax,
                                            max_daily_turnover=to, impact_weighted=iw))
        return out["report"], out["nav"], out["weights"], out["sentiment_state"]

    if sig_df.empty:
        st.info("Generate the signal stream first.")
    else:
        report, nav, weights, state = backtest(lam, hl, wmax, to, iw, os.path.getmtime(SIGNALS))
        oos = report["out_of_sample"]
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Excess return (out-of-sample)", f"{oos['active']['excess_total_return']:+.2%}")
        k2.metric("Information ratio (OOS)", f"{oos['active']['information_ratio']:.2f}")
        k3.metric("Sharpe: strategy vs EW (OOS)", f"{oos['strategy']['sharpe']:.2f}",
                  f"{oos['strategy']['sharpe'] - oos['equal_weight_benchmark']['sharpe']:+.2f}")
        k4.metric("Sentiment IC (OOS)", f"{oos['sentiment_ic']['mean']:.3f}", f"t = {oos['sentiment_ic']['t_stat']:.2f}",
                  delta_color="off")
        k5.metric("Avg daily turnover (OOS)", f"{oos['avg_daily_turnover']:.2%}")

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

        w = weights.copy()
        w.index.name = "date"
        long = w.reset_index().melt(id_vars="date", var_name="ticker", value_name="weight")
        st.plotly_chart(px.area(long, x="date", y="weight", color="ticker", title="Index weights over time"),
                        width="stretch")

        c1, c2 = st.columns([1, 2])
        pick = c1.selectbox("Inspect a stock", list(weights.columns), index=list(weights.columns).index("NVDA"))
        latest = pd.DataFrame({"weight": weights.iloc[-1], "sentiment state": state.iloc[-1]}).sort_values("weight")
        c1.dataframe(latest.style.format({"weight": "{:.2%}", "sentiment state": "{:+.3f}"}), width="stretch")
        fig = go.Figure()
        fig.add_scatter(x=weights.index, y=weights[pick], name="weight")
        fig.add_scatter(x=state.index, y=state[pick], name="sentiment state", yaxis="y2", line=dict(dash="dot"))
        fig.add_hline(y=1 / len(weights.columns), line_dash="dash", line_color="gray", annotation_text="equal weight")
        fig.update_layout(title=f"{pick}: weight follows sentiment",
                          yaxis=dict(tickformat=".0%"), yaxis2=dict(overlaying="y", side="right", title="sentiment"))
        c2.plotly_chart(fig, width="stretch")

        with st.expander("Full backtest report"):
            st.json(report)

# ==== Tab 3: Module B ======================================================================
with tab_b:
    stress = get_stress()
    st.subheader("Event-driven stress testing of a wholesale banking portfolio")
    st.write(f"Subscribes to `event_type` and `impact_score`. A signal with impact ≥ "
             f"{stress.trigger['min_impact']} in {', '.join(stress.trigger['event_types'])}, with adverse sentiment "
             f"and event confidence ≥ {stress.trigger.get('min_event_confidence', 0):.0%}, triggers the matching "
             "scenario, scaled by impact / 10, across loans (ECL with IFRS 9 staging), bonds (duration + convexity) "
             "and derivatives (sensitivities).")

    port = stress.portfolio
    with st.expander("Synthetic portfolio", expanded=False):
        c1, c2 = st.columns([2, 3])
        c1.plotly_chart(px.sunburst(port, path=["asset_class", "sector"], values="notional",
                                    title=f"{len(port)} positions · {money(port['notional'].sum())} notional"),
                        width="stretch")
        c2.dataframe(port, hide_index=True, width="stretch", height=380)

    mode = st.radio("Trigger source", ["Headline through the engine", "Historical triggered signal", "Manual scenario"],
                    horizontal=True)
    result, trigger_info = None, None
    if mode == "Headline through the engine":
        h = st.text_input("Headline", "Russia launches full-scale invasion; US and EU impose sweeping sanctions")
        force = st.checkbox("Run even if the trigger rule is not met", value=False)
        if h:
            sig = engine.analyze_texts([h], source="dashboard", source_type="event_feed")[0]
            trigger_info = sig
            result = stress.run_for_signal(sig, force=force)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("event_type", sig.event_type, f"conf {sig.event_confidence:.0%}", delta_color="off")
            m2.metric("impact_score", sig.impact_score, f"conf {sig.impact_confidence:.0%}", delta_color="off")
            m3.metric("sentiment_score", f"{sig.sentiment_score:+.2f}")
            m4.metric("Stress test", "TRIGGERED" if stress.should_trigger(sig) else "not triggered")
            if result is None:
                st.info("Signal below the trigger rule. Tick the box to run the scenario anyway.")
    elif mode == "Historical triggered signal":
        trig = load_csv("data/outputs/module_b_triggers.csv", parse_dates=["timestamp"])
        if trig.empty:
            st.info("No historical triggers. Run `python -m scripts.run_module_b`.")
        else:
            fig = px.scatter(trig, x="timestamp", y="total_pnl", color="event_type", size="impact_score",
                             hover_data=["ticker", "text_excerpt"], color_discrete_map=EVENT_COLORS,
                             title=f"{len(trig)} stress tests the live engine would have triggered")
            fig.update_yaxes(title="stressed P&L ($)")
            st.plotly_chart(fig, width="stretch")
            labels = [f"{r.timestamp:%Y-%m-%d} · {r.event_type} · impact {r.impact_score} · {r.text_excerpt[:90]}"
                      for r in trig.itertuples()]
            pick = st.selectbox("Select a triggering signal", range(len(trig)), format_func=lambda i: labels[i],
                                index=int(trig["total_pnl"].values.argmin()))
            row = trig.iloc[pick]
            result = stress.run(stress.shock_for(row["event_type"], int(row["impact_score"])))
    else:
        c1, c2 = st.columns(2)
        et = c1.selectbox("Event type", list(stress.scenarios))
        imp = c2.slider("Impact score", 1, 10, 9)
        result = stress.run(stress.shock_for(et, imp))

    if result:
        s = result["shock"]
        st.caption(f"**Scenario:** {s['narrative']} Severity {s['severity']:.0%} → equity {s['equity']:+.1%}, "
                   f"rates {s['rates_bp']:+.0f}bp, IG {s['ig_spread_bp']:+.0f}bp, HY {s['hy_spread_bp']:+.0f}bp, "
                   f"USD {s['fx_usd']:+.1%}, commodities {s['commodity']:+.1%}")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Portfolio value", money(result["portfolio_value_after"]), money(result["total_pnl"]))
        k2.metric("Stressed P&L", f"{result['pnl_pct_of_value']:.2%}")
        k3.metric("Expected credit loss", money(result["ecl_after"]),
                  money(result["ecl_after"] - result["ecl_before"]), delta_color="inverse")
        k4.metric("CET1 ratio", f"{result['cet1_ratio_after']:.2%}",
                  f"{(result['cet1_ratio_after'] - result['cet1_ratio_before']) * 100:+.2f} pp")
        k5.metric("Loans moved to Stage 2", result["loans_moved_to_stage2"])
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
        wf.update_layout(title="Portfolio value before → after stress", yaxis=dict(range=[lo * 0.97, result["portfolio_value_before"] * 1.01]))
        c1.plotly_chart(wf, width="stretch")
        sec = pd.Series(result["pnl_by_sector"]).sort_values()
        c2.plotly_chart(px.bar(x=sec.values, y=sec.index, orientation="h", title="Stressed P&L by sector",
                               labels={"x": "P&L ($)", "y": ""}), width="stretch")

        cap = go.Figure()
        cap.add_bar(x=["Before", "After"], y=[result["cet1_ratio_before"], result["cet1_ratio_after"]],
                    marker_color=["#1f77b4", "#d62728" if result["breaches_buffer"] else "#ff7f0e"])
        cap.add_hline(y=result["cet1_minimum_with_buffer"], line_dash="dash", annotation_text="4.5% min + 2.5% buffer")
        cap.update_layout(title="CET1 capital ratio", yaxis=dict(tickformat=".1%"), height=320)
        c1, c2 = st.columns([1, 2])
        c1.plotly_chart(cap, width="stretch")
        c2.write("Largest position losses")
        c2.dataframe(pd.DataFrame(result["top_losses"]), hide_index=True, width="stretch")

# ==== Tab 4: model performance =============================================================
with tab_perf:
    ev = load_json("reports/engine_eval.json")
    if not ev:
        st.info("Run `python -m scripts.evaluate_engine` to populate this tab.")
    else:
        st.subheader("Transformer vs. TF-IDF baseline on the held-out chronological test split")
        rows = []
        for task, metrics in [("sentiment", ["pearson_r", "mae", "macro_f1"]),
                              ("event_type", ["macro_f1", "weighted_f1", "accuracy"]),
                              ("impact_score", ["spearman_rho", "mae", "high_impact_precision", "high_impact_recall"]),
                              ("impact_score_company", ["spearman_rho", "high_impact_precision"]),
                              ("impact_score_market", ["spearman_rho", "high_impact_precision"])]:
            if task not in ev["test"]["baseline"]:
                continue
            for m in metrics:
                rows.append({"task": task, "metric": m,
                             "baseline": ev["test"]["baseline"][task].get(m),
                             "transformer": ev["test"].get("transformer", {}).get(task, {}).get(m)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        if "transformer" in ev["test"]:
            c1, c2 = st.columns(2)
            et = ev["test"]["transformer"]["event_type"]
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
        if ev.get("hand_labelled"):
            st.subheader("Human-labelled evaluation set")
            st.json(ev["hand_labelled"])
        if ev.get("latency"):
            st.subheader("Inference latency")
            st.dataframe(pd.DataFrame(ev["latency"]).T.rename_axis("backend / device"), width="stretch")
