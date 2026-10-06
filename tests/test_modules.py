from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.modules.rebalancer import RebalancerConfig, SentimentRebalancer, project_to_bounds
from src.modules.stress import StressTestEngine
from src.schemas import Signal


def make_signal(ticker, sentiment, ts, event_type="Other/None", impact=5):
    return Signal(
        signal_id=f"{ticker}-{ts.isoformat()}", doc_id="d", timestamp=ts, source="test", source_type="news",
        entity_type="company" if ticker else "event", ticker=ticker, sentiment_score=sentiment,
        sentiment_label="positive" if sentiment > 0 else "negative", event_type=event_type,
        event_confidence=0.9, impact_score=impact, impact_confidence=0.5, text_excerpt="x", model_version="test",
    )


# ---- Module A ----------------------------------------------------------------------------

def test_projection_respects_bounds_and_budget():
    w = project_to_bounds(np.array([10.0, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 0.02, 0.2)
    assert abs(w.sum() - 1) < 1e-9
    assert w.max() <= 0.2 + 1e-9 and w.min() >= 0.02 - 1e-9


def test_positive_sentiment_overweights_and_turnover_is_capped():
    tickers = ["A", "B", "C", "D"]
    model = SentimentRebalancer(tickers, RebalancerConfig(max_weight=0.6, max_daily_turnover=0.05))
    w0 = np.full(4, 0.25)
    w1, turnover = model.step(w0, np.array([0.9, 0.0, 0.0, -0.9]))
    assert w1[0] > 0.25 > w1[3]
    assert turnover <= 0.05 + 1e-12
    assert abs(w1.sum() - 1) < 1e-9


def test_backtest_has_no_lookahead():
    """Weights set on day t must only earn day t+1's return, never day t's."""
    days = pd.bdate_range("2020-01-01", periods=6)
    prices = pd.DataFrame({"A": [100, 100, 150, 150, 150, 150], "B": [100] * 6}, index=days, dtype=float)
    # Bullish news on A published after the close of day 1, i.e. after the jump is already known
    # to arrive on day 2: the strategy must not capture the day-2 move.
    ts = datetime(days[1].year, days[1].month, days[1].day, 21, 0, tzinfo=timezone.utc)
    model = SentimentRebalancer(["A", "B"], RebalancerConfig(max_weight=0.99, min_weight=0.01,
                                                             max_daily_turnover=1.0, cost_bps=0))
    out = model.backtest([make_signal("A", 1.0, ts)], prices, str(days[0].date()), str(days[-1].date()))
    nav = out["nav"]
    assert nav["strategy"].iloc[2] == pytest.approx(nav["equal_weight"].iloc[2])
    assert out["weights"]["A"].iloc[2] > 0.5


# ---- Module B ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stress():
    return StressTestEngine()


def test_trigger_rule(stress):
    ts = datetime(2020, 3, 1, tzinfo=timezone.utc)
    assert stress.should_trigger(make_signal(None, -0.8, ts, "Geopolitical", 9))
    assert not stress.should_trigger(make_signal(None, -0.8, ts, "Geopolitical", 7))
    assert not stress.should_trigger(make_signal(None, -0.8, ts, "Product Launch", 10))
    assert not stress.should_trigger(make_signal(None, 0.6, ts, "Macroeconomic", 9))  # benign news


def test_scan_runs_one_scenario_per_event_day(stress):
    ts = datetime(2020, 3, 9, 14, tzinfo=timezone.utc)
    same_day = [make_signal(None, -0.7, ts, "Macroeconomic", 8), make_signal(None, -0.9, ts, "Macroeconomic", 10),
                make_signal(None, -0.5, ts, "Geopolitical", 9)]
    out = stress.scan(same_day)
    assert len(out) == 2
    assert out.loc[out["event_type"] == "Macroeconomic", "impact_score"].item() == 10


def test_impact_weighting_lets_high_impact_news_dominate():
    from src.modules.rebalancer import daily_sentiment_matrix
    ts = datetime(2020, 3, 2, 14, tzinfo=timezone.utc)
    sigs = [make_signal("AAPL", 0.8, ts, impact=9), make_signal("AAPL", -0.8, ts, impact=1)]
    days = pd.DatetimeIndex(["2020-03-02"])
    plain = daily_sentiment_matrix(sigs, ["AAPL"], days).iloc[0, 0]
    weighted = daily_sentiment_matrix(sigs, ["AAPL"], days, impact_weighted=True).iloc[0, 0]
    assert abs(plain) < 1e-9 and weighted > 0.6


def test_shock_scales_with_impact(stress):
    mild, severe = stress.shock_for("Macroeconomic", 5), stress.shock_for("Macroeconomic", 10)
    assert severe.rates_bp == pytest.approx(200) and mild.rates_bp == pytest.approx(100)
    assert stress.run(severe)["total_pnl"] < stress.run(mild)["total_pnl"] < 0


def test_rate_shock_revaluation_signs(stress):
    r = stress.run(stress.shock_for("Macroeconomic", 10))
    pos = r["positions"].set_index("position_id")
    portfolio = stress.portfolio.set_index("position_id")
    payer = portfolio.index[portfolio["instrument"].str.contains("pay fixed")]
    receiver = portfolio.index[portfolio["instrument"].str.contains("receive fixed")]
    assert (pos.loc[payer, "pnl"] > 0).all() and (pos.loc[receiver, "pnl"] < 0).all()
    bonds = portfolio.index[portfolio["asset_class"] == "Bond"]
    assert (pos.loc[bonds, "pnl"] < 0).all()
    assert r["ecl_after"] > r["ecl_before"]
    assert r["cet1_ratio_after"] < r["cet1_ratio_before"]


def test_neutral_scenario_is_flat(stress):
    r = stress.run(stress.shock_for("Other/None", 10))
    assert r["total_pnl"] == pytest.approx(0, abs=1)
    assert r["ecl_after"] == pytest.approx(r["ecl_before"])
