"""
Module A: tactical index rebalancing driven by the engine's sentiment_score.

The mock index is the 20-name universe in config/universe.yaml. Each day the module:

  1. aggregates that day's company signals into one sentiment per ticker (optionally
     weighted by each headline's predicted impact_score),
  2. updates an exponentially-weighted sentiment state (no news decays toward neutral),
  3. tilts weights away from equal-weight: w_i ∝ (1/N) · exp(λ · s_i),
  4. enforces per-name bounds and a daily turnover budget, and
  5. holds those weights over the *next* session's returns (no lookahead).
"""
import os
from dataclasses import asdict, dataclass
from datetime import date
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from src.labeling.impact import MarketImpactEngine
from src.schemas import Signal

TRADING_DAYS = 252


@dataclass
class RebalancerConfig:
    tilt_lambda: float = 3.0          # strength of the sentiment tilt
    halflife_days: float = 5.0        # EWMA half-life of the sentiment state
    min_weight: float = 0.01
    max_weight: float = 0.12
    max_daily_turnover: float = 0.10  # one-way fraction of the book that may trade per day
    cost_bps: float = 5.0             # transaction cost per unit of one-way turnover
    min_signal_confidence: float = 0.0
    impact_weighted: bool = False     # weight each headline's sentiment by its predicted impact_score


def load_universe(path: str = "config/universe.yaml") -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [e["ticker"] for e in yaml.safe_load(f)["universe"]]


def load_close_prices(tickers: Iterable[str], prices_dir: str = "data/prices") -> pd.DataFrame:
    frames = {}
    for t in tickers:
        df = pd.read_csv(os.path.join(prices_dir, f"{t}.csv"), parse_dates=["Date"])
        frames[t] = df.set_index("Date")["Close"]
    return pd.DataFrame(frames).sort_index()


def project_to_bounds(w: np.ndarray, lo: float, hi: float, iters: int = 50) -> np.ndarray:
    """Clips weights into [lo, hi] while keeping them summing to one (iterative water-filling)."""
    w = w / w.sum()
    for _ in range(iters):
        clipped = np.clip(w, lo, hi)
        free = (clipped > lo) & (clipped < hi)
        excess = 1.0 - clipped.sum()
        if abs(excess) < 1e-12 or not free.any():
            return clipped / clipped.sum()
        clipped[free] += excess * clipped[free] / clipped[free].sum()
        w = clipped
    return w / w.sum()


def daily_sentiment_matrix(signals: Iterable[Signal], tickers: List[str], trading_days: pd.DatetimeIndex,
                           min_conf: float = 0.0, impact_weighted: bool = False) -> pd.DataFrame:
    """
    Sentiment per (session, ticker); a headline counts on the first session that can react to it.
    With impact weighting, headlines the engine expects to move the stock dominate routine chatter.
    """
    rows = []
    for s in signals:
        if s.ticker not in tickers or s.entity_type != "company" or s.event_confidence < min_conf:
            continue
        weight = float(s.impact_score) if impact_weighted else 1.0
        rows.append((pd.Timestamp(MarketImpactEngine.effective_event_date(s.timestamp)), s.ticker,
                     s.sentiment_score * weight, weight))
    if not rows:
        return pd.DataFrame(np.nan, index=trading_days, columns=tickers)
    df = pd.DataFrame(rows, columns=["date", "ticker", "weighted", "weight"])
    pos = trading_days.searchsorted(df["date"])
    df = df[pos < len(trading_days)]
    df["session"] = trading_days[pos[pos < len(trading_days)]]
    sums = df.groupby(["session", "ticker"])[["weighted", "weight"]].sum()
    mat = (sums["weighted"] / sums["weight"]).unstack("ticker")
    return mat.reindex(index=trading_days, columns=tickers)


class SentimentRebalancer:
    def __init__(self, tickers: List[str], config: RebalancerConfig = None):
        self.tickers = tickers
        self.cfg = config or RebalancerConfig()
        self.alpha = 1 - 0.5 ** (1 / self.cfg.halflife_days)

    def target_weights(self, sentiment_state: np.ndarray) -> np.ndarray:
        n = len(self.tickers)
        raw = np.full(n, 1.0 / n) * np.exp(self.cfg.tilt_lambda * sentiment_state)
        return project_to_bounds(raw, self.cfg.min_weight, self.cfg.max_weight)

    def step(self, prev_weights: np.ndarray, sentiment_state: np.ndarray):
        """One rebalance: returns (new_weights, one_way_turnover)."""
        target = self.target_weights(sentiment_state)
        delta = target - prev_weights
        turnover = 0.5 * np.abs(delta).sum()
        if turnover > self.cfg.max_daily_turnover:
            delta *= self.cfg.max_daily_turnover / turnover
            turnover = self.cfg.max_daily_turnover
        return prev_weights + delta, turnover

    def backtest(self, signals: List[Signal], prices: pd.DataFrame, start: str, end: str) -> Dict[str, pd.DataFrame]:
        px = prices.loc[start:end, self.tickers].dropna(how="any")
        rets = px.pct_change().fillna(0.0)
        days = px.index
        sent = daily_sentiment_matrix(signals, self.tickers, days, self.cfg.min_signal_confidence,
                                      self.cfg.impact_weighted)

        n = len(self.tickers)
        w = np.full(n, 1.0 / n)
        state = np.zeros(n)
        nav, bench_nav = [1.0], [1.0]
        weights, states, turnovers = [], [], []
        for i, day in enumerate(days):
            if i > 0:
                r = rets.iloc[i].values
                nav.append(nav[-1] * (1 + w @ r))
                bench_nav.append(bench_nav[-1] * (1 + r.mean()))
                w = w * (1 + r)
                w /= w.sum()
            obs = sent.iloc[i].values
            state = (1 - self.alpha) * state + self.alpha * np.nan_to_num(obs, nan=0.0)
            w, to = self.step(w, state)
            nav[-1] *= 1 - to * self.cfg.cost_bps / 1e4
            weights.append(w.copy())
            states.append(state.copy())
            turnovers.append(to)

        return {
            "nav": pd.DataFrame({"strategy": nav, "equal_weight": bench_nav, "turnover": turnovers}, index=days),
            "weights": pd.DataFrame(weights, index=days, columns=self.tickers),
            "sentiment_state": pd.DataFrame(states, index=days, columns=self.tickers),
            "daily_sentiment": sent,
            "returns": rets,
        }


def performance(nav: pd.Series) -> Dict[str, float]:
    r = nav.pct_change().dropna()
    years = max(len(r) / TRADING_DAYS, 1e-9)
    dd = nav / nav.cummax() - 1
    vol = r.std() * np.sqrt(TRADING_DAYS)
    return {
        "total_return": round(float(nav.iloc[-1] / nav.iloc[0] - 1), 4),
        "cagr": round(float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1), 4),
        "volatility": round(float(vol), 4),
        "sharpe": round(float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else 0.0, 3),
        "max_drawdown": round(float(dd.min()), 4),
    }


def summarize(result: Dict[str, pd.DataFrame], start: Optional[str] = None) -> Dict:
    nav = result["nav"].loc[start:] if start else result["nav"]
    strat, bench = nav["strategy"], nav["equal_weight"]
    active = strat.pct_change().dropna() - bench.pct_change().dropna()
    te = active.std() * np.sqrt(TRADING_DAYS)

    # Information coefficient: does today's sentiment state rank tomorrow's returns?
    st = result["sentiment_state"].loc[nav.index]
    fwd = result["returns"].shift(-1).loc[nav.index]
    ics = []
    for d in st.index[:-1]:
        s, f = st.loc[d], fwd.loc[d]
        if s.abs().sum() > 0 and s.nunique() > 2:
            ics.append(spearmanr(s, f)[0])
    ics = [x for x in ics if np.isfinite(x)]

    return {
        "period": {"start": str(nav.index[0].date()), "end": str(nav.index[-1].date()), "days": int(len(nav))},
        "strategy": performance(strat),
        "equal_weight_benchmark": performance(bench),
        "active": {
            "excess_total_return": round(float(strat.iloc[-1] / strat.iloc[0] - bench.iloc[-1] / bench.iloc[0]), 4),
            "tracking_error": round(float(te), 4),
            "information_ratio": round(float(active.mean() * TRADING_DAYS / te) if te > 0 else 0.0, 3),
            "daily_hit_rate": round(float((active > 0).mean()), 4),
        },
        "avg_daily_turnover": round(float(nav["turnover"].mean()), 4),
        "sentiment_ic": {"mean": round(float(np.mean(ics)), 4) if ics else None,
                         "t_stat": round(float(np.mean(ics) / (np.std(ics) / np.sqrt(len(ics)))), 3) if len(ics) > 2 else None,
                         "days": len(ics)},
    }


def run_backtest(signals: List[Signal], start: str, end: str, oos_start: str,
                 config: RebalancerConfig = None) -> Dict:
    tickers = load_universe()
    prices = load_close_prices(tickers)
    model = SentimentRebalancer(tickers, config)
    result = model.backtest(signals, prices, start, end)
    coverage = result["daily_sentiment"].notna().sum().to_dict()
    report = {
        "config": asdict(model.cfg),
        "full_period": summarize(result),
        "out_of_sample": summarize(result, oos_start),
        "signal_days_per_ticker": {k: int(v) for k, v in coverage.items()},
    }
    return {"report": report, **result}
