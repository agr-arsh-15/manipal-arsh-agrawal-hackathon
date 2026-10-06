"""
Module B: event-driven stress testing of a synthetic wholesale-banking portfolio.

Subscribes to `event_type` and `impact_score`. When a signal crosses the trigger (default:
impact_score > 7 for a configured event type, adverse sentiment and a confident event
classification) the matching scenario from
config/stress_scenarios.yaml is scaled by severity = impact_score / 10 and applied:

  Bonds        dP = MV * (-D_mod * dy + 0.5 * C * dy^2), dy = rates + spread shock by bucket
  Loans        IFRS 9-style ECL = PD * LGD * EAD; stressed PD via sector multipliers and a
               floating-rate debt-service add-on; a >= 2x PD rise (SICR) moves the loan to
               Stage 2 and lifetime PD
  Derivatives  first/second-order sensitivities: DV01 x dRates, delta/gamma x dEquity,
               delta x dFX / dCommodity, CDS spread DV01 x dSpread
  Capital      CET1 ratio before/after, with rating migration feeding risk-weighted assets
"""
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yaml

from src.schemas import Signal

PD_BY_RATING = {"AAA": 0.0001, "AA": 0.0002, "A": 0.0006, "BBB": 0.0020, "BB": 0.0090, "B": 0.035, "CCC": 0.15}
RISK_WEIGHT = {"AAA": 0.2, "AA": 0.2, "A": 0.5, "BBB": 1.0, "BB": 1.0, "B": 1.5, "CCC": 1.5}
SICR_MULTIPLE = 2.0                 # relative PD increase treated as a significant increase in credit risk
FLOATING_PD_PER_100BP = 0.15        # PD uplift per +100bp for floating-rate borrowers


@dataclass
class Shock:
    event_type: str
    severity: float
    narrative: str
    equity: float = 0.0
    rates_bp: float = 0.0
    ig_spread_bp: float = 0.0
    hy_spread_bp: float = 0.0
    fx_usd: float = 0.0
    commodity: float = 0.0
    pd_multiplier: Dict[str, float] = field(default_factory=lambda: {"all": 1.0})

    def pd_mult(self, sector: str) -> float:
        return self.pd_multiplier.get(sector, self.pd_multiplier.get("all", 1.0))


def rating_for_pd(pd_value: float) -> str:
    """Nearest rating bucket in log-PD space (used for stressed rating migration)."""
    names = list(PD_BY_RATING)
    logs = np.log([PD_BY_RATING[n] for n in names])
    return names[int(np.argmin(np.abs(logs - np.log(max(pd_value, 1e-6)))))]


class StressTestEngine:
    def __init__(self, portfolio_path: str = "data/portfolio/synthetic_portfolio.csv",
                 scenarios_path: str = "config/stress_scenarios.yaml"):
        self.portfolio = pd.read_csv(portfolio_path)
        with open(scenarios_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.scenarios = cfg["scenarios"]
        self.trigger = cfg["trigger"]
        self.capital_cfg = cfg["capital"]

    # ---- subscription -------------------------------------------------------------------
    def should_trigger(self, signal: Signal) -> bool:
        return (signal.impact_score >= self.trigger["min_impact"]
                and signal.event_type in self.trigger["event_types"]
                and signal.sentiment_score < self.trigger.get("max_sentiment", 1.0)
                and signal.event_confidence >= self.trigger.get("min_event_confidence", 0.0))

    def shock_for(self, event_type: str, impact_score: float) -> Shock:
        spec = self.scenarios.get(event_type, self.scenarios["Other/None"])
        sev = float(np.clip(impact_score / 10.0, 0.0, 1.0))
        return Shock(
            event_type=event_type, severity=sev, narrative=spec["narrative"],
            equity=spec["equity"] * sev, rates_bp=spec["rates_bp"] * sev,
            ig_spread_bp=spec["ig_spread_bp"] * sev, hy_spread_bp=spec["hy_spread_bp"] * sev,
            fx_usd=spec["fx_usd"] * sev, commodity=spec["commodity"] * sev,
            pd_multiplier={k: 1.0 + (v - 1.0) * sev for k, v in spec["pd_multiplier"].items()},
        )

    # ---- revaluation --------------------------------------------------------------------
    def _bond_pnl(self, p: pd.DataFrame, s: Shock) -> np.ndarray:
        spread = np.select([p["spread_bucket"] == "IG", p["spread_bucket"] == "HY"],
                           [s.ig_spread_bp, s.hy_spread_bp], 0.0)
        dy = (s.rates_bp + spread) / 1e4
        return p["market_value"].values * (-p["modified_duration"].values * dy + 0.5 * p["convexity"].values * dy ** 2)

    def _loan_ecl(self, p: pd.DataFrame, s: Optional[Shock]) -> pd.DataFrame:
        base_pd = p["pd_1y"].values
        if s is None:
            pd_s = base_pd
        else:
            mult = np.array([s.pd_mult(sec) for sec in p["sector"]])
            rate_addon = np.where(p["rate_type"] == "floating", 1 + FLOATING_PD_PER_100BP * max(s.rates_bp, 0) / 100, 1.0)
            pd_s = np.minimum(base_pd * mult * rate_addon, 0.99)
        stage2 = pd_s >= SICR_MULTIPLE * base_pd
        lifetime_pd = 1 - (1 - pd_s) ** np.maximum(p["maturity_years"].values, 1.0)
        eff_pd = np.where(stage2, lifetime_pd, pd_s)
        ecl = eff_pd * p["lgd"].values * p["notional"].values
        ratings = [rating_for_pd(x) for x in pd_s]
        rwa = p["notional"].values * np.array([RISK_WEIGHT[r] for r in ratings])
        return pd.DataFrame({"pd": pd_s, "stage": np.where(stage2, 2, 1), "ecl": ecl,
                             "rating": ratings, "rwa": rwa}, index=p.index)

    def _derivative_pnl(self, p: pd.DataFrame, s: Shock) -> np.ndarray:
        n, d, g, dv01 = p["notional"].values, p["delta"].values, p["gamma"].values, p["dv01_usd"].values
        f = p["risk_factor"].values
        return np.select(
            [f == "rates", f == "equity", f == "fx", f == "commodity", f == "credit_spread"],
            [dv01 * s.rates_bp, n * (d * s.equity + 0.5 * g * s.equity ** 2), n * d * s.fx_usd,
             n * d * s.commodity, dv01 * s.ig_spread_bp],
            0.0,
        )

    def run(self, shock: Shock) -> Dict:
        p = self.portfolio
        loans, bonds, derivs = p[p.asset_class == "Loan"], p[p.asset_class == "Bond"], p[p.asset_class == "Derivative"]

        base_ecl, stressed_ecl = self._loan_ecl(loans, None), self._loan_ecl(loans, shock)
        pnl = pd.Series(0.0, index=p.index)
        pnl[loans.index] = -(stressed_ecl["ecl"] - base_ecl["ecl"])
        pnl[bonds.index] = self._bond_pnl(bonds, shock)
        pnl[derivs.index] = self._derivative_pnl(derivs, shock)

        bond_rwa = (bonds["market_value"] * bonds["risk_weight"]).sum()
        rwa_0 = base_ecl["rwa"].sum() + bond_rwa
        rwa_1 = stressed_ecl["rwa"].sum() + bond_rwa
        capital_0 = self.capital_cfg["cet1_ratio_start"] * rwa_0
        total_pnl = float(pnl.sum())
        capital_1 = capital_0 + total_pnl * (1 - self.capital_cfg["tax_rate"])

        value_0 = float(bonds["market_value"].sum() + loans["notional"].sum() - base_ecl["ecl"].sum())
        detail = p[["position_id", "asset_class", "instrument", "counterparty", "sector", "rating", "notional"]].copy()
        detail["pnl"] = pnl.round(0)
        detail.loc[loans.index, "stressed_rating"] = stressed_ecl["rating"]
        detail.loc[loans.index, "stage_after"] = stressed_ecl["stage"]

        by_class = pnl.groupby(p["asset_class"]).sum()
        top = detail.nsmallest(10, "pnl").astype(object)
        top = top.where(top.notna(), None)
        return {
            "shock": asdict(shock),
            "portfolio_value_before": round(value_0, 0),
            "portfolio_value_after": round(value_0 + total_pnl, 0),
            "total_pnl": round(total_pnl, 0),
            "pnl_pct_of_value": round(total_pnl / value_0, 5),
            "pnl_by_asset_class": {k: round(float(v), 0) for k, v in by_class.items()},
            "pnl_by_sector": {k: round(float(v), 0) for k, v in pnl.groupby(p["sector"]).sum().sort_values().items()},
            "ecl_before": round(float(base_ecl["ecl"].sum()), 0),
            "ecl_after": round(float(stressed_ecl["ecl"].sum()), 0),
            "loans_moved_to_stage2": int((stressed_ecl["stage"] == 2).sum()),
            "rwa_before": round(float(rwa_0), 0),
            "rwa_after": round(float(rwa_1), 0),
            "cet1_ratio_before": round(float(capital_0 / rwa_0), 4),
            "cet1_ratio_after": round(float(capital_1 / rwa_1), 4),
            "cet1_minimum_with_buffer": self.capital_cfg["cet1_minimum_with_buffer"],
            "breaches_buffer": bool(capital_1 / rwa_1 < self.capital_cfg["cet1_minimum_with_buffer"]),
            "top_losses": top.to_dict(orient="records"),
            "positions": detail,
        }

    def run_for_signal(self, signal: Signal, force: bool = False) -> Optional[Dict]:
        if not force and not self.should_trigger(signal):
            return None
        result = self.run(self.shock_for(signal.event_type, signal.impact_score))
        result["trigger_signal"] = {
            "signal_id": signal.signal_id, "timestamp": signal.timestamp.isoformat(),
            "text_excerpt": signal.text_excerpt, "ticker": signal.ticker, "event_type": signal.event_type,
            "event_confidence": signal.event_confidence, "impact_score": signal.impact_score,
            "impact_confidence": signal.impact_confidence, "sentiment_score": signal.sentiment_score,
        }
        return result

    def scan(self, signals: Iterable[Signal]) -> pd.DataFrame:
        """
        Replays a signal stream and records every stress test it would have triggered. Several
        headlines about the same event type on the same day are one event: only the most severe
        (then most confident) signal of each (day, event_type) runs a scenario.
        """
        events: Dict[tuple, Signal] = {}
        for s in signals:
            if not self.should_trigger(s):
                continue
            day_key = (s.timestamp.date(), s.event_type)
            best = events.get(day_key)
            if best is None or (s.impact_score, s.event_confidence) > (best.impact_score, best.event_confidence):
                events[day_key] = s

        rows = []
        cache: Dict[tuple, Dict] = {}
        for s in events.values():
            key = (s.event_type, s.impact_score)
            if key not in cache:
                cache[key] = self.run(self.shock_for(*key))
            r = cache[key]
            rows.append({
                "timestamp": s.timestamp, "signal_id": s.signal_id, "ticker": s.ticker, "source": s.source,
                "event_type": s.event_type, "impact_score": s.impact_score, "event_confidence": s.event_confidence,
                "text_excerpt": s.text_excerpt, "total_pnl": r["total_pnl"], "pnl_pct_of_value": r["pnl_pct_of_value"],
                "cet1_ratio_after": r["cet1_ratio_after"],
            })
        return pd.DataFrame(rows).sort_values("timestamp") if rows else pd.DataFrame()
