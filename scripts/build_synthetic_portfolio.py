"""
Generates a seeded, fully synthetic wholesale-banking book for Module B.

    python -m scripts.build_synthetic_portfolio

No real counterparties, clients or proprietary data are used: borrower names are generated
identifiers and all risk parameters are drawn from illustrative ranges documented below.

Output: data/portfolio/synthetic_portfolio.csv (one row per position) with columns shared
across asset classes plus class-specific risk sensitivities.

Assumptions (illustrative, rounded):
  - 1-year PD by rating: AAA 0.01%, AA 0.02%, A 0.06%, BBB 0.20%, BB 0.90%, B 3.5%, CCC 15%
  - LGD: senior secured 35%, senior unsecured 45%
  - Standardised-approach style risk weights: AAA-AA 20%, A 50%, BBB-BB 100%, B-CCC 150%
  - Bond modified duration ~ 0.9 x maturity (capped), convexity ~ duration^2 / 100 scale
"""
import os

import numpy as np
import pandas as pd

SEED = 2026
OUT = "data/portfolio/synthetic_portfolio.csv"

RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
PD_1Y = {"AAA": 0.0001, "AA": 0.0002, "A": 0.0006, "BBB": 0.0020, "BB": 0.0090, "B": 0.035, "CCC": 0.15}
RISK_WEIGHT = {"AAA": 0.2, "AA": 0.2, "A": 0.5, "BBB": 1.0, "BB": 1.0, "B": 1.5, "CCC": 1.5}
SECTORS = ["Energy", "Industrials", "Consumer Discretionary", "Technology", "Financials",
           "Healthcare", "Real Estate", "Utilities"]
REGIONS = ["North America", "Europe", "Asia-Pacific", "India", "Latin America"]


def build(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    # Corporate loans (banking book, amortised cost): credit risk via PD / LGD / EAD.
    loan_rating_p = [0.02, 0.05, 0.18, 0.35, 0.25, 0.12, 0.03]
    for i in range(60):
        rating = rng.choice(RATINGS, p=loan_rating_p)
        secured = rng.random() < 0.55
        notional = float(np.round(rng.lognormal(mean=np.log(90e6), sigma=0.6), -5))
        rows.append({
            "position_id": f"LN-{i + 1:03d}", "asset_class": "Loan", "instrument": "Term loan" if rng.random() < 0.6 else "Revolver",
            "counterparty": f"Borrower-{i + 1:03d}", "sector": rng.choice(SECTORS), "region": rng.choice(REGIONS),
            "rating": rating, "notional": notional, "market_value": notional,
            "maturity_years": round(float(rng.uniform(1, 7)), 1),
            "rate_type": "floating" if rng.random() < 0.7 else "fixed",
            "pd_1y": PD_1Y[rating], "lgd": 0.35 if secured else 0.45, "risk_weight": RISK_WEIGHT[rating],
            "modified_duration": 0.0, "convexity": 0.0, "spread_bucket": "IG" if rating in RATINGS[:4] else "HY",
            "delta": 0.0, "gamma": 0.0, "dv01_usd": 0.0, "risk_factor": "credit",
        })

    # Bonds (trading / fair-value book): rate and spread duration.
    for i in range(45):
        kind = rng.choice(["Sovereign", "Corporate IG", "Corporate HY"], p=[0.35, 0.45, 0.20])
        if kind == "Sovereign":
            rating, sector = rng.choice(["AAA", "AA"]), "Sovereign"
        elif kind == "Corporate IG":
            rating, sector = rng.choice(["AA", "A", "BBB"], p=[0.15, 0.4, 0.45]), rng.choice(SECTORS)
        else:
            rating, sector = rng.choice(["BB", "B", "CCC"], p=[0.55, 0.35, 0.10]), rng.choice(SECTORS)
        maturity = float(rng.choice([2, 3, 5, 7, 10, 20, 30], p=[0.15, 0.15, 0.25, 0.15, 0.18, 0.07, 0.05]))
        duration = round(min(0.9 * maturity, 18.0) * float(rng.uniform(0.85, 1.0)), 2)
        notional = float(np.round(rng.lognormal(mean=np.log(60e6), sigma=0.5), -5))
        price = float(rng.uniform(0.92, 1.04))
        rows.append({
            "position_id": f"BD-{i + 1:03d}", "asset_class": "Bond", "instrument": kind,
            "counterparty": f"Issuer-{i + 1:03d}", "sector": sector, "region": rng.choice(REGIONS),
            "rating": rating, "notional": notional, "market_value": round(notional * price, 0),
            "maturity_years": maturity, "rate_type": "fixed",
            "pd_1y": PD_1Y[rating], "lgd": 0.6, "risk_weight": 0.0 if kind == "Sovereign" else RISK_WEIGHT[rating],
            "modified_duration": duration, "convexity": round(duration ** 2 / 100 * 1.2, 3),
            "spread_bucket": "SOV" if kind == "Sovereign" else ("IG" if rating in RATINGS[:4] else "HY"),
            "delta": 0.0, "gamma": 0.0, "dv01_usd": 0.0, "risk_factor": "rates+spread",
        })

    # Derivatives (trading book): signed sensitivities to each market risk factor.
    deriv_specs = [
        ("Interest rate swap (pay fixed)", "rates", 10), ("Interest rate swap (receive fixed)", "rates", 4),
        ("Equity index total return swap", "equity", 5), ("Equity index put option", "equity", 4),
        ("FX forward (long USD)", "fx", 5), ("FX forward (short USD)", "fx", 3),
        ("Commodity swap (long oil)", "commodity", 3), ("Credit default swap (protection bought)", "credit_spread", 5),
    ]
    k = 0
    for name, factor, count in deriv_specs:
        for _ in range(count):
            k += 1
            notional = float(np.round(rng.lognormal(mean=np.log(120e6), sigma=0.5), -5))
            delta = gamma = dv01 = 0.0
            if factor == "rates":
                tenor = float(rng.choice([2, 5, 10]))
                dv01 = notional / 1e6 * tenor * 95 * (1 if "pay fixed" in name else -1)  # $ per bp
            elif factor == "equity":
                delta = 1.0 if "total return" in name else -float(rng.uniform(0.25, 0.5))
                gamma = 0.0 if "total return" in name else float(rng.uniform(1.0, 3.0))
            elif factor == "fx":
                delta = 1.0 if "long USD" in name else -1.0
            elif factor == "commodity":
                delta = 1.0
            elif factor == "credit_spread":
                tenor = 5.0
                dv01 = notional / 1e6 * tenor * 90  # protection buyer gains as spreads widen
            rows.append({
                "position_id": f"DV-{k:03d}", "asset_class": "Derivative", "instrument": name,
                "counterparty": f"Dealer-{int(rng.integers(1, 9)):02d}", "sector": "Markets", "region": rng.choice(REGIONS[:3]),
                "rating": rng.choice(["AA", "A"]), "notional": notional, "market_value": 0.0,
                "maturity_years": float(rng.choice([1, 2, 5, 10])), "rate_type": "n/a",
                "pd_1y": 0.0, "lgd": 0.0, "risk_weight": 0.0, "modified_duration": 0.0, "convexity": 0.0,
                "spread_bucket": "IG", "delta": round(delta, 3), "gamma": round(gamma, 3),
                "dv01_usd": round(dv01, 1), "risk_factor": factor,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} positions to {OUT}")
    print(df.groupby("asset_class")["notional"].agg(["count", "sum"]).map(lambda x: f"{x:,.0f}"))
