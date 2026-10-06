# AI/NLP Financial Risk Engine — S&P Global × Crisil Campus Hackathon 2026

**Candidate Name:** Arsh Agrawal  
**College Email ID:** ARSH.23FE10CDS00069@muj.manipal.edu  
**College / Campus:** Manipal University Jaipur  
**Demo Video Link:** _to be added — unlisted YouTube link (script in [`docs/demo_script.md`](docs/demo_script.md))_  
**Slide Deck Link:** [`docs/presentation.pdf`](docs/presentation.pdf) (7 slides, built reproducibly by `scripts/build_deck.py`)

---

## 1. Project Overview — Problem Statement & Approach

Market-moving information arrives first as unstructured text: an analyst downgrade, a sanctions
headline, a CEO's tweet. It reaches prices, spreads and ratings later. This project builds an
**AI/NLP risk engine** that reads that text from multiple sources and turns every item into three
quantified, auditable signals:

| Field | Range | Meaning |
|---|---|---|
| `sentiment_score` | [-1, 1] | Tone of the text towards the linked company (or the market) |
| `event_type` | 9 classes | Geopolitical · Macroeconomic · Credit Event · Merger/Acquisition · Product Launch · Regulatory/Legal · Earnings/Guidance · Operational/Supply-chain · Other/None |
| `impact_score` | 1–10 | Expected severity of the market reaction, **calibrated to realised returns** (see below) |

Each signal also carries `event_confidence`, `impact_confidence` (from a predicted variance),
ticker, timestamp, source and model version. Signals are served by a **FastAPI** service and
written to a **JSONL file**. Two downstream risk modules consume them, and both are implemented:

- **Module A — Sentiment-driven index rebalancing.** A 20-stock S&P 100 index tilts towards
  improving impact-weighted sentiment. Weights are bounded, turnover is capped and costs are
  charged, and weights-over-time are shown on the dashboard.
- **Module B — Event-driven stress testing.** A high-impact systemic event (impact ≥ 8 with a
  Geopolitical, Macroeconomic, Credit Event, Operational or Regulatory type, adverse sentiment and
  a confident event classification) triggers a shock vector.
  The shock revalues a synthetic wholesale banking book of loans, bonds and derivatives and reports
  P&L, IFRS 9 ECL, Stage 2 migration and CET1 before and after.

**Design choices that matter to a risk user**

- **Impact is measured, not guessed.** Labels come from an event study over the [0, +1] session
  window:
  - **Company news:** the cumulative abnormal return against an SPY market model, divided by the
    stock's 60-day idiosyncratic volatility.
  - **Market-wide news:** the SPY move divided by SPY's own 60-day volatility. A market model would
    net a systemic shock out of the label.
  - Both are binned into train-set deciles, so an impact of 9 means a top-decile surprise.
- **No look-ahead.**
  - The train/validation/test split is chronological (train before 2019-07, validation before
    2019-11, test after that, which includes the COVID-19 crash).
  - Headlines published at or after 16:00 New York time roll to the next session.
  - Module A weights formed on day *t* only earn day *t+1* returns.
- **Honest labels.** Event classes come from a zero-shot NLI model (`facebook/bart-large-mnli`)
  reconciled with keyword rules. A 268-row hand-labelled set is the independent check.
- **Uncertainty is first-class.** The impact head predicts a mean and a variance, so every signal
  says how sure it is.

## 2. Architecture & Tech Stack

![Architecture](docs/architecture.png)

| Layer | Implementation |
|---|---|
| Ingestion | `src/ingestion/adapters.py` (PhraseBank, Benzinga, Stock Tweets) and `src/ingestion/gdelt.py`. GDELT uses the live DOC API, falls back to the raw 15-minute GKG exports, and falls back again to a snapshot. Everything is normalised to one `Document` schema (`src/schemas.py`). |
| Entity linking | `src/linking/matcher.py` maps cashtags, aliases and context rules to the 20-stock universe (`config/universe.yaml`). Ambiguous names such as Apple or Meta need financial context. Unlinked macro news becomes an event-level signal. |
| Label factory | `src/labeling/impact.py` (event study), `src/labeling/zeroshot.py` with `scripts/relabel_events_zeroshot.py` (BART-MNLI with rule reconciliation), and `scripts/build_unified_dataset.py`. |
| Model | `src/models/multitask.py` is a fine-tuned **DistilRoBERTa** encoder with mean pooling and three heads: a tanh sentiment regressor, a 9-way event softmax, and a heteroscedastic impact head (Gaussian NLL). The loss is masked per task and the event weights are class-balanced. `src/models/baselines.py` is a TF-IDF + linear baseline, used as the yardstick and as a fallback. |
| Engine & delivery | `src/engine/pipeline.py` (`RiskEngine`), `src/engine/store.py` (JSONL store) and `src/api/main.py` (FastAPI). |
| Modules | `src/modules/rebalancer.py` (Module A) and `src/modules/stress.py` with `config/stress_scenarios.yaml` (Module B). |
| Dashboard | `app/dashboard.py` (Streamlit and Plotly), with tabs for the live engine, Module A, Module B and model performance. |

**Tech stack:** Python 3.11, PyTorch 2.14 (Apple-silicon MPS, CUDA or CPU), Hugging Face
Transformers 5, scikit-learn, pandas 3, FastAPI and Pydantic v2, Streamlit and Plotly,
matplotlib and ReportLab for the deck, and pytest. Exact versions are pinned in `requirements.txt`.

**API**

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Backend (`transformer` or `baseline`), model version and signal count |
| POST | `/analyze` | Score up to 256 texts (`{"items":[{"text": "...", "source_type": "news"}]}`); optionally persist them |
| GET | `/signals` | Query the stream (filters: `event_type`, `min_impact`, `since`, `limit`) |
| GET | `/signals/{ticker}` | Signals for one universe ticker |
| POST | `/ingest/gdelt` | Pull live GDELT headlines, score and persist them |
| POST | `/modules/stress-test` | Run Module B from a headline or from an `event_type` and `impact_score` |
| GET | `/modules/rebalancer/weights` | Latest Module A target weights |

## 3. Dataset Used

All data is public or synthetic. No S&P Global or Crisil proprietary data is used. Every CSV/JSON
the system consumes is in [`data/`](data). The raw Kaggle dumps (~1.1 GB) are gitignored, but
format-preserving 300-row extracts are committed in [`data/raw_samples/`](data/raw_samples).

| Source | Licence | Role | Rows used |
|---|---|---|---|
| [FinancialPhraseBank](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news) (Malo et al.) | CC BY-NC-SA 3.0 | Human sentiment labels (news sentences) | 4,846 |
| [Benzinga "Massive Stock News"](https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests) (analyst-ratings and partner feeds) | Kaggle, research use | Company news (event + company impact) and market-wide news (market impact) | 13,000 company + 3,655 market-wide (training); ~24k in the signal stream |
| [Tweet Sentiment's Impact on Stock Returns](https://www.kaggle.com/datasets/thedevastator/tweet-sentiment-s-impact-on-stock-returns) | CC0 | Social posts (event + company impact) | 3,481 |
| [GDELT 2.0](https://www.gdeltproject.org/) DOC API and GKG raw exports | Open | Live global event feed (inference) — snapshot in `data/samples/gdelt_snapshot.json` | live |
| Yahoo Finance daily prices (20 stocks, SPY, ^VIX) | Public | Event-study labels and the Module A backtest — cached in `data/prices/` | 2018-2020 |
| Synthetic wholesale portfolio (`scripts/build_synthetic_portfolio.py`, seed 2026) | Synthetic | Module B book — `data/portfolio/synthetic_portfolio.csv` | 144 positions (60 loans, 45 bonds, 39 derivatives) |

**Key files**

- `data/samples/unified_risk_dataset.csv`: the labelled training table. It has task masks and a
  shared `split` column.
- `data/labels/event_labels_zeroshot.csv`: zero-shot event labels.
- `data/labels/hand_labelled_eval.csv`: the 268-row human evaluation set.
- `data/samples/signals.jsonl`: the full signal stream.
- `data/outputs/`: Module A weights, NAV and sentiment state, plus the Module B triggers.

**Assumptions and limitations of the data**

- Benzinga and Stock Tweets end in mid-2020. Tweets cover 2018 H2 only.
- Market-wide impact labels are day-level, so all market headlines published on the same day
  share one label.
- The PhraseBank licence is non-commercial, which is fine for this research prototype.

## 4. Quickstart & Installation

Runtime: Python 3.11 on macOS (Apple-silicon MPS), Linux (CUDA) or CPU.

```bash
git clone https://github.com/agr-arsh-15/manipal-arsh-agrawal-hackathon.git
cd manipal-arsh-agrawal-hackathon
make setup && source .venv/bin/activate      # python3.11 -m venv .venv && pip install -r requirements.txt
make test                                     # full pytest suite

make dashboard                                # Streamlit UI on http://localhost:8501
make api                                      # FastAPI on http://localhost:8000  (docs at /docs)
curl -s -X POST localhost:8000/analyze -H 'content-type: application/json' \
  -d '{"items":[{"text":"Moody'\''s downgrades Boeing to junk as cash burn accelerates"}]}'
```

**About the model weights.** The fine-tuned transformer (~330 MB) is gitignored. Without it the
engine **automatically falls back to the committed TF-IDF baseline**, so every command above still
works. To reproduce the transformer:

```bash
make train          # ~25 min on an Apple M3 (MPS); writes models/risk_engine/
make evaluate       # reports/engine_eval.{json,md} + docs/figures/
make signals        # re-score the full stream -> data/samples/signals.jsonl
make modules        # Module A backtest + Module B scenarios -> reports/, data/outputs/
make deck           # docs/presentation.pdf + docs/architecture.png
```

To rebuild the dataset from scratch:

1. Run `pip install kaggle`.
2. Download the three Kaggle datasets into `data/raw/` with `python scripts/download_data.py` and
   `python scripts/download_benzinga.py`. Both scripts read `KAGGLE_USERNAME` / `KAGGLE_KEY` from
   `.env` (see `.env.example`).
3. Run `make data label`. The zero-shot labelling step takes about 1 hour on MPS.

## 5. Key Results & Domain Impact

Every number below is produced by the scripts in this repository. The underlying files are
`reports/engine_eval.md`, `reports/module_a_backtest.json` and `reports/module_b_stress.json`.

**NLP engine on the held-out chronological test split** (Nov 2019 – Jun 2020, never seen in
training or model selection)

| Task | Metric | TF-IDF baseline | Multi-task DistilRoBERTa |
|---|---|---|---|
| Sentiment (n = 730) | Pearson r / MAE / 3-class macro-F1 | 0.58 / 0.38 / 0.62 | **0.84 / 0.17 / 0.83** |
| Event type (n = 2,839) | macro-F1 / accuracy | 0.40 / 0.80 | **0.78 / 0.90** |
| Impact, company news (n = 6,421) | Spearman ρ | 0.08 | **0.12** |
| Impact, all (n = 7,114) | recall / precision of "impact > 7" (base rate 0.30) | 0.09 / 0.44 | **0.32** / 0.38 |
| Latency | p50 per headline | 1 ms (CPU) | **9 ms (MPS), 12 ms (CPU)**; 900 headlines/s batched |

![Model comparison](docs/figures/model_comparison.png)

Notes on the engine results:

- The transformer's per-class event F1 is 0.57–0.97, including the rare systemic classes:
  Macroeconomic 0.69, Geopolitical 0.69, Credit Event 0.65.
- Predicted impact deciles are monotone in realised severity. The mean realised impact rises from
  about 5.2 in the lowest buckets to 6.5 in the top bucket
  (`docs/figures/impact_calibration.png`).
- **Caveat:** the market-wide impact label (one per day) does not yet generalise to the
  COVID-era test window (Spearman ρ ≈ 0). See Limitations.

**Module A: sentiment-tilted index vs equal weight**

- The configuration is selected on the in-sample window (before 2019-11-01) from a 24-point grid:
  λ = 1, half-life 5 days, impact-weighted sentiment.
- In-sample, the five best configurations all weighted each headline's sentiment by its
  predicted impact rather than plain averaging.
- Out of sample (2019-11-01 to 2020-06-10, through the COVID crash):
  - excess return +0.25%;
  - information ratio 0.30;
  - Sharpe 0.81 vs 0.80;
  - tracking error 1.1%;
  - daily hit rate 56%;
  - 1.5% average daily turnover;
  - sentiment IC 0.021 (t = 0.9).
- The edge is small and not statistically significant. It is reported as such.

![Module A](docs/figures/module_a_nav.png)
![Weights over time](docs/figures/module_a_weights.png)

**Module B: event-driven stress test** on the synthetic $9.1bn book, with reference scenarios at
impact 9:

| Scenario | P&L | Loans to Stage 2 | ECL | CET1 (from 12.5%) |
|---|---|---|---|---|
| Macroeconomic (rates +, spreads wider) | **−$482m (−5.3%)** | 5 | $23m → $54m | **6.1% — breaches the 7% buffer** |
| Credit Event (spread blow-out, PD ×) | −$316m (−3.5%) | 12 | $23m → $60m | 8.4% |
| Regulatory/Legal | −$71m | 0 | $23m → $26m | 11.6% |
| Geopolitical | −$54m | 0 | $23m → $31m | 11.8% |
| Operational/Supply-chain | −$49m | 0 | $23m → $28m | 11.9% |

![Module B](docs/figures/module_b_scenarios.png)

- Pay-fixed swaps and CDS protection offset part of the losses, so the hedges show up in the
  waterfall.
- **Historical replay.** The 27,669-signal stream would have fired 162 stress tests (about 6 per
  month). Each one required impact ≥ 8, an adverse sentiment, event confidence ≥ 0.6, and counted
  once per event-day. Examples include "Dow Jones plunges further as Fed rate cut fails to boost
  confidence" (Mar 16 2020) and "Jobs report: −701K in March" (Apr 3 2020).

**Domain impact**

- **Time to impact.** Headline-to-capital-impact time falls from days of manual analysis to
  seconds. A severe adverse headline automatically yields a sized P&L, ECL and CET1 read-out.
- **Credit early warning.** Company-level sentiment and Credit Event signals flag SICR candidates
  before ratings move. The IFRS 9 staging effect is quantified immediately.
- **Capital planning.** The sector and asset-class waterfall shows where the buffer is consumed,
  which feeds hedging and limit decisions.
- **Audit trail.** Every signal carries its source, timestamp, model version, confidence and a
  text excerpt.

---

### Repository layout

```
app/dashboard.py            Streamlit dashboard (4 tabs)
config/                     engine, taxonomy, universe, impact bins, stress scenarios
data/                       prices, labels, samples (dataset, signals, GDELT snapshot), portfolio, outputs, raw_samples
docs/                       architecture.{dot,png}, presentation.pdf, figures/, demo_script.md
reports/                    engine_eval, module_a_backtest, module_b_stress, event_labels, training_history
scripts/                    dataset build, labelling, training, evaluation, signal generation, modules, deck
src/                        ingestion, linking, labeling, models, engine, api, modules, eval
tests/                      pytest suite (adapters, impact, GDELT, pipeline, API, modules, eval)
```

### Limitations

- **Event labels are weak labels.** Event macro-F1 on the test split measures agreement with the
  zero-shot and rule teacher. The hand-labelled set is the ground-truth check (see
  `reports/engine_eval.md`).
- **Headline-level impact is inherently noisy.** Most daily return variance is not news-driven,
  and market-wide labels are day-level. Intraday prices would sharpen both.
- **Coverage is limited.** The data covers 2018 to mid-2020 and 20 large caps. GDELT is live but
  rate-limited, which is why the raw GKG fallback exists.
- **Module B is a sensitivity model.** It uses duration and convexity, delta-gamma and CS01
  revaluation, a static balance sheet and a hand-calibrated scenario library. There is no full
  revaluation, liquidity modelling or second-round effects.
- **Module A is a 2-year, 20-name backtest with simple costs.** It is evidence of signal value,
  not an investable strategy.

### Academic Integrity & AI Assistance Disclosure

In line with Section 6 of the submission guidelines: AI coding assistants were used to write and
refactor code, tests and documentation under my direction. The problem framing, the event
taxonomy, the event-study labelling design (company vs market scope, leakage controls), the model
and evaluation design, the stress-testing methodology and every reported number were reviewed and
verified by me by running the code in this repository.
