# AI/NLP Financial Risk Engine - S&P Global & Crisil Campus Hackathon
**Candidate Name:** Arsh Agrawal  
**College Email ID:** ARSH.23FE10CDS00069@muj.manipal.edu  
**College / Campus:** Manipal University Jaipur  
**Demo Video Link:** [YouTube / Unlisted - Pending Recording]  
**Slide Deck Link (if hosted externally):** [Pending docs/presentation.pdf]

## 1. Project Overview / Problem Statement & Approach
Financial markets are continuously moved by fast-breaking, unstructured text streams ranging from corporate disclosures and news alerts to social sentiment. Traditional risk surveillance pipelines struggle to extract actionable, calibrated risk indicators in real time across heterogeneous sources.

This project delivers a unified, production-grade AI/NLP Financial Risk Engine engineered to ingest multi-source unstructured text (financial news, social posts/tweets, global event feeds) and output calibrated financial risk signals containing exactly three standard fields:
1. `sentiment_score`: Normalized numerical polarity in range [-1.0, 1.0].
2. `event_type`: Categorical classification across 9 taxonomy classes (Geopolitical, Macroeconomic, Credit Event, Merger/Acquisition, Product Launch, Regulatory/Legal, Earnings/Guidance, Operational/Supply-chain, Other/None).
3. `impact_score`: Calibrated market-impact severity on a 1–10 scale, derived from empirical cumulative abnormal return (CAR) event-study metrics against trailing volatility.

The signals are delivered via an asynchronous FastAPI service (`POST /analyze`, `GET /signals`, `GET /signals/{ticker}`) and reproducible structured file streams (`signals.jsonl`).

## 2. Architecture & Tech Stack
- **Architecture Overview**: The pipeline decouples source ingestion through an extensible adapter pattern, normalizes raw text into a standard `Document` schema, executes rule-based disambiguated entity linking against a large-cap universe, and feeds a multi-task text transformer with masked task heads for sentiment regression, event classification, and impact severity estimation.
- **Data Flow Diagram**: (See `docs/architecture.png` embedded below).
- **Tech Stack**:
  - Language & Runtime: Python 3.11 on macOS arm64 (Apple Silicon MPS) and Linux CUDA (Google Colab).
  - Deep Learning: PyTorch, Hugging Face Transformers (`distilroberta-base`).
  - Data Processing: Pandas, NumPy, Scikit-Learn, yfinance, pandas-market-calendars.
  - Serving & Schemas: FastAPI, Pydantic v2, Uvicorn.
  - Verification & Visualization: PyTest, Graphviz.

## 3. Dataset Used
- **Financial News Sentiment**: FinancialPhraseBank (Malo et al., CC BY-NC-SA 4.0), providing human-annotated sentiment benchmarks for financial statements.
- **Stock Tweets Social Impact**: "Tweet Sentiment's Impact on Stock Returns" (CC0 Public Domain), providing social feed volume, ticker mentions, and empirical return horizons.
- **Historical Stock News Stream**: "Daily Financial News for 6000+ Stocks" (Benzinga / CC0 research archive), providing timestamped corporate headlines across multi-year cycles.
- **Macro & Geopolitical Event Feeds**: GDELT 2.0 API (Global Database of Events, Language, and Tone) for global macro theme tracking.
- **Market Price & Benchmark Data**: Yahoo Finance (yfinance) for SPY benchmark and large-cap equity histories (cached locally in `data/prices` to ensure complete reproducibility).
- *Assumptions*: All datasets used are strictly public research archives or simulated benchmarks. No proprietary S&P Global or CRISIL client data or identifiers are utilized.

## 4. Quickstart & Installation
Runtime: Python 3.11 on macOS (Apple Silicon MPS) or Linux (Ubuntu / CUDA)

Step-by-step commands to set up the environment and run locally:
```bash
git clone https://github.com/agr-arsh-15/manipal-arsh-agrawal-hackathon.git
cd manipal-arsh-agrawal-hackathon
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run test suite
pytest tests/

# Start the Risk Engine API server
python -m src.api.main
```

## 5. Key Results & Domain Impact
- **Standardized Risk Ingestion**: Converts heterogeneous text feeds into uniform risk signals in sub-50ms inference latency on consumer hardware.
- **Empirically Grounded Impact**: Solves the synthetic label pitfall by training severity regression directly against market model abnormal returns normalized by idiosyncratic volatility.
- **Downstream Ready**: Schema output matches requirements for Module A (Tactical Portfolio Rebalancing) and Module B (Event-Driven Stress Testing).

---
### Academic Integrity & AI Assistance Disclosure
In accordance with Hackathon Submission Guidelines Section 6, AI assistance (Google Antigravity Agent) was utilized for scaffolding boilerplate, exploratory data analysis scripting, and test case framing. All architectural designs, event taxonomy definitions, market-timing event study formulations, and validation splits were designed with professional rigor and verified through empirical execution.

### Limitations
- Impact scores represent historical statistical proxies under event-study market assumptions; unobserved concurrent market events may introduce residual noise.
- Social media datasets contain automated model sentiment labels, handled via weak supervision masks during training.
