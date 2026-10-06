# Demo video script (10 minutes)

Record at 1080p with the dashboard in a full-screen browser window and a terminal beside it.
Upload to YouTube as **Unlisted** and paste the link into the README header.

**Before recording**

Activate the virtual environment first (`source .venv/bin/activate`, or `.venv\Scripts\Activate.ps1`
on Windows), then:

```bash
python run.py test             # show the green suite in the first minute
python run.py api              # FastAPI on :8000 (in a second terminal)
python run.py dashboard        # Streamlit on :8501
```

Optionally show the green CI badge (Windows, macOS, Ubuntu) on the GitHub page.

Open `docs/presentation.pdf` in a second window for the opening and closing slides.

---

## 0:00 – 1:00 · Problem and what was built (slides 1–2)

- "Risk-relevant information shows up first as text: a downgrade, a sanctions headline, a CEO tweet."
- "I built a risk engine that turns every text into three numbers: sentiment from -1 to 1, one of nine
  event types, and an impact score from 1 to 10. The impact score is anchored to how markets actually
  reacted to similar news."
- "Two downstream modules use these signals. Module A rebalances an index. Module B stress-tests a
  wholesale banking book."

## 1:00 – 2:30 · Architecture and data (slides 3–4)

- Walk the diagram top to bottom: three sources → adapters → entity linker → multi-task model → API and
  file → modules → dashboard.
- Explain the offline label factory, because this is where the domain judgement sits:
  - **Impact labels** come from an event study: the cumulative abnormal return over the headline day
    and the next day versus SPY, divided by the stock's own 60-day idiosyncratic volatility, then
    bucketed into deciles.
  - **Event labels** come from a BART-MNLI zero-shot model reconciled with keyword rules.
  - **Splits are chronological**, so the test set is 2019-11 to 2020-06 and includes the COVID crash.

## 2:30 – 4:30 · Live engine (dashboard tab "Risk Engine")

1. Click **Run engine** on the five default headlines and read the results:
   - Russia invasion → Geopolitical, negative sentiment, high impact.
   - Fed 75bp hike → Macroeconomic.
   - Moody's downgrades Boeing to junk → Credit Event, linked to BA.
   - Apple record → positive sentiment, linked to AAPL.
   - $TSLA recall → Operational/Supply-chain, linked to TSLA.
   - Point out the confidence columns, and that the impact score carries an uncertainty (σ).
2. Type one headline of your own, ideally today's news.
3. Click **Fetch GDELT headlines** to score live global news from the GDELT raw feed.
4. Scroll to the **signal stream explorer**. Filter to Credit Event with impact ≥ 7. Pick a ticker and
   show sentiment plotted over its price.
5. In the terminal, show the API working:
   ```bash
   curl -s 127.0.0.1:8000/health
   curl -s -X POST 127.0.0.1:8000/analyze -H 'content-type: application/json' \
     -d '{"items":[{"text":"Rating agency cuts Boeing outlook to negative on cash burn"}]}' | python -m json.tool
   curl -s '127.0.0.1:8000/signals/BA?limit=3' | python -m json.tool
   ```

## 4:30 – 6:30 · Module A (tab "Module A · Index Rebalancer")

- Explain the rule: each stock has an EWMA of its impact-weighted sentiment (high-impact headlines
  count more); weights tilt as w ∝ exp(λ·s), with
  per-stock caps, a daily turnover budget and transaction costs. Weights are set after the close and
  only earn the next day's return, so there is no look-ahead.
- Read the KPIs: out-of-sample excess return, information ratio, sentiment IC and its t-stat. Say
  plainly that parameters were chosen in-sample only and that the out-of-sample edge is small and
  not statistically significant.
- Show the NAV chart, the shaded out-of-sample window and the COVID drawdown.
- Show the **weights-over-time** chart, which is the deliverable the problem statement asks for. Then
  inspect one stock to show its weight following its sentiment.
- Move the λ slider live to show the trade-off between tilt strength, turnover and tracking error.

## 6:30 – 8:30 · Module B (tab "Module B · Stress Testing")

1. Show the portfolio sunburst: loans, bonds and derivatives by sector.
2. **Headline through the engine**: type *"Fed delivers surprise 100bp hike as inflation spirals"*. Show
   the trigger rule firing (impact ≥ 8, a systemic event type, adverse sentiment and event confidence
   ≥ 60%), or tick "run anyway". Mention that the historical replay fires about 6 stress tests a
   month because same-day headlines on the same event type count once.
3. Read the before/after view:
   - The P&L waterfall by asset class: bonds lose on duration and spreads, while pay-fixed swaps gain.
   - Loans: ECL rises and some loans move to IFRS 9 Stage 2.
   - The CET1 ratio before and after, against the 7% line (4.5% minimum + 2.5% buffer).
4. Switch to **Manual scenario → Credit Event at impact 9**. Show the Stage 2 migrations and the CDS
   hedge offset.
5. Switch to **Historical triggered signal** and replay a real headline from the stream.

## 8:30 – 9:20 · Model performance (tab "Model Performance")

- Baseline vs transformer on the same held-out test split, task by task.
- The impact calibration chart: higher predicted buckets should realise higher impact.
- Latency: p50 per headline on MPS and on CPU.
- Hand-labelled set: be explicit that event labels are teacher labels, and that the hand-labelled set
  is the independent check.

## 9:20 – 10:00 · Impact and limitations (slides 6–7)

- "For a bank this cuts the time from headline to capital impact from days to seconds, and it gives an
  early-warning signal for SICR staging before ratings move."
- Limitations, said honestly:
  - the event labels are weak labels;
  - impact is noisy at the headline level;
  - the data window is 2018 to 2020 for 20 names;
  - Module B is a sensitivity model.
- Close with the repo URL.
