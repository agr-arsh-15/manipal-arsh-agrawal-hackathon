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

The public deployment at https://manipal-arsh-agrawal-hackathon.streamlit.app/ runs the same code
and the same transformer. Recording from it shows judges exactly what they will open. Load it once
a few minutes before recording so it is awake.

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

0. Start at the header: the four headline numbers, the one-line data flow, and the
   "Evaluate this in 60 seconds" guide.
1. Click **Score headlines** on the six default headlines and read the results:
   - Microsoft beats and raises → Earnings/Guidance, strongly positive, linked to MSFT.
   - Intel plunges on weak guidance → Earnings/Guidance, strongly negative, linked to INTC.
   - US unemployment surges → Macroeconomic, negative, high impact.
   - S&P cuts Ford to junk → Credit Event, negative.
   - Factory fire halts a chip supplier → Operational/Supply-chain, negative.
   - Disney to acquire Fox assets → Merger/Acquisition, linked to DIS.
   - Point out the confidence columns, and that these are model predictions, not ground truth.
2. Type one headline of your own, ideally today's news.
3. Click **Fetch GDELT headlines** to score live global news. If GDELT rate-limits the request,
   the dashboard says it is replaying the saved snapshot instead.
4. Scroll to the **signal stream explorer**. Read the coverage line (which source covers which
   dates). Filter to Credit Event with impact ≥ 7, pick a ticker, show sentiment plotted over its
   price, and use **Download filtered signals** to export the view.
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
  Point out the warning that the out-of-sample figures are no longer clean once parameters change,
  then click **Reset to selected parameters**.

## 6:30 – 8:30 · Module B (tab "Module B · Stress Testing")

1. Show the portfolio sunburst: loans, bonds and derivatives by sector.
2. **Headline through the engine**: the default headline, *"US Federal Government Posts Widest
   Deficit Since 2012"*, has already been scored. Walk the trigger-gate table: a systemic event type,
   impact ≥ 8, adverse sentiment and event confidence ≥ 60% all pass, so the label reads "Triggered
   by the model". Then enter an earnings headline to show which gate fails, and tick the what-if box
   to show a forced run labelled as such. Mention that the historical replay fires about 6 stress
   tests a month because same-day headlines on the same event type count once.
3. Read the before/after view:
   - The P&L waterfall by asset class: bonds lose on duration and spreads, while pay-fixed swaps gain.
   - Loans: ECL rises and some loans move to IFRS 9 Stage 2.
   - The CET1 ratio before and after, against the 7% line (4.5% minimum + 2.5% buffer).
4. Switch to **Manual scenario → Credit Event at impact 9**. Show the Stage 2 migrations and the CDS
   hedge offset.
5. Switch to **Historical triggered signal**: 162 replayed triggers, 51 of which breach the CET1
   buffer (marked x). Replay one and download its result as JSON.

## 8:30 – 9:20 · Model performance (tab "Model Performance")

- The model card: where the model is strong, where it is weak, and what it is appropriate for.
- Baseline vs transformer on the same held-out test split, metric by metric, with the
  better/worse column.
- The impact calibration chart: the slope is upward but flat, so impact is a relative priority,
  not an absolute severity forecast.
- Latency: median per headline on MPS and on CPU.
- Be explicit that event labels are teacher labels and that the 268-row hand-labelled check is
  still being labelled.

## 9:20 – 10:00 · Impact and limitations (slides 6–7)

- "For a bank this cuts the time from headline to capital impact from days to seconds, and it gives an
  early-warning signal for SICR staging before ratings move."
- Limitations, said honestly:
  - the event labels are weak labels;
  - impact is noisy at the headline level;
  - the data window is 2018 to 2020 for 20 names;
  - Module B is a sensitivity model.
- Close with the repo URL.
