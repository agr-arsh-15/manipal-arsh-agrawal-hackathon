# Engine evaluation (Gate 4)

Test split: chronological (Benzinga published on or after 2019-11-01) plus a hash-held-out 15% of FinancialPhraseBank. Event labels on the test split are BART-MNLI zero-shot labels reconciled with keyword rules; the human-labelled set below is the independent check.

| Task | Metric | baseline | transformer |
|---|---|---|---|
| sentiment | n | 730 | 730 |
| sentiment | pearson_r | 0.5811 | 0.8373 |
| sentiment | mae | 0.3798 | 0.1715 |
| sentiment | macro_f1 | 0.6182 | 0.8313 |
| event_type | n | 2839 | 2839 |
| event_type | macro_f1 | 0.3963 | 0.7827 |
| event_type | weighted_f1 | 0.767 | 0.9023 |
| event_type | accuracy | 0.7978 | 0.901 |
| impact_score | n | 7114 | 7114 |
| impact_score | spearman_rho | 0.0759 | 0.1112 |
| impact_score | mae | 2.5904 | 3.0254 |
| impact_score | high_impact_precision | 0.4365 | 0.3801 |
| impact_score | high_impact_recall | 0.0908 | 0.3201 |
| impact_score | high_impact_base_rate | 0.3035 | 0.3035 |
| impact_score_company | n | 6421 | 6421 |
| impact_score_company | spearman_rho | 0.0807 | 0.1241 |
| impact_score_company | high_impact_precision | 0.4572 | 0.4012 |
| impact_score_market | n | 693 | 693 |
| impact_score_market | spearman_rho | 0.0288 | 0.0015 |
| impact_score_market | high_impact_precision | 0.225 | 0.1656 |

## Human-labelled event set

```json
{
  "status": "pending",
  "labelled_rows": 0,
  "note": "Run `streamlit run scripts/label_helper.py` to label the 268-row set."
}
```

## Latency (single headline)

```json
{
  "baseline_cpu": {
    "single_p50_ms": 1.25,
    "single_p95_ms": 1.47,
    "batch_throughput_per_s": 23019.6
  },
  "transformer_mps": {
    "single_p50_ms": 9.45,
    "single_p95_ms": 12.24,
    "batch_throughput_per_s": 912.2
  },
  "transformer_cpu": {
    "single_p50_ms": 11.76,
    "single_p95_ms": 12.13,
    "batch_throughput_per_s": 579.1
  }
}
```

## Impact calibration

| predicted | realised mean | n |
|---|---|---|
| 1 | 5.36 | 673 |
| 2 | 5.441 | 706 |
| 3 | 5.191 | 738 |
| 4 | 5.188 | 966 |
| 5 | 5.563 | 737 |
| 6 | 5.795 | 732 |
| 7 | 5.794 | 744 |
| 8 | 6.14 | 749 |
| 9 | 5.891 | 642 |
| 10 | 6.496 | 427 |
