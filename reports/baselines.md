# Gate 3 — Baseline Models & Benchmark Evaluation Report

**Date:** 2026-10-06  
**Auditor:** Senior Data Science Engineer (Antigravity AI)  
**Execution Context:** Local macOS arm64 environment (Python 3.11.17)

---

## 1. Executive Summary & Purpose
To establish an empirical reference point for the multi-task deep transformer architecture (Gate 4), we implemented and executed a transparent, lightweight linear benchmark suite using **TF-IDF n-gram vectorization + Ridge / Logistic Regression** across the three required risk engine tasks:
1. **Sentiment Score Regression & 3-Class Discretization**
2. **Event Type Taxonomy Classification**
3. **Market Impact Severity Regression (1–10 Scale)**

All trained baseline weights have been serialized and committed to `models/baseline/` (~500 KB total), ensuring reviewers can run the pipeline without external model downloads.

---

## 2. Empirical Benchmark Results

Metrics were produced strictly via automated execution of `scripts/train_eval_baselines.py` on held-out test splits:

| Task / Field | Model Architecture | Test Set Size | Primary Evaluation Metric | Secondary Evaluation Metric |
|---|---|---|---|---|
| **`sentiment_score`** | TF-IDF (1-2 gram) + Ridge Regression | 970 sentences (held-out PhraseBank) | **MAE: 0.3668** | **Pearson $r$: +0.6352**<br>Macro-F1: 0.6700 |
| **`event_type`** | TF-IDF (1-2 gram) + Multinomial Logistic Reg. | 1,129 headlines | **Macro-F1: 0.2883** | Accuracy: 92.38% *(high accuracy skewed by majority class)* |
| **`impact_score`** | TF-IDF (1-2 gram) + Ridge Regression | 160 headlines (held-out chronological period) | **Spearman $\rho$: +0.1762** | **MAE: 2.5722** (on 1–10 scale) |

---

## 3. Key Technical Insights & Architecture Motivation
1. **Sentiment Baseline is Robust**: The linear TF-IDF sentiment model achieves $r = 0.635$, demonstrating that lexical polarity captures significant sentiment signal. The neural model will improve on subtle financial context (e.g., "debt increased to fund growth" vs "debt downgraded").
2. **Event Classification Suffers from Class Imbalance in Linear Models**: Despite 92.38% overall accuracy, the baseline macro-F1 is low (0.2883) because rare categories (e.g., Credit Events, Operational Disruption) lack expressive representations in bag-of-words space. This justifies the shared contextual representations of a pre-trained transformer.
3. **Impact Prediction Requires Deep Context**: The baseline Spearman correlation of $0.1762$ and MAE of $2.57$ reflects the noisy nature of market reactions. An attention-based transformer that models financial semantic nuances is expected to better separate minor drift from high-severity tail shocks.

---

## 4. Benchmark Delivery Status
- Baseline artifacts saved in `models/baseline/`:
  - `tfidf_sent.joblib` & `ridge_sent.joblib`
  - `tfidf_event.joblib` & `logreg_event.joblib`
  - `tfidf_impact.joblib` & `ridge_impact.joblib`
- Raw metrics persisted to `reports/baselines_results.json`.
