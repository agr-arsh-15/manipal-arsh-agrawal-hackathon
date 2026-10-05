import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, f1_score, accuracy_score
from scipy.stats import spearmanr, pearsonr
from src.models.baselines import BaselineModelSuite

def run_baseline_evaluation():
    print("Executing Gate 3: Training & Evaluating Benchmark Baselines...")
    df = pd.read_csv("data/samples/unified_risk_dataset.csv")
    
    suite = BaselineModelSuite()
    results = {}
    
    # 1. Sentiment Baseline Evaluation (PhraseBank)
    df_sent = df[df["has_sentiment_label"]].copy()
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        df_sent["text"].tolist(), df_sent["sentiment_score"].tolist(),
        test_size=0.2, random_state=42
    )
    suite.fit_sentiment(X_train_s, y_train_s)
    preds_s = suite.predict_sentiment(X_test_s)
    mae_s = float(mean_absolute_error(y_test_s, preds_s))
    p_corr_s, _ = pearsonr(y_test_s, preds_s)
    
    # Discrete 3-class mapping
    def to_class(scores):
        return ["positive" if s > 0.33 else ("negative" if s < -0.33 else "neutral") for s in scores]
        
    f1_s = float(f1_score(to_class(y_test_s), to_class(preds_s), average="macro"))
    results["sentiment"] = {
        "model": "TF-IDF + Ridge",
        "sample_size": len(y_test_s),
        "mae": round(mae_s, 4),
        "pearson_corr": round(float(p_corr_s), 4),
        "macro_f1": round(f1_s, 4)
    }
    
    # 2. Event Type Baseline Evaluation
    df_event = df[df["has_event_label"]].copy()
    X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
        df_event["text"].tolist(), df_event["event_type"].tolist(),
        test_size=0.2, random_state=42
    )
    suite.fit_event(X_train_e, y_train_e)
    preds_e = suite.predict_event(X_test_e)
    macro_f1_e = float(f1_score(y_test_e, preds_e, average="macro"))
    acc_e = float(accuracy_score(y_test_e, preds_e))
    results["event_type"] = {
        "model": "TF-IDF + LogisticRegression",
        "sample_size": len(y_test_e),
        "macro_f1": round(macro_f1_e, 4),
        "accuracy": round(acc_e, 4)
    }
    
    # 3. Impact Severity Baseline Evaluation (Time-based split on Benzinga)
    df_impact = df[df["has_impact_label"]].sort_values(by="published_at").copy()
    split_idx = int(len(df_impact) * 0.8)
    train_imp = df_impact.iloc[:split_idx]
    test_imp = df_impact.iloc[split_idx:]
    
    suite.fit_impact(train_imp["text"].tolist(), train_imp["impact_score"].tolist())
    preds_imp = suite.predict_impact(test_imp["text"].tolist())
    y_test_imp = test_imp["impact_score"].values
    
    mae_imp = float(mean_absolute_error(y_test_imp, preds_imp))
    spearman_imp, _ = spearmanr(y_test_imp, preds_imp)
    
    results["impact_score"] = {
        "model": "TF-IDF + Ridge",
        "sample_size": len(y_test_imp),
        "mae": round(mae_imp, 4),
        "spearman_corr": round(float(spearman_imp), 4)
    }
    
    # Save model weights to models/baseline
    suite.save("models/baseline")
    
    # Save metrics report
    os.makedirs("reports", exist_ok=True)
    with open("reports/baselines_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n=== Baseline Results ===")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    run_baseline_evaluation()
