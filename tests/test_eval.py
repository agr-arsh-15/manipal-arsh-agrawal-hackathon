import numpy as np
import pandas as pd

from src.eval import evaluate
from src.eval.metrics import event_metrics, impact_metrics, sentiment_metrics


def test_sentiment_metrics_perfect_prediction():
    y = np.array([-0.8, 0.0, 0.7, 0.2])
    m = sentiment_metrics(y, y)
    assert m["mae"] == 0 and m["pearson_r"] > 0.999 and m["accuracy"] == 1


def test_event_macro_f1_ignores_classes_absent_from_labels():
    labels = ["Macroeconomic", "Credit Event", "Other/None"]
    m = event_metrics(["Macroeconomic", "Other/None"], ["Macroeconomic", "Other/None"], labels)
    assert m["macro_f1"] == 1.0


def test_impact_high_impact_precision_recall():
    m = impact_metrics(np.array([9, 8, 2, 3]), np.array([9, 3, 8, 3]))
    assert m["high_impact_precision"] == 0.5
    assert m["high_impact_recall"] == 0.5


def test_impact_calibration_buckets_predictions():
    df = pd.DataFrame({"pred": [1.2, 1.4, 8.6, 9.2], "true": [2, 4, 9, 7]})
    cal = {c["predicted_bucket"]: c for c in evaluate.impact_calibration(df)}
    assert cal[1]["realised_mean"] == 3.0 and cal[1]["n"] == 2
    assert set(cal) == {1, 9}


def test_hand_labelled_eval_reports_pending_when_unlabelled(tmp_path, monkeypatch):
    path = tmp_path / "hand.csv"
    pd.DataFrame({"text": ["Fed hikes rates"], "rule_suggested_event": ["Macroeconomic"],
                  "user_ground_truth_event": [""]}).to_csv(path, index=False)
    monkeypatch.setattr(evaluate, "HAND", str(path))
    assert evaluate.hand_labelled_eval({})["status"] == "pending"


def test_hand_labelled_eval_scores_rules_against_truth(tmp_path, monkeypatch):
    path = tmp_path / "hand.csv"
    pd.DataFrame({"text": ["Fed hikes rates", "Moody's cuts Boeing"],
                  "rule_suggested_event": ["Macroeconomic", "Other/None"],
                  "user_ground_truth_event": ["Macroeconomic", "Credit Event"]}).to_csv(path, index=False)
    monkeypatch.setattr(evaluate, "HAND", str(path))
    monkeypatch.setattr(evaluate, "ZS", str(tmp_path / "missing.csv"))
    res = evaluate.hand_labelled_eval({})
    assert res["labelled_rows"] == 2
    assert res["keyword_rules"]["accuracy"] == 0.5
