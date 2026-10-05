import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, f1_score
from scipy.stats import spearmanr
import joblib
import os


class BaselineModelSuite:
    """
    Standard benchmark models for the 3 target tasks:
    1. Sentiment: TF-IDF + Ridge Regression
    2. Event Type: TF-IDF + Multinomial Logistic Regression
    3. Impact: TF-IDF + Ridge Regression on continuous |z|
    """

    def __init__(self):
        self.tfidf_sentiment = TfidfVectorizer(max_features=2500, ngram_range=(1, 2))
        self.model_sentiment = Ridge(alpha=1.0)
        
        self.tfidf_event = TfidfVectorizer(max_features=2500, ngram_range=(1, 2))
        self.model_event = LogisticRegression(max_iter=500, C=1.0)
        
        self.tfidf_impact = TfidfVectorizer(max_features=2500, ngram_range=(1, 2))
        self.model_impact = Ridge(alpha=1.0)

    def fit_sentiment(self, texts: list, y_scores: list):
        X = self.tfidf_sentiment.fit_transform(texts)
        self.model_sentiment.fit(X, y_scores)

    def predict_sentiment(self, texts: list) -> np.ndarray:
        X = self.tfidf_sentiment.transform(texts)
        preds = self.model_sentiment.predict(X)
        return np.clip(preds, -1.0, 1.0)

    def fit_event(self, texts: list, y_events: list):
        X = self.tfidf_event.fit_transform(texts)
        self.model_event.fit(X, y_events)

    def predict_event(self, texts: list) -> np.ndarray:
        X = self.tfidf_event.transform(texts)
        return self.model_event.predict(X)

    def fit_impact(self, texts: list, y_impact: list):
        X = self.tfidf_impact.fit_transform(texts)
        self.model_impact.fit(X, y_impact)

    def predict_impact(self, texts: list) -> np.ndarray:
        X = self.tfidf_impact.transform(texts)
        preds = self.model_impact.predict(X)
        return np.clip(preds, 1.0, 10.0)

    def save(self, out_dir: str = "models/baseline"):
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(self.tfidf_sentiment, os.path.join(out_dir, "tfidf_sent.joblib"))
        joblib.dump(self.model_sentiment, os.path.join(out_dir, "ridge_sent.joblib"))
        joblib.dump(self.tfidf_event, os.path.join(out_dir, "tfidf_event.joblib"))
        joblib.dump(self.model_event, os.path.join(out_dir, "logreg_event.joblib"))
        joblib.dump(self.tfidf_impact, os.path.join(out_dir, "tfidf_impact.joblib"))
        joblib.dump(self.model_impact, os.path.join(out_dir, "ridge_impact.joblib"))
        print(f"Saved baseline suite to {out_dir}")

    def load(self, in_dir: str = "models/baseline"):
        self.tfidf_sentiment = joblib.load(os.path.join(in_dir, "tfidf_sent.joblib"))
        self.model_sentiment = joblib.load(os.path.join(in_dir, "ridge_sent.joblib"))
        self.tfidf_event = joblib.load(os.path.join(in_dir, "tfidf_event.joblib"))
        self.model_event = joblib.load(os.path.join(in_dir, "logreg_event.joblib"))
        self.tfidf_impact = joblib.load(os.path.join(in_dir, "tfidf_impact.joblib"))
        self.model_impact = joblib.load(os.path.join(in_dir, "ridge_impact.joblib"))
