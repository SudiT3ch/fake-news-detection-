"""Pipelines (vectorizer + classifier), split, CV tuning, held-out evaluation.

Leakage control: the vectorizer lives INSIDE each Pipeline, so it is fit only on
training data (and only on the training folds during cross-validation).
Model selection uses cross-validation on the TRAIN split; the test split is used
only once, for reporting.
"""
import tempfile
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .data_loader import FAKE, LABEL_NAMES
from .preprocessing import clean_text

SEED = 42


def make_vectorizer(kind: str, max_features=20000, min_df=2):
    common = dict(preprocessor=clean_text, lowercase=False, ngram_range=(1, 1),
                  max_features=max_features, min_df=min_df)
    if kind == "bow":      # word COUNTS; vocabulary learned from training text only
        return CountVectorizer(**common)
    if kind == "tfidf":    # counts re-weighted so words common to all documents matter less
        return TfidfVectorizer(max_df=0.9, sublinear_tf=True, **common)
    raise ValueError(kind)


def make_classifier(name: str):
    if name == "logreg":
        return LogisticRegression(max_iter=1000, random_state=SEED)
    if name == "nb":
        return MultinomialNB()
    if name == "rf":
        return RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=SEED)
    raise ValueError(name)


# Small grids (tuned by CV on the training split only).
GRIDS = {
    "logreg": {"clf__C": [0.1, 1, 10]},
    "nb": {"clf__alpha": [0.1, 0.5, 1.0]},
    "rf": {"clf__n_estimators": [50]},   # single config: RF is slow on sparse text
}


def split(df: pd.DataFrame, test_size=0.2):
    return train_test_split(df["text"], df["label"], test_size=test_size,
                            stratify=df["label"], random_state=SEED)


def evaluate(pipe, X_test, y_test) -> dict:
    pred = pipe.predict(X_test)
    kw = dict(pos_label=FAKE, zero_division=0)   # positive class = Fake News
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision_fake": precision_score(y_test, pred, **kw),
        "recall_fake": recall_score(y_test, pred, **kw),
        "f1_fake": f1_score(y_test, pred, **kw),
        "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        "report": classification_report(y_test, pred, labels=[0, 1],
                                        target_names=[LABEL_NAMES[0], LABEL_NAMES[1]],
                                        zero_division=0),
    }


def run_experiments(X_train, X_test, y_train, y_test, vectorizers=("bow", "tfidf"),
                    classifiers=("logreg", "nb", "rf"), cv=3, max_features=20000, min_df=2):
    """Train every combination. Returns (results_df, fitted_pipelines, details)."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)
    rows, fitted, details = [], {}, {}
    cache = tempfile.mkdtemp()  # caches per-fold vectorizer fits (still train-fold only)
    for v in vectorizers:
        for c in classifiers:
            name = f"{v}+{c}"
            pipe = Pipeline([("vec", make_vectorizer(v, max_features, min_df)),
                             ("clf", make_classifier(c))], memory=cache)
            t0 = time.time()
            gs = GridSearchCV(pipe, GRIDS[c], cv=skf, scoring="f1_macro", refit=True)
            gs.fit(X_train, y_train)
            m = evaluate(gs.best_estimator_, X_test, y_test)
            rows.append({"model": name, "cv_f1_macro_train": gs.best_score_,
                         **{k: m[k] for k in ("accuracy", "precision_fake", "recall_fake",
                                              "f1_fake", "f1_macro")},
                         "best_params": str(gs.best_params_),
                         "fit_seconds": round(time.time() - t0, 1)})
            fitted[name], details[name] = gs.best_estimator_, m
            print(f"done {name}")
    return pd.DataFrame(rows), fitted, details


def select_best(results: pd.DataFrame) -> str:
    """Select by cross-validation score on the TRAIN split (never the test set)."""
    return results.sort_values("cv_f1_macro_train", ascending=False).iloc[0]["model"]
