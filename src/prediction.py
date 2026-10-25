"""Inference using the saved pipeline."""
from pathlib import Path

import joblib

from .data_loader import LABEL_NAMES

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "fake_news_pipeline.joblib"
MIN_WORDS = 3
DISCLAIMER = ("This is a statistical prediction of writing patterns learned from the training "
              "dataset. It does NOT verify facts and can be wrong, especially on news from "
              "sources or topics unlike the training data.")

_cache = {}


def load_pipeline(path=DEFAULT_MODEL):
    """Load a pipeline saved by train.py. Only load files you created yourself:
    joblib/pickle files can execute code when loaded."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run: python train.py ...")
    key = str(path)
    if key not in _cache:
        _cache[key] = joblib.load(path)
    return _cache[key]


def predict_news(news_text, pipeline=None, model_path=DEFAULT_MODEL) -> dict:
    """Return {'label': 'Fake News'|'Real News', 'class_id', 'probability' (or None), 'disclaimer'}."""
    if not isinstance(news_text, str) or not news_text.strip():
        raise ValueError("Please enter a non-empty news headline or article.")
    if len(news_text.split()) < MIN_WORDS:
        raise ValueError(f"Input too short; enter at least {MIN_WORDS} words.")
    pipe = pipeline or load_pipeline(model_path)
    cls = int(pipe.predict([news_text])[0])
    prob = None
    if hasattr(pipe, "predict_proba"):
        classes = list(pipe.classes_)
        prob = float(pipe.predict_proba([news_text])[0][classes.index(cls)])
    return {"label": LABEL_NAMES[cls], "class_id": cls, "probability": prob,
            "disclaimer": DISCLAIMER}
