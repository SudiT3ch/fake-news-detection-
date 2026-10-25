"""Text preprocessing. ONE function is used for training AND prediction
(it is passed to the vectorizer as `preprocessor`), so both always match.

Deliberately simple: lowercase, strip HTML/URLs, keep letters/digits/apostrophes.
Stop words are NOT removed here, so negations ("not", "never") are kept.
"""
import re

import pandas as pd

_HTML = re.compile(r"<[^>]+>")
_URL = re.compile(r"(https?://\S+|www\.\S+)")
# Dataset artifacts (verified in the supplied Fake/True CSVs): ~99% of Real articles start with
# "CITY (Reuters) -" and ~15% of Fake ones contain pic.twitter links. A model could "cheat" on these,
# so they are removed. Removing them lowers dataset-specific scores but gives a fairer estimate.
_DATELINE = re.compile(r"^\s*[^()]{0,60}\(reuters\)\s*[-\u2013\u2014]?\s*", re.I)
_REUTERS = re.compile(r"\breuters\b", re.I)
_PIC_TWITTER = re.compile(r"pic\.twitter\.com/\S+", re.I)
_NON_TEXT = re.compile(r"[^a-z0-9' ]+")
_SPACES = re.compile(r"\s+")


def clean_text(text) -> str:
    """Return cleaned text. None/NaN/non-strings become ''."""
    if text is None or (not isinstance(text, str) and pd.isna(text)):
        return ""
    text = _DATELINE.sub("", str(text))
    text = _REUTERS.sub(" ", _PIC_TWITTER.sub(" ", text)).lower()
    text = _HTML.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _NON_TEXT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def combine_title_text(title, text) -> str:
    """Join headline and body into one raw string (cleaning happens later)."""
    parts = [p for p in (title, text) if isinstance(p, str) and p.strip()]
    return " ".join(parts)
