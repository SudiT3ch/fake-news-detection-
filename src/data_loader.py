"""Dataset loading and label normalisation.

Label convention used everywhere in this project:
    0 = Fake News,  1 = Real News   (positive class for metrics = Fake, see evaluation)
"""
import pandas as pd

from .preprocessing import clean_text, combine_title_text

LABEL_NAMES = {0: "Fake News", 1: "Real News"}
FAKE, REAL = 0, 1

_FAKE_VALUES = {"fake", "false", "f", "0", "unreliable", "0.0"}
_REAL_VALUES = {"real", "true", "t", "1", "reliable", "1.0"}
# NOTE: some datasets use 1 = fake. If yours does, pass label_map explicitly.


def normalize_labels(series: pd.Series, label_map: dict | None = None) -> pd.Series:
    """Map raw labels to 0 (Fake) / 1 (Real). Unknown values raise ValueError."""
    def conv(v):
        key = str(v).strip().lower()
        if label_map is not None:
            m = {str(k).strip().lower(): val for k, val in label_map.items()}
            if key in m:
                return m[key]
        elif key in _FAKE_VALUES:
            return FAKE
        elif key in _REAL_VALUES:
            return REAL
        raise ValueError(f"Unrecognised label {v!r}; pass label_map={{raw: 0 or 1}} (0=Fake, 1=Real).")
    return series.map(conv).astype(int)


def _find_col(df, names):
    lower = {c.lower(): c for c in df.columns}
    return next((lower[n] for n in names if n in lower), None)


def prepare_frame(df: pd.DataFrame, label_col="label", label_map=None) -> pd.DataFrame:
    """Return DataFrame[text, label]. Uses title+text when present; ignores metadata
    (subject, date, ...) which can leak the label."""
    title_c = _find_col(df, ["title", "headline"])
    text_c = _find_col(df, ["text", "content", "article", "body"])
    if title_c is None and text_c is None:
        raise ValueError(f"Need a 'title' and/or 'text' column; found {list(df.columns)}")
    lab_c = _find_col(df, [label_col.lower()])
    if lab_c is None:
        raise ValueError(f"Label column '{label_col}' not found; found {list(df.columns)}")
    raw = [combine_title_text(df[title_c].iloc[i] if title_c else None,
                              df[text_c].iloc[i] if text_c else None) for i in range(len(df))]
    out = pd.DataFrame({"text": raw, "label": df[lab_c].values})
    out = out[out["label"].notna()]
    out["label"] = normalize_labels(out["label"], label_map)
    return out


def drop_unusable_and_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove empty texts and duplicates (judged on CLEANED text, so near-identical
    copies can't land in both train and test). Also drops texts whose cleaned
    form appears with conflicting labels."""
    info = {"rows_in": len(df)}
    df = df.copy()
    df["_key"] = df["text"].map(clean_text)
    df = df[df["_key"] != ""]
    info["after_empty_removed"] = len(df)
    conflict = df.groupby("_key")["label"].nunique()
    conflict_keys = conflict[conflict > 1].index
    df = df[~df["_key"].isin(conflict_keys)]
    df = df.drop_duplicates("_key")
    info["rows_out"] = len(df)
    info["label_counts"] = {LABEL_NAMES[k]: int(v) for k, v in df["label"].value_counts().items()}
    return df.drop(columns="_key").reset_index(drop=True), info


def load_dataset(path=None, fake_path=None, real_path=None, label_col="label", label_map=None):
    """Either `path` (one CSV with a label column) or `fake_path`+`real_path`
    (two CSVs, e.g. Fake.csv / True.csv). Returns (DataFrame[text,label], info)."""
    if fake_path and real_path:
        f, r = pd.read_csv(fake_path), pd.read_csv(real_path)
        f["label"], r["label"] = FAKE, REAL
        frames = [prepare_frame(f, "label"), prepare_frame(r, "label")]
        df = pd.concat(frames, ignore_index=True)
    elif path:
        df = prepare_frame(pd.read_csv(path), label_col, label_map)
    else:
        raise ValueError("Provide --data CSV or both --fake and --real CSVs.")
    return drop_unusable_and_duplicates(df)
