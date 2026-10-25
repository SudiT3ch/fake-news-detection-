import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loader import drop_unusable_and_duplicates, normalize_labels, prepare_frame
from src.model_training import run_experiments, select_best, split
from src.prediction import predict_news
from src.preprocessing import clean_text


def toy_df(n=40):
    fake = ["shocking secret cure doctors hate click now", "aliens control the government says anonymous post"]
    real = ["parliament passes annual budget after debate", "central bank holds interest rates steady on tuesday"]
    rows = []
    for i in range(n):
        rows.append((f"{fake[i % 2]} number {i}", 0))
        rows.append((f"{real[i % 2]} report {i}", 1))
    return pd.DataFrame(rows, columns=["text", "label"])


def test_clean_text():
    assert clean_text("<b>Hello</b> WORLD!! http://x.com not") == "hello world not"
    assert clean_text(None) == "" and clean_text(float("nan")) == ""


def test_label_normalization():
    assert normalize_labels(pd.Series(["FAKE", "real", "True", "0"])).tolist() == [0, 1, 1, 0]
    with pytest.raises(ValueError):
        normalize_labels(pd.Series(["maybe"]))


def test_prepare_and_dedupe():
    raw = pd.DataFrame({"title": ["A b c", "A b c", None, "x y z"], "text": ["body", "body", "", "w"],
                        "subject": ["s"] * 4, "label": ["fake", "fake", "real", "real"]})
    df, info = drop_unusable_and_duplicates(prepare_frame(raw))
    assert list(df.columns) == ["text", "label"]
    assert info["rows_out"] == 2


def test_split_stratified_no_overlap():
    df = toy_df()
    Xtr, Xte, ytr, yte = split(df)
    assert set(Xtr).isdisjoint(set(Xte))
    assert abs(ytr.mean() - yte.mean()) < 0.05


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    df = toy_df()
    Xtr, Xte, ytr, yte = split(df)
    res, fitted, _ = run_experiments(Xtr, Xte, ytr, yte, cv=2, min_df=1)
    best = select_best(res)
    path = tmp_path_factory.mktemp("m") / "p.joblib"
    joblib.dump(fitted[best], path)
    return res, fitted, best, path


def test_all_six_models_trained(trained):
    res = trained[0]
    assert len(res) == 6 and res["accuracy"].between(0, 1).all()


def test_predict_format_and_loading(trained):
    _, _, _, path = trained
    r = predict_news("shocking secret cure doctors hate", model_path=path)
    assert r["label"] in ("Fake News", "Real News") and "disclaimer" in r
    assert r["probability"] is None or 0 <= r["probability"] <= 1


def test_invalid_input(trained):
    for bad in ["", "   ", None, "hi"]:
        with pytest.raises(ValueError):
            predict_news(bad, pipeline=trained[1]["tfidf+nb"])


def test_missing_model_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        predict_news("some valid news words", model_path=tmp_path / "nope.joblib")


def test_source_artifacts_removed():
    assert clean_text("WASHINGTON (Reuters) - The senate voted") == "the senate voted"
    assert "reuters" not in clean_text("said Reuters today") and "twitter" not in clean_text("look pic.twitter.com/abc123")
