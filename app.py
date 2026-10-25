import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.prediction import DISCLAIMER, predict_news

ROOT = Path(__file__).parent
st.set_page_config(page_title="Fake News Detection", layout="centered")
st.title("Fake News Detection using NLP")
page = st.sidebar.radio("Section", ["Predict", "Model performance", "About"])

if page == "Predict":
    st.write("Paste a news headline or article. The saved model predicts whether it looks like "
             "**Fake** or **Real** news based on patterns in its training data.")
    text = st.text_area("News text", height=220)
    if st.button("Predict"):
        try:
            r = predict_news(text)
            (st.error if r["class_id"] == 0 else st.success)(f"Prediction: {r['label']}")
            if r["probability"] is not None:
                st.caption(f"Model probability for this class: {r['probability']:.2f} "
                           "(not a measure of truth)")
        except FileNotFoundError as e:
            st.warning(f"{e}")
        except ValueError as e:
            st.warning(str(e))
    st.info(DISCLAIMER)

elif page == "Model performance":
    cmp_f, info_f, img = (ROOT / "reports" / n for n in
                          ("model_comparison.csv", "run_info.json", "confusion_matrix.png"))
    if not cmp_f.exists():
        st.warning("No results yet. Run `python train.py ...` first.")
    else:
        info = json.loads(info_f.read_text())
        st.subheader("Dataset"); st.json(info["dataset"])
        st.subheader("Model comparison (held-out test set)")
        st.dataframe(pd.read_csv(cmp_f).drop(columns=["best_params"]))
        st.write(f"Selected model: **{info['selected_model']}** ({info['selection_rule']}). "
                 "Precision/recall/F1 'fake' treat Fake News as the positive class.")
        if img.exists():
            st.image(str(img))
        st.text(info["selected_report"])

else:
    st.markdown("""
**Bag of Words** counts how often each vocabulary word appears in a document (word order is ignored).
**TF-IDF** re-weights those counts so words that appear in almost every document count less.
Models compared: Logistic Regression, Multinomial Naive Bayes, Random Forest.

**Limitations:** the model learns writing-style patterns of one dataset; it does not check facts,
and performance on other news sources may be much lower than on the test split.
""")
