# Fake News Detection using NLP and Machine Learning

Classifies a news headline/article as **Fake News (0)** or **Real News (1)** using
Bag of Words / TF-IDF features and Logistic Regression, Multinomial Naive Bayes and Random Forest.

> Trained and evaluated on the supplied `Fake.csv` / `True.csv` (44,898 rows; looks like the ISOT-style
> Fake/Real news dataset - verify the original source and license before citing). Tests use tiny synthetic fixtures only.

## Actual results (held-out 20% test split, 7,765 articles, seed 42, 2-fold CV)
After cleaning: 38,825 usable articles (17,900 Fake, 20,925 Real); 6,073 rows removed (empty/duplicate).
| model | accuracy | precision (fake) | recall (fake) | F1 (fake) |
|---|---|---|---|---|
| bow+logreg | 0.9854 | 0.9838 | 0.9846 | 0.9842 |
| bow+nb | 0.9455 | 0.9323 | 0.9508 | 0.9415 |
| bow+rf | 0.9710 | 0.9762 | 0.9606 | 0.9683 |
| tfidf+logreg | 0.9916 | 0.9930 | 0.9888 | 0.9909 |
| tfidf+nb | 0.9420 | 0.9321 | 0.9430 | 0.9375 |
| tfidf+rf | 0.9706 | 0.9751 | 0.9609 | 0.9679 |

Selected by cross-validation on the training split: **tfidf+logreg**. Full numbers: `reports/model_comparison.csv`.
**Caution:** these scores are within this one dataset. The dataset has strong source artifacts (nearly all Real articles begin
with a "CITY (Reuters) -" tag; some Fake ones contain pic.twitter links). These are stripped in `clean_text`, but other style
differences between the two sources remain, so expect lower accuracy on news from other sources.
Settings: unigram features (max 20,000), Random Forest 50 trees (kept small for speed on a single CPU core).

## Dataset format (you must supply it; source not verified here)
Either
- one CSV with `title` and/or `text` columns plus a label column (default name `label`;
  values like FAKE/REAL, fake/real, true/false, 0/1 with **0=Fake, 1=Real**), or
- two CSVs (e.g. `Fake.csv`, `True.csv`) with `title`/`text` columns.

Other columns (subject, date, ...) are ignored because they can leak the label.
If your dataset uses 1 = fake, pass `--label-map '{"fake":0,"real":1}'` style mapping explicitly.
Put files in `data/` (CSV files are git-ignored).

## How it works
1. **Load/clean** (`src/data_loader.py`): title+text combined, labels normalised, empty rows removed,
   duplicates removed on *cleaned* text (and texts with conflicting labels dropped), so copies can't sit in both train and test.
2. **Preprocess** (`src/preprocessing.py`): lowercase, strip HTML/URLs/punctuation. Stop words are kept so negations ("not") survive.
   The same function is embedded in the vectorizer, so training and prediction behave identically.
3. **Bag of Words** (`CountVectorizer`): each document becomes word counts over a vocabulary learned from training text; word order is mostly ignored.
4. **TF-IDF** (`TfidfVectorizer`): term frequency × inverse document frequency, so words common to nearly all documents get less weight.
5. **Models**: Logistic Regression (linear decision function), Multinomial Naive Bayes (Bayes' theorem with feature independence), Random Forest (ensemble of trees; slow on sparse text, not assumed best).
6. **No leakage**: 80/20 stratified split (seed 42) first. Vectorizer + classifier are one `Pipeline`, so the vocabulary/IDF are fitted only on training data (and only on training folds in CV).
7. **Tuning/selection**: small `GridSearchCV` (3-fold stratified, macro-F1) on the **train split only**. The final model is chosen by CV score, **not** by test score. The test set is used once for reporting.
8. **Metrics**: accuracy, precision/recall/F1 with **Fake News as the positive class**, macro-F1, confusion matrix, classification report.
   Accuracy alone is not enough; very high scores should be checked for duplicates and source artifacts (e.g. news-agency tags in the text).

## Install and run
```bash
pip install -r requirements.txt
pytest -q

# train (choose one)
python train.py --fake data/Fake.csv --real data/True.csv
python train.py --data data/news.csv --label-col label
# optional: --skip-rf  --max-rows 10000  --cv 3  --max-features 20000

python predict.py "Parliament passes annual budget after long debate"
streamlit run app.py
```
Outputs: `models/fake_news_pipeline.joblib`, `reports/model_comparison.csv`, `reports/run_info.json`, `reports/confusion_matrix.png`.
Python: `from src.prediction import predict_news; predict_news("text ...")` returns label, probability (if supported) and a disclaimer.
Only load `.joblib` files you created yourself (pickle can execute code).

## Structure
`train.py`, `predict.py`, `app.py`, `src/{data_loader,preprocessing,model_training,prediction}.py`, `tests/test_pipeline.py`

## Limitations
- Predicts writing patterns of the training data; it does **not** fact-check. Probability is not truth.
- Scores on a test split of the same dataset usually overstate performance on other sources/time periods.
- Bag-of-words ignores context; English only.

## Future improvements
Group/source-aware or time-based splits, evaluation on a second independent dataset, probability calibration, error analysis.
