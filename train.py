"""Train, compare, select and save. Examples:
  python train.py --fake data/Fake.csv --real data/True.csv
  python train.py --data data/news.csv --label-col label
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import LABEL_NAMES, load_dataset
from src.model_training import SEED, run_experiments, select_best, split

p = argparse.ArgumentParser()
p.add_argument("--data"); p.add_argument("--fake"); p.add_argument("--real")
p.add_argument("--label-col", default="label")
p.add_argument("--label-map", help='JSON, e.g. \'{"FAKE":0,"REAL":1}\' (0=Fake, 1=Real)')
p.add_argument("--max-rows", type=int, help="random subsample for quick runs")
p.add_argument("--skip-rf", action="store_true", help="skip Random Forest (slow)")
p.add_argument("--cv", type=int, default=3)
p.add_argument("--max-features", type=int, default=20000)
a = p.parse_args()

df, info = load_dataset(a.data, a.fake, a.real, a.label_col,
                        json.loads(a.label_map) if a.label_map else None)
print("Data cleaning summary:", info)
if a.max_rows and len(df) > a.max_rows:
    df = df.sample(a.max_rows, random_state=SEED).reset_index(drop=True)
    info["subsampled_to"] = len(df)

X_tr, X_te, y_tr, y_te = split(df)
info.update(train_size=len(X_tr), test_size=len(X_te))
clfs = ("logreg", "nb") if a.skip_rf else ("logreg", "nb", "rf")
results, fitted, details = run_experiments(X_tr, X_te, y_tr, y_te, classifiers=clfs,
                                           cv=a.cv, max_features=a.max_features)
best = select_best(results)

out_m, out_r = Path("models"), Path("reports")
out_m.mkdir(exist_ok=True); out_r.mkdir(exist_ok=True)
joblib.dump(fitted[best], out_m / "fake_news_pipeline.joblib")
results.to_csv(out_r / "model_comparison.csv", index=False)
(out_r / "run_info.json").write_text(json.dumps(
    {"dataset": info, "selected_model": best, "selection_rule": "highest CV macro-F1 on train split",
     "confusion_matrix_labels": [LABEL_NAMES[0], LABEL_NAMES[1]],
     "selected_confusion_matrix": details[best]["confusion_matrix"],
     "selected_report": details[best]["report"]}, indent=2))
plt.figure(figsize=(4, 3.5))
sns.heatmap(details[best]["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
            xticklabels=list(LABEL_NAMES.values()), yticklabels=list(LABEL_NAMES.values()))
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(best); plt.tight_layout()
plt.savefig(out_r / "confusion_matrix.png", dpi=120)

print(results.drop(columns="best_params").round(4).to_string(index=False))
print(f"\nSelected (by train-CV): {best}\n{details[best]['report']}")
print("Saved models/fake_news_pipeline.joblib and reports/*")
