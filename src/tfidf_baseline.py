"""TF-IDF + Logistic Regression baseline with 5-fold stratified CV grid search."""
import json
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from data import RESULTS, SEED, load_data

train, test = load_data()
X, y = train["reviews"], train["sentiments"].values

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, sublinear_tf=True, min_df=2)),
    ("clf", LogisticRegression(max_iter=2000, solver="liblinear")),
])
grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "clf__C": [1, 4, 16],
    "clf__class_weight": [None, "balanced"],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

t0 = time.time()
search = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=2)
search.fit(X, y)
fit_time = time.time() - t0

res = pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "std_test_score"]]
res = res.sort_values("mean_test_score", ascending=False)
print(res.to_string(index=False))
print("best:", search.best_params_)

# out-of-fold predictions for the best setting, so val metrics are comparable with the other models
oof = cross_val_predict(search.best_estimator_, X, y, cv=cv, n_jobs=2)
metrics = {
    "model": "tfidf_logreg",
    "best_params": {k: str(v) for k, v in search.best_params_.items()},
    "cv_acc": accuracy_score(y, oof),
    "cv_f1_macro": f1_score(y, oof, average="macro"),
    "grid_time_s": fit_time,
    "n_features": len(search.best_estimator_.named_steps["tfidf"].vocabulary_),
}
print(metrics)

test_pred = search.best_estimator_.predict(test["reviews"])
test_prob = search.best_estimator_.predict_proba(test["reviews"])[:, 1]
pd.DataFrame({"prob": test_prob, "pred": test_pred}).to_csv(RESULTS / "tfidf_test_probs.csv", index=False)
res.assign(params=res["params"].astype(str)).to_csv(RESULTS / "tfidf_grid.csv", index=False)
with open(RESULTS / "tfidf_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
