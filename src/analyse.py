"""Turn results/ into the tables, figures and examples used in the report.

Reads results/runs.jsonl, results/probs/*.npz and results/cv_probs.npz (written by run_experiments.py).
Writes results/report_tables.json, results/figures/*.png and results/error_examples.json.
"""
import json

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from data import RESULTS, load_data

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = RESULTS / "figures"
FIG.mkdir(exist_ok=True)
SHORT = {"distilbert-base-uncased": "DistilBERT", "bert-base-uncased": "BERT-base",
         "roberta-base": "RoBERTa-base", "microsoft/deberta-v3-base": "DeBERTa-v3-base"}


def probs_path(run_id):
    return RESULTS / "probs" / (run_id.replace("/", "_").replace("|", "__") + ".npz")


runs = pd.DataFrame(map(json.loads, (RESULTS / "runs.jsonl").read_text().splitlines()))
runs["name"] = runs["model"].map(SHORT)
tables = {}

# 1. encoder comparison, mean and std over 3 seeds
cmp_ = runs[runs.stage == "compare"]
t = (cmp_.groupby("name")
     .agg(f1_mean=("val_f1_macro", "mean"), f1_std=("val_f1_macro", "std"),
          acc_mean=("val_acc", "mean"), acc_std=("val_acc", "std"),
          sec_per_run=("seconds", "mean"), params=("n_params", "first"))
     .sort_values("f1_mean", ascending=False))
tables["compare"] = t.reset_index().to_dict(orient="records")
print(t.round(4), "\n")

# 2. tuning runs for the chosen encoder (seed 42), including the lr=2e-5 compare run as the baseline
best_model = cmp_.groupby("model")["val_f1_macro"].mean().idxmax()
tune = runs[(runs.model == best_model) & (runs.seed == 42) & runs.stage.isin(["compare", "tune"])]
tune = tune[["lr", "max_len", "class_weight", "best_epoch", "val_acc", "val_f1_macro", "seconds"]].sort_values(
    ["class_weight", "max_len", "lr"])
tables["tune"] = tune.to_dict(orient="records")
tables["best_model"] = best_model
print(tune.round(4), "\n")

# 3. cross-validation summary and per-fold scores
cv = json.loads((RESULTS / "cv_summary.json").read_text())
folds = runs[runs.stage == "cv"].sort_values("fold")
cv["fold_f1"] = folds["val_f1_macro"].round(4).tolist()
cv["fold_acc"] = folds["val_acc"].round(4).tolist()
cv["fold_seconds"] = folds["seconds"].round(0).tolist()
tables["cv"] = cv
print(json.dumps(cv, indent=2))

train, test = load_data()
y = train["sentiments"].values
z = np.load(RESULTS / "cv_probs.npz")
oof, p_test = z["oof"], z["p_test"]
tn, fp, fn, tp = confusion_matrix(y, oof > 0.5).ravel()
pr, rc, f1, _ = precision_recall_fscore_support(y, oof > 0.5, labels=[0, 1])
tables["cv_confusion"] = {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
                          "neg_precision": pr[0], "neg_recall": rc[0], "pos_precision": pr[1], "pos_recall": rc[1]}

# threshold check on out-of-fold probabilities (reported, not used for the submission)
grid = np.round(np.arange(0.1, 0.91, 0.05), 2)
tables["threshold_scan"] = {str(th): f1_score(y, oof > th, average="macro") for th in grid}

# 4. likely label errors: out-of-fold model very sure the label is wrong (confident-learning style)
susp = np.where(((oof > 0.95) & (y == 0)) | ((oof < 0.05) & (y == 1)))[0]
tables["suspect_labels"] = {"count": int(len(susp)), "share": float(len(susp) / len(y)),
                            "labelled_neg_pred_pos": int(((oof > 0.95) & (y == 0)).sum()),
                            "labelled_pos_pred_neg": int(((oof < 0.05) & (y == 1)).sum())}
f1_clean = f1_score(np.delete(y, susp), np.delete(oof, susp) > 0.5, average="macro")
tables["suspect_labels"]["oof_f1_without_suspects"] = f1_clean

# 5. test-set predictions: confidence profile and agreement with the TF-IDF baseline
tables["test"] = {"n": int(len(p_test)), "pred_pos": int((p_test > 0.5).sum()), "pred_neg": int((p_test <= 0.5).sum()),
                  "uncertain_0.2_0.8": int(((p_test > 0.2) & (p_test < 0.8)).sum())}
tf_path = RESULTS / "tfidf_test_probs.csv"
if tf_path.exists():
    p_tf = pd.read_csv(tf_path)["prob"].values
    tables["test"]["disagree_with_tfidf"] = int(((p_tf > 0.5) != (p_test > 0.5)).sum())
else:
    p_tf = np.full(len(p_test), np.nan)

examples = {
    "oof_errors": [{"label": int(y[i]), "p": float(oof[i]), "text": train.reviews[i]} for i in np.argsort(-np.abs(oof - y))[:40]],
    "test_uncertain": [{"row": int(i), "p": float(p_test[i]), "p_tfidf": float(p_tf[i]), "text": test.reviews[i]}
                       for i in np.where((p_test > 0.2) & (p_test < 0.8))[0]],
    "test_disagree": [{"row": int(i), "p": float(p_test[i]), "p_tfidf": float(p_tf[i]), "text": test.reviews[i]}
                      for i in np.where((p_tf > 0.5) != (p_test > 0.5))[0]],
}
(RESULTS / "error_examples.json").write_text(json.dumps(examples, indent=1))
(RESULTS / "report_tables.json").write_text(json.dumps(tables, indent=2, default=float))

# figure: validation macro-F1 per epoch for each encoder (seed 42)
fig, ax = plt.subplots(figsize=(6.4, 3.6))
styles = ["-", "--", "-.", ":"]
for (name, g), ls in zip(cmp_[cmp_.seed == 42].groupby("name"), styles):
    h = g.iloc[0]["history"]
    ax.plot([r["epoch"] for r in h], [r["val_f1_macro"] for r in h], ls, color="black", marker="o", label=name)
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation macro-F1")
ax.set_xticks([1, 2, 3])
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG / "epochs.png", dpi=200)

# figure: encoder comparison with seed spread
fig, ax = plt.subplots(figsize=(6.4, 3.0))
order = t.sort_values("f1_mean")
ax.barh(order.index, order["f1_mean"], xerr=order["f1_std"], color="0.75", edgecolor="black", capsize=4)
for i, v in enumerate(order["f1_mean"]):
    ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=9)
ax.set_xlim(0.80, 0.95)
ax.set_xlabel("Validation macro-F1 (mean of 3 seeds, bar = std)")
fig.tight_layout()
fig.savefig(FIG / "encoders.png", dpi=200)
print("figures written to", FIG)
