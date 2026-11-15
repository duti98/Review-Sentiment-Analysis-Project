"""Rebuild the report tables, figures, error examples and submission.csv from results/colab_export.txt.

Use this when only the compact export (see export_compact.py) came back from the GPU machine.
Writes results/runs_summary.jsonl, report_tables.json, error_examples.json, figures/*.png and submission.csv.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)
train = pd.read_json(REPO / "data/train.json")
test = pd.read_json(REPO / "data/test.json")
y = train["sentiments"].values

lines = (RES / "colab_export.txt").read_text().splitlines()
SHORT = {"distilbert-base-uncased": "DistilBERT", "bert-base-uncased": "BERT-base",
         "roberta-base": "RoBERTa-base", "deberta-v3-base": "DeBERTa-v3-base"}
FULL = {"deberta-v3-base": "microsoft/deberta-v3-base"}
runs = []
for l in lines:
    if not l.startswith("R|"):
        continue
    _, stage, model, lr, ml, cw, seed, fold, be, acc, f1, sec, npar, hist = l.split("|")
    runs.append({"stage": stage, "model": FULL.get(model, model), "name": SHORT[model], "lr": float(lr),
                 "max_len": int(ml), "class_weight": bool(int(cw)), "seed": int(seed),
                 "fold": None if fold == "-1" else int(fold), "best_epoch": int(be), "val_acc": float(acc),
                 "val_f1_macro": float(f1), "seconds": float(sec), "n_params": int(npar),
                 "history_f1": [float(x) for x in hist.split(",")]})
(RES / "runs_summary.jsonl").write_text("\n".join(json.dumps(r) for r in runs) + "\n")
df = pd.DataFrame(runs)

# submission
bline = next(l for l in lines if l.startswith("B|"))
_, hx, n = bline.split("|")
bits = bin(int(hx, 16))[2:].zfill(int(n))
pred = np.array([int(b) for b in bits])
assert len(pred) == len(test) == 1851
sub = pd.DataFrame({"reviews": test["reviews"], "predictions": pred})
sub.to_csv(RES / "submission.csv", index=False)
sub.to_csv(REPO / "submission.csv", index=False)

# test probabilities: exact for 0.03 < p < 0.97, otherwise known only to be confident
T = {int(k): float(v) for k, v in (x.split(":") for x in next(l for l in lines if l.startswith("T|"))[2:].split(";"))}
for i, p in T.items():
    assert (p > 0.5) == bool(pred[i]), i
# out-of-fold: every misclassified review plus every review with 0.2 < p < 0.8
O = {int(k): float(v) for k, v in (x.split(":") for x in next(l for l in lines if l.startswith("O|"))[2:].split(";"))}
wrong = {i: p for i, p in O.items() if (p > 0.5) != (y[i] == 1)}
oof_pred = y.copy()
for i in wrong:
    oof_pred[i] = 1 - y[i]
acc = (oof_pred == y).mean()
f1 = f1_score(y, oof_pred, average="macro")
tn = int(((y == 0) & (oof_pred == 0)).sum()); fp = int(((y == 0) & (oof_pred == 1)).sum())
fn = int(((y == 1) & (oof_pred == 0)).sum()); tp = int(((y == 1) & (oof_pred == 1)).sum())
susp = [i for i, p in wrong.items() if (p > 0.95 and y[i] == 0) or (p < 0.05 and y[i] == 1)]
keep = np.setdiff1d(np.arange(len(y)), susp)
f1_clean = f1_score(y[keep], oof_pred[keep], average="macro")

cmp_ = df[df.stage == "compare"]
t = (cmp_.groupby("name").agg(f1_mean=("val_f1_macro", "mean"), f1_std=("val_f1_macro", "std"),
                              acc_mean=("val_acc", "mean"), acc_std=("val_acc", "std"),
                              sec_per_run=("seconds", "mean"), params=("n_params", "first"))
     .sort_values("f1_mean", ascending=False))
best_model = "microsoft/deberta-v3-base"
tune = df[(df.model == best_model) & (df.seed == 42) & df.stage.isin(["compare", "tune"])]
tune = tune[["lr", "max_len", "class_weight", "best_epoch", "val_acc", "val_f1_macro", "seconds"]].sort_values(
    ["class_weight", "max_len", "lr"])
folds = df[df.stage == "cv"].sort_values("fold")
p_tf = pd.read_csv(RES / "tfidf_test_probs.csv")["prob"].values
tables = {
    "compare": t.reset_index().to_dict(orient="records"),
    "tune": tune.to_dict(orient="records"),
    "best_model": best_model,
    "cv": {"model": best_model, "lr": 3e-5, "oof_acc": acc, "oof_f1_macro": f1,
           "fold_f1": folds.val_f1_macro.tolist(), "fold_acc": folds.val_acc.tolist(),
           "fold_seconds": folds.seconds.tolist()},
    "cv_confusion": {"TN": tn, "FP": fp, "FN": fn, "TP": tp,
                     "neg_precision": tn / (tn + fn), "neg_recall": tn / (tn + fp),
                     "pos_precision": tp / (tp + fp), "pos_recall": tp / (tp + fn)},
    "suspect_labels": {"count": len(susp), "share": len(susp) / len(y),
                       "labelled_neg_pred_pos": sum(1 for i in susp if y[i] == 0),
                       "labelled_pos_pred_neg": sum(1 for i in susp if y[i] == 1),
                       "oof_f1_without_suspects": f1_clean},
    "test": {"n": 1851, "pred_pos": int(pred.sum()), "pred_neg": int((1 - pred).sum()),
             "uncertain_0.2_0.8": sum(1 for p in T.values() if 0.2 < p < 0.8),
             "not_confident_0.03_0.97": len(T),
             "disagree_with_tfidf": int(((p_tf > 0.5).astype(int) != pred).sum())},
    "threshold_scan": {"0.5": f1},
}
(RES / "report_tables.json").write_text(json.dumps(tables, indent=2, default=float))
print(json.dumps({k: tables[k] for k in ["cv", "cv_confusion", "suspect_labels", "test"]}, indent=1, default=float))
print(t.round(4))
print(tune)

# examples for (e)
ex = {
    "oof_wrong": [{"i": i, "label": int(y[i]), "p": p, "text": train.reviews[i][:400]} for i, p in sorted(wrong.items(), key=lambda kv: -abs(kv[1] - y[kv[0]]))],
    "test_uncertain": [{"row": i, "p": p, "p_tfidf": float(p_tf[i]), "text": test.reviews[i][:400]} for i, p in sorted(T.items()) if 0.2 < p < 0.8],
    "test_disagree": [{"row": int(i), "pred": int(pred[i]), "p": T.get(int(i)), "p_tfidf": float(p_tf[i]), "text": test.reviews[i][:400]}
                      for i in np.where((p_tf > 0.5).astype(int) != pred)[0]],
}
(RES / "error_examples.json").write_text(json.dumps(ex, indent=1))

# figures (black and white)
fig, ax = plt.subplots(figsize=(6.4, 3.4))
styles = {"DistilBERT": ":", "BERT-base": "-.", "RoBERTa-base": "--", "DeBERTa-v3-base": "-"}
for name, g in cmp_.groupby("name"):
    h = np.array(g.history_f1.tolist()).mean(axis=0)
    ax.plot([1, 2, 3], h, styles[name], color="black", marker="o", label=name)
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation macro-F1 (mean of 3 seeds)")
ax.set_xticks([1, 2, 3]); ax.legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout(); fig.savefig(FIG / "epochs.png", dpi=200)

fig, ax = plt.subplots(figsize=(6.4, 2.8))
order = t.sort_values("f1_mean")
ax.barh(order.index, order["f1_mean"], xerr=order["f1_std"], color="0.8", edgecolor="black", capsize=4)
ax.axvline(0.875, color="black", ls="--", lw=1)
ax.text(0.8745, -0.45, "TF-IDF baseline 0.875", fontsize=8.5, va="center", ha="right")
for i, (v, sd) in enumerate(zip(order["f1_mean"], order["f1_std"])):
    ax.text(v + sd + 0.003, i, f"{v:.3f}", va="center", fontsize=9)
ax.set_xlim(0.85, 0.96)
ax.set_xlabel("Validation macro-F1 (mean of 3 seeds, error bar = std)")
fig.tight_layout(); fig.savefig(FIG / "encoders.png", dpi=200)
print("done")
