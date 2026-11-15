"""GPU experiments for the transformer models.

Stages (run in order with --stage all):
  compare  4 pretrained encoders x 3 seeds on the fixed 90/10 split
  tune     learning rate, max length and class weighting for the best encoder
  cv       5-fold cross-validation of the final setting on all 7,401 reviews;
           the 5 fold models are averaged to predict the test set

Every run appends one line to results/runs.jsonl and saves its probabilities to results/probs/.
Finished runs are skipped, so the script can be restarted after a disconnect.
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from data import RESULTS, cv_folds, load_data, load_split
from transformer import train_eval

ENCODERS = ["distilbert-base-uncased", "bert-base-uncased", "roberta-base", "microsoft/deberta-v3-base"]
SEEDS = [42, 43, 44]
BASE = {"lr": 2e-5, "epochs": 3, "batch_size": 16, "max_len": 256, "class_weight": False}

RUNS = RESULTS / "runs.jsonl"
PROBS = RESULTS / "probs"
PROBS.mkdir(parents=True, exist_ok=True)


def run_id(stage, model, cfg, seed, fold=None):
    tag = f"{stage}|{model}|lr{cfg['lr']}|len{cfg['max_len']}|cw{int(cfg['class_weight'])}|ep{cfg['epochs']}|s{seed}"
    return tag if fold is None else f"{tag}|f{fold}"


def done():
    if not RUNS.exists():
        return {}
    return {r["run_id"]: r for r in map(json.loads, RUNS.read_text().splitlines())}


def record(rid, stage, model, cfg, seed, out, fold=None):
    row = {"run_id": rid, "stage": stage, "model": model, "seed": seed, "fold": fold, **cfg,
           **{k: out[k] for k in ["best_epoch", "val_acc", "val_f1_macro", "seconds", "n_params", "fp16", "history"]}}
    with RUNS.open("a") as f:
        f.write(json.dumps(row) + "\n")
    np.savez(PROBS / (rid.replace("/", "_").replace("|", "__") + ".npz"), p_val=out["p_val"], p_test=out["p_test"])
    return row


def run(stage, model, cfg, seed, tr, va, te, fold=None, select="best"):
    rid = run_id(stage, model, cfg, seed, fold)
    prev = done()
    if rid in prev:
        print(f"skip {rid}")
        return prev[rid]
    print(f"run  {rid}", flush=True)
    try:
        out = train_eval(model, tr, va, te, seed=seed, select=select, **cfg)
    except FloatingPointError as e:  # fp16 overflow: redo this run in full precision
        print(f"  {e}; retrying in fp32")
        out = train_eval(model, tr, va, te, seed=seed, select=select, fp16=False, **cfg)
    return record(rid, stage, model, cfg, seed, out, fold)


def best_model():
    rows = pd.DataFrame([r for r in done().values() if r["stage"] == "compare"])
    return rows.groupby("model")["val_f1_macro"].mean().idxmax()


def best_lr(model):
    rows = pd.DataFrame([r for r in done().values()
                         if r["model"] == model and r["seed"] == 42 and r["max_len"] == BASE["max_len"]
                         and not r["class_weight"] and r["stage"] in ("compare", "tune")])
    return float(rows.groupby("lr")["val_f1_macro"].mean().idxmax())


def stage_compare(models):
    tr, va, te = load_split()
    for model in models:
        for seed in SEEDS:
            run("compare", model, BASE, seed, tr, va, te)


def stage_tune(model):
    tr, va, te = load_split()
    for lr in [1e-5, 3e-5, 5e-5]:
        run("tune", model, dict(BASE, lr=lr), 42, tr, va, te)
    lr = best_lr(model)
    print(f"best learning rate for {model}: {lr}")
    run("tune", model, dict(BASE, lr=lr, max_len=128), 42, tr, va, te)
    run("tune", model, dict(BASE, lr=lr, max_len=512), 42, tr, va, te)
    run("tune", model, dict(BASE, lr=lr, class_weight=True), 42, tr, va, te)


def stage_cv(model, lr):
    train, te = load_data()
    cfg = dict(BASE, lr=lr)
    oof = np.zeros(len(train))
    test_probs = []
    for k, (i_tr, i_va) in enumerate(cv_folds(train)):
        tr, va = train.iloc[i_tr].reset_index(drop=True), train.iloc[i_va].reset_index(drop=True)
        row = run("cv", model, cfg, 42, tr, va, te, fold=k, select="last")
        z = np.load(PROBS / (row["run_id"].replace("/", "_").replace("|", "__") + ".npz"))
        oof[i_va] = z["p_val"]
        test_probs.append(z["p_test"])
    y = train["sentiments"].values
    p_test = np.mean(test_probs, axis=0)
    summary = {"model": model, "lr": lr, "oof_acc": accuracy_score(y, oof > 0.5),
               "oof_f1_macro": f1_score(y, oof > 0.5, average="macro"),
               "test_positive_rate": float((p_test > 0.5).mean())}
    print(json.dumps(summary, indent=2))
    (RESULTS / "cv_summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(RESULTS / "cv_probs.npz", oof=oof, p_test=p_test)
    pd.DataFrame({"reviews": te["reviews"], "predictions": (p_test > 0.5).astype(int)}).to_csv(
        RESULTS / "submission.csv", index=False)
    print("wrote results/submission.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["compare", "tune", "cv", "all"], default="all")
    ap.add_argument("--model", help="encoder for tune/cv (default: best from compare)")
    ap.add_argument("--lr", type=float, help="learning rate for cv (default: best from tune)")
    ap.add_argument("--models", nargs="+", default=ENCODERS)
    args = ap.parse_args()

    if args.stage in ("compare", "all"):
        stage_compare(args.models)
    model = args.model or best_model()
    print(f"encoder: {model}")
    if args.stage in ("tune", "all"):
        stage_tune(model)
    if args.stage in ("cv", "all"):
        stage_cv(model, args.lr or best_lr(model))
