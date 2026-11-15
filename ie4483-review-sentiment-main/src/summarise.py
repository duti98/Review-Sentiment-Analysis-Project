"""Print result tables from results/runs.jsonl. With --export, also print a base64 bundle of all results."""
import argparse
import base64
import gzip
import io
import json

import numpy as np
import pandas as pd

from data import RESULTS

ap = argparse.ArgumentParser()
ap.add_argument("--export", action="store_true")
args = ap.parse_args()

runs = pd.DataFrame(map(json.loads, (RESULTS / "runs.jsonl").read_text().splitlines()))
pd.set_option("display.width", 200)

cmp_ = runs[runs.stage == "compare"]
if len(cmp_):
    t = cmp_.groupby("model").agg(f1_mean=("val_f1_macro", "mean"), f1_std=("val_f1_macro", "std"),
                                  acc_mean=("val_acc", "mean"), acc_std=("val_acc", "std"),
                                  minutes=("seconds", lambda s: s.mean() / 60), params_m=("n_params", lambda s: s.iloc[0] / 1e6))
    print("Encoder comparison (3 seeds, 90/10 split)\n", t.sort_values("f1_mean", ascending=False).round(4), "\n")

tune = runs[runs.stage == "tune"]
if len(tune):
    print("Tuning (seed 42)\n", tune[["model", "lr", "max_len", "class_weight", "best_epoch", "val_acc", "val_f1_macro", "seconds"]].to_string(float_format=lambda x: f"{x:.4g}"), "\n")

cv = RESULTS / "cv_summary.json"
if cv.exists():
    print("Cross-validation\n", cv.read_text())

if args.export:
    bundle = {"runs": (RESULTS / "runs.jsonl").read_text()}
    if cv.exists():
        bundle["cv_summary"] = cv.read_text()
    arrays = {}
    for f in sorted((RESULTS / "probs").glob("*.npz")):
        z = np.load(f)
        arrays[f.stem + "::p_val"] = z["p_val"].astype(np.float16)
        arrays[f.stem + "::p_test"] = z["p_test"].astype(np.float16)
    if (RESULTS / "cv_probs.npz").exists():
        z = np.load(RESULTS / "cv_probs.npz")
        arrays["cv::oof"], arrays["cv::p_test"] = z["oof"].astype(np.float32), z["p_test"].astype(np.float32)
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    bundle["arrays_b64"] = base64.b64encode(buf.getvalue()).decode()
    blob = base64.b64encode(gzip.compress(json.dumps(bundle).encode())).decode()
    (RESULTS / "export_b64.txt").write_text(blob)
    print(f"export written: {len(blob):,} characters")
