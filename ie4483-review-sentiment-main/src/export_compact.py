"""Run on the GPU machine after run_experiments.py: write a small text summary of all results.

Colab sessions can disconnect before large files are copied off, so this packs what the report needs
into ~6 KB of plain text (results/colab_export.txt):
  R|...  one line per run: stage, model, settings, best epoch, val accuracy, val macro-F1, time, params, per-epoch F1
  B|...  test predictions of the 5-fold ensemble as a hex bit string
  T|...  test probabilities that are not near 0 or 1 (0.03 < p < 0.97)
  O|...  out-of-fold probabilities for every misclassified or borderline training review
"""
import json

import numpy as np

from data import RESULTS, load_data

train, _ = load_data()
y = train["sentiments"].values
out = []
for line in (RESULTS / "runs.jsonl").read_text().splitlines():
    r = json.loads(line)
    hist = ",".join(f"{e['val_f1_macro']:.4f}" for e in r["history"])
    fold = -1 if r["fold"] is None else r["fold"]
    out.append(f"R|{r['stage']}|{r['model'].split('/')[-1]}|{r['lr']:g}|{r['max_len']}|{int(r['class_weight'])}|"
               f"{r['seed']}|{fold}|{r['best_epoch']}|{r['val_acc']:.4f}|{r['val_f1_macro']:.4f}|{r['seconds']:.0f}|"
               f"{r['n_params']}|{hist}")

z = np.load(RESULTS / "cv_probs.npz")
oof, p_test = z["oof"], z["p_test"]
bits = "".join("1" if p > 0.5 else "0" for p in p_test)
out.append(f"B|{int(bits, 2):x}|{len(bits)}")
out.append("T|" + ";".join(f"{i}:{p:.3f}" for i, p in enumerate(p_test) if 0.03 < p < 0.97))
out.append("O|" + ";".join(f"{i}:{p:.3f}" for i, p in enumerate(oof) if (p > 0.5) != (y[i] == 1) or 0.2 < p < 0.8))
(RESULTS / "colab_export.txt").write_text("\n".join(out))
print(f"wrote {sum(map(len, out)):,} characters to results/colab_export.txt")
