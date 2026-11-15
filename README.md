# Sentiment classification of product reviews

Artificial Intelligence and Data Mining Project, NTU.

Binary sentiment classification (0 = negative, 1 = positive) of 7,401 labelled product reviews, with predictions for 1,851 unlabelled test reviews. We compare three feature formats (TF-IDF, GloVe word vectors, pretrained transformer encoders) and submit a fine-tuned transformer ensemble.

## Results

| Model | Feature format | Validation macro-F1 | Validation accuracy |
|---|---|---|---|
| Logistic regression | TF-IDF, unigrams + bigrams | 0.875 | 93.7% |
| BiLSTM | GloVe-100 word vectors | 0.852 | 92.2% |
| DistilBERT (3 seeds) | contextual subwords | 0.900 ± 0.012 | 95.0% |
| BERT-base (3 seeds) | contextual subwords | 0.917 ± 0.008 | 95.9% |
| RoBERTa-base (3 seeds) | contextual subwords | 0.925 ± 0.001 | 96.3% |
| DeBERTa-v3-base (3 seeds) | contextual subwords | **0.934 ± 0.004** | **96.7%** |

Validation set: a fixed stratified 10% of train.json (741 reviews). The submitted model is a 5-fold DeBERTa-v3-base ensemble (lr 3e-5, max length 256, 3 epochs): out-of-fold accuracy 96.7%, macro-F1 0.935 on all 7,401 training reviews, against 92.8% / 0.851 for TF-IDF on the same folds. Predictions are in `submission.csv`.

## Layout

```
data/                 train.json, test.json (course-provided)
src/data.py           loading, seeding, the fixed 90/10 split and the 5 CV folds
src/tfidf_baseline.py TF-IDF + logistic regression, 5-fold grid search (CPU, ~15 s)
src/bilstm.py         BiLSTM on GloVe-100 embeddings (CPU, ~10 min)
src/transformer.py    fine-tuning loop for any Hugging Face encoder (dynamic padding, fp16 on GPU)
src/run_experiments.py encoder comparison, tuning and 5-fold CV ensemble (GPU)
src/summarise.py      result tables from results/runs.jsonl
src/analyse.py        report tables, figures and error examples
src/draw_architecture.py  architecture figure for the report
src/export_compact.py  packs GPU results into a small text file (results/colab_export.txt)
src/rebuild_from_export.py  rebuilds tables, figures and submission.csv from that file
notebooks/            Colab runner for the GPU experiments
results/              run log (runs_summary.jsonl), metrics, figures, error examples
submission.csv        final test predictions (reviews, predictions)
```

## Running it

```bash
pip install -r requirements.txt
cd src
python tfidf_baseline.py
python bilstm.py                     # add --emb random for the random-embedding comparison
python run_experiments.py --stage all  # needs a GPU; about 1.5 h on a Colab T4
python analyse.py
```

After the GPU run, `python export_compact.py` (on the GPU machine) and `python rebuild_from_export.py` (anywhere) recreate every table and figure in the report from a 6 KB text file. That file for our run is committed as `results/colab_export.txt`, so the report numbers can be checked without a GPU.

`run_experiments.py` appends every finished run to `results/runs.jsonl` and skips runs that are already there, so it can be restarted after a disconnect. Set `RESULTS_DIR` to write results somewhere else (for example a mounted Google Drive folder).

On Colab, open `notebooks/colab_gpu_experiments.ipynb`, set the runtime to T4 GPU, and run all cells.

All randomness we control is seeded (42 for the split and folds; 42, 43 and 44 for the encoder comparison). Mixed-precision GPU training is not bit-for-bit repeatable, so reruns land close to, not exactly on, our numbers.

## Group

Hong Si Wei Nicholas, Gabriel d'Hauteville
