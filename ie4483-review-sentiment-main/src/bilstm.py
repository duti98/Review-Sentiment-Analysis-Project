"""BiLSTM + linear head on GloVe-100 word embeddings."""
import argparse
import json
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from data import RESULTS, load_split, set_seed

p = argparse.ArgumentParser()
p.add_argument("--emb", default="glove", choices=["glove", "random"])
p.add_argument("--max_len", type=int, default=200)
p.add_argument("--hidden", type=int, default=128)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--dropout", type=float, default=0.3)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--threads", type=int, default=1)
p.add_argument("--tag", default="bilstm_glove")
args = p.parse_args()

set_seed(args.seed)
torch.set_num_threads(args.threads)
tr, va, te = load_split()


def tokenize(s):
    return re.findall(r"[a-z0-9']+|[!?.,]", s.lower())


counts = Counter(w for s in tr["reviews"] for w in tokenize(s))
itos = ["<pad>", "<unk>"] + [w for w, c in counts.items() if c >= 2]
stoi = {w: i for i, w in enumerate(itos)}


def encode(texts):
    ids = np.zeros((len(texts), args.max_len), dtype=np.int64)
    lens = np.zeros(len(texts), dtype=np.int64)
    for i, s in enumerate(texts):
        t = [stoi.get(w, 1) for w in tokenize(s)][: args.max_len]
        ids[i, : len(t)] = t
        lens[i] = max(len(t), 1)
    return torch.from_numpy(ids), torch.from_numpy(lens)


emb_dim = 100
emb_matrix = np.random.normal(0, 0.1, (len(itos), emb_dim)).astype(np.float32)
emb_matrix[0] = 0
hit = 0
if args.emb == "glove":
    from gensim.models import KeyedVectors
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("fse/glove-wiki-gigaword-100", "glove-wiki-gigaword-100.model")
    hf_hub_download("fse/glove-wiki-gigaword-100", "glove-wiki-gigaword-100.model.vectors.npy")
    kv = KeyedVectors.load(path)
    for w, i in stoi.items():
        if w in kv.key_to_index:
            emb_matrix[i] = kv[w]; hit += 1
    print(f"GloVe coverage {hit}/{len(itos)}")


class BiLSTMClassifier(nn.Module):
    def __init__(self, emb, hidden, dropout):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.from_numpy(emb), freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(emb.shape[1], hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(4 * hidden, 2)  # [last fwd, last bwd] + mean pool

    def forward(self, ids, lens):
        x = self.drop(self.emb(ids))
        packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, (h, _) = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        mask = (torch.arange(out.size(1))[None, :] < lens[:, None]).unsqueeze(-1)
        mean = (out * mask).sum(1) / lens[:, None]
        feat = torch.cat([h[0], h[1], mean], dim=1)
        return self.fc(self.drop(feat))


Xtr, Ltr = encode(tr["reviews"]); ytr = torch.tensor(tr["sentiments"].values)
Xva, Lva = encode(va["reviews"]); yva = va["sentiments"].values
Xte, Lte = encode(te["reviews"])
g = torch.Generator().manual_seed(args.seed)
train_dl = DataLoader(TensorDataset(Xtr, Ltr, ytr), batch_size=args.bs, shuffle=True, generator=g)

model = BiLSTMClassifier(emb_matrix, args.hidden, args.dropout)
# class weights (inverse frequency) because only ~15% of reviews are negative
w = torch.tensor(len(ytr) / (2 * np.bincount(ytr.numpy())), dtype=torch.float32)
loss_fn = nn.CrossEntropyLoss(weight=w)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)


@torch.no_grad()
def predict_probs(X, L):
    model.eval()
    return torch.cat([torch.softmax(model(X[i:i + 256], L[i:i + 256]), -1)[:, 1]
                      for i in range(0, len(X), 256)]).numpy()


history, best_f1, bad, t0 = [], -1, 0, time.time()
for ep in range(1, args.epochs + 1):
    model.train(); tot = 0.0
    for ids, lens, yb in train_dl:
        loss = loss_fn(model(ids, lens), yb)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        tot += loss.item()
    vp = predict_probs(Xva, Lva)
    rec = {"epoch": ep, "train_loss": tot / len(train_dl), "val_acc": accuracy_score(yva, vp > 0.5),
           "val_f1_macro": f1_score(yva, vp > 0.5, average="macro"), "elapsed_s": time.time() - t0}
    history.append(rec); print(rec, flush=True)
    if rec["val_f1_macro"] > best_f1:
        best_f1, bad = rec["val_f1_macro"], 0
        torch.save(model.state_dict(), RESULTS / f"{args.tag}_best.pt")
        pd.DataFrame({"prob": vp, "label": yva}).to_csv(RESULTS / f"{args.tag}_val_probs.csv", index=False)
    else:
        bad += 1
        if bad >= 3:  # early stopping
            break

model.load_state_dict(torch.load(RESULTS / f"{args.tag}_best.pt"))
tp = predict_probs(Xte, Lte)
pd.DataFrame({"prob": tp, "pred": (tp > 0.5).astype(int)}).to_csv(RESULTS / f"{args.tag}_test_probs.csv", index=False)
json.dump({"args": vars(args), "history": history, "vocab": len(itos), "glove_hits": hit,
           "train_time_s": time.time() - t0, "n_params": sum(p.numel() for p in model.parameters())},
          open(RESULTS / f"{args.tag}_metrics.json", "w"), indent=2)
