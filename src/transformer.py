"""Fine-tuning a pretrained transformer encoder with a classification head.

Works on CPU or GPU. Reviews are tokenised once and padded per batch; batches are built from
reviews of similar length, which roughly halves the cost compared with padding everything.
"""
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from data import set_seed


class Reviews(Dataset):
    def __init__(self, tok, texts, labels, max_len):
        self.ids = tok(list(texts), truncation=True, max_length=max_len)["input_ids"]
        self.labels = labels if labels is not None else np.full(len(self.ids), -1)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i], int(self.labels[i])


class LengthBucketSampler(Sampler):
    """Shuffle, sort inside chunks of 50 batches, then shuffle the batch order."""

    def __init__(self, lengths, batch_size, seed):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        idx = self.rng.permutation(len(self.lengths))
        chunk = self.batch_size * 50
        batches = []
        for s in range(0, len(idx), chunk):
            part = idx[s:s + chunk]
            part = part[np.argsort(self.lengths[part], kind="stable")]
            batches += [part[k:k + self.batch_size].tolist() for k in range(0, len(part), self.batch_size)]
        for b in self.rng.permutation(len(batches)):
            yield batches[b]

    def __len__(self):
        return -(-len(self.lengths) // self.batch_size)


def make_collate(pad_id):
    def collate(batch):
        width = max(len(ids) for ids, _ in batch)
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for j, (x, _) in enumerate(batch):
            ids[j, :len(x)] = torch.tensor(x)
            mask[j, :len(x)] = 1
        return ids, mask, torch.tensor([y for _, y in batch])
    return collate


@torch.no_grad()
def predict(model, loader, device, fp16):
    model.eval()
    out = []
    for ids, mask, _ in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        out.append(torch.softmax(logits.float(), -1)[:, 1].cpu())
    return torch.cat(out).numpy()


def train_eval(model_name, train_df, val_df, test_df, lr=2e-5, epochs=3, batch_size=16, max_len=256,
               seed=42, class_weight=False, warmup=0.1, weight_decay=0.01, fp16=None, select="best", log=print):
    """Fine-tune and return validation and test probabilities.

    select="best" keeps the epoch with the best validation macro-F1 (used for model selection);
    select="last" keeps the final epoch (used in cross-validation, so the held-out fold is never used to choose).
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = device.type == "cuda" if fp16 is None else fp16
    tok = AutoTokenizer.from_pretrained(model_name)
    collate = make_collate(tok.pad_token_id)

    train_ds = Reviews(tok, train_df["reviews"], train_df["sentiments"].values, max_len)
    train_dl = DataLoader(train_ds, batch_sampler=LengthBucketSampler([len(x) for x in train_ds.ids], batch_size, seed),
                          collate_fn=collate)
    val_dl = DataLoader(Reviews(tok, val_df["reviews"], val_df["sentiments"].values, max_len), batch_size=64, collate_fn=collate)
    test_dl = DataLoader(Reviews(tok, test_df["reviews"], None, max_len), batch_size=64, collate_fn=collate)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).float().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total = len(train_dl) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(warmup * total), total)
    scaler = torch.amp.GradScaler(device.type, enabled=fp16)

    weight = None
    if class_weight:
        y = train_df["sentiments"].values
        weight = torch.tensor(len(y) / (2 * np.bincount(y)), dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

    y_val = val_df["sentiments"].values
    history, best = [], None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for ids, mask, yb in train_dl:
            ids, mask, yb = ids.to(device), mask.to(device), yb.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
                logits = model(input_ids=ids, attention_mask=mask).logits
            loss = loss_fn(logits.float(), yb)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss in epoch {ep}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()

        p_val = predict(model, val_dl, device, fp16)
        rec = {"epoch": ep, "train_loss": running / len(train_dl),
               "val_acc": accuracy_score(y_val, p_val > 0.5),
               "val_f1_macro": f1_score(y_val, p_val > 0.5, average="macro"),
               "seconds": time.time() - t0}
        history.append(rec)
        log(f"  epoch {ep}: loss {rec['train_loss']:.4f}  val acc {rec['val_acc']:.4f}  val F1 {rec['val_f1_macro']:.4f}  ({rec['seconds']:.0f}s)")
        if (select == "last" and ep == epochs) or (select == "best" and (best is None or rec["val_f1_macro"] > best["val_f1_macro"])):
            best = dict(rec, p_val=p_val, p_test=predict(model, test_dl, device, fp16))

    n_params = sum(p.numel() for p in model.parameters())
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"history": history, "best_epoch": best["epoch"], "val_acc": best["val_acc"],
            "val_f1_macro": best["val_f1_macro"], "p_val": best["p_val"], "p_test": best["p_test"],
            "seconds": time.time() - t0, "n_params": n_params, "fp16": fp16}
