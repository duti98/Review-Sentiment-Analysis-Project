"""Shared data loading, seeding and the fixed train/validation split."""
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
SEED = 42


def set_seed(seed=SEED):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_data():
    train = pd.read_json(DATA / "train.json")
    test = pd.read_json(DATA / "test.json")
    return train, test


def load_split(val_size=0.1, seed=SEED):
    """Stratified 90/10 split used by every model, so validation scores are comparable."""
    train, test = load_data()
    tr, va = train_test_split(train, test_size=val_size, stratify=train["sentiments"], random_state=seed)
    return tr.reset_index(drop=True), va.reset_index(drop=True), test


def cv_folds(train, n_splits=5, seed=SEED):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(train["reviews"], train["sentiments"]))


RESULTS.mkdir(parents=True, exist_ok=True)
