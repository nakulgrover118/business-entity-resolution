"""Stage 4b: learned blocking filter (candidate pruning).

The two-pass TF-IDF blocking (blocking.py) is recall-oriented and returns ~62 candidates per Source-1
entity on test.  A small LightGBM on cheap pair features only -- blocking scores and ranks, IDF cosines
of the name / address vectors, the numba token alignment, house-number / city / state / legal-form
lookups, empty-address and source flags; no rapidfuzz, word tables or cross-encoders -- scores every
candidate (out of fold on train), and filter_pairs.py keeps only the pairs above a per-country
threshold, before the matcher: they are the candidates the matcher scores, written to candidate_pairs.tsv.

Test: ~62 -> 4.4 candidates / S1 (US 4.19, India 4.50, France 4.71); out-of-fold macro F0.5 change
-0.00003 (US) / -0.00008 (India).

Usage: python prune.py train | score
"""
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np

from config import cache_path
from model import fold_of
from pairs import pairs_dir
from records import split_countries

CHEAP = ["blk_score", "q_rank", "q_ncand", "q_gap", "q_second", "q_best", "b_rank", "b_ncand", "b_gap", "b_second",
         "b_best", "n_cos", "a_cos", "n_idf_q", "n_idf_b", "n_len_q", "n_len_b", "a_len_q", "a_len_b", "hnum_eq",
         "city_eq", "state_eq", "legal_same", "legal_rel", "legal_q_empty", "legal_b_empty", "q_addr_empty", "q_src",
         "n_cov_q", "n_cov_b", "n_soft_q", "n_soft_b", "n_unm_q_max", "n_unm_b_max", "n_exact_q", "n_first_sim",
         "a_cov_q", "a_cov_b", "a_soft_q", "a_soft_b", "num_cov_q", "num_cov_b", "hnum_rel", "hnum_sub", "hnum_lev"]
PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=3, num_threads=8)
# India test candidate lists are longer than in training, where the filter is less exact: lower threshold
THRESHOLD = {"US": 0.01, "India": 0.003, "France": 0.01}
SAMPLE = 1_500_000
CHUNK = 2_000_000


def _model_path(k):
    return cache_path(f"prune2_model_f{k}.txt")


def _cheap(d, rows=None):
    with open(os.path.join(d, "cols.json")) as fh:
        cols = json.load(fh)
    ix = [cols.index(n) for n in CHEAP]
    X = np.load(os.path.join(d, "X.npy"), mmap_mode="r")
    if rows is not None:
        return np.asarray(X[rows])[:, ix]
    return (np.asarray(X[a:a + CHUNK])[:, ix] for a in range(0, X.shape[0], CHUNK))


def train():
    t0 = time.time()
    rng = np.random.default_rng(0)
    folds = {c: fold_of(np.load(os.path.join(pairs_dir("train", c), "B.npy"))) for c in split_countries("train")}
    for k in (0, 1):
        Xs, ys = [], []
        for c, f in folds.items():
            d = pairs_dir("train", c)
            idx = np.flatnonzero(f == k)
            idx = np.sort(rng.choice(idx, min(SAMPLE, len(idx)), replace=False))
            Xs.append(_cheap(d, idx))
            ys.append(np.load(os.path.join(d, "y.npy"))[idx])
        m = lgb.train(PARAMS, lgb.Dataset(np.vstack(Xs), label=np.concatenate(ys)), num_boost_round=300)
        m.save_model(_model_path(k))
        print(f"  prune model fold {k} ({time.time() - t0:.0f}s)", flush=True)
    models = [lgb.Booster(model_file=_model_path(k)) for k in (0, 1)]
    for c, f in folds.items():  # out-of-fold scores for evaluation
        d = pairs_dir("train", c)
        s = np.empty(len(f), np.float32)
        a = 0
        for xc in _cheap(d):
            fc = f[a:a + len(xc)]
            out = np.empty(len(xc), np.float32)
            for k in (0, 1):
                sel = fc == 1 - k
                if sel.any():
                    out[sel] = models[k].predict(xc[sel])
            s[a:a + len(xc)] = out
            a += len(xc)
        np.save(os.path.join(d, "prune2_oof.npy"), s)


def score(split="test"):
    models = [lgb.Booster(model_file=_model_path(k)) for k in (0, 1)]
    for c in split_countries(split):
        d = pairs_dir(split, c)
        parts = [(models[0].predict(xc) + models[1].predict(xc)) / 2 for xc in _cheap(d)]
        s = np.concatenate(parts).astype(np.float32)
        np.save(os.path.join(d, "prune2_score.npy"), s)
        src = np.load(os.path.join(d, "src.npy"))
        print(f"  {split} {c}: {(s >= THRESHOLD.get(c, 0.01)).sum() / (src == 1).sum():.2f} candidates per S1 kept", flush=True)


if __name__ == "__main__":
    {"train": train, "score": score}[sys.argv[1]]()
