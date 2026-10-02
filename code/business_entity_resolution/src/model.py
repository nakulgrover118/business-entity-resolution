"""Stage 5: two-stage LightGBM matcher.

Stage A: pairwise classifier on the pair features.
Stage B: same features + "competition" features derived from stage-A
probabilities (best competing S1 entity for the same S2/S3 record, rank and
mass of the candidate within its S1 entity's list, ...).

Training uses 2 folds split by Source-1 entity so that every training pair gets
an out-of-fold probability; stage-B features for training are built from those
out-of-fold stage-A probabilities.  At inference both fold models are averaged.
"""
import json
import os

import lightgbm as lgb
import numpy as np

from config import N_JOBS, SEED
from pairs import pairs_dir

PARAMS = dict(
    objective="binary",
    learning_rate=0.06,
    num_leaves=255,
    min_data_in_leaf=200,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    max_bin=255,
    num_threads=N_JOBS,
    verbose=-1,
    seed=SEED,
)

STAGEB_COLS = [
    "pA", "pA_q_max_other", "pA_q_margin", "pA_q_rank", "pA_q_n05", "pA_q_sum",
    "pA_b_rank", "pA_b_n05", "pA_b_sum", "pA_b_max_other", "pA_is_qmax", "pA_b_n05_same_src",
]


def load_pairs(split, country):
    d = pairs_dir(split, country)
    out = dict(
        B=np.load(os.path.join(d, "B.npy")),
        Q=np.load(os.path.join(d, "Q.npy")),
        S=np.load(os.path.join(d, "S.npy")),
        X=np.load(os.path.join(d, "X.npy"), mmap_mode="r"),
    )
    with open(os.path.join(d, "cols.json")) as fh:
        out["cols"] = json.load(fh)
    yp = os.path.join(d, "y.npy")
    if os.path.exists(yp):
        out["y"] = np.load(yp)
    return out


def fold_of(idx, n_folds=2):
    h = (idx.astype(np.uint64) * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)
    return ((h >> np.uint64(7)) % np.uint64(n_folds)).astype(np.int8)


def predict(model, X, chunk=2_000_000, cols_idx=None):
    out = np.empty(X.shape[0], np.float32)
    for s in range(0, X.shape[0], chunk):
        xc = np.asarray(X[s:s + chunk])
        if cols_idx is not None:
            xc = xc[:, cols_idx]
        out[s:s + chunk] = model.predict(xc, num_threads=N_JOBS)
    return out


def _group_stats(key, p, n):
    """per-group max, second max, sum, count>0.5 and rank of each element."""
    P = len(key)
    order = np.lexsort((-p, key))
    ks = key[order]
    ps = p[order]
    first = np.r_[True, ks[1:] != ks[:-1]]
    start = np.flatnonzero(first)
    grp = np.cumsum(first) - 1
    rank = np.empty(P, np.float32)
    rank[order] = np.arange(P) - start[grp]
    gmax = np.full(n, 0.0, np.float32)
    gmax[ks[start]] = ps[start]
    second = np.zeros(n, np.float32)
    idx2 = start + 1
    ends = np.r_[start[1:], P]
    has2 = idx2 < ends
    second[ks[start[has2]]] = ps[idx2[has2]]
    gsum = np.bincount(key, weights=p, minlength=n).astype(np.float32)
    n05 = np.bincount(key, weights=(p > 0.5), minlength=n).astype(np.float32)
    return rank, gmax, second, gsum, n05


def stage_b_features(B, Q, pA, n_records, src):
    rq, qmax, qsec, qsum, qn05 = _group_stats(Q, pA, n_records)
    rb, bmax, bsec, bsum, bn05 = _group_stats(B, pA, n_records)
    is_qmax = (rq == 0).astype(np.float32)
    q_max_other = np.where(rq == 0, qsec[Q], qmax[Q]).astype(np.float32)
    b_max_other = np.where(rb == 0, bsec[B], bmax[B]).astype(np.float32)
    # confident candidates of the same S1 entity coming from the same source as q
    key = B.astype(np.int64) * 4 + src[Q]
    n05_same = np.bincount(key, weights=(pA > 0.5), minlength=4 * n_records)[key] - (pA > 0.5)
    F = np.column_stack([
        pA, q_max_other, pA - q_max_other, rq, qn05[Q] - (pA > 0.5), qsum[Q] - pA,
        rb, bn05[B] - (pA > 0.5), bsum[B] - pA, b_max_other, is_qmax, n05_same,
    ]).astype(np.float32)
    return F


def sample_rows(fold_mask, B, max_rows, rng):
    """Sample whole S1 entities from the pairs in fold_mask up to ~max_rows pairs."""
    idx = np.flatnonzero(fold_mask)
    if len(idx) <= max_rows:
        return idx
    ents = np.unique(B[idx])
    keep_frac = max_rows / len(idx)
    chosen = ents[rng.random(len(ents)) < keep_frac]
    m = np.isin(B[idx], chosen)
    return idx[m]


def make_dataset(X, y, cat_idx, reference=None, weight=None):
    """Build and bin an lgb.Dataset right away so the raw float matrix can be freed."""
    cat = list(cat_idx) if cat_idx else "auto"
    ds = lgb.Dataset(X, label=y, weight=weight, free_raw_data=True, categorical_feature=cat, reference=reference,
                     params={"max_bin": PARAMS["max_bin"], "num_threads": N_JOBS, "verbose": -1})
    ds.construct()
    return ds


def train_datasets(dtr, dva, rounds=4000, params=None):
    p = dict(PARAMS)
    if params:
        p.update(params)
    return lgb.train(p, dtr, num_boost_round=rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(200)])
