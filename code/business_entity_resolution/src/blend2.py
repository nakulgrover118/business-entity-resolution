"""Stage 9a: LightGBM blender over the hard pairs (used by final.py and expand.py).

Inputs per hard pair (all out-of-fold on train):
  * stage-B probability of the v3 model (or of the seed bag, bag.py) and its margin to the record's
    best competitor,
  * one or more cross-encoder scores (rounds) + their per-record margins,
  * optionally raw pair features ("base", "base2" = more features + the signed house-number offset).
The hard set of the *last* cross-encoder round defines which pairs are re-scored;
scores of other rounds are joined by pair index (NaN when a pair was not scored).

Usage:
  python blend2.py eval CONFIG      out-of-fold macro F0.5 of the blend
CONFIG: comma list of inputs among  ce1, ce2, ce3, ce4, ce5, base, base2
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq

from config import cache_path
from decision import argmax_per_query, select_matches
from evaluate import f05_per_entity
from pairs import pairs_dir
from records import split_countries

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=7)
CE_DIRS = {"ce1": cache_path("crossenc"), "ce2": cache_path("crossenc_r2"), "ce3": cache_path("crossenc_r3"),
           "ce4": cache_path("crossenc_r4"), "ce5": cache_path("crossenc_r5")}
BASE_COLS = ["hx_min", "te_x_max", "legal_rel", "hnum_rel", "a_cos", "n_cos", "q_src", "q_addr_empty"]
# "base2": more pair features + the signed house-number offset (true copies change digits both ways,
# generator decoys shift the S1 number up by K in KSET)
BASE2_COLS = ["n_soft_q", "n_soft_b", "a_soft_q", "a_soft_b", "r_full", "r_tset", "r_a_tset", "r_concat", "legal_same",
              "hnum_eq", "hnum_lev", "hnum_sub", "num_cov_q", "num_cov_b", "city_eq", "n_unm_q_max", "n_unm_b_max",
              "n_first_sim", "q_rank", "b_rank", "q_ncand", "blk_score", "blk_expanded"]
KSET = np.array([3, 4, 5, 7, 9, 11, 13, 21])
W = cache_path("..")


def _logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _other_max(v, q):
    o = np.lexsort((-v, q))
    qs = q[o]
    f = np.r_[True, qs[1:] != qs[:-1]]
    st = np.flatnonzero(f)
    g = np.cumsum(f) - 1
    best = v[o][st][g]
    sec = np.full(len(st), np.nan)
    has2 = np.r_[st[1:], len(v)] - st > 1
    sec[has2] = v[o][st[has2] + 1]
    isbest = (np.arange(len(v)) - st[g]) == 0
    res = np.empty(len(v))
    res[o] = np.where(isbest, sec[g], best)
    return res


def v3_prob(split, c, tag=None):
    """Stage-B probabilities of the tree model.  tag selects a trained variant ("bag" = seed bag, see bag.py);
    default: ER_V3_TAG or the original model "v3"."""
    tag = tag or os.environ.get("ER_V3_TAG", "v3")
    d = pairs_dir(split, c)
    return np.load(os.path.join(d, f"pB_{tag}.npy" if split == "train" else f"pB_{tag}_test.npy"))


def features(split, c, cfg, v3_tag=None):
    d = pairs_dir(split, c)
    Q = np.load(os.path.join(d, "Q.npy"))
    last = next(ce for ce in ("ce5", "ce4", "ce3", "ce2", "ce1") if ce in cfg)
    t = pq.read_table(os.path.join(CE_DIRS[last], f"hard_{split}_{c}.parquet"),
                      columns=["pair", "fold"] + (["y"] if split == "train" else []))
    pair = t.column("pair").to_numpy()
    q = Q[pair]
    lp = _logit(v3_prob(split, c, v3_tag)[pair])
    cols, names = [lp, lp - _other_max(lp, q)], ["lp_v3", "lp_margin_q"]
    for ce in ("ce1", "ce2", "ce3", "ce4", "ce5"):
        if ce not in cfg:
            continue
        tt = pq.read_table(os.path.join(CE_DIRS[ce], f"hard_{split}_{c}.parquet"), columns=["pair"])
        sc = np.load(os.path.join(CE_DIRS[ce], f"ce_{split}_{c}.npy"))
        full = np.full(len(Q), np.nan)
        full[tt.column("pair").to_numpy()] = sc
        v = full[pair]
        vv = np.where(np.isnan(v), -99.0, v)
        m = v - _other_max(vv, q)
        cols += [v, np.where(np.isnan(v), np.nan, m)]
        names += [ce, f"{ce}_margin_q"]
    if "base" in cfg:
        with open(os.path.join(d, "cols.json")) as fh:
            xc = json.load(fh)
        X = np.load(os.path.join(d, "X.npy"), mmap_mode="r")
        XE = np.load(os.path.join(d, "XE.npy"), mmap_mode="r")
        order = np.argsort(pair)
        rows = np.asarray(X[pair[order]])
        rows_e = np.asarray(XE[pair[order]])
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        from wordte import WORD_COLS
        for n in BASE_COLS:
            if n in xc:
                cols.append(rows[inv, xc.index(n)])
            else:
                cols.append(rows_e[inv, WORD_COLS.index(n)])
            names.append(n)
        if "base2" in cfg:
            for n in BASE2_COLS:
                if n in xc:
                    cols.append(rows[inv, xc.index(n)])
                    names.append(n)
            hv = np.load(os.path.join(d, "hnum_val.npy"))
            bb = np.load(os.path.join(d, "B.npy"))[pair]
            qq = Q[pair]
            off = np.where((hv[qq] >= 0) & (hv[bb] >= 0), hv[qq] - hv[bb], np.nan).astype(np.float64)
            cols += [np.clip(off, -1000, 1000), np.isin(off, KSET).astype(np.float64), np.isin(-off, KSET).astype(np.float64)]
            names += ["hnum_offset", "hnum_plusK", "hnum_minusK"]
    Xf = np.column_stack(cols).astype(np.float32)
    y = t.column("y").to_numpy() if split == "train" else None
    return pair, t.column("fold").to_numpy(), Xf, y, names


def _decide(B, Q, p):
    keep = argmax_per_query(B, Q, p)
    ki = np.flatnonzero(keep)
    s = select_matches(B[ki], Q[ki], p[ki])
    return B[ki[s]], Q[ki[s]]


def _f05(d, pb, pq_):
    ts = np.load(os.path.join(d, "true_s1.npy"))
    src = np.load(os.path.join(d, "src.npy"))
    n = len(src)
    tp = np.bincount(pb, weights=(ts[pq_] == pb), minlength=n)
    npred = np.bincount(pb, minlength=n)
    nt = np.bincount(ts[ts >= 0], minlength=n)
    s1 = np.flatnonzero(src == 1)
    return f05_per_entity(tp[s1], npred[s1], nt[s1])


def fit(cfg, verbose=True, return_oof=False, v3_tag=None, params=None):
    tr = {c: features("train", c, cfg, v3_tag) for c in split_countries("train")}
    names = next(iter(tr.values()))[4]
    X = np.vstack([v[2] for v in tr.values()])
    y = np.concatenate([v[3] for v in tr.values()])
    f = np.concatenate([v[1] for v in tr.values()])
    models, oof = [], np.empty(len(y))
    for k in (0, 1):
        m = lgb.train(params or PARAMS, lgb.Dataset(X[f == k], label=y[f == k]), num_boost_round=500)
        oof[f != k] = m.predict(X[f != k])
        models.append(m)
    o, allF, oofs = 0, [], {}
    for c, (pair, fold, Xc, yc, _) in tr.items():
        d = pairs_dir("train", c)
        B = np.load(os.path.join(d, "B.npy"))
        Q = np.load(os.path.join(d, "Q.npy"))
        p = v3_prob("train", c, v3_tag).astype(np.float64)
        p[pair] = oof[o:o + len(pair)]
        o += len(pair)
        oofs[c] = p
        F = _f05(d, *_decide(B, Q, p))
        allF.append(F)
        if verbose:
            print(f"  [{','.join(cfg)}] train {c}: {F.mean():.5f}", flush=True)
    tot = np.concatenate(allF).mean()
    imp = models[0].feature_importance("gain")
    if verbose:
        print(f"  [{','.join(cfg)}] train ALL: {tot:.5f}   gain: " +
              ", ".join(f"{n}={g:.3f}" for n, g in zip(names, imp / imp.sum())), flush=True)
    return (models, tot, oofs) if return_oof else (models, tot)


if __name__ == "__main__":
    # python blend2.py eval CONFIG  -> out-of-fold macro F0.5 of the blend (ER_V3_TAG=bag for the bagged v3)
    fit(sys.argv[2].split(","))
