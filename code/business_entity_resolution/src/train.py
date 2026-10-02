"""Train the two-stage matcher on the training split and report out-of-fold macro F0.5.

Usage: python train.py [max_rows_per_country_per_fold] [decoy_weight]

decoy_weight: negative pairs whose S2/S3 record matches no S1 entity (decoys /
siblings) are up-weighted.  The test set contains about twice as many sibling
decoys per entity as the training set, so a weight ~2 calibrates the
probabilities for the test distribution.  Besides the plain out-of-fold score we
report a "test-like" score in which every decoy false positive counts that many
times.
"""
import os
import pickle
import sys
import time

import numpy as np

from config import SEED, cache_path
from decision import argmax_per_query, select_matches
from evaluate import f05_per_entity, true_s1_index
from features import CATEGORICAL
from model import (STAGEB_COLS, fold_of, load_pairs, make_dataset, sample_rows, stage_b_features,
                   train_datasets)
from pairs import pairs_dir
from records import load_records, split_countries
from wordte import SUP_COLS, WORD_COLS

# extra seeds of the seed bag (bag.py) are trained under a tag: ER_SEED=7 ER_MODEL_TAG=s7
#   -> models_s7.pkl, pA_s7.npy, pB_s7.npy (the original model: models.pkl, pA_v3.npy, pB_v3.npy)
MODEL_TAG = os.environ.get("ER_MODEL_TAG", "")
MODEL_FILE = cache_path(f"models_{MODEL_TAG}.pkl" if MODEL_TAG else "models.pkl")
MASK_FRAC = 0.15
SUP_MASK_FRAC = 0.3
VAL_FRAC = 0.05


def aux_arrays(split, country):
    """src (per record) and, for train, the true S1 index of every record."""
    d = pairs_dir(split, country)
    fs = os.path.join(d, "src.npy")
    ft = os.path.join(d, "true_s1.npy")
    if not os.path.exists(fs) or (split == "train" and not os.path.exists(ft)):
        rec = load_records(split, country)
        np.save(fs, rec["src"])
        if split == "train":
            np.save(ft, true_s1_index(rec).astype(np.int32))
        del rec
    src = np.load(fs)
    ts = np.load(ft) if split == "train" else None
    return src, ts


def load_extra(split, country, name="XE"):
    return np.load(os.path.join(pairs_dir(split, country), f"{name}.npy"))


def sibling_entities(d):
    """S1 entities that have a near-identical decoy among their candidates
    (the situation that dominates the test set)."""
    cols = d["cols"]
    js, ja = cols.index("n_soft_q"), cols.index("a_soft_q")
    near = np.zeros(len(d["B"]), bool)
    for s in range(0, len(near), 2_000_000):
        x = np.asarray(d["X"][s:s + 2_000_000])
        near[s:s + len(x)] = (x[:, js] >= 0.8) & (x[:, ja] >= 0.8)
    decoy = d["true_s1"][d["Q"]] < 0
    ent = np.zeros(len(d["src"]), bool)
    ent[d["B"][near & decoy]] = True
    return ent


def rows(d, sub, extras):
    x = np.asarray(d["X"][sub])
    if extras:
        x = np.hstack([x] + [d[k][sub] for k in extras])
    return x


def predict_oof(models, d, extras, chunk=1_000_000):
    fold = d["fold"]
    out = np.empty(len(fold), np.float32)
    for s in range(0, len(fold), chunk):
        sl = slice(s, s + chunk)
        xc = np.asarray(d["X"][sl])
        if extras:
            xc = np.hstack([xc] + [d[k][sl] for k in extras])
        fc = fold[sl]
        oc = out[sl]
        for f, m in enumerate(models):
            sel = fc != f
            if sel.any():
                oc[sel] = m.predict(xc[sel])
    return out


def _gather(data, parts, extras, n_cols, decoy_w):
    """rows, labels and weights (decoy negatives x decoy_w)"""
    n = sum(len(ix) for _, ix in parts)
    X = np.empty((n, n_cols), np.float32)
    y = np.empty(n, np.int8)
    w = np.empty(n, np.float32)
    o = 0
    for c, ix in parts:
        d = data[c]
        for s in range(0, len(ix), 1_000_000):
            sub = ix[s:s + 1_000_000]
            X[o:o + len(sub)] = rows(d, sub, extras)
            yy = d["y"][sub]
            y[o:o + len(sub)] = yy
            decoy = (yy == 0) & (d["true_s1"][d["Q"][sub]] < 0)
            w[o:o + len(sub)] = np.where(decoy, decoy_w, 1.0)
            o += len(sub)
    return X, y, w


def fit_stage(data, rng, max_rows, extras, decoy_w, tag=""):
    models = []
    first = next(iter(data.values()))
    cols = first["cols"]
    n_cols = len(cols) + sum(first[k].shape[1] for k in extras)
    cat_idx = [cols.index(c) for c in CATEGORICAL if c in cols]
    for f in (0, 1):
        tr_parts, va_parts = [], []
        for c, d in data.items():
            idx = sample_rows(d["fold"] == f, d["B"], max_rows, rng)
            ents = d["B"][idx]
            u = np.unique(ents)
            isva = np.isin(ents, u[rng.random(len(u)) < VAL_FRAC])
            tr_parts.append((c, idx[~isva]))
            va_parts.append((c, idx[isva]))
        t0 = time.time()
        datasets = []
        for parts in (tr_parts, va_parts):
            X, y, w = _gather(data, parts, extras, n_cols, decoy_w)
            # hide the categorical legal ids on a random subset so the model also learns
            # to rely on the generic legal-relation features (used for unseen legal forms)
            if cat_idx:
                mrow = rng.random(len(y)) < MASK_FRAC
                for j in cat_idx:
                    X[mrow, j] = np.nan
            # hide the supervised word encodings on a subset so the model also learns to
            # use the label-free ones (the only ones available for words never seen in
            # training, e.g. French)
            if "XE" in extras:
                mrow = rng.random(len(y)) < SUP_MASK_FRAC
                for j in range(len(cols), len(cols) + len(SUP_COLS)):
                    X[mrow, j] = np.nan
            datasets.append(make_dataset(X, y, cat_idx, reference=datasets[0] if datasets else None, weight=w))
            if len(datasets) == 1:
                n_tr, pos_tr = len(y), float(y.mean())
            del X, y, w
        m = train_datasets(datasets[0], datasets[1])
        print(f"  stage{tag} fold {f}: {n_tr} rows, pos rate {pos_tr:.3f}, best iter {m.best_iteration}, "
              f"({time.time() - t0:.0f}s)", flush=True)
        models.append(m)
        del datasets
    return models


def macro_f05_w(d, pred_b, pred_q, fp_decoy_w):
    """Macro F0.5 where a false positive on a decoy record counts fp_decoy_w times
    (emulates a test set with that many times more decoys)."""
    ts = d["true_s1"]
    n = len(d["src"])
    is_tp = ts[pred_q] == pred_b
    is_decoy = ts[pred_q] < 0
    wpred = np.where(is_tp, 1.0, np.where(is_decoy, fp_decoy_w, 1.0))
    n_pred = np.bincount(pred_b, weights=wpred, minlength=n)
    tp = np.bincount(pred_b, weights=is_tp, minlength=n)
    n_true = np.bincount(ts[ts >= 0], minlength=n)
    s1 = np.flatnonzero(d["src"] == 1)
    return f05_per_entity(tp[s1], n_pred[s1], n_true[s1])


def evaluate_all(data, pkey, miss=0.0, eval_w=(1.0, 2.0), verbose=True):
    tots = {w: [] for w in eval_w}
    sib = []
    for c, d in data.items():
        B, Q, p = d["B"], d["Q"], d[pkey]
        keep = argmax_per_query(B, Q, p)
        Bk, Qk, pk = B[keep], Q[keep], p[keep]
        sel = select_matches(Bk, Qk, pk, miss=miss)
        msg = []
        for w in eval_w:
            F = macro_f05_w(d, Bk[sel], Qk[sel], w)
            tots[w].append(F)
            msg.append(f"decoyFPx{w:g}={F.mean():.4f}")
            if w == 1.0:
                se = d["sib_ent"][d["src"] == 1]
                sib.append(F[se])
                msg.append(f"sibling-entities({se.mean():.1%})={F[se].mean():.4f}")
        if verbose:
            print(f"  {c}: " + "  ".join(msg), flush=True)
    res = {w: np.concatenate(v).mean() for w, v in tots.items()}
    if verbose:
        print("  ALL " + "  ".join(f"decoyFPx{w:g}={v:.4f}" for w, v in res.items())
              + f"  sibling-entities={np.concatenate(sib).mean():.4f}", flush=True)
    return res


def main(max_rows=2_200_000, decoy_w=2.0):
    """Model v3 (used by the final blend): pair features + word-identity features."""
    ext = ["XE"]
    tag = f"_{MODEL_TAG}" if MODEL_TAG else "_v3"
    rng = np.random.default_rng(SEED)
    countries = split_countries("train")
    data = {}
    for c in countries:
        d = load_pairs("train", c)
        d["src"], d["true_s1"] = aux_arrays("train", c)
        d["fold"] = fold_of(d["B"])
        d["XE"] = load_extra("train", c)
        d["sib_ent"] = sibling_entities(d)
        data[c] = d
        print(f"{c}: {len(d['B'])} pairs, pos {d['y'].sum()}, features {len(d['cols'])}+{d['XE'].shape[1]}", flush=True)
    cols = data[countries[0]]["cols"]

    print(f"stage A (decoy weight {decoy_w})", flush=True)
    models_A = fit_stage(data, rng, max_rows, ext, decoy_w, tag="A")
    for c, d in data.items():
        d["pA"] = predict_oof(models_A, d, ext)
        d["FB"] = stage_b_features(d["B"], d["Q"], d["pA"], len(d["src"]), d["src"])
    evaluate_all(data, "pA")

    print("stage B", flush=True)
    models_B = fit_stage(data, rng, max_rows, ext + ["FB"], decoy_w, tag="B")
    for c, d in data.items():
        d["pB"] = predict_oof(models_B, d, ext + ["FB"])
    evaluate_all(data, "pB")

    imp = models_B[0].feature_importance("gain")
    names = cols + WORD_COLS + STAGEB_COLS
    top = np.argsort(-imp)[:30]
    print("top features (gain):", [(names[i], round(float(imp[i]) / imp.sum(), 4)) for i in top])

    with open(MODEL_FILE, "wb") as fh:
        pickle.dump(dict(A=[m.model_to_string() for m in models_A], B=[m.model_to_string() for m in models_B],
                         cols=cols, extra_cols=WORD_COLS, miss=0.0, decoy_w=decoy_w), fh)
    for c, d in data.items():
        np.save(os.path.join(pairs_dir("train", c), f"pA{tag}.npy"), d["pA"])
        np.save(os.path.join(pairs_dir("train", c), f"pB{tag}.npy"), d["pB"])
    print("saved models", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2_200_000,
         float(sys.argv[2]) if len(sys.argv) > 2 else 2.0)
