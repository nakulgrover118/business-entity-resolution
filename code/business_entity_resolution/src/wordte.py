"""Word-identity features ("which word was added / dropped").

Sibling decoys are usually the S1 name plus a *branch-like* word ("westgate",
"midtown", "north", "board") while true-match noise adds fillers ("services",
"center", "group").  For every candidate pair we list the name words of the
S2/S3 record that have no counterpart in the S1 name (extra words) and the S1
words missing from the record (dropped words), then encode each word by how often
it occurs in true matches among near-identical training pairs.  Training pairs
get out-of-fold encodings (folds by S1 entity); test pairs use all training pairs.

Output per (split, country): XE.npy with columns WORD_COLS.
"""
import json
import os
import sys
import time

import numpy as np

from features import token_char_table, unmatched_tokens
from model import fold_of, load_pairs
from pairs import pairs_dir
from records import load_records, split_countries

K = 4
SMOOTH = 10.0
# supervised (training labels, out-of-fold) word encodings
SUP_COLS = ["te_x_min", "te_x_max", "te_x_logn", "te_d_min", "te_d_max", "te_d_logn"]
# label-free encodings computed on each split's own pairs: how often the house number
# still agrees when this word is added / dropped (sibling decoys almost always move the
# house number, noise fillers usually keep it).  Transfers to unseen languages (France).
UNSUP_COLS = ["hx_min", "hx_max", "hx_logn", "hd_min", "hd_max", "hd_logn"]
WORD_COLS = SUP_COLS + UNSUP_COLS


def compute_unmatched(split, country, gvocab):
    d = pairs_dir(split, country)
    fx, fd = os.path.join(d, "ux.npy"), os.path.join(d, "ud.npy")
    rec = load_records(split, country)
    B = np.load(os.path.join(d, "B.npy")).astype(np.int64)
    Q = np.load(os.path.join(d, "Q.npy")).astype(np.int64)
    chars, lens = token_char_table(rec["n_vocab"])
    ex, dr = unmatched_tokens(B, Q, rec["n_ptr"], rec["n_idx"], chars, lens, K)
    # local token id -> global word id (shared across countries / splits)
    used = np.unique(np.concatenate([ex[ex >= 0], dr[dr >= 0]]))
    lut = np.full(len(rec["n_vocab"]), -1, np.int32)
    for t in used:
        w = rec["n_vocab"][t]
        g = gvocab.get(w)
        if g is None:
            g = len(gvocab)
            gvocab[w] = g
        lut[t] = g
    ex = np.where(ex >= 0, lut[np.maximum(ex, 0)], -1).astype(np.int32)
    dr = np.where(dr >= 0, lut[np.maximum(dr, 0)], -1).astype(np.int32)
    np.save(fx, ex)
    np.save(fd, dr)
    del rec


def _near_masks(dd):
    X = dd["X"]
    cols = dd["cols"]
    jb, jq, jh = cols.index("n_cov_b"), cols.index("n_cov_q"), cols.index("hnum_eq")
    nb = np.empty(X.shape[0], np.float32)
    nq = np.empty(X.shape[0], np.float32)
    he = np.empty(X.shape[0], np.float32)
    for s in range(0, X.shape[0], 2_000_000):
        x = np.asarray(X[s:s + 2_000_000])
        nb[s:s + len(x)] = x[:, jb]
        nq[s:s + len(x)] = x[:, jq]
        he[s:s + len(x)] = x[:, jh]
    base = np.ones(X.shape[0], bool)
    if "blk_expanded" in cols:
        # label-free statistics only from the first-pass candidates: the second-pass search covers a
        # different share of records in train and test, which would shift these per-split rates
        je = cols.index("blk_expanded")
        for s in range(0, X.shape[0], 2_000_000):
            base[s:s + 2_000_000] = np.asarray(X[s:s + 2_000_000, je]) == 0
    return nb >= 0.999, nq >= 0.999, he, base


def _counts(tok, mask, y, G):
    t = tok[mask]
    yy = np.repeat(y[mask][:, None], tok.shape[1], axis=1)
    v = t >= 0
    n = np.bincount(t[v], minlength=G).astype(np.float64)
    pos = np.bincount(t[v], weights=yy[v], minlength=G)
    return n, pos


def _unsup_stats(ex, dr, near_x, near_d, he, G):
    """Per word: number of near pairs where it is the extra / dropped word (with both
    house numbers present) and how many of them keep the same house number."""
    hv = he >= 0
    agree = (he == 1).astype(np.float64)
    nx, hx = _counts(ex, near_x & hv, agree, G)
    nd, hd = _counts(dr, near_d & hv, agree, G)
    px = hx.sum() / max(nx.sum(), 1)
    pd_ = hd.sum() / max(nd.sum(), 1)
    return nx, hx, nd, hd, px, pd_


def _encode(tok, n, pos, prior):
    rate = (pos + SMOOTH * prior) / (n + SMOOTH)
    known = n > 0
    t = np.maximum(tok, 0)
    r = np.where((tok >= 0) & known[t], rate[t], np.nan)
    c = np.where((tok >= 0) & known[t], n[t], np.nan)
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore", RuntimeWarning)
        mn = np.nanmin(r, axis=1)
        mx = np.nanmax(r, axis=1)
        lg = np.log1p(np.nanmax(c, axis=1))
    return mn.astype(np.float32), mx.astype(np.float32), lg.astype(np.float32)


def build_all():
    t0 = time.time()
    runs = [("train", c) for c in split_countries("train")] + [("test", c) for c in split_countries("test")]
    vfile = os.path.join(pairs_dir("train", split_countries("train")[0]), "..", "word_vocab.json")
    have = os.path.exists(vfile) and all(os.path.exists(os.path.join(pairs_dir(s, c), "ux.npy")) for s, c in runs)
    if have:
        with open(vfile, encoding="utf-8") as fh:
            gvocab = json.load(fh)
    else:
        gvocab = {}
        for split, c in runs:
            compute_unmatched(split, c, gvocab)
            print(f"  unmatched words {split} {c} ({time.time() - t0:.0f}s)", flush=True)
    G = len(gvocab)
    unsup = {}
    # ---- statistics from training pairs, per fold
    stats = {f: [np.zeros(G), np.zeros(G), np.zeros(G), np.zeros(G)] for f in (0, 1)}
    train_meta = {}
    for c in split_countries("train"):
        dd = load_pairs("train", c)
        d = pairs_dir("train", c)
        ex = np.load(os.path.join(d, "ux.npy"))
        dr = np.load(os.path.join(d, "ud.npy"))
        y = dd["y"].astype(np.float64)
        fold = fold_of(dd["B"])
        near_x, near_d, he, base = _near_masks(dd)
        near_x &= ex[:, 0] >= 0
        near_d &= dr[:, 0] >= 0
        unsup[("train", c)] = _unsup_stats(ex, dr, near_x & base, near_d & base, he, G)
        for f in (0, 1):
            m = fold == f
            nx, px = _counts(ex, near_x & m, y, G)
            nd, pd_ = _counts(dr, near_d & m, y, G)
            s = stats[f]
            s[0] += nx
            s[1] += px
            s[2] += nd
            s[3] += pd_
        train_meta[c] = (fold,)
    tot = [stats[0][i] + stats[1][i] for i in range(4)]
    prior_x = tot[1].sum() / max(tot[0].sum(), 1)
    prior_d = tot[3].sum() / max(tot[2].sum(), 1)
    print(f"  near-pair match rate: extra-word {prior_x:.3f}, dropped-word {prior_d:.3f}", flush=True)
    inv = {g: w for w, g in gvocab.items()}
    rx = (tot[1] + SMOOTH * prior_x) / (tot[0] + SMOOTH)
    top = [g for g in np.argsort(-tot[0])[:40]]
    print("  frequent extra words (count, match rate):",
          [(inv[g], int(tot[0][g]), round(float(rx[g]), 3)) for g in top], flush=True)
    for split, c in runs:
        if (split, c) not in unsup:
            dd = load_pairs(split, c)
            d = pairs_dir(split, c)
            ex = np.load(os.path.join(d, "ux.npy"))
            dr = np.load(os.path.join(d, "ud.npy"))
            near_x, near_d, he, base = _near_masks(dd)
            unsup[(split, c)] = _unsup_stats(ex, dr, near_x & (ex[:, 0] >= 0) & base, near_d & (dr[:, 0] >= 0) & base, he, G)
            del dd
    for (split, c), (nx, hx, nd, hd, px, pd_) in unsup.items():
        rx = (hx + SMOOTH * px) / (nx + SMOOTH)
        top = np.argsort(-nx)[:12]
        print(f"  label-free {split} {c}: extra words (count, house-number agreement):",
              [(inv[g], int(nx[g]), round(float(rx[g]), 3)) for g in top], flush=True)
    for split, c in runs:
        d = pairs_dir(split, c)
        ex = np.load(os.path.join(d, "ux.npy"))
        dr = np.load(os.path.join(d, "ud.npy"))
        out = np.empty((len(ex), len(WORD_COLS)), np.float32)
        nx, hx, nd, hd, px, pd_ = unsup[(split, c)]
        out[:, 6], out[:, 7], out[:, 8] = _encode(ex, nx, hx, px)
        out[:, 9], out[:, 10], out[:, 11] = _encode(dr, nd, hd, pd_)
        if split == "train":
            fold = train_meta[c][0]
            for f in (0, 1):
                m = fold == f
                s = stats[1 - f]  # out-of-fold statistics
                out[m, 0], out[m, 1], out[m, 2] = _encode(ex[m], s[0], s[1], prior_x)
                out[m, 3], out[m, 4], out[m, 5] = _encode(dr[m], s[2], s[3], prior_d)
        else:
            out[:, 0], out[:, 1], out[:, 2] = _encode(ex, tot[0], tot[1], prior_x)
            out[:, 3], out[:, 4], out[:, 5] = _encode(dr, tot[2], tot[3], prior_d)
        np.save(os.path.join(d, "XE.npy"), out)
        print(f"  XE {split} {c}: known extra-word rate {np.isfinite(out[:, 0]).mean():.3f} ({time.time() - t0:.0f}s)",
              flush=True)
    with open(os.path.join(pairs_dir("train", split_countries("train")[0]), "..", "word_vocab.json"), "w",
              encoding="utf-8") as fh:
        json.dump(gvocab, fh)


if __name__ == "__main__":
    build_all()
