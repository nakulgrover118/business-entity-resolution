"""Stage 3+4 driver: blocking + pair features for one (split, country), stored on disk.

Outputs in work/cache/pairs_{split}_{country}/:
  B.npy, Q.npy (int32 record indices), S.npy (blocking score), X.npy (float32
  feature matrix, memory-mapped), y.npy (labels, train only), cols.json
"""
import json
import os
import sys
import time

import numpy as np

from blocking import generate
from config import cache_path
from evaluate import true_s1_index
from features import PairFeaturizer, context_features
from records import load_records, split_countries

CHUNK = 1_500_000


def pairs_dir(split, country):
    return cache_path(f"pairs_{split}_{country}")


def build(split, country, force=False, k=10, rel=0.6, max_df=4000):
    """If cache/expand_{split}_{country}.npy exists (boolean mask over records, written by expand.py
    from a first full pipeline run), the unmatched query records get a wider second search."""
    d = pairs_dir(split, country)
    if os.path.exists(os.path.join(d, "done")) and not force:
        return
    os.makedirs(d, exist_ok=True)
    t0 = time.time()
    rec = load_records(split, country)
    ef = cache_path(f"expand_{split}_{country}.npy")
    if os.path.exists(ef):
        B, Q, S, E = generate(rec, k_fwd=k, rel=rel, max_df=max_df, expand_q=np.load(ef))
    else:
        B, Q, S = generate(rec, k_fwd=k, rel=rel, max_df=max_df)
        E = None
    order = np.lexsort((Q, B))
    B, Q, S = B[order], Q[order], S[order]
    P = len(B)
    print(f"[{split} {country}] {P} pairs, {P / max(1, (rec['src'] == 1).sum()):.2f}/S1 ({time.time() - t0:.0f}s)", flush=True)
    np.save(os.path.join(d, "B.npy"), B)
    np.save(os.path.join(d, "Q.npy"), Q)
    np.save(os.path.join(d, "S.npy"), S)
    np.save(os.path.join(d, "src.npy"), rec["src"])
    if split == "train":
        true_s1 = true_s1_index(rec)
        np.save(os.path.join(d, "true_s1.npy"), true_s1.astype(np.int32))
        np.save(os.path.join(d, "y.npy"), (true_s1[Q] == B).astype(np.int8))
    ctx = context_features(B, Q, S, rec["N"])
    if E is not None:
        ctx["blk_expanded"] = E[order].astype(np.float32)
    fz = PairFeaturizer(rec)
    cols = None
    X = None
    for s in range(0, P, CHUNK):
        e = min(P, s + CHUNK)
        f = fz.compute(B[s:e], Q[s:e])
        for kname, v in ctx.items():
            f[kname] = v[s:e]
        if cols is None:
            cols = list(f.keys())
            X = np.lib.format.open_memmap(os.path.join(d, "X.npy"), mode="w+", dtype=np.float32, shape=(P, len(cols)))
        X[s:e] = np.column_stack([f[c] for c in cols])
        print(f"   features {e}/{P} ({time.time() - t0:.0f}s)", flush=True)
    X.flush()
    del X
    with open(os.path.join(d, "cols.json"), "w") as fh:
        json.dump(cols, fh)
    open(os.path.join(d, "done"), "w").close()
    print(f"[{split} {country}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    split = sys.argv[1]
    countries = sys.argv[2:] or split_countries(split)
    for c in countries:
        build(split, c)
