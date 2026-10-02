"""Stage 4c: apply the learned candidate filter BEFORE the matcher.

pairs.py computes blocking candidates and their features; prune.py scores every candidate with a
cheap-feature LightGBM (out of fold on train).  This step keeps only the pairs above the per-country
threshold, so the matcher (v3), the cross-encoders and the decision see exactly the filtered set --
the set written to candidate_pairs.tsv.  Blocking-context features (ranks / counts among the blocking
candidates) keep the values computed on the full blocking output; everything downstream (word
features, v3 and its stage-B competition features, cross-encoder hard pairs, blend) is recomputed
on the filtered set.

Usage: python filter_pairs.py [OUT_CACHE_DIR]   (default: filter the current cache in place)
"""
import json
import os
import shutil
import sys

import numpy as np

from pairs import pairs_dir
from prune import THRESHOLD
from records import split_countries

PER_PAIR = ("B.npy", "Q.npy", "S.npy", "y.npy")
PER_RECORD = ("src.npy", "true_s1.npy", "hnum_val.npy")
CHUNK = 2_000_000


def filter_one(split, country, out_dir=None):
    d = pairs_dir(split, country)
    o = os.path.join(out_dir, os.path.basename(d)) if out_dir else d + "_filtered_tmp"
    os.makedirs(o, exist_ok=True)
    score = np.load(os.path.join(d, "prune2_oof.npy" if split == "train" else "prune2_score.npy"))
    keep = np.flatnonzero(score >= THRESHOLD.get(country, 0.01))
    for f in PER_PAIR:
        if os.path.exists(os.path.join(d, f)):
            np.save(os.path.join(o, f), np.load(os.path.join(d, f))[keep])
    for f in PER_RECORD:
        if os.path.exists(os.path.join(d, f)):
            shutil.copyfile(os.path.join(d, f), os.path.join(o, f))
    shutil.copyfile(os.path.join(d, "cols.json"), os.path.join(o, "cols.json"))
    X = np.load(os.path.join(d, "X.npy"), mmap_mode="r")
    Xo = np.lib.format.open_memmap(os.path.join(o, "X.npy"), mode="w+", dtype=np.float32, shape=(len(keep), X.shape[1]))
    for s in range(0, len(keep), CHUNK):
        Xo[s:s + CHUNK] = np.asarray(X[keep[s:s + CHUNK]])
    Xo.flush()
    del Xo, X
    open(os.path.join(o, "done"), "w").close()
    src = np.load(os.path.join(d, "src.npy"))
    print(f"  {split} {country}: {len(score)} -> {len(keep)} pairs ({len(keep) / (src == 1).sum():.2f} per S1)", flush=True)
    if not out_dir:  # in place: replace the full pairs directory by the filtered one
        shutil.rmtree(d)
        os.rename(o, d)


def main(out_dir=None):
    for split in ("train", "test"):
        for c in split_countries(split):
            filter_one(split, c, out_dir)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
