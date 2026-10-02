"""Masks for the second-pass ("expanded") search: query records that the first full pipeline
(blend + decision rules of final.py) leaves unmatched.  99 % of the true pairs missed by the first
blocking belong to such records (India train), so only they get the wide pass-3 search.

Run with the FIRST run's cache (default ER_WORK_DIR) and write into the second run's cache:
    python expand.py <second-run cache dir>
Train masks use out-of-fold blend probabilities, test masks the final test decisions.
"""
import os
import sys

import numpy as np

import final
from blend2 import _decide, _logit, features, fit, v3_prob
from pairs import pairs_dir
from records import load_records, split_countries


def _assigned(B, Q, p, n):
    b, q = _decide(B, Q, p)
    m = np.zeros(n, bool)
    m[q] = True
    return m


# the first-pass decision is a fixed recipe, whatever the final blend of pass 2 is: the CE1-CE3 blend with
# logit -0.25 and the +K penalty of that blend (US -2, India -1; the final base2 blend needs US 0 instead)
CFG = ["ce1", "ce2", "ce3", "base"]
PASS1_BASE_SHIFT = -0.25
PASS1_K_SHIFT = {"US": -2.0, "India": -1.0}


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cfg = CFG
    models, _, oofs = fit(cfg, verbose=True, return_oof=True)
    for split in ("train", "test"):
        for c in split_countries(split):
            d = pairs_dir(split, c)
            B = np.load(os.path.join(d, "B.npy"))
            Q = np.load(os.path.join(d, "Q.npy"))
            src = np.load(os.path.join(d, "src.npy"))
            rec = load_records(split, c)
            if c not in PASS1_K_SHIFT:  # France and any other country unseen in training
                _, _, p = final.country_probs(c, rec)
            else:
                if split == "train":
                    p = oofs[c].astype(np.float64)
                else:
                    p = v3_prob("test", c).astype(np.float64)
                    pair, _, X, _, _ = features("test", c, cfg)
                    p[pair] = np.mean([m.predict(X) for m in models], axis=0)
                hv = final.house_numbers(split, c, rec)
                lp = _logit(p) + PASS1_BASE_SHIFT
                off = np.where((hv[Q] >= 0) & (hv[B] >= 0), hv[Q] - hv[B], 10 ** 9)
                lp[np.isin(off, final.KSET)] += PASS1_K_SHIFT[c]
                p = 1 / (1 + np.exp(-lp))
            del rec
            m = (src != 1) & ~_assigned(B, Q, p, len(src))
            np.save(os.path.join(out_dir, f"expand_{split}_{c}.npy"), m)
            print(f"  {split} {c}: {m.sum()} of {(src != 1).sum()} query records to expand", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
