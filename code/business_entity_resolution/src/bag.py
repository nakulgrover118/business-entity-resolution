"""Stage 7b: seed bag of the v3 matcher.

train.py / predict.py are run for extra seeds with ER_SEED=<s> ER_MODEL_TAG=s<s> (pB_s7.npy, pB_s7_test.npy, ...);
this step logit-averages the stage-B probabilities of the tags into pB_bag.npy (train, out of fold) and
pB_bag_test.npy (test).  The bag feeds the US / India blend (final.py); France keeps the original v3.

Usage: python bag.py [TAGS]      (default: v3,s7,s13)
"""
import os
import sys

import numpy as np

from pairs import pairs_dir
from records import split_countries

TAGS = ("v3", "s7", "s13")


def main(tags=TAGS):
    for split in ("train", "test"):
        for c in split_countries(split):
            d = pairs_dir(split, c)
            ps = [np.load(os.path.join(d, f"pB_{t}.npy" if split == "train" else f"pB_{t}_test.npy")).astype(np.float64)
                  for t in tags]
            lg = np.mean([np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1)) for p in ps], axis=0)
            np.save(os.path.join(d, "pB_bag.npy" if split == "train" else "pB_bag_test.npy"),
                    (1 / (1 + np.exp(-lg))).astype(np.float32))
            print(f"  {split} {c}: bag of {','.join(tags)} (corr first/last seed {np.corrcoef(ps[0], ps[-1])[0, 1]:.5f})", flush=True)


if __name__ == "__main__":
    main(tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else TAGS)
