"""Stage 9: final submission.

US / India : the probabilities of the hard pairs are replaced by a LightGBM blend (blend2.py) of the
             seed-bagged v3 (seeds 42 / 7 / 13, bag.py) + cross-encoder scores CE1-CE5 + base pair
             features + the signed house-number offset; the blender is trained with 3 seeds and
             averaged; then
             * logit - 0.25 (slightly stricter; tuned out of fold with decoys counted twice),
             * "+K" house-number decoys: the generator makes decoys by shifting an S1 house
               number upwards by K in KSET, true copies change digits symmetrically; pairs
               whose record house number = S1 house number + K get a logit penalty
               (US 0 -- the blend learns it from the offset feature --, India -1, tuned out of fold).
France     : v3 probabilities (the cross-encoders were trained on US/India, where a swapped
             business-type word means a decoy; they re-admit French decoys), logit - 1.0, and
             two word rules for one-word name swaps (one extra word, one dropped word):
             * the new word is a French *filler* the generator uses for true copies
               (FR_NOISE) and house number and address agree -> logit + 2,
             * the new word is a real word (>= 4 letters, name document frequency >= 100)
               that is not a filler -> a different business of the same family -> logit - 5.
Candidates: only pairs that pass the learned blocking filter (prune.py + filter_pairs.py, ~4.4 per S1
on test) are scored; they are written to candidate_pairs.tsv.
Every S2/S3 record is offered only to its best S1 entity; each entity keeps the subset that
maximises its expected F0.5 (decision.py).

Usage: python final.py [OUT_PATH] [US_CFG] [INDIA_CFG]      (CFG e.g. ce1,ce2,ce3,ce4,ce5,base,base2)
"""
import json
import os
import sys

import numpy as np
import pyarrow as pa

import blend2
from blend2 import _decide, _logit, features, fit, v3_prob
from config import OUTPUT_DIR, cache_path
from io_utils import load_source
from pairs import pairs_dir
from predict import _lists
from records import load_records, split_countries

CFG = {"US": ["ce1", "ce2", "ce3", "ce4", "ce5", "base", "base2"],
       "India": ["ce1", "ce2", "ce3", "ce4", "ce5", "base", "base2"]}
BLEND_V3 = "bag"            # US / India: the seed-bagged v3 (bag.py) in the blend; France keeps the original v3
BLEND_SEEDS = (7, 11, 23)   # blender trained with 3 LightGBM seeds, predictions averaged (2 folds x 3 seeds)
_MODELS = {}
BASE_SHIFT = {"US": -0.25, "India": -0.25, "France": -1.0}
# with the base2 blend (signed house-number offset as a feature) the US needs no extra penalty (OOF-tuned)
K_SHIFT = {"US": 0.0, "India": -1.0}
KSET = np.array([3, 4, 5, 7, 9, 11, 13, 21])
FR_NOISE = ["groupe", "services", "developpement", "france", "compagnie", "cie", "fils", "associes", "st",
            "et", "e", "frs", "and", "freres"]
FR_TYPE_SHIFT, FR_NOISE_SHIFT = -5.0, 2.0


def house_numbers(split, country, rec):
    """Integer value of every record's house number (-1 if none); cached."""
    f = os.path.join(pairs_dir(split, country), "hnum_val.npy")
    if os.path.exists(f):
        return np.load(f)
    vals = np.array([int(t) if t.isdigit() and len(t) <= 9 else -1 for t in rec["a_vocab"]], np.int64)
    h = rec["hnum_id"]
    hv = np.where(h >= 0, vals[np.maximum(h, 0)], -1)
    np.save(f, hv)
    return hv


def blend_models(cfg):
    """Blender models for one input configuration (trained once, shared by US and India)."""
    key = tuple(cfg)
    if key not in _MODELS:
        ms = []
        for sd in BLEND_SEEDS:
            m, _ = fit(cfg, verbose=True, v3_tag=BLEND_V3, params=dict(blend2.PARAMS, seed=sd))
            ms += m
        _MODELS[key] = ms
    return _MODELS[key]


def blend_country(c, rec):
    d = pairs_dir("test", c)
    B = np.load(os.path.join(d, "B.npy"))
    Q = np.load(os.path.join(d, "Q.npy"))
    models = blend_models(CFG[c])
    p = v3_prob("test", c, BLEND_V3).astype(np.float64)
    pair, _, X, _, _ = features("test", c, CFG[c], BLEND_V3)
    p[pair] = np.mean([m.predict(X) for m in models], axis=0)
    np.save(os.path.join(d, "pB_final_blend.npy"), p.astype(np.float32))  # raw blend probabilities (team_stack.py)
    lp = _logit(p) + BASE_SHIFT[c]
    hv = house_numbers("test", c, rec)
    off = np.where((hv[Q] >= 0) & (hv[B] >= 0), hv[Q] - hv[B], 10 ** 9)
    isk = np.isin(off, KSET)
    lp[isk] += K_SHIFT[c]
    print(f"  {c}: blend {','.join(CFG[c])}, {isk.sum()} +K pairs penalised", flush=True)
    return B, Q, 1 / (1 + np.exp(-lp))


def unseen_country(c, rec):
    """A country absent from training (other than France, which has its own word rules): v3 probabilities only
    (the cross-encoders learned US/India decoy patterns) with the conservative France strictness."""
    d = pairs_dir("test", c)
    B = np.load(os.path.join(d, "B.npy"))
    Q = np.load(os.path.join(d, "Q.npy"))
    lp = _logit(v3_prob("test", c)) + BASE_SHIFT.get(c, BASE_SHIFT["France"])
    print(f"  {c}: unseen country, v3 probabilities with logit {BASE_SHIFT.get(c, BASE_SHIFT['France']):+.2f}", flush=True)
    return B, Q, 1 / (1 + np.exp(-lp))


def country_probs(c, rec):
    if c == "France":
        return france(rec)
    if c in K_SHIFT:
        return blend_country(c, rec)
    return unseen_country(c, rec)


def france(rec):
    d = pairs_dir("test", "France")
    B = np.load(os.path.join(d, "B.npy"))
    Q = np.load(os.path.join(d, "Q.npy"))
    ux = np.load(os.path.join(d, "ux.npy"))
    ud = np.load(os.path.join(d, "ud.npy"))
    X = np.load(os.path.join(d, "X.npy"), mmap_mode="r")
    with open(os.path.join(d, "cols.json")) as fh:
        cols = json.load(fh)
    he = np.empty(len(B), np.float32)
    aq = np.empty(len(B), np.float32)
    for s in range(0, len(B), 2_000_000):
        x = np.asarray(X[s:s + 2_000_000])
        he[s:s + len(x)] = x[:, cols.index("hnum_eq")]
        aq[s:s + len(x)] = x[:, cols.index("a_soft_q")]
    with open(cache_path("word_vocab.json"), encoding="utf-8") as fh:
        gvocab = json.load(fh)
    inv = {g: w for w, g in gvocab.items()}
    wlen = np.array([len(inv.get(g, "")) for g in range(len(gvocab))])
    # log document frequency of each (global) word among the country's names
    gdf = np.zeros(len(gvocab), np.float32)
    for i, t in enumerate(rec["n_vocab"]):
        g = gvocab.get(t)
        if g is not None:
            gdf[g] = np.log1p(rec["ndf"][i])
    noise_ids = np.array([gvocab[w] for w in FR_NOISE if w in gvocab])
    swap = (ux[:, 0] >= 0) & (ux[:, 1] < 0) & (ud[:, 0] >= 0) & (ud[:, 1] < 0)
    w = np.maximum(ux[:, 0], 0)
    is_noise = np.isin(ux[:, 0], noise_ids)
    real_word = (ux[:, 0] >= 0) & (wlen[w] >= 4) & (gdf[w] >= np.log1p(100))
    type_swap = swap & ~is_noise & real_word
    noise_swap = swap & is_noise & (he == 1) & (aq >= 0.9)
    lp = _logit(v3_prob("test", "France")) + BASE_SHIFT["France"]
    lp[type_swap] += FR_TYPE_SHIFT
    lp[noise_swap] += FR_NOISE_SHIFT
    print(f"  France: {type_swap.sum()} type-word swaps penalised, {noise_swap.sum()} filler swaps boosted", flush=True)
    return B, Q, 1 / (1 + np.exp(-lp))


def main(out_path, cand_path=None):
    rows, cands = {}, {}
    for c in split_countries("test"):  # country is an open set: every test country is processed
        rec = load_records("test", c)
        B, Q, p = country_probs(c, rec)   # the candidates are already filtered (filter_pairs.py)
        pb, pq_ = _decide(B, Q, p)
        src = np.load(os.path.join(pairs_dir("test", c), "src.npy"))
        ids = rec["ids"]
        ids = ids.combine_chunks() if isinstance(ids, pa.ChunkedArray) else ids
        del rec
        s1_rows = np.flatnonzero(src == 1)
        mp = _lists(ids, len(s1_rows), pb, pq_, s1_rows)
        cp = _lists(ids, len(s1_rows), B, Q, s1_rows)
        for r, sid in zip(s1_rows.tolist(), ids.take(pa.array(s1_rows)).to_pylist()):
            rows[sid] = mp.get(r, "")
            cands[sid] = cp.get(r, "")
        print(f"  {c}: {len(B) / len(s1_rows):.2f} candidates and {len(pb)} matches ({len(pb) / len(s1_rows):.3f}) per S1", flush=True)
    s1 = load_source("test", "source1").column("entity_id").to_pylist()
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1:
            fh.write(f"{sid}\t{rows.get(sid, '')}\n")
    print("wrote", out_path, flush=True)
    cand_path = cand_path or os.path.join(os.path.dirname(os.path.abspath(out_path)), "candidate_pairs.tsv")
    with open(cand_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in s1:
            fh.write(f"{sid}\t{cands.get(sid, '')}\n")
    print("wrote", cand_path, flush=True)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUTPUT_DIR, "matching_results.tsv")
    if len(sys.argv) > 2:
        CFG["US"] = sys.argv[2].split(",")
    if len(sys.argv) > 3:
        CFG["India"] = sys.argv[3].split(",")
    main(out)
