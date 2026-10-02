"""Inference on the test split and writing of the two submission files.

output/matching_results.tsv : source1_entity_id, matched_entity_ids
output/candidate_pairs.tsv  : source1_entity_id, candidate_entity_ids
"""
import os
import pickle
import time

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from config import OUTPUT_DIR, cache_path
from decision import argmax_per_query, select_matches
from io_utils import load_source
from model import load_pairs, stage_b_features
from pairs import pairs_dir
from records import load_records, split_countries
from train import MODEL_FILE


def _avg_predict(models, X, extras=(), chunk=1_000_000):
    out = np.zeros(X.shape[0], np.float32)
    for s in range(0, X.shape[0], chunk):
        xc = np.asarray(X[s:s + chunk])
        if extras:
            xc = np.hstack([xc] + [e[s:s + chunk] for e in extras])
        acc = np.zeros(len(xc), np.float64)
        for m in models:
            acc += m.predict(xc)
        out[s:s + chunk] = acc / len(models)
    return out


def predict_country(country, mdl):
    t0 = time.time()
    d = load_pairs("test", country)
    assert d["cols"] == mdl["cols"], "feature columns differ between train and test"
    A = [lgb.Booster(model_str=s) for s in mdl["A"]]
    Bm = [lgb.Booster(model_str=s) for s in mdl["B"]]
    src = np.load(os.path.join(pairs_dir("test", country), "src.npy"))
    XE = np.load(os.path.join(pairs_dir("test", country), "XE.npy"))
    ext = [XE]
    pA = _avg_predict(A, d["X"], ext)
    FB = stage_b_features(d["B"], d["Q"], pA, len(src), src)
    pB = _avg_predict(Bm, d["X"], ext + [FB])
    np.save(os.path.join(pairs_dir("test", country), f"pB_{os.environ.get('ER_MODEL_TAG')}_test.npy" if os.environ.get("ER_MODEL_TAG") else "pB_v3_test.npy"), pB)
    keep = argmax_per_query(d["B"], d["Q"], pB)
    Bk, Qk, pk = d["B"][keep], d["Q"][keep], pB[keep]
    sel = select_matches(Bk, Qk, pk, miss=mdl.get("miss", 0.0))
    print(f"[{country}] {len(d['B'])} pairs -> {sel.sum()} matches ({time.time() - t0:.0f}s)", flush=True)
    return d["B"], d["Q"], Bk[sel], Qk[sel], pk[sel], src


def _lists(ids, n_s1_rows, b, q, s1_rows_all):
    """Group q ids by b -> dict of s1 row -> comma-joined ids."""
    order = np.lexsort((q, b))
    b, q = b[order], q[order]
    qid = ids.take(pa.array(q)).to_pylist()
    out = {}
    if len(b) == 0:
        return out
    first = np.r_[True, b[1:] != b[:-1]]
    starts = np.flatnonzero(first)
    ends = np.r_[starts[1:], len(b)]
    for s, e in zip(starts, ends):
        out[int(b[s])] = ",".join(qid[s:e])
    return out


def main():
    with open(MODEL_FILE, "rb") as fh:
        mdl = pickle.load(fh)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    match_rows = {}
    cand_rows = {}
    for country in split_countries("test"):
        B, Q, Bs, Qs, ps, src = predict_country(country, mdl)
        rec = load_records("test", country)
        ids = rec["ids"]
        if isinstance(ids, pa.ChunkedArray):
            ids = ids.combine_chunks()
        del rec
        s1_rows = np.flatnonzero(src == 1)
        m = _lists(ids, len(s1_rows), Bs, Qs, s1_rows)
        c = _lists(ids, len(s1_rows), B, Q, s1_rows)
        s1_ids = ids.take(pa.array(s1_rows)).to_pylist()
        for r, sid in zip(s1_rows.tolist(), s1_ids):
            match_rows[sid] = m.get(r, "")
            cand_rows[sid] = c.get(r, "")
    # every Source-1 test entity gets exactly one row, in the order of the input file
    s1 = load_source("test", "source1").column("entity_id").to_pylist()
    with open(os.path.join(OUTPUT_DIR, "matching_results.tsv"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1:
            fh.write(f"{sid}\t{match_rows.get(sid, '')}\n")
    with open(os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in s1:
            fh.write(f"{sid}\t{cand_rows.get(sid, '')}\n")
    n_match = sum(1 for v in match_rows.values() if v)
    print(f"wrote {len(s1)} rows; {n_match} entities with >=1 match")


if __name__ == "__main__":
    main()
