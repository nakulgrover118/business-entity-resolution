"""Stage 10: team ensemble -- decision-level stack of our output with team-mate submissions, then three French
stages.  Produces the leaderboard file (v12i = 0.990336) from final.py's output.

Stack (US / India, one team file at a time, "conservative" rule):
  * for every S1 entity where our list and the team file differ, each disputed pair gets a probability:
    our blend probability if the pair is one of our candidates, otherwise the cross-encoder CE3 score
    (isotonic-calibrated on its out-of-fold training scores);
  * a pair only the team file has is added if (ours: p >= 0.5 and not a +K house-number shift) or
    (not ours: calibrated CE3 >= 0.7);
  * a pair only we have is removed only if it is a +K house-number shift (a generator decoy);
  * every record keeps at most one S1 (the most probable).
France (the matcher has never seen French labels, so French decisions get label-validated rules):
  1. refinement (team file 2): add the team file's French pairs whose record name is the initials of the S1
     name, with the same house number and a non-empty address, if the record is unmatched; remove our French
     pairs with a +K house-number shift that the team file does not have;
  2. team additions (team file 3, a French audit): its extra pairs in two label-supported classes
     (france_team_additions);
  3. level matching and filler additions (france_level_and_fillers).
US / India rows keep final.py's / the stack's id order; France rows are written as sorted id lists.
The candidate file is our filtered candidates plus the matched pairs the team stages added (every match is a
candidate).

Usage: python team_stack.py OUR_MATCHES OUR_CANDIDATES TEAM_FILE_1 TEAM_FILE_2 TEAM_FILE_3 OUT_MATCHES OUT_CANDIDATES
       (the leaderboard file used team_inputs/team_submission_1.tsv, _2.tsv and _3.tsv)
"""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from config import cache_path
from io_utils import load_source
from pairs import pairs_dir
from records import load_records

KSET = np.array([3, 4, 5, 7, 9, 11, 13, 21])
T_IN, T_OUT, T_DROP = 0.50, 0.70, 0.0
PROB_FILE = "pB_final_blend.npy"   # raw blend probabilities saved by final.py
LEGAL = {"sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "ei", "and", "cie", "de", "du", "des", "la", "le",
         "les", "l", "d", "et"}
# French words the generator adds to true copies (team-file classification) / adds to +K decoy siblings
TEAM_FILL = {"groupe", "france", "developpement", "services", "compagnie", "cie", "fils", "associes", "freres",
             "etablissements", "ets", "st"}
DECOY_WORDS = {"france", "holding", "participations"}
# train truth share (mean of US / India) of "record without address, core tokens = those of exactly one S1",
# per legal-form transition, for the cells where the team additions overshoot it
LEVEL_TARGET = {"same": 0.990, "dropped": 0.934}
# noise words whose same-house additions France under-assigns (S1/S2-S3 frequency ratio 0.09 / 0.21, fingerprint
# of a filler; "federation" is a US organisation filler, 100 % true there)
FR_ADD_FILLERS = {"developpement", "groupe", "federation"}


def _read(path, country):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df["country"] = df.source1_entity_id.map(country)
    return df.set_index("source1_entity_id")[["country", "matched_entity_ids"]]


def _ce3(tb, tq, bs=64):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    ced = cache_path("crossenc_r3")
    tok = AutoTokenizer.from_pretrained(os.path.join(ced, "model_f0"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = np.zeros(len(tb), np.float32)
    order = np.argsort([len(x) + len(y) for x, y in zip(tb, tq)])
    for f in (0, 1):
        m = AutoModelForSequenceClassification.from_pretrained(os.path.join(ced, f"model_f{f}")).to(dev).eval()
        if dev == "cuda":
            m = m.half()
        with torch.no_grad():
            for s in range(0, len(order), bs):
                idx = order[s:s + bs]
                enc = tok([tb[i] for i in idx], [tq[i] for i in idx], truncation=True, max_length=128, padding=True,
                          return_tensors="pt")
                out[idx] += m(**{k: v.to(dev) for k, v in enc.items()}).logits.squeeze(-1).float().cpu().numpy() / 2
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()
    return out


def stack(base_path, team_path, country, raw_test):
    """One conservative stacking pass (US / India); returns {S1 id: comma list} for the changed entities."""
    from sklearn.isotonic import IsotonicRegression
    base, ext = _read(base_path, country), _read(team_path, country)
    rows = {}
    for c in ("US", "India"):
        a = base[base.country == c].matched_entity_ids
        b = ext.matched_entity_ids.reindex(a.index)
        diff = a.index[(a != b).values]
        recs = []
        for e in diff:
            sa = set(a[e].split(",")) if a[e] else set()
            sb = set(b[e].split(",")) if b[e] else set()
            if sa == sb:
                continue
            for q in sa | sb:
                recs.append((e, q, q in sa, q in sb))
        u = pd.DataFrame(recs, columns=["e", "q", "ours", "ext"])
        d = pairs_dir("test", c)
        rec = load_records("test", c)
        ids = rec["ids"]
        ids = ids.combine_chunks() if isinstance(ids, pa.ChunkedArray) else ids
        idmap = pd.Series(np.arange(len(ids)), index=ids.to_pylist())
        hv = np.load(os.path.join(d, "hnum_val.npy"))
        del rec
        Bc = np.load(os.path.join(d, "B.npy")).astype(np.int64)
        Qc = np.load(os.path.join(d, "Q.npy")).astype(np.int64)
        N = len(idmap)
        key = Bc * N + Qc
        o = np.argsort(key)
        ks = key[o]
        pbl = np.load(os.path.join(d, PROB_FILE)).astype(np.float64)
        s1 = idmap.reindex(u.e).to_numpy()
        qq = idmap.reindex(u.q).to_numpy()
        kk = s1 * N + qq
        pos = np.minimum(np.searchsorted(ks, kk), len(ks) - 1)
        inc = ks[pos] == kk
        pour = np.where(inc, pbl[o[pos]], np.nan)
        off = np.where((hv[qq] >= 0) & (hv[s1] >= 0), hv[qq] - hv[s1], 10 ** 9)
        isk = np.isin(off, KSET)
        need = pa.array(sorted(set(u.e) | set(u.q)))
        txt = {}
        for s in ("source1", "source2", "source3"):
            t = raw_test[s].filter(pc.is_in(raw_test[s].column("entity_id"), value_set=need))
            for i, n, ad in zip(t.column("entity_id").to_pylist(), t.column("business_name").to_pylist(),
                                t.column("business_address").to_pylist()):
                txt[i] = f"{n} | {ad}" if ad else n
        sc = _ce3([txt[x] for x in u.e], [txt[x] for x in u.q])
        ced = cache_path("crossenc_r3")
        y_tr = pq.read_table(os.path.join(ced, f"hard_train_{c}.parquet"), columns=["y"]).column("y").to_numpy()
        pce = IsotonicRegression(out_of_bounds="clip").fit(np.load(os.path.join(ced, f"ce_train_{c}.npy")), y_tr).predict(sc)
        mine, mex = u.ours.to_numpy(), u.ext.to_numpy()
        add, drop = mex & ~mine, mine & ~mex
        pA = np.where(inc, pour, pce)
        keep_add = add & ((inc & (np.nan_to_num(pour) >= T_IN) & ~isk) | (~inc & (pce >= T_OUT)))
        rm = drop & (isk | (inc & (np.nan_to_num(pour, nan=1.0) < T_DROP)))
        m = (mine & ~rm) | keep_add
        idx = np.flatnonzero(m)
        qarr = u.q.to_numpy()
        best = pd.Series(pA[idx], index=idx).groupby(qarr[idx]).idxmax().to_numpy()
        m2 = np.zeros(len(m), bool)
        m2[best] = True
        print(f"  stack {os.path.basename(team_path)} {c}: +{(m2 & ~mine).sum()} added, -{(mine & ~m2).sum()} removed", flush=True)
        new = u[m2].groupby("e").q.apply(lambda s: ",".join(sorted(s)))
        for e in u.e.unique():
            rows[e] = new.get(e, "")
    return rows


def french_refinement(rows, team_path, fr_ids):
    """Returns the France rows (sorted id lists); other countries are left to the caller unchanged."""
    team = pd.read_csv(team_path, sep="\t", dtype=str, keep_default_na=False).set_index("source1_entity_id").matched_entity_ids
    base = {e: set(rows[e].split(",")) if rows[e] else set() for e in fr_ids}
    ext = {e: set(team[e].split(",")) if team[e] else set() for e in fr_ids}
    rec = load_records("test", "France")
    ids = rec["ids"].to_pylist()
    idx = {x: i for i, x in enumerate(ids)}
    core = rec["n_core"].to_pylist()
    ast = rec["a_str"].to_pylist()
    hv = np.load(os.path.join(pairs_dir("test", "France"), "hnum_val.npy"))
    kset = set(KSET.tolist())

    def initials(s):
        return "".join(w[0] for w in s.split() if w not in LEGAL)

    assigned = {q for e in fr_ids for q in base[e]}
    add = rem = 0
    for e in fr_ids:
        i = idx[e]
        for q in ext[e] - base[e]:
            j = idx[q]
            qw = [w for w in core[j].split() if w not in LEGAL]
            acro = (bool(qw) and len(qw[0]) >= 2 and len(qw) <= 2 and qw[0] == initials(core[i])[:len(qw[0])]
                    and len(qw[0]) >= len(initials(core[i])) - 1)
            if acro and ast[j] and hv[j] >= 0 and hv[j] == hv[i] and q not in assigned:
                base[e].add(q)
                assigned.add(q)
                add += 1
        for q in list(base[e] - ext[e]):
            j = idx[q]
            if hv[j] >= 0 and hv[i] >= 0 and (hv[j] - hv[i]) in kset:
                base[e].discard(q)
                rem += 1
    print(f"  France refinement: +{add} initials matches, -{rem} disputed +K pairs", flush=True)
    return {e: ",".join(sorted(s)) for e, s in base.items()}


def _legal(ll, lid, b, j):
    lb, lq = ll[lid[b]], ll[lid[j]]
    return ("same" if lb else "none") if lb == lq else "dropped" if not lq else "added" if not lb else "changed"


def france_team_additions(cur, used, team_path, fr_ids, rec, hv):
    """Adds the team file's extra French pairs of two label-supported classes (cur: France S1 -> set, edited in place).
      * a filler word added / swapped in (TEAM_FILL, not a decoy-pattern word), same house number, record address
        present -- train analogue (US / India fillers at the same house) 98.5-100 % true;
      * identical core-name tokens, record without address, S1 core name unique in France -- train analogue 97-98 %.
    A record already assigned anywhere (used) is never given to a second S1."""
    team = pd.read_csv(team_path, sep="\t", dtype=str, keep_default_na=False).set_index("source1_entity_id").matched_entity_ids
    ids = rec["ids"].to_pylist()
    idx = {x: i for i, x in enumerate(ids)}
    core = rec["n_core"].to_pylist()
    ast = rec["a_str"].to_pylist()
    name_cnt = Counter(core[i] for i in np.flatnonzero(rec["src"] == 1))
    added = 0
    for e in fr_ids:
        ext = set(team[e].split(",")) if team[e] else set()
        for q in sorted(ext - cur[e]):
            i, j = idx[e], idx[q]
            tq, tb = set(core[j].split()), set(core[i].split())
            extra, dropped = tq - tb, tb - tq
            same_h = hv[j] >= 0 and hv[i] >= 0 and hv[j] == hv[i]
            filler = (extra and extra <= TEAM_FILL and (not dropped or (len(extra) == 1 and len(dropped) == 1))
                      and same_h and ast[j] and not (extra & DECOY_WORDS))
            ident = not extra and not dropped and not ast[j] and name_cnt[core[i]] == 1
            if (filler or ident) and q not in used:
                cur[e].add(q)
                used.add(q)
                added += 1
    print(f"  France team additions (file 3): +{added}", flush=True)


def france_level_and_fillers(cur, base, fr_ids, rec, hv):
    """Final French edits (cur: France S1 -> set, edited in place; base: the France sets before the team additions).
      1. remove team additions that swap in a filler while dropping the S1 legal form (train analogue ~2 % true);
      2. record without address whose core tokens equal those of exactly one S1: per legal-form transition, keep the
         additions (highest v3 probability first) only until France's assigned share reaches the train truth share
         (LEVEL_TARGET) -- beyond it an addition is more likely false than true;
      3. add record = S1 core + FR_ADD_FILLERS words at the same house number, same street, same legal form, when
         the record is unassigned and exactly one S1 qualifies.  Train analogue (same-house, same-legal filler
         additions incl. the mixed-use "partners" / "federation"): 99-100 % true; France assigned only 11 % / 81 % / 0 %
         of these (the matcher has no labelled French word identities) vs 100 % for services / associes / fils."""
    ids = rec["ids"].to_pylist()
    idx = {x: i for i, x in enumerate(ids)}
    core = rec["n_core"].to_pylist()
    ast = rec["a_str"].to_pylist()
    lid, ll, src, N = rec["legal_id"], rec["legal_list"], rec["src"], rec["N"]
    d = pairs_dir("test", "France")
    B = np.load(os.path.join(d, "B.npy")).astype(np.int64)
    Q = np.load(os.path.join(d, "Q.npy")).astype(np.int64)
    with open(os.path.join(d, "cols.json")) as fh:
        cols = json.load(fh)
    aq = np.asarray(np.load(os.path.join(d, "X.npy"), mmap_mode="r")[:, cols.index("a_soft_q")])
    from blend2 import v3_prob
    pmap = dict(zip((B * N + Q).tolist(), v3_prob("test", "France", "v3").tolist()))
    # 1. filler swaps that drop the legal form
    rm1 = 0
    for e in fr_ids:
        for q in list(cur[e] - base[e]):
            i, j = idx[e], idx[q]
            tb, tq = set(core[i].split()), set(core[j].split())
            if len(tq - tb) == 1 and len(tb - tq) == 1 and _legal(ll, lid, i, j) == "dropped":
                cur[e].discard(q)
                rm1 += 1
    # 2. level-match the empty-address identical-name cells
    key = lambda s: " ".join(sorted(set(s.split())))
    s1_by = defaultdict(list)
    for i in np.flatnonzero(src == 1):
        s1_by[key(core[i])].append(int(i))
    cell_n, cell_base, cell_adds = Counter(), Counter(), defaultdict(list)
    for j in range(N):
        if src[j] == 1 or ast[j] or not core[j]:
            continue
        c = s1_by.get(key(core[j]), [])
        if len(c) != 1:
            continue
        b = c[0]
        lg = _legal(ll, lid, b, j)
        e, q = ids[b], ids[j]
        cell_n[lg] += 1
        cell_base[lg] += q in base[e]
        if q in cur[e] and q not in base[e]:
            cell_adds[lg].append((pmap.get(b * N + j, -1.0), e, q))
    rm2 = 0
    for lg, t in LEVEL_TARGET.items():
        room = max(0, int(round(t * cell_n[lg])) - cell_base[lg])
        for _, e, q in sorted(cell_adds[lg], reverse=True)[room:]:
            cur[e].discard(q)
            rm2 += 1
    # 3. filler additions at the same house number
    assigned = {q for e in cur for q in cur[e]}
    cand = defaultdict(list)
    for k in np.flatnonzero((aq >= 0.9) & (hv[Q] >= 0) & (hv[Q] == hv[B]) & (lid[B] == lid[Q])):
        b, j = int(B[k]), int(Q[k])
        if ids[j] in assigned:
            continue
        tb, tq = set(core[b].split()), set(core[j].split())
        ex = tq - tb
        if ex and not (tb - tq) and ex <= FR_ADD_FILLERS:
            cand[j].append(b)
    new = [(ids[c[0]], ids[j]) for j, c in cand.items() if len(set(c)) == 1]
    for e, q in new:
        cur[e].add(q)
    print(f"  France level match: -{rm1} legal-dropping filler swaps, -{rm2} over-level identical-name additions; "
          f"filler additions: +{len(new)}", flush=True)


def _write(path, order, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for e in order:
            fh.write(f"{e}\t{rows[e]}\n")


def main(ours, our_cands, team1, team2, team3, out_matches, out_cands):
    s1 = load_source("test", "source1").select(["entity_id", "country"])
    order = s1.column("entity_id").to_pylist()
    country = dict(zip(order, s1.column("country").to_pylist()))
    raw_test = {s: load_source("test", s).select(["entity_id", "business_name", "business_address"])
                for s in ("source1", "source2", "source3")}
    cur = pd.read_csv(ours, sep="\t", dtype=str, keep_default_na=False).set_index("source1_entity_id").matched_entity_ids.to_dict()
    tmp = out_matches + ".stage.tsv"
    for team in (team1, team2):
        changed = stack(ours if team == team1 else tmp, team, country, raw_test)
        cur.update(changed)
        _write(tmp, order, cur)
    os.remove(tmp)
    fr_ids = [e for e in order if country[e] == "France"]
    cur.update(french_refinement(cur, team2, fr_ids))
    rows = cur
    if fr_ids:
        rec = load_records("test", "France")
        hv = np.load(os.path.join(pairs_dir("test", "France"), "hnum_val.npy"))
        base = {e: set(rows[e].split(",")) if rows[e] else set() for e in fr_ids}
        fr = {e: set(s) for e, s in base.items()}
        used = {q for v in rows.values() for q in (v.split(",") if v else [])}
        france_team_additions(fr, used, team3, fr_ids, rec, hv)
        france_level_and_fillers(fr, base, fr_ids, rec, hv)
        assert max(Counter(q for s in fr.values() for q in s).values()) == 1, "a record is matched to two S1"
        rows.update({e: ",".join(sorted(s)) for e, s in fr.items()})
    _write(out_matches, order, rows)
    # candidate file: our candidates + any matched pair the stack added
    cands = pd.read_csv(our_cands, sep="\t", dtype=str, keep_default_na=False).set_index("source1_entity_id").candidate_entity_ids.to_dict()
    added = 0
    with open(out_cands, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("source1_entity_id\tcandidate_entity_ids\n")
        for e in order:
            c = cands.get(e, "")
            have = set(c.split(",")) if c else set()
            extra = [q for q in (rows[e].split(",") if rows[e] else []) if q not in have]
            added += len(extra)
            fh.write(f"{e}\t{','.join(([c] if c else []) + extra)}\n")
    print(f"wrote {out_matches} and {out_cands} (+{added} stacked pairs added to the candidates)", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:8])
