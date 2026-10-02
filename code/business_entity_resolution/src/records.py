"""Stage 2: turn normalised records into compact per-country arrays.

For every (split, country) we build one structure holding S1, S2 and S3 records:
  * name strings (core / full / alt) as Arrow arrays, legal family id, flags
  * address with the state component removed (data-driven state vocabulary),
    canonical state id, city id, house-number token id, address string
  * int32 token-id CSR arrays for name / alt-name / address tokens + IDF vectors
Concatenated web/handle names ("modernconstructionenterprises") are segmented
into vocabulary words with a unigram DP.  Everything is streamed in chunks to
keep memory low.
"""
import array
import collections
import math
import os
import pickle
import time

import numba as nb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from config import SOURCES, cache_path
from prep import prep_path

CHUNK = 250_000


def parse_comps(s):
    if not s:
        return []
    return [c.split() for c in s.split(" | ") if c]


# ----------------------------------------------------------------------------- states
def split_state(comps, states):
    """Remove state components wherever they appear (components get reordered).
    Returns (comps, state) where state is the last state component found."""
    state = ""
    out = []
    for c in comps:
        s = " ".join(c)
        if s in states:
            state = s
        else:
            out.append(c)
    return out, state


def guess_city(comps):
    for c in reversed(comps):
        if not any(t.isdigit() for t in c):
            return " ".join(c)
    return ""


def house_number(comps):
    for c in comps:
        for t in c:
            if t.isdigit():
                return t
    return ""


# ----------------------------------------------------------------------------- segmentation
class Segmenter:
    def __init__(self, counter, min_count=3):
        tot = max(1, sum(counter.values()))
        self.logp = {w: math.log(c / tot) for w, c in counter.items() if c >= min_count and len(w) >= 2}
        self.maxlen = 20
        self.unk = math.log(1.0 / tot) - 4.0

    def __call__(self, s):
        if s in self.logp or len(s) < 5:
            return [s]
        n = len(s)
        best = [0.0] + [-1e18] * n
        back = [0] * (n + 1)
        for i in range(1, n + 1):
            for j in range(max(0, i - self.maxlen), i):
                w = s[j:i]
                lp = self.logp.get(w)
                if lp is None:
                    if i - j != 1:
                        continue
                    lp = self.unk
                v = best[j] + lp
                if v > best[i]:
                    best[i] = v
                    back[i] = j
        out, i = [], n
        while i > 0:
            j = back[i]
            out.append(s[j:i])
            i = j
        out.reverse()
        merged = []
        for w in out:
            if len(w) == 1 and w not in self.logp and merged and merged[-1] not in self.logp:
                merged[-1] += w
            else:
                merged.append(w)
        known = sum(len(w) for w in merged if w in self.logp)
        if known < 0.6 * n:
            return [s]
        return merged


# ----------------------------------------------------------------------------- build
@nb.njit(cache=True)
def doc_freq(ptr, idx, V):
    df = np.zeros(V, np.int64)
    mark = np.full(V, -1, np.int64)
    for i in range(len(ptr) - 1):
        for p in range(ptr[i], ptr[i + 1]):
            t = idx[p]
            if mark[t] != i:
                mark[t] = i
                df[t] += 1
    return df


class _Vocab(dict):
    def id(self, t):
        j = self.get(t)
        if j is None:
            j = len(self)
            self[t] = j
        return j


def rec_path(split, country):
    return cache_path(f"rec_{split}_{country}.pkl")


def split_countries(split):
    countries = set()
    for s in SOURCES:
        col = pq.read_table(prep_path(split, s), columns=["country"]).column("country")
        countries |= set(pc.unique(col).to_pylist())
    return sorted(countries)


def build_split(split, force=False):
    for country in split_countries(split):
        fn = rec_path(split, country)
        if os.path.exists(fn) and not force:
            continue
        t0 = time.time()
        tabs = []
        for k, s in enumerate(SOURCES, start=1):
            t = pq.read_table(prep_path(split, s), filters=[("country", "=", country)])
            t = t.append_column("src", pa.array(np.full(t.num_rows, k, dtype=np.int8)))
            tabs.append(t)
        sub = pa.concat_tables(tabs).combine_chunks()
        del tabs, t
        rec = build_country(sub)
        rec["country"] = country
        with open(fn, "wb") as fh:
            pickle.dump(rec, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"records {split} {country}: {sub.num_rows} rows in {time.time() - t0:.0f}s", flush=True)
        del sub, rec


def _chunks(col, n):
    for i in range(0, n, CHUNK):
        yield col.slice(i, CHUNK).to_pylist()


def build_country(sub):
    N = sub.num_rows
    src = sub.column("src").to_numpy()
    comps_col = sub.column("a_comps")

    # ---- pass 1: state vocabulary (frequent, almost-always-last components)
    last_cnt = collections.Counter()
    for lst in _chunks(comps_col, N):
        for s in lst:
            if s:
                j = s.rfind(" | ")
                last_cnt[s[j + 3:] if j >= 0 else s] += 1
    cand = {s for s, n in last_cnt.items() if n >= 150 and not any(ch.isdigit() for ch in s)}
    tot = collections.Counter()
    for lst in _chunks(comps_col, N):
        for s in lst:
            if s:
                for c in s.split(" | "):
                    if c in cand:
                        tot[c] += 1
    states = {s for s in cand if last_cnt[s] >= 0.6 * tot[s]}

    # ---- Source-1 name vocabulary for segmentation of concatenated names
    wc = collections.Counter()
    core_col = sub.column("n_core")
    s1_mask = pa.array(src == 1)
    for lst in _chunks(pc.filter(core_col, s1_mask), int((src == 1).sum())):
        for c in lst:
            wc.update(c.split())
    seg = Segmenter(wc)
    del wc

    # ---- pass 2: per-record features
    flags = sub.column("n_flags").to_numpy()
    nvocab, avocab, cityv, statev = _Vocab(), _Vocab(), _Vocab(), _Vocab()
    cityv.id("")
    statev.id("")
    n_idx, alt_idx, a_idx = array.array("i"), array.array("i"), array.array("i")
    n_ptr = np.zeros(N + 1, dtype=np.int64)
    alt_ptr = np.zeros(N + 1, dtype=np.int64)
    a_ptr = np.zeros(N + 1, dtype=np.int64)
    city_id = np.zeros(N, dtype=np.int32)
    state_id = np.zeros(N, dtype=np.int32)
    hnum_id = np.full(N, -1, dtype=np.int32)
    n_core_out, n_alt_out, a_str_out = [], [], []
    alt_col = sub.column("n_alt")
    i = 0
    for cores, alts, comps_s in zip(_chunks(core_col, N), _chunks(alt_col, N), _chunks(comps_col, N)):
        cc, aa, ss = [], [], []
        for core, alt, cs in zip(cores, alts, comps_s):
            toks = core.split()
            if flags[i] & 8:
                nt = []
                for w in toks:
                    nt.extend(seg(w) if len(w) >= 7 else [w])
                toks = nt
                core = " ".join(toks)
            for tk in toks:
                n_idx.append(nvocab.id(tk))
            n_ptr[i + 1] = len(n_idx)
            atoks = []
            for w in alt.split():
                atoks.extend(seg(w) if len(w) >= 7 else [w])
            for tk in atoks:
                alt_idx.append(nvocab.id(tk))
            alt_ptr[i + 1] = len(alt_idx)
            comps, st = split_state(parse_comps(cs), states)
            state_id[i] = statev.id(st)
            city_id[i] = cityv.id(guess_city(comps))
            hn = house_number(comps)
            for c in comps:
                for tk in c:
                    a_idx.append(avocab.id(tk))
            a_ptr[i + 1] = len(a_idx)
            if hn:
                hnum_id[i] = avocab[hn]
            cc.append(core)
            aa.append(" ".join(atoks))
            ss.append(", ".join(" ".join(c) for c in comps))
            i += 1
        n_core_out.append(pa.array(cc, pa.string()))
        n_alt_out.append(pa.array(aa, pa.string()))
        a_str_out.append(pa.array(ss, pa.string()))

    n_idx = np.frombuffer(n_idx, dtype=np.int32).copy()
    alt_idx = np.frombuffer(alt_idx, dtype=np.int32).copy()
    a_idx = np.frombuffer(a_idx, dtype=np.int32).copy()

    # ---- canonical states via city co-occurrence (unsupervised)
    state_list = list(statev.keys())
    city_list = list(cityv.keys())
    cs1 = collections.Counter(zip(city_id[src == 1].tolist(), state_id[src == 1].tolist()))
    city_state = collections.defaultdict(collections.Counter)
    for (c, s), n in cs1.items():
        if c and s:
            city_state[c][s] += n
    city_major = {}
    for c, cn in city_state.items():
        s, n = cn.most_common(1)[0]
        if n >= 3 and n >= 0.8 * sum(cn.values()):
            city_major[c] = s
    s1_states = {s for (c, s) in cs1 if s}
    votes = collections.defaultdict(collections.Counter)
    cso = collections.Counter(zip(city_id[src != 1].tolist(), state_id[src != 1].tolist()))
    for (c, s), n in cso.items():
        if s and c in city_major:
            votes[s][city_major[c]] += n
    canon = np.arange(len(state_list), dtype=np.int32)
    for s, vc in votes.items():
        if s in s1_states:
            continue
        best, n = vc.most_common(1)[0]
        if n >= 20 and n >= 0.6 * sum(vc.values()):
            canon[s] = best
    state_canon = canon[state_id]

    # ---- document frequencies / idf
    ndf = doc_freq(n_ptr, n_idx, len(nvocab))
    adf = doc_freq(a_ptr, a_idx, len(avocab))
    n_idf = (np.log((N + 1) / (ndf + 1)) + 1.0).astype(np.float32)
    a_idf = (np.log((N + 1) / (adf + 1)) + 1.0).astype(np.float32)
    a_isnum = np.array([t.isdigit() for t in avocab.keys()], dtype=bool)

    legal_vocab = _Vocab()
    legal_id = np.array([legal_vocab.id(l) for l in sub.column("n_legal").to_pylist()], dtype=np.int32)

    return dict(
        ids=sub.column("entity_id"), src=src, N=N,
        n_core=pa.concat_arrays(n_core_out), n_full=sub.column("n_full").combine_chunks(),
        n_alt=pa.concat_arrays(n_alt_out), a_str=pa.concat_arrays(a_str_out),
        legal_id=legal_id, legal_list=list(legal_vocab.keys()),
        n_flags=flags, n_unk=sub.column("n_unk").to_numpy(),
        n_ptr=n_ptr, n_idx=n_idx, alt_ptr=alt_ptr, alt_idx=alt_idx, n_idf=n_idf, ndf=ndf,
        n_vocab=list(nvocab.keys()),
        a_ptr=a_ptr, a_idx=a_idx, a_idf=a_idf, adf=adf, a_isnum=a_isnum, a_vocab=list(avocab.keys()),
        hnum_id=hnum_id, city_id=city_id, city_list=city_list,
        state_id=state_id, state_canon=state_canon, state_list=state_list,
        states=sorted(states),
    )


def load_records(split, country):
    with open(rec_path(split, country), "rb") as fh:
        return pickle.load(fh)


if __name__ == "__main__":
    import sys
    for sp in (sys.argv[1:] or ["train", "test"]):
        build_split(sp)
