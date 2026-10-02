"""Stage 4: pairwise features for candidate pairs (b = Source-1 record, q = Source-2/3 record).

Feature groups
  * name token alignment (IDF weighted, typo tolerant via Jaro-Winkler on tokens):
    coverage both ways, soft coverage, max IDF of unmatched tokens, counts, cosine
  * alt-name (website domain / alias) coverage of the S1 name
  * address alpha-token alignment, numeric-token overlap, house-number agreement,
    city / canonical-state agreement
  * rapidfuzz string similarities (ratio, token-set, partial, JW, concatenated,
    full-name, address strings, city)
  * legal-form agreement, record flags, source
  * blocking context: score, ranks within the S1 record's and the query's lists
All features are country-agnostic.
"""
import numba as nb
import numpy as np
import pyarrow as pa
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

TOK_L = 24
THETA = 0.86


def token_char_table(vocab):
    V = len(vocab)
    chars = np.zeros((V, TOK_L), np.uint8)
    lens = np.zeros(V, np.int32)
    for i, t in enumerate(vocab):
        b = t.encode("ascii", "ignore")[:TOK_L]
        chars[i, :len(b)] = np.frombuffer(b, np.uint8)
        lens[i] = len(b)
    return chars, lens


def numeric_values(vocab):
    out = np.full(len(vocab), -1, np.int64)
    for i, t in enumerate(vocab):
        if t.isdigit() and len(t) <= 12:
            out[i] = int(t)
    return out


@nb.njit(cache=True, inline="always")
def _jw(chars, lens, x, y):
    if x == y:
        return 1.0
    la = lens[x]
    lb = lens[y]
    if la == 0 or lb == 0:
        return 0.0
    a = chars[x]
    b = chars[y]
    md = max(la, lb) // 2 - 1
    if md < 0:
        md = 0
    af = np.zeros(TOK_L, np.bool_)
    bf = np.zeros(TOK_L, np.bool_)
    m = 0
    for i in range(la):
        lo = max(0, i - md)
        hi = min(i + md + 1, lb)
        for j in range(lo, hi):
            if not bf[j] and a[i] == b[j]:
                af[i] = True
                bf[j] = True
                m += 1
                break
    if m == 0:
        return 0.0
    t = 0
    k = 0
    for i in range(la):
        if af[i]:
            while not bf[k]:
                k += 1
            if a[i] != b[k]:
                t += 1
            k += 1
    jaro = (m / la + m / lb + (m - t / 2.0) / m) / 3.0
    p = 0
    for i in range(min(4, min(la, lb))):
        if a[i] == b[i]:
            p += 1
        else:
            break
    sim = jaro + p * 0.1 * (1.0 - jaro)
    # abbreviation / truncation: one token is a prefix of the other
    s = min(la, lb)
    if s >= 3:
        pre = True
        for i in range(s):
            if a[i] != b[i]:
                pre = False
                break
        if pre and sim < 0.9:
            sim = 0.9
    return sim


@nb.njit(cache=True)
def _align(chars, lens, idf, ta, tb, skip_num, isnum):
    """For tokens ta (list of ids) against tb: returns
    hard_cov_a, soft_cov_a, unm_max_a, unm_cnt_a, exact_cnt_a, idf_sum_a, n_a."""
    wsum = 0.0
    hard = 0.0
    soft = 0.0
    umax = 0.0
    ucnt = 0
    ex = 0
    n = 0
    for x in ta:
        if skip_num and isnum[x]:
            continue
        n += 1
        w = idf[x]
        wsum += w
        best = 0.0
        exact = False
        for y in tb:
            if skip_num and isnum[y]:
                continue
            if x == y:
                best = 1.0
                exact = True
                break
            s = _jw(chars, lens, x, y)
            if s > best:
                best = s
        if exact:
            ex += 1
        soft += w * best
        if best >= THETA:
            hard += w
        else:
            ucnt += 1
            if w > umax:
                umax = w
    if wsum > 0:
        hard /= wsum
        soft /= wsum
    return hard, soft, umax, ucnt, ex, wsum, n


@nb.njit(cache=True)
def _cos(ta, tb, idf):
    na = 0.0
    nb_ = 0.0
    dot = 0.0
    for x in ta:
        na += idf[x] * idf[x]
    for y in tb:
        nb_ += idf[y] * idf[y]
    for x in ta:
        for y in tb:
            if x == y:
                dot += idf[x] * idf[x]
                break
    if na == 0 or nb_ == 0:
        return 0.0
    return dot / np.sqrt(na * nb_)


@nb.njit(cache=True)
def _lev(chars, lens, x, y):
    la = lens[x]
    lb = lens[y]
    prev = np.arange(lb + 1)
    cur = np.zeros(lb + 1, np.int64)
    for i in range(1, la + 1):
        cur[0] = i
        for j in range(1, lb + 1):
            c = 0 if chars[x, i - 1] == chars[y, j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + c)
        prev[:] = cur
    return prev[lb]


N_TOKFEAT = 37


@nb.njit(parallel=True, cache=True)
def token_features(B, Q, n_ptr, n_idx, n_idf, n_chars, n_lens, alt_ptr, alt_idx,
                   a_ptr, a_idx, a_idf, a_chars, a_lens, a_isnum, a_numval,
                   hnum_id, city_id, state_canon, n_isnum):
    P = len(B)
    out = np.full((P, N_TOKFEAT), np.nan, np.float32)
    for p in nb.prange(P):
        b = B[p]
        q = Q[p]
        tb = n_idx[n_ptr[b]:n_ptr[b + 1]]
        tq = n_idx[n_ptr[q]:n_ptr[q + 1]]
        # --- name
        h, s, um, uc, ex, ws, n = _align(n_chars, n_lens, n_idf, tq, tb, False, n_isnum)
        out[p, 0] = h
        out[p, 2] = s
        out[p, 4] = um
        out[p, 6] = uc
        out[p, 8] = ex / n if n > 0 else 0.0
        out[p, 10] = n
        out[p, 12] = ws
        h, s, um, uc, ex, ws, n = _align(n_chars, n_lens, n_idf, tb, tq, False, n_isnum)
        out[p, 1] = h
        out[p, 3] = s
        out[p, 5] = um
        out[p, 7] = uc
        out[p, 9] = _jw(n_chars, n_lens, tq[0], tb[0]) if len(tq) > 0 and len(tb) > 0 else 0.0
        out[p, 11] = n
        out[p, 13] = ws
        out[p, 14] = _cos(tq, tb, n_idf)
        # --- alt name of q vs S1 name (coverage of S1 tokens by alt tokens)
        ta = alt_idx[alt_ptr[q]:alt_ptr[q + 1]]
        if len(ta) > 0:
            h, s, um, uc, ex, ws, n = _align(n_chars, n_lens, n_idf, tb, ta, False, n_isnum)
            out[p, 15] = s
        else:
            out[p, 15] = -1.0
        # --- address alpha tokens
        ab = a_idx[a_ptr[b]:a_ptr[b + 1]]
        aq = a_idx[a_ptr[q]:a_ptr[q + 1]]
        h, s, um, uc, ex, ws, n = _align(a_chars, a_lens, a_idf, aq, ab, True, a_isnum)
        out[p, 16] = h
        out[p, 18] = s
        out[p, 20] = um
        out[p, 22] = n
        out[p, 34] = uc
        h, s, um, uc, ex, ws, n = _align(a_chars, a_lens, a_idf, ab, aq, True, a_isnum)
        out[p, 17] = h
        out[p, 19] = s
        out[p, 21] = um
        out[p, 23] = n
        out[p, 35] = uc
        out[p, 24] = _cos(aq, ab, a_idf)
        # --- numbers
        nq = 0
        nbb = 0
        hitq = 0
        hitb = 0
        for x in aq:
            if a_isnum[x]:
                nq += 1
                for y in ab:
                    if y == x:
                        hitq += 1
                        break
        for y in ab:
            if a_isnum[y]:
                nbb += 1
                for x in aq:
                    if y == x:
                        hitb += 1
                        break
        out[p, 25] = nq
        out[p, 26] = nbb
        out[p, 27] = hitq / nq if nq > 0 else -1.0
        out[p, 28] = hitb / nbb if nbb > 0 else -1.0
        hb = hnum_id[b]
        hq = hnum_id[q]
        if hb < 0 or hq < 0:
            out[p, 29] = -1.0
            out[p, 30] = -1.0
            out[p, 31] = -1.0
            out[p, 36] = -1.0
        else:
            out[p, 29] = 1.0 if hb == hq else 0.0
            out[p, 36] = _lev(a_chars, a_lens, hb, hq)
            vb = a_numval[hb]
            vq = a_numval[hq]
            if vb >= 0 and vq >= 0:
                mx = max(vb, vq)
                out[p, 30] = abs(vb - vq) / mx if mx > 0 else 0.0
            # digit-substring relation (977 vs 3977, 2 vs 23)
            sub = 0.0
            lb_ = a_lens[hb]
            lq_ = a_lens[hq]
            if lb_ != lq_ or hb == hq:
                if lb_ < lq_:
                    sa, la_, sb, lb2 = hb, lb_, hq, lq_
                else:
                    sa, la_, sb, lb2 = hq, lq_, hb, lb_
                for st in range(lb2 - la_ + 1):
                    okk = True
                    for i in range(la_):
                        if a_chars[sa, i] != a_chars[sb, st + i]:
                            okk = False
                            break
                    if okk:
                        sub = 1.0
                        break
            out[p, 31] = sub
        # --- city / state
        cb = city_id[b]
        cq = city_id[q]
        out[p, 32] = -1.0 if (cb == 0 or cq == 0) else (1.0 if cb == cq else 0.0)
        sb_ = state_canon[b]
        sq_ = state_canon[q]
        out[p, 33] = -1.0 if (sb_ == 0 or sq_ == 0) else (1.0 if sb_ == sq_ else 0.0)
    return out


@nb.njit(parallel=True, cache=True)
def unmatched_tokens(B, Q, n_ptr, n_idx, n_chars, n_lens, K):
    """For each pair: up to K name tokens of q with no match in b ("extra" words) and
    up to K tokens of b with no match in q ("dropped" words); -1 padded."""
    P = len(B)
    ex = np.full((P, K), -1, np.int32)
    dr = np.full((P, K), -1, np.int32)
    for p in nb.prange(P):
        tb = n_idx[n_ptr[B[p]]:n_ptr[B[p] + 1]]
        tq = n_idx[n_ptr[Q[p]]:n_ptr[Q[p] + 1]]
        c = 0
        for x in tq:
            best = 0.0
            for y in tb:
                s = _jw(n_chars, n_lens, x, y)
                if s > best:
                    best = s
            if best < THETA and c < K:
                ex[p, c] = x
                c += 1
        c = 0
        for y in tb:
            best = 0.0
            for x in tq:
                s = _jw(n_chars, n_lens, y, x)
                if s > best:
                    best = s
            if best < THETA and c < K:
                dr[p, c] = y
                c += 1
    return ex, dr


TOKFEAT_NAMES = [
    "n_cov_q", "n_cov_b", "n_soft_q", "n_soft_b", "n_unm_q_max", "n_unm_b_max", "n_unm_q_cnt",
    "n_unm_b_cnt", "n_exact_q", "n_first_sim", "n_len_q", "n_len_b", "n_idf_q", "n_idf_b", "n_cos",
    "alt_cov", "a_cov_q", "a_cov_b", "a_soft_q", "a_soft_b", "a_unm_q_max", "a_unm_b_max", "a_len_q",
    "a_len_b", "a_cos", "num_cnt_q", "num_cnt_b", "num_cov_q", "num_cov_b", "hnum_eq", "hnum_rel",
    "hnum_sub", "city_eq", "state_eq", "a_unm_q_cnt", "a_unm_b_cnt", "hnum_lev",
]
CATEGORICAL = ["lg_b", "lg_q"]


def legal_vocab(min_count=100):
    """Global id for every legal-family combination seen in the training data
    (fixed vocabulary shared by train and test; unseen combinations -> NaN)."""
    import collections
    import json
    import os

    import pyarrow.parquet as pq

    from config import SOURCES, cache_path
    from prep import prep_path
    fn = cache_path("legal_vocab.json")
    if os.path.exists(fn):
        with open(fn) as fh:
            return json.load(fh)
    cnt = collections.Counter()
    for s in SOURCES:
        col = pq.read_table(prep_path("train", s), columns=["n_legal"]).column("n_legal")
        cnt.update(col.to_pylist())
    vocab = {k: i for i, k in enumerate(sorted(k for k, v in cnt.items() if v >= min_count))}
    with open(fn, "w") as fh:
        json.dump(vocab, fh)
    return vocab


def legal_relation(sa, sb):
    if not sa and not sb:
        return 0
    if not sa:
        return 1
    if not sb:
        return 2
    if sa == sb:
        return 3
    if sb < sa:
        return 4
    if sb > sa:
        return 5
    if sa & sb:
        return 6
    return 7


class PairFeaturizer:
    """Holds per-country lookup tables; computes feature blocks for pair chunks."""

    def __init__(self, rec):
        self.rec = rec
        self.n_chars, self.n_lens = token_char_table(rec["n_vocab"])
        self.a_chars, self.a_lens = token_char_table(rec["a_vocab"])
        self.a_numval = numeric_values(rec["a_vocab"])
        self.n_isnum = np.array([t.isdigit() for t in rec["n_vocab"]], dtype=np.bool_)
        self.city_arr = pa.array(rec["city_list"], pa.string())
        legal = rec["legal_list"]
        self.legal_sets = [set(l.split()) for l in legal]
        gv = legal_vocab()
        self.legal_gid = np.array([gv.get(l, np.nan) for l in legal], dtype=np.float32)

    def compute(self, B, Q):
        r = self.rec
        B = np.ascontiguousarray(B, dtype=np.int64)
        Q = np.ascontiguousarray(Q, dtype=np.int64)
        tok = token_features(
            B, Q, r["n_ptr"], r["n_idx"], r["n_idf"], self.n_chars, self.n_lens, r["alt_ptr"], r["alt_idx"],
            r["a_ptr"], r["a_idx"], r["a_idf"], self.a_chars, self.a_lens, r["a_isnum"], self.a_numval,
            r["hnum_id"], r["city_id"], r["state_canon"], self.n_isnum,
        )
        feats = {n: tok[:, i] for i, n in enumerate(TOKFEAT_NAMES)}
        bi = pa.array(B)
        qi = pa.array(Q)
        cb = r["n_core"].take(bi).to_pylist()
        cq = r["n_core"].take(qi).to_pylist()
        kw = dict(workers=-1, dtype=np.float32)
        feats["r_ratio"] = cpdist(cq, cb, scorer=fuzz.ratio, **kw)
        feats["r_tset"] = cpdist(cq, cb, scorer=fuzz.token_set_ratio, **kw)
        feats["r_tsort"] = cpdist(cq, cb, scorer=fuzz.token_sort_ratio, **kw)
        feats["r_partial"] = cpdist(cq, cb, scorer=fuzz.partial_ratio, **kw)
        feats["r_jw"] = cpdist(cq, cb, scorer=JaroWinkler.normalized_similarity, **kw)
        cbc = [s.replace(" ", "") for s in cb]
        cqc = [s.replace(" ", "") for s in cq]
        feats["r_concat"] = cpdist(cqc, cbc, scorer=fuzz.ratio, **kw)
        feats["r_concat_partial"] = cpdist(cqc, cbc, scorer=fuzz.partial_ratio, **kw)
        del cb, cq, cbc, cqc
        fb = r["n_full"].take(bi).to_pylist()
        fq = r["n_full"].take(qi).to_pylist()
        feats["r_full"] = cpdist(fq, fb, scorer=fuzz.token_sort_ratio, **kw)
        del fb, fq
        ab = r["a_str"].take(bi).to_pylist()
        aq = r["a_str"].take(qi).to_pylist()
        feats["r_a_tset"] = cpdist(aq, ab, scorer=fuzz.token_set_ratio, **kw)
        feats["r_a_ratio"] = cpdist(aq, ab, scorer=fuzz.ratio, **kw)
        feats["r_a_partial"] = cpdist(aq, ab, scorer=fuzz.partial_ratio, **kw)
        empty_q = np.array([len(s) == 0 for s in aq])
        del ab, aq
        cyb = self.city_arr.take(pa.array(r["city_id"][B])).to_pylist()
        cyq = self.city_arr.take(pa.array(r["city_id"][Q])).to_pylist()
        feats["r_city"] = cpdist(cyq, cyb, scorer=JaroWinkler.normalized_similarity, **kw)
        del cyb, cyq
        # legal forms
        lb = r["legal_id"][B]
        lq = r["legal_id"][Q]
        ls = self.legal_sets
        same = (lb == lq).astype(np.float32)
        jac = np.empty(len(B), np.float32)
        conflict = np.zeros(len(B), np.float32)
        pub = np.zeros(len(B), np.float32)
        rel = np.zeros(len(B), np.float32)
        cache = {}
        for i in range(len(B)):
            key = (lb[i], lq[i])
            v = cache.get(key)
            if v is None:
                sa, sb = ls[lb[i]], ls[lq[i]]
                u = len(sa | sb)
                j = len(sa & sb) / u if u else 1.0
                c = 1.0 if (sa and sb and not (sa & sb)) else 0.0
                pm = 1.0 if (("public" in sa) != ("public" in sb)) else 0.0
                v = (j, c, pm, legal_relation(sa, sb))
                cache[key] = v
            jac[i], conflict[i], pub[i], rel[i] = v
        feats["legal_same"] = same
        feats["legal_jacc"] = jac
        feats["legal_conflict"] = conflict
        feats["legal_public_mm"] = pub
        feats["legal_rel"] = rel
        feats["lg_b"] = self.legal_gid[lb]
        feats["lg_q"] = self.legal_gid[lq]
        feats["legal_q_empty"] = np.array([not ls[x] for x in lq], np.float32)
        feats["legal_b_empty"] = np.array([not ls[x] for x in lb], np.float32)
        # flags / source
        fl = r["n_flags"][Q]
        feats["q_indic"] = (fl & 1).astype(np.float32)
        feats["q_pipe"] = ((fl & 2) > 0).astype(np.float32)
        feats["q_alias"] = ((fl & 4) > 0).astype(np.float32)
        feats["q_web"] = ((fl & 8) > 0).astype(np.float32)
        feats["q_unk"] = r["n_unk"][Q].astype(np.float32)
        feats["q_src"] = r["src"][Q].astype(np.float32)
        feats["q_addr_empty"] = empty_q.astype(np.float32)
        return feats


def context_features(B, Q, score, n_records):
    """Rank / competition features from the blocking scores."""
    P = len(B)
    out = {}
    out["blk_score"] = score.astype(np.float32)
    for name, key in (("q", Q), ("b", B)):
        order = np.lexsort((-score, key))
        ks = key[order]
        first = np.r_[True, ks[1:] != ks[:-1]]
        start = np.flatnonzero(first)
        grp = np.cumsum(first) - 1
        rank = np.empty(P, np.float32)
        rank[order] = np.arange(P) - start[grp]
        cnt = np.bincount(key, minlength=n_records)
        best = np.full(n_records, -1.0, np.float32)
        np.maximum.at(best, key, score)
        # second best
        sc_sorted = score[order]
        second = np.full(n_records, 0.0, np.float32)
        idx2 = start + 1
        has2 = idx2 < np.r_[start[1:], P]
        second[ks[start[has2]]] = sc_sorted[idx2[has2]]
        out[f"{name}_rank"] = rank
        out[f"{name}_ncand"] = cnt[key].astype(np.float32)
        out[f"{name}_gap"] = (best[key] - score).astype(np.float32)
        out[f"{name}_second"] = second[key]
        out[f"{name}_best"] = best[key]
    return out
