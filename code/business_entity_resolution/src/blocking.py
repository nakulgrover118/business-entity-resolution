"""Stage 3: candidate generation (blocking).

Each record becomes a sparse IDF-weighted vector over
  * name core tokens and adjacent name-token bigrams,
  * alt-name tokens (website domains / aliases),
  * address tokens (state removed) and house-number+street-word combos.
Name and address parts are L2-normalised separately and mixed, so the dot
product is a weighted sum of a name cosine and an address cosine.  Very common
features (high Source-1 document frequency) are dropped from the *index* only to
bound the cost of the sparse product; the norms still include them.

For every Source 2/3 record we keep its top-K Source 1 records (same country)
with a multithreaded sparse top-n matrix product, then also add, for every
Source 1 record, its best-scoring Source 2/3 records (reverse direction) so that
entities with many look-alikes keep their true matches.
"""
import time

import numba as nb
import numpy as np
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn

from config import N_JOBS


@nb.njit(cache=True)
def _bigrams(ptr, idx, V):
    n = len(ptr) - 1
    cnt = 0
    for i in range(n):
        k = ptr[i + 1] - ptr[i]
        if k > 1:
            cnt += k - 1
    rows = np.empty(cnt, np.int64)
    keys = np.empty(cnt, np.int64)
    c = 0
    for i in range(n):
        for p in range(ptr[i], ptr[i + 1] - 1):
            rows[c] = i
            keys[c] = np.int64(idx[p]) * V + idx[p + 1]
            c += 1
    return rows, keys


@nb.njit(cache=True)
def _hnum_combo(ptr, idx, isnum, common, V):
    """house number (first numeric token) + the next non-numeric, non-common token
    (skips street types / articles such as 'rue', 'de', 'st' that precede the
    street name in some address styles)."""
    n = len(ptr) - 1
    rows = np.empty(n, np.int64)
    keys = np.empty(n, np.int64)
    c = 0
    for i in range(n):
        h = -1
        for p in range(ptr[i], ptr[i + 1]):
            t = idx[p]
            if h < 0:
                if isnum[t]:
                    h = t
            elif not isnum[t] and not common[t]:
                rows[c] = i
                keys[c] = np.int64(h) * V + t
                c += 1
                break
    return rows[:c], keys[:c]


def _group(rows, keys, n_rows):
    """Deduplicate (row,key) and re-index keys to 0..F-1. Returns rows, fids, F."""
    order = np.lexsort((keys, rows))
    rows, keys = rows[order], keys[order]
    keep = np.ones(len(rows), bool)
    keep[1:] = (rows[1:] != rows[:-1]) | (keys[1:] != keys[:-1])
    rows, keys = rows[keep], keys[keep]
    uk, fid = np.unique(keys, return_inverse=True)
    return rows, fid.astype(np.int64), len(uk)


@nb.njit(cache=True)
def _combo(ptr, idx, other, V2):
    """(token, other[row]) combos for every token of every row with other[row] >= 0."""
    n = len(ptr) - 1
    cnt = 0
    for i in range(n):
        if other[i] >= 0:
            cnt += ptr[i + 1] - ptr[i]
    rows = np.empty(cnt, np.int64)
    keys = np.empty(cnt, np.int64)
    c = 0
    for i in range(n):
        o = other[i]
        if o < 0:
            continue
        for p in range(ptr[i], ptr[i + 1]):
            rows[c] = i
            keys[c] = np.int64(idx[p]) * V2 + o
            c += 1
    return rows, keys


GROUPS = {"n": 0, "alt": 0, "a": 1, "c": 2, "g": 3}


def token_ngrams(vocab, n=4, min_len=5):
    """CSR (ptr, codes) of character n-gram codes (with boundary marks) per vocab token."""
    ptr = np.zeros(len(vocab) + 1, np.int64)
    codes = []
    for i, t in enumerate(vocab):
        if len(t) >= min_len:
            s = "^" + t + "$"
            for j in range(len(s) - n + 1):
                g = s[j:j + n]
                c = 0
                for ch in g:
                    o = ord(ch)
                    v = 37 if ch == "^" else 38 if ch == "$" else (o - 48 if o < 58 else o - 97 + 10)
                    c = c * 40 + v
                codes.append(c)
        ptr[i + 1] = len(codes)
    return ptr, np.asarray(codes, np.int64)


@nb.njit(cache=True)
def _expand_ngrams(ptr, idx, tptr, tcodes):
    n = len(ptr) - 1
    cnt = 0
    for i in range(n):
        for p in range(ptr[i], ptr[i + 1]):
            t = idx[p]
            cnt += tptr[t + 1] - tptr[t]
    rows = np.empty(cnt, np.int64)
    keys = np.empty(cnt, np.int64)
    c = 0
    for i in range(n):
        for p in range(ptr[i], ptr[i + 1]):
            t = idx[p]
            for k in range(tptr[t], tptr[t + 1]):
                rows[c] = i
                keys[c] = tcodes[k]
                c += 1
    return rows, keys


def build_block_matrix(rec, weights=(0.4, 0.35, 0.25, 0.15), max_df=4000, verbose=False, ngrams=False,
                       with_addr=False, groups_out=()):
    """Returns (Q, B, s1_rows, q_rows) : CSR float32 matrices for queries (S2/S3) and index (S1)."""
    N = rec["N"]
    src = rec["src"]
    is_s1 = src == 1
    parts = []  # list of (rows, fid, weight_per_fid, group) with fid local to part

    def add_part(rows, fids, F, group):
        # idf from df over all records, df_s1 for pruning
        df = np.bincount(fids, minlength=F)
        df1 = np.bincount(fids[is_s1[rows]], minlength=F)
        idf = np.log((N + 1) / (df + 1)) + 1.0
        parts.append((rows, fids, idf, df1, F, group))

    # name tokens
    rows = np.repeat(np.arange(N, dtype=np.int64), np.diff(rec["n_ptr"]))
    r, f, F = _group(rows, rec["n_idx"].astype(np.int64), N)
    add_part(r, f, F, "n")
    # name bigrams
    rb, kb = _bigrams(rec["n_ptr"], rec["n_idx"], np.int64(len(rec["n_vocab"])))
    if len(rb):
        r, f, F = _group(rb, kb, N)
        add_part(r, f, F, "n")
    # alt-name tokens (websites / aliases) -> name group, down-weighted later
    rows = np.repeat(np.arange(N, dtype=np.int64), np.diff(rec["alt_ptr"]))
    if len(rows):
        r, f, F = _group(rows, rec["alt_idx"].astype(np.int64) + 0, N)
        add_part(r, f, F, "alt")
    # address tokens
    rows = np.repeat(np.arange(N, dtype=np.int64), np.diff(rec["a_ptr"]))
    r, f, F = _group(rows, rec["a_idx"].astype(np.int64), N)
    add_part(r, f, F, "a")
    # house number + street word
    common = rec["adf"] > 0.01 * N
    rh, kh = _hnum_combo(rec["a_ptr"], rec["a_idx"], rec["a_isnum"], common, np.int64(len(rec["a_vocab"])))
    if len(rh):
        r, f, F = _group(rh, kh, N)
        add_part(r, f, F, "a")
    # name token x city, name token x house number  (disambiguate generic names)
    city = np.where(rec["city_id"] > 0, rec["city_id"], -1).astype(np.int64)
    rc, kc = _combo(rec["n_ptr"], rec["n_idx"], city, np.int64(len(rec["city_list"]) + 1))
    if len(rc):
        r, f, F = _group(rc, kc, N)
        add_part(r, f, F, "c")
    hn = rec["hnum_id"].astype(np.int64)
    rc, kc = _combo(rec["n_ptr"], rec["n_idx"], hn, np.int64(len(rec["a_vocab"]) + 1))
    if len(rc):
        r, f, F = _group(rc, kc, N)
        add_part(r, f, F, "c")
    del rc, kc
    # character 4-grams of name tokens (typo tolerance)
    if ngrams:
        tptr, tcodes = token_ngrams(rec["n_vocab"])
        rg, kg = _expand_ngrams(rec["n_ptr"], rec["n_idx"], tptr, tcodes)
        if len(rg):
            r, f, F = _group(rg, kg, N)
            add_part(r, f, F, "g")
        del rg, kg
    if not ngrams:
        weights = tuple(weights[:3]) + (0.0,)

    # assemble: global feature ids, weights; norms per group include pruned features
    all_rows, all_cols, all_w, all_grp, all_keep = [], [], [], [], []
    off = 0
    cost = 0.0
    for rows, fids, idf, df1, F, group in parts:
        w = idf[fids].astype(np.float32)
        if group == "alt":
            w *= 0.5
        all_rows.append(rows)
        all_cols.append(fids + off)
        all_w.append(w)
        all_grp.append(np.full(len(rows), GROUPS[group], np.int8))
        k = df1[fids] <= max_df
        all_keep.append(k)
        dfq = np.bincount(fids[~is_s1[rows]], minlength=F)
        kept = df1 <= max_df
        cost += float((dfq[kept].astype(np.float64) * df1[kept]).sum())
        off += F
    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    w = np.concatenate(all_w)
    grp = np.concatenate(all_grp)
    keep = np.concatenate(all_keep)
    del all_rows, all_cols, all_w, all_grp, all_keep, parts
    scale = np.zeros(len(w), np.float32)
    for g, wg in enumerate(weights):
        m = grp == g
        nrm = np.zeros(N, np.float64)
        np.add.at(nrm, rows[m], w[m].astype(np.float64) ** 2)
        nrm = np.sqrt(nrm)
        scale[m] = (np.sqrt(wg) / np.maximum(nrm[rows[m]], 1e-9)).astype(np.float32)
    w = w * scale
    rows, cols, w, grp = rows[keep], cols[keep], w[keep], grp[keep]
    if verbose:
        print(f"  features {off}, kept nnz {len(w)}, est. cost {cost / 1e9:.2f}G mult-adds", flush=True)
    s1_rows = np.where(is_s1)[0]
    q_rows = np.where(~is_s1)[0]
    X = sp.csr_matrix((w, (rows, cols)), shape=(N, off), dtype=np.float32)
    X.sum_duplicates()
    B = X[s1_rows]
    Q = X[q_rows]
    del X
    if groups_out:
        # extra per-group matrices (for supplementary retrievers): {group id: (Q_g, B_g)}
        extra = {}
        for g in groups_out:
            m = grp == g
            Xg = sp.csr_matrix((w[m], (rows[m], cols[m])), shape=(N, off), dtype=np.float32)
            Xg.sum_duplicates()
            extra[g] = (Xg[q_rows], Xg[s1_rows])
        return Q, B, s1_rows, q_rows, extra
    if not with_addr:
        return Q, B, s1_rows, q_rows
    m = grp == 1
    Xa = sp.csr_matrix((w[m], (rows[m], cols[m])), shape=(N, off), dtype=np.float32)
    Xa.sum_duplicates()
    return Q, B, s1_rows, q_rows, Xa[q_rows], Xa[s1_rows]


def rowwise_dot(Q, B, qi, bi, chunk=2_000_000):
    out = np.empty(len(qi), np.float32)
    for s in range(0, len(qi), chunk):
        a = Q[qi[s:s + chunk]]
        b = B[bi[s:s + chunk]]
        out[s:s + chunk] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out


def topk_candidates(Q, B, k=10, threshold=0.05, chunk=400_000, rel=0.0):
    """For every query row the top-k index rows. Returns (q, b, score) arrays."""
    BT = B.T.tocsr()
    BT.indices = BT.indices.astype(np.int32)
    BT.indptr = BT.indptr.astype(np.int64) if BT.indptr.max() > 2**31 - 1 else BT.indptr.astype(np.int32)
    qs, bs, ss = [], [], []
    for s in range(0, Q.shape[0], chunk):
        Qc = Q[s:s + chunk]
        C = sp_matmul_topn(Qc, BT, top_n=k, threshold=threshold, sort=True, n_threads=N_JOBS)
        C = C.tocoo()
        q = C.row.astype(np.int64) + s
        b = C.col.astype(np.int64)
        v = C.data.astype(np.float32)
        if rel > 0:
            best = np.zeros(Qc.shape[0], np.float32)
            np.maximum.at(best, C.row, v)
            m = v >= rel * best[C.row]
            q, b, v = q[m], b[m], v[m]
        qs.append(q.astype(np.int32))
        bs.append(b.astype(np.int32))
        ss.append(v)
    return np.concatenate(qs), np.concatenate(bs), np.concatenate(ss)


def reverse_topk(Q, B, k=8, threshold=0.05, chunk=200_000):
    """For every index row (S1) the top-k query rows (S2/S3)."""
    QT = Q.T.tocsr()
    qs, bs, ss = [], [], []
    for s in range(0, B.shape[0], chunk):
        Bc = B[s:s + chunk]
        C = sp_matmul_topn(Bc, QT, top_n=k, threshold=threshold, sort=True, n_threads=N_JOBS).tocoo()
        bs.append((C.row.astype(np.int64) + s).astype(np.int32))
        qs.append(C.col.astype(np.int32))
        ss.append(C.data.astype(np.float32))
    return np.concatenate(qs), np.concatenate(bs), np.concatenate(ss)


def generate(rec, k_fwd=10, k_rev=0, threshold=0.05, max_df=4000, rel=0.6, verbose=True, ngrams=False,
             k_addr=1, addr_min_cos=0.3, expand_q=None, k_exp=20, k_exp_addr=5, k_exp_combo=3, exp_max_df=20000):
    """Candidate pairs (s1_record_idx, query_record_idx, blocking_score).
    Pass 1: top k_fwd S1 records per query on the combined score (>= rel * best).
    Pass 2: top k_addr S1 records per query on the address-only score, so that
    records whose name was replaced / generic still reach the right address.
    Pass 3 (only if expand_q is given, a boolean mask over records): for the query records that a
    first full pipeline left unmatched, a much wider search -- top k_exp on the combined score without
    the relative cut, top k_exp_addr on the address score and top k_exp_combo on the name x city /
    name x house-number score.  The combined search of pass 3 keeps index features up to an S1
    document frequency of exp_max_df (common words / city names carry signal for these records).
    Returns a 4th array flagging the pass-3-only pairs."""
    t0 = time.time()
    wide = None
    if expand_q is not None and k_exp > 0:
        Qw, Bw, _, qw = build_block_matrix(rec, max_df=exp_max_df, verbose=verbose, ngrams=ngrams)
        subw = np.flatnonzero(expand_q[qw])
        qx, bx, _ = topk_candidates(Qw[subw], Bw, k=k_exp, threshold=threshold)
        wide = (subw[qx].astype(np.int64), bx.astype(np.int64))
        del Qw, Bw
        if verbose:
            print(f"  wide search (max_df {exp_max_df}): {len(qx)} pairs ({time.time() - t0:.0f}s)", flush=True)
    if expand_q is not None:
        Q, B, s1_rows, q_rows, extra = build_block_matrix(rec, max_df=max_df, verbose=verbose, ngrams=ngrams,
                                                          groups_out=(1, 2))
        res = (Q, B, s1_rows, q_rows) + extra[1]
    else:
        res = build_block_matrix(rec, max_df=max_df, verbose=verbose, ngrams=ngrams, with_addr=k_addr > 0)
    Q, B, s1_rows, q_rows = res[:4]
    if verbose:
        print(f"  block matrix: Q {Q.shape} nnz {Q.nnz}, B {B.shape} nnz {B.nnz} ({time.time() - t0:.0f}s)", flush=True)
    q1, b1, s1 = topk_candidates(Q, B, k=k_fwd, threshold=threshold, rel=rel)
    if verbose:
        print(f"  forward: {len(q1)} pairs ({time.time() - t0:.0f}s)", flush=True)
    if k_addr > 0:
        Qa, Ba = res[4], res[5]
        del res
        # address part of the score is w_addr * cosine (w_addr = 0.35)
        q3, b3, _ = topk_candidates(Qa, Ba, k=k_addr, threshold=addr_min_cos * 0.35, rel=rel)
        del Qa, Ba
        key1 = b1.astype(np.int64) * np.int64(Q.shape[0]) + q1
        key3 = b3.astype(np.int64) * np.int64(Q.shape[0]) + q3
        new = ~np.isin(key3, key1)
        q3, b3 = q3[new], b3[new]
        s3 = rowwise_dot(Q, B, q3, b3)
        if verbose:
            print(f"  address pass: +{len(q3)} pairs ({time.time() - t0:.0f}s)", flush=True)
        q1 = np.concatenate([q1, q3])
        b1 = np.concatenate([b1, b3])
        s1 = np.concatenate([s1, s3])
    if k_rev > 0:
        q2, b2, s2 = reverse_topk(Q, B, k=k_rev, threshold=threshold)
        if verbose:
            print(f"  reverse: {len(q2)} pairs ({time.time() - t0:.0f}s)", flush=True)
        q = np.concatenate([q1, q2])
        b = np.concatenate([b1, b2])
        s = np.concatenate([s1, s2])
        key = b.astype(np.int64) * np.int64(Q.shape[0]) + q
        key, first = np.unique(key, return_index=True)
        q, b, s = q[first], b[first], s[first]
    else:
        q, b, s = q1, b1, s1
    if expand_q is None:
        # map to record indices within rec
        return s1_rows[b].astype(np.int32), q_rows[q].astype(np.int32), s
    sub = np.flatnonzero(expand_q[q_rows])
    qe, be = ([wide[0]], [wide[1]]) if wide is not None else ([], [])
    for Qx, Bx, k, thr in ((extra[1][0], extra[1][1], k_exp_addr, 0.35 * 0.2),
                           (extra[2][0], extra[2][1], k_exp_combo, 0.02)):
        if k <= 0:
            continue
        qx, bx, _ = topk_candidates(Qx[sub], Bx, k=k, threshold=thr)
        qe.append(sub[qx].astype(np.int64))
        be.append(bx.astype(np.int64))
    del extra
    ke = np.unique(np.concatenate(be) * np.int64(Q.shape[0]) + np.concatenate(qe))
    key = b.astype(np.int64) * np.int64(Q.shape[0]) + q
    ke = ke[~np.isin(ke, key)]
    be_, qe_ = ke // Q.shape[0], ke % Q.shape[0]
    se = rowwise_dot(Q, B, qe_, be_)
    if verbose:
        print(f"  expansion: {len(sub)} query records -> +{len(ke)} pairs ({time.time() - t0:.0f}s)", flush=True)
    q = np.concatenate([q, qe_]).astype(np.int64)
    b = np.concatenate([b, be_]).astype(np.int64)
    s = np.concatenate([s, se])
    flag = np.r_[np.zeros(len(key), np.int8), np.ones(len(ke), np.int8)]
    return s1_rows[b].astype(np.int32), q_rows[q].astype(np.int32), s, flag
