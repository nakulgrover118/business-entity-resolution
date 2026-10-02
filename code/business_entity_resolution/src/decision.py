"""Stage 6: turn pair probabilities into final match lists.

1. One-to-many constraint: a Source 2/3 record belongs to at most one Source 1
   entity, so each record is only offered to its highest-probability S1 entity.
2. Per S1 entity we choose the subset that maximises the *expected* F0.5 given the
   (calibrated) probabilities of its remaining candidates.  Under independence the
   optimum is a top-k set; k = 0 (predict "no match") is chosen when the entity is
   probably a singleton.  The expectation is computed exactly with Poisson-binomial
   dynamic programming.
"""
import numba as nb
import numpy as np


def argmax_per_query(B, Q, p):
    """Mask of pairs where B is the best S1 for Q (ties -> first)."""
    order = np.lexsort((-p, Q))
    qs = Q[order]
    first = np.r_[True, qs[1:] != qs[:-1]]
    keep = np.zeros(len(B), bool)
    keep[order[first]] = True
    return keep


@nb.njit(cache=True)
def _poibin(p):
    n = len(p)
    d = np.zeros(n + 1)
    d[0] = 1.0
    for i in range(n):
        pi = p[i]
        for j in range(i + 1, 0, -1):
            d[j] = d[j] * (1 - pi) + d[j - 1] * pi
        d[0] *= 1 - pi
    return d


@nb.njit(cache=True)
def best_k(p, miss, beta2):
    """p sorted descending. miss = expected true matches outside the list.
    Returns (k, expected F)."""
    n = len(p)
    # suffix distributions
    best_e = 0.0
    # k = 0: score 1 only when there is no true match at all
    e0 = 1.0
    for i in range(n):
        e0 *= 1 - p[i]
    e0 *= np.exp(-miss)
    best_e = e0
    bk = 0
    px = np.zeros(n + 1)
    px[0] = 1.0
    for k in range(1, n + 1):
        pk = p[k - 1]
        for j in range(k, 0, -1):
            px[j] = px[j] * (1 - pk) + px[j - 1] * pk
        px[0] *= 1 - pk
        py = _poibin(p[k:])
        e = 0.0
        for x in range(1, k + 1):
            if px[x] == 0.0:
                continue
            acc = 0.0
            for y in range(0, n - k + 1):
                acc += py[y] * (1 + beta2) * x / (beta2 * (x + y + miss) + k)
            e += px[x] * acc
        if e > best_e:
            best_e = e
            bk = k
    return bk, best_e


@nb.njit(cache=True)
def _select(starts, ends, p_sorted, miss, beta2, max_n):
    ne = len(starts)
    ks = np.zeros(ne, np.int64)
    es = np.zeros(ne)
    for g in range(ne):
        s = starts[g]
        e = min(ends[g], s + max_n)
        k, ev = best_k(p_sorted[s:e], miss, beta2)
        ks[g] = k
        es[g] = ev
    return ks, es


def select_matches(B, Q, p, miss=0.0, beta2=0.25, max_n=16, min_p=0.0):
    """Given pairs (already filtered by argmax_per_query) return mask of selected pairs."""
    m = p > min_p
    idx = np.flatnonzero(m)
    Bm, pm = B[idx], p[idx]
    order = np.lexsort((-pm, Bm))
    bs = Bm[order]
    ps = pm[order].astype(np.float64)
    first = np.r_[True, bs[1:] != bs[:-1]]
    starts = np.flatnonzero(first)
    ends = np.r_[starts[1:], len(bs)]
    ks, _ = _select(starts, ends, ps, miss, beta2, max_n)
    rank = np.arange(len(bs)) - np.repeat(starts, ends - starts)
    kk = np.repeat(ks, ends - starts)
    sel_sorted = rank < kk
    sel = np.zeros(len(B), bool)
    sel[idx[order[sel_sorted]]] = True
    return sel
