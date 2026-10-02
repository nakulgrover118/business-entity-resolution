"""Evaluation helpers: macro F0.5 (challenge definition) and ground-truth lookup."""
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from io_utils import load_ground_truth


def id_keys(ids):
    """'S2-123' -> 2*10**10 + 123 (int64); works on pyarrow string arrays."""
    if isinstance(ids, pa.ChunkedArray):
        ids = ids.combine_chunks()
    src = pc.cast(pc.utf8_slice_codeunits(ids, 1, 2), pa.int64()).to_numpy()
    num = pc.cast(pc.utf8_slice_codeunits(ids, 3), pa.int64()).to_numpy()
    return src * np.int64(10**10) + num


_GT = None


def gt_key_pairs():
    """Arrays (s1_key, other_key) of all ground-truth pairs, plus all S1 keys in GT."""
    global _GT
    if _GT is None:
        gt = load_ground_truth()
        s1 = gt.column("source1_entity_id").combine_chunks()
        m = gt.column("matched_entity_ids").combine_chunks()
        lists = pc.split_pattern(m, ",")
        lens = pc.list_value_length(lists).to_numpy(zero_copy_only=False)
        flat = pc.list_flatten(lists)
        s1_rep = pc.take(s1, pa.array(np.repeat(np.arange(len(s1)), lens)))
        nonempty = pc.not_equal(flat, "")
        flat = pc.filter(flat, nonempty)
        s1_rep = pc.filter(s1_rep, nonempty)
        _GT = (id_keys(s1_rep), id_keys(flat), id_keys(s1))
    return _GT


class Lookup:
    def __init__(self, keys):
        self.order = np.argsort(keys, kind="stable")
        self.sk = keys[self.order]

    def __call__(self, k):
        pos = np.searchsorted(self.sk, k)
        pos_c = np.minimum(pos, len(self.sk) - 1)
        ok = (pos < len(self.sk)) & (self.sk[pos_c] == k)
        return np.where(ok, self.order[pos_c], -1)


def true_s1_index(rec):
    """For every record of a country, index (within rec) of its true S1 record, -1 if none."""
    s1k, ok, _ = gt_key_pairs()
    lk = Lookup(id_keys(rec["ids"]))
    oi = lk(ok)
    si = lk(s1k)
    v = (oi >= 0) & (si >= 0)
    out = np.full(rec["N"], -1, np.int64)
    out[oi[v]] = si[v]
    return out


def f05_per_entity(tp, n_pred, n_true):
    tp = tp.astype(np.float64)
    P = np.divide(tp, n_pred, out=np.zeros_like(tp), where=n_pred > 0)
    R = np.divide(tp, n_true, out=np.zeros_like(tp), where=n_true > 0)
    den = 0.25 * P + R
    F = np.divide(1.25 * P * R, den, out=np.zeros_like(tp), where=den > 0)
    F = np.where((n_pred == 0) & (n_true == 0), 1.0, F)
    return F


def macro_f05(s1_idx_eval, pred_b, pred_q, true_s1, n_records):
    """s1_idx_eval: record indices of S1 entities to score.
    pred_b/pred_q: predicted pairs (S1 record idx, other record idx).
    true_s1: array from true_s1_index."""
    is_tp = true_s1[pred_q] == pred_b
    n_pred = np.bincount(pred_b, minlength=n_records)
    tp = np.bincount(pred_b, weights=is_tp, minlength=n_records)
    t = true_s1[true_s1 >= 0]
    n_true = np.bincount(t, minlength=n_records)
    F = f05_per_entity(tp[s1_idx_eval], n_pred[s1_idx_eval], n_true[s1_idx_eval])
    return F.mean(), F
