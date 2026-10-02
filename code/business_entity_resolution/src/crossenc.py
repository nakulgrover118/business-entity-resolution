"""Stage 8: multilingual cross-encoders on the hard candidate pairs.

A small multilingual transformer reads both *raw* records jointly ("name | address" of the S1
entity and of the S2/S3 record) and is fine-tuned to predict a match.  It only runs on the hard
pairs: each record's top-k candidates whose v3 stage-B probability is not already ~0 or ~1.
Training is 2-fold by S1 entity so every training pair gets an out-of-fold score; test pairs get
the average of both fold models.  The five rounds (models, top-k, epochs, seeds) are configured
through the environment, see run_all.CE_ROUNDS: CE1 / CE2 paraphrase-multilingual-MiniLM-L12-v2
(Apache-2.0, 118 M parameters), CE3 / CE5 multilingual-e5-small (MIT, 118 M), CE4
multilingual-e5-base (MIT, 278 M).

Usage: python crossenc.py select | train FOLD [MAX_PAIRS] | score
"""
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from config import cache_path
from io_utils import load_source
from model import fold_of
from pairs import pairs_dir
from records import load_records, split_countries

MODEL_NAME = os.environ.get("CE_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
MAX_LEN = 128
BS = int(os.environ.get("CE_BS", "64"))
LR = float(os.environ.get("CE_LR", "4e-5"))
# round configuration (defaults = round 1; the other rounds: run_all.CE_ROUNDS)
TAG = os.environ.get("CE_TAG", "")
TOPK = int(os.environ.get("CE_TOPK", "2"))
LO = float(os.environ.get("CE_LO", "0.003"))
HI = float(os.environ.get("CE_HI", "0.997"))
EPOCHS = int(os.environ.get("CE_EPOCHS", "1"))
SEED = int(os.environ.get("CE_SEED", "0"))
EMB_HALF = os.environ.get("CE_EMB_HALF", "0") == "1"      # frozen embeddings in fp16 (saves ~0.4 GB for base models)
PRED_BS = int(os.environ.get("CE_PRED_BS", "256"))
# reuse scores of an earlier run for identical (S1 record, S2/S3 record) pairs: "<old CE dir>|<old cache dir>"
REUSE = os.environ.get("CE_REUSE", "")
CKPT_EVERY = int(os.environ.get("CE_CKPT_EVERY", "0"))     # save a partial model every N steps (crash insurance)
CE_DIR = cache_path("crossenc" + (f"_{TAG}" if TAG else ""))
os.makedirs(CE_DIR, exist_ok=True)


def prob_file(split, country):
    d = pairs_dir(split, country)
    return os.path.join(d, "pB_v3.npy") if split == "train" else os.path.join(d, "pB_v3_test.npy")


def hard_mask(B, Q, p):
    """top-TOPK candidates of each record (by p) with p in [LO, HI]."""
    order = np.lexsort((-p, Q))
    qs = Q[order]
    first = np.r_[True, qs[1:] != qs[:-1]]
    start = np.flatnonzero(first)
    grp = np.cumsum(first) - 1
    rank = np.empty(len(p), np.int64)
    rank[order] = np.arange(len(p)) - start[grp]
    return (rank < TOPK) & (p >= LO) & (p <= HI)


def _texts(split, country, rows):
    """raw 'name | address' for record indices `rows` of a country."""
    rec = load_records(split, country)
    ids = rec["ids"]
    ids = ids.combine_chunks() if isinstance(ids, pa.ChunkedArray) else ids
    del rec
    want = ids.take(pa.array(rows))
    out = {}
    for s in ("source1", "source2", "source3"):
        t = load_source(split, s).select(["entity_id", "business_name", "business_address"])
        t = t.filter(pc.is_in(t.column("entity_id"), value_set=want))
        for i, n, a in zip(t.column("entity_id").to_pylist(), t.column("business_name").to_pylist(),
                           t.column("business_address").to_pylist()):
            out[i] = f"{n} | {a}" if a else n
    idl = want.to_pylist()
    return [out[i] for i in idl]


def select():
    """Write the hard-pair tables (pair index, S1 text, record text, label, fold)."""
    t0 = time.time()
    for split in ("train", "test"):
        for c in split_countries(split):
            d = pairs_dir(split, c)
            B = np.load(os.path.join(d, "B.npy"))
            Q = np.load(os.path.join(d, "Q.npy"))
            p = np.load(prob_file(split, c))
            m = hard_mask(B, Q, p)
            idx = np.flatnonzero(m)
            rows = np.unique(np.concatenate([B[idx], Q[idx]]))
            txt = _texts(split, c, rows)
            pos = {int(r): k for k, r in enumerate(rows)}
            tb = [txt[pos[int(r)]] for r in B[idx]]
            tq = [txt[pos[int(r)]] for r in Q[idx]]
            cols = {"pair": pa.array(idx), "tb": pa.array(tb), "tq": pa.array(tq),
                    "fold": pa.array(fold_of(B[idx])), "p": pa.array(p[idx])}
            if split == "train":
                cols["y"] = pa.array(np.load(os.path.join(d, "y.npy"))[idx])
            pq.write_table(pa.table(cols), os.path.join(CE_DIR, f"hard_{split}_{c}.parquet"))
            print(f"  {split} {c}: {len(idx)} hard pairs of {len(B)} ({time.time() - t0:.0f}s)", flush=True)


def _load(split):
    ts = [pq.read_table(os.path.join(CE_DIR, f"hard_{split}_{c}.parquet")).append_column(
        "country", pa.array([c] * pq.read_metadata(os.path.join(CE_DIR, f"hard_{split}_{c}.parquet")).num_rows))
          for c in split_countries(split)]
    return pa.concat_tables(ts)


class _Pairs:
    def __init__(self, tok, tb, tq):
        self.tok, self.tb, self.tq = tok, tb, tq

    def batch(self, idx):
        enc = self.tok([self.tb[i] for i in idx], [self.tq[i] for i in idx], truncation=True,
                       max_length=MAX_LEN, padding=True, return_tensors="pt")
        return enc


def _amp_dtype():
    import torch
    return torch.float16


def train(fold, max_pairs=900_000, epochs=None, bs=None, lr=None):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    epochs = epochs or EPOCHS
    bs = bs or BS
    lr = lr or LR
    amp = _amp_dtype()
    torch.manual_seed(fold + 100 * SEED)
    t = _load("train")
    t = t.filter(pc.equal(t.column("fold"), fold))
    n = t.num_rows
    rng = np.random.default_rng(fold + 100 * SEED)
    sel = np.sort(rng.choice(n, min(n, max_pairs), replace=False))
    t = t.take(pa.array(sel))
    tb, tq = t.column("tb").to_pylist(), t.column("tq").to_pylist()
    y = torch.tensor(t.column("y").to_numpy().astype(np.float32))
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
    for prm in model.base_model.embeddings.parameters():
        prm.requires_grad = False
    if EMB_HALF:
        model.base_model.embeddings.half()
    model.cuda()
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=lr, weight_decay=0.01)
    steps = epochs * (len(tb) // bs)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp == torch.float16))
    lossf = torch.nn.BCEWithLogitsLoss()
    data = _Pairs(tok, tb, tq)
    model.train()
    t0 = time.time()
    step = 0
    for ep in range(epochs):
        perm = rng.permutation(len(tb))
        for s in range(0, len(perm) - bs + 1, bs):
            idx = perm[s:s + bs]
            enc = {k: v.cuda(non_blocking=True) for k, v in data.batch(idx).items()}
            with torch.autocast("cuda", dtype=amp):
                out = model(**enc).logits.squeeze(-1)
            loss = lossf(out.float(), y[idx].cuda())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            step += 1
            if step % 500 == 0:
                print(f"  fold {fold} step {step}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
            if CKPT_EVERY and step % CKPT_EVERY == 0:
                model.save_pretrained(os.path.join(CE_DIR, f"model_f{fold}_partial"))
                tok.save_pretrained(os.path.join(CE_DIR, f"model_f{fold}_partial"))
    model.save_pretrained(os.path.join(CE_DIR, f"model_f{fold}"))
    tok.save_pretrained(os.path.join(CE_DIR, f"model_f{fold}"))
    print(f"  fold {fold} trained on {len(tb)} pairs in {time.time() - t0:.0f}s", flush=True)


def _predict(model, tok, tb, tq, bs=None):
    bs = bs or PRED_BS
    import torch
    amp = _amp_dtype()
    data = _Pairs(tok, tb, tq)
    out = np.empty(len(tb), np.float32)
    order = np.argsort([len(a) + len(b) for a, b in zip(tb, tq)])  # length bucketing for speed
    with torch.no_grad():
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            enc = {k: v.cuda(non_blocking=True) for k, v in data.batch(idx).items()}
            with torch.autocast("cuda", dtype=amp):
                out[idx] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
    return out


def _old_scores(split, c):
    """(sorted pair keys, scores) of an earlier run, keyed by (S1 record, S2/S3 record)."""
    old_ce, old_cache = REUSE.split("|")
    f = os.path.join(old_ce, f"ce_{split}_{c}.npy")
    if not os.path.exists(f):
        return None
    pair = pq.read_table(os.path.join(old_ce, f"hard_{split}_{c}.parquet"), columns=["pair"]).column("pair").to_numpy()
    od = os.path.join(old_cache, f"pairs_{split}_{c}")
    key = np.load(os.path.join(od, "B.npy"))[pair].astype(np.int64) * (1 << 32) + np.load(os.path.join(od, "Q.npy"))[pair]
    v = np.load(f)
    o = np.argsort(key)
    return key[o], v[o]


def score():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    t0 = time.time()
    models = []
    for f in (0, 1):
        p = os.path.join(CE_DIR, f"model_f{f}")
        m = AutoModelForSequenceClassification.from_pretrained(p)
        m = (m.half() if EMB_HALF else m).cuda().eval()
        models.append((m, AutoTokenizer.from_pretrained(p)))
    for split in ("train", "test"):
        for c in split_countries(split):
            t = pq.read_table(os.path.join(CE_DIR, f"hard_{split}_{c}.parquet"))
            tb, tq = t.column("tb").to_pylist(), t.column("tq").to_pylist()
            fold = t.column("fold").to_numpy()
            ce = np.zeros(len(tb), np.float32)
            todo = np.ones(len(tb), bool)
            if REUSE:
                old = _old_scores(split, c)
                if old is not None:
                    d = pairs_dir(split, c)
                    pair = t.column("pair").to_numpy()
                    key = np.load(os.path.join(d, "B.npy"))[pair].astype(np.int64) * (1 << 32) + np.load(os.path.join(d, "Q.npy"))[pair]
                    ok, ov = old
                    pos = np.minimum(np.searchsorted(ok, key), len(ok) - 1)
                    hit = ok[pos] == key
                    ce[hit] = ov[pos[hit]]
                    todo = ~hit
                    print(f"  {split} {c}: reused {hit.sum()} scores, scoring {todo.sum()} new pairs", flush=True)
            if split == "train":
                for f, (m, tok) in enumerate(models):  # model f scores the other fold (out-of-fold)
                    sel = np.flatnonzero((fold != f) & todo)
                    if len(sel):
                        ce[sel] = _predict(m, tok, [tb[i] for i in sel], [tq[i] for i in sel])
            else:
                sel = np.flatnonzero(todo)
                if len(sel):
                    acc = np.zeros(len(sel), np.float32)
                    for m, tok in models:
                        acc += _predict(m, tok, [tb[i] for i in sel], [tq[i] for i in sel]) / len(models)
                    ce[sel] = acc
            np.save(os.path.join(CE_DIR, f"ce_{split}_{c}.npy"), ce)
            print(f"  scored {split} {c}: {len(tb)} pairs ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "select":
        select()
    elif cmd == "train":
        train(int(sys.argv[2]), max_pairs=int(sys.argv[3]) if len(sys.argv) > 3 else 900_000)
    elif cmd == "score":
        score()
