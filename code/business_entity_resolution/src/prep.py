"""Stage 1: normalise every record of every source (parallel), cache as Parquet."""
import os
import sys
import time
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq

from config import N_JOBS, SOURCES, cache_path
from io_utils import load_source
from normalize import Normalizer
from translit import build_dictionary

_NORM = None


def _init():
    global _NORM
    _NORM = Normalizer(build_dictionary())


def _work(args):
    names, addrs = args
    out = []
    for n, a in zip(names, addrs):
        core, full, alt, legal, flags, unk = _NORM.name(n)
        comps = _NORM.address(a)
        out.append((core, full, alt, legal, flags, unk, " | ".join(" ".join(c) for c in comps)))
    return out


def prep_path(split, source):
    return cache_path(f"norm_{split}_{source}.parquet")


def run(split, source, pool, chunk=20000):
    fn = prep_path(split, source)
    if os.path.exists(fn):
        return
    t0 = time.time()
    t = load_source(split, source)
    names = t.column("business_name").to_pylist()
    addrs = t.column("business_address").to_pylist()
    jobs = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(names), chunk)]
    res = []
    for r in pool.imap(_work, jobs, chunksize=1):
        res.extend(r)
    cols = list(zip(*res))
    out = pa.table({
        "entity_id": t.column("entity_id"),
        "country": t.column("country"),
        "n_core": pa.array(cols[0], pa.string()),
        "n_full": pa.array(cols[1], pa.string()),
        "n_alt": pa.array(cols[2], pa.string()),
        "n_legal": pa.array(cols[3], pa.string()),
        "n_flags": pa.array(cols[4], pa.int8()),
        "n_unk": pa.array(cols[5], pa.int8()),
        "a_comps": pa.array(cols[6], pa.string()),
    })
    pq.write_table(out, fn, compression="zstd")
    print(f"prep {split} {source}: {len(names)} rows in {time.time() - t0:.0f}s", flush=True)


def main(splits=("train", "test")):
    build_dictionary()
    with Pool(N_JOBS, initializer=_init) as pool:
        for split in splits:
            for s in SOURCES:
                run(split, s, pool)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("train", "test"))
