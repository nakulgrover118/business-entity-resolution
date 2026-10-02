"""Loading the raw TSV files (cached as Parquet for fast re-reads)."""
import os

import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq

from config import cache_path, tsv_path

_PARSE = pv.ParseOptions(delimiter="\t", quote_char=False, newlines_in_values=False)
_READ = pv.ReadOptions(block_size=1 << 25)


def _read_tsv(fn, cols):
    conv = pv.ConvertOptions(
        column_types={c: pa.string() for c in cols},
        strings_can_be_null=False,
        quoted_strings_can_be_null=False,
    )
    return pv.read_csv(fn, read_options=_READ, parse_options=_PARSE, convert_options=conv)


def load_source(split, source):
    """Return a pyarrow Table with entity_id, business_name, business_address, country."""
    pq_fn = cache_path(f"raw_{split}_{source}.parquet")
    if os.path.exists(pq_fn):
        return pq.read_table(pq_fn)
    cols = ["entity_id", "business_name", "business_address", "country"]
    t = _read_tsv(tsv_path(split, source), cols)
    pq.write_table(t, pq_fn, compression="zstd")
    return t


def load_ground_truth():
    pq_fn = cache_path("raw_train_ground_truth.parquet")
    if os.path.exists(pq_fn):
        return pq.read_table(pq_fn)
    cols = ["source1_entity_id", "matched_entity_ids"]
    t = _read_tsv(tsv_path("train", "ground_truth"), cols)
    pq.write_table(t, pq_fn, compression="zstd")
    return t


def gt_pairs():
    """Dict: S2/S3 entity_id -> S1 entity_id, and list of S1 ids."""
    gt = load_ground_truth()
    s1 = gt.column("source1_entity_id").to_pylist()
    m = gt.column("matched_entity_ids").to_pylist()
    rev = {}
    for a, ids in zip(s1, m):
        if ids:
            for x in ids.split(","):
                if x:
                    rev[x] = a
    return rev, s1
