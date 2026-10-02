"""Learn an Indic-script -> Latin token dictionary from the labelled training pairs.

Source 2/3 records for Indian businesses often write the name (and the state) in a
native script (Devanagari, Tamil, Telugu, ...).  Source 1 is always Latin.  The
generator transliterates word by word, so the same Latin word always maps to the
same native token.  We align native tokens with the Latin tokens of the matched
Source 1 record (positionally when token counts agree, otherwise by
co-occurrence) and keep the dominant mapping.  Unseen tokens fall back to
`unidecode` at normalisation time.
"""
import collections
import json
import os
import re
import unicodedata

from unidecode import unidecode

from config import cache_path
from io_utils import gt_pairs, load_source

INDIC_RE = re.compile(r"[ऀ-෿]")
_ZW = dict.fromkeys([0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x00AD], None)
_STRIP = ".,()[]{}|-_:;\"'*#@<>!?/\\+&~`"

DICT_FILE = cache_path("translit_dict.json")


def pre(s):
    return unicodedata.normalize("NFKC", s).translate(_ZW)


def latin_tokens(s):
    s = unidecode(pre(s)).lower().replace(".", "").replace("'", "")
    return re.sub(r"[^a-z0-9]+", " ", s).split()


def raw_tokens(s):
    out = []
    for t in re.split(r"[\s,]+", pre(s)):
        t = t.strip(_STRIP)
        if t:
            out.append(t)
    return out


def _learn(pairs):
    pos = collections.defaultdict(collections.Counter)
    co = collections.defaultdict(collections.Counter)
    cf = collections.Counter()
    ce = collections.Counter()
    for latin, other in pairs:
        F = raw_tokens(other)
        if not any(INDIC_RE.search(f) for f in F):
            continue
        E = latin_tokens(latin)
        if not E:
            continue
        if len(E) == len(F):
            for e, f in zip(E, F):
                if INDIC_RE.search(f):
                    pos[f][e] += 1
        Es = set(E)
        for e in Es:
            ce[e] += 1
        for f in set(F):
            if INDIC_RE.search(f):
                cf[f] += 1
                for e in Es:
                    co[f][e] += 1
    mapping = {}
    for f, n in cf.items():
        p = pos.get(f)
        if p:
            e, c = p.most_common(1)[0]
            if c >= 2 and c >= 0.5 * sum(p.values()):
                mapping[f] = e
                continue
        cands = co[f]
        best, key = None, None
        for e, c in cands.items():
            r = c / n
            if r < 0.6:
                continue
            k = (round(r, 2), c / ce[e])
            if key is None or k > key:
                best, key = e, k
        if best is not None and n >= 1:
            mapping[f] = best
    return mapping


def _learn_components(pairs):
    """Map whole native-script address components (e.g. a state name written as one
    word) to the Latin Source 1 component they co-occur with."""
    co = collections.defaultdict(collections.Counter)
    cf = collections.Counter()
    ce = collections.Counter()
    for latin, other in pairs:
        F = {c.strip().strip(_STRIP) for c in pre(other).split(",")}
        F = {f for f in F if INDIC_RE.search(f)}
        if not F:
            continue
        E = {" ".join(latin_tokens(c)) for c in latin.split(",")}
        E.discard("")
        for e in E:
            ce[e] += 1
        for f in F:
            cf[f] += 1
            for e in E:
                co[f][e] += 1
    out = {}
    for f, n in cf.items():
        best, key = None, None
        for e, c in co[f].items():
            r = c / n
            if r < 0.6:
                continue
            k = (round(r, 2), c / ce[e])
            if key is None or k > key:
                best, key = e, k
        if best is not None and n >= 3:
            out[f] = best
    return out


def build_dictionary(force=False):
    if os.path.exists(DICT_FILE) and not force:
        with open(DICT_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    rev, _ = gt_pairs()
    s1 = load_source("train", "source1")
    s1_name = dict(zip(s1.column("entity_id").to_pylist(), s1.column("business_name").to_pylist()))
    s1_addr = dict(zip(s1.column("entity_id").to_pylist(), s1.column("business_address").to_pylist()))
    del s1
    name_pairs, addr_pairs = [], []
    for src in ("source2", "source3"):
        t = load_source("train", src)
        ids = t.column("entity_id").to_pylist()
        names = t.column("business_name").to_pylist()
        addrs = t.column("business_address").to_pylist()
        for i, n, a in zip(ids, names, addrs):
            s1id = rev.get(i)
            if s1id is None:
                continue
            if INDIC_RE.search(n):
                name_pairs.append((s1_name[s1id], n))
            if INDIC_RE.search(a):
                addr_pairs.append((s1_addr[s1id], a))
        del t, ids, names, addrs
    name_map = _learn(name_pairs)
    addr_map = _learn(addr_pairs)
    comp_map = _learn_components(addr_pairs)
    # names and addresses share one token dictionary; the name mapping wins on conflicts
    tokens = dict(addr_map)
    tokens.update(name_map)
    mapping = {"tokens": tokens, "components": comp_map}
    with open(DICT_FILE, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh, ensure_ascii=False)
    print(f"translit: {len(name_pairs)} name pairs, {len(addr_pairs)} addr pairs, "
          f"{len(name_map)} name tokens, {len(addr_map)} addr tokens, {len(comp_map)} components")
    return mapping


if __name__ == "__main__":
    m = build_dictionary(force=True)
    for k, v in m["components"].items():
        print(k, "->", v)
