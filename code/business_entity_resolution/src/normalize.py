"""Text normalisation for business names and addresses.

Everything here is country-agnostic: the same rules run for every record, and
nothing assumes a closed set of countries.  Abbreviation tables cover common
English / French / Indian street vocabulary; they only canonicalise spellings
(both sides of a comparison go through the same mapping).
"""
import re
import unicodedata

from unidecode import unidecode

INDIC_RE = re.compile(r"[ऀ-෿]")
_ZW = dict.fromkeys([0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x00AD], None)
_STRIP = ".,()[]{}|-_:;\"'*#@<>!?/\\+&~`"

# ----------------------------------------------------------------------------- names
ALIAS_RE = re.compile(
    r"\s+(?:f/k/a|fka|d/b/a|dba|a/k/a|aka|t/a|trading\s+as|formerly\s+known\s+as|"
    r"formerly|n[é�]e)\s*:?\s+",
    re.I,
)
ID_RE = re.compile(r"[\(\[]\s*id\s*:\s*\d+\s*[\)\]]", re.I)
URL_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:com|net|org|co\.in|in|co|fr|io|biz|us|info)$"
)
# website names whose dot was lost ("GSVISIONCOM", "departmentbuildingscom")
DOTLESS_URL_RE = re.compile(r"^[#@]?([a-z0-9]{5,})(?:com|net|org)$")
MS_RE = re.compile(r"\bm\s*/\s*s\b\.?", re.I)

LEGAL = {
    "inc", "incorporated", "llc", "corp", "corporation", "co", "company", "ltd", "limited",
    "pvt", "private", "lp", "llp", "pc", "pllc", "plc", "pa", "lc", "opc", "ltda",
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei", "scop", "selarl", "sca", "gie",
    "scm", "scp", "sem", "gmbh", "ag", "bv", "nv", "cie",
}
LEGAL_FAMILY = {
    "inc": "inc", "incorporated": "inc", "corp": "inc", "corporation": "inc",
    "llc": "llc", "lc": "llc", "co": "co", "company": "co", "cie": "co",
    "ltd": "ltd", "limited": "ltd", "plc": "plc", "pvt": "pvt", "private": "pvt",
    "lp": "lp", "llp": "llp", "pc": "pc", "pllc": "pllc", "pa": "pa", "opc": "opc",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl", "sci": "sci",
    "snc": "snc", "ei": "ei", "scop": "scop", "selarl": "selarl", "sca": "sca", "gie": "gie",
    "scm": "scm", "scp": "scp", "sem": "sem", "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv",
    "ltda": "ltd", "public": "public",
}
HONORIFIC = {"ms", "mr", "mrs", "sri", "shri", "shree", "smt", "the"}


def pre(s):
    return unicodedata.normalize("NFKC", s).translate(_ZW)


_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t", "8": "b"})
_LEET_RE = re.compile(r"^(?=(?:[^0-9]*[0-9]){1,2}[^0-9]*$)(?=(?:.*[a-z]){3,})[a-z0-9]{4,}$")


def deleet(t):
    """Typo noise sometimes swaps a letter for a look-alike digit ('s0lutions',
    '8lack', 'helpin6').  Tokens with >=3 letters and 1-2 such digits are mapped back."""
    if _LEET_RE.match(t):
        u = t.translate(_LEET)
        if u.isalpha():
            return u
    return t


class Normalizer:
    def __init__(self, translit):
        self.tok_map = translit.get("tokens", {})
        self.comp_map = translit.get("components", {})

    # -- helpers ------------------------------------------------------------------
    def _translit_tokens(self, s):
        if not INDIC_RE.search(s):
            return s, 0
        out, unknown = [], 0
        for t in s.split():
            if INDIC_RE.search(t):
                core = t.strip(_STRIP)
                m = self.tok_map.get(core)
                if m is None:
                    unknown += 1
                    m = unidecode(core)
                out.append(m)
            else:
                out.append(t)
        return " ".join(out), unknown

    @staticmethod
    def _name_tokens(s):
        s = unidecode(s).lower()
        s = s.replace("&", " and ").replace("+", " and ")
        s = s.replace(".", "").replace("'", "").replace("`", "")
        s = re.sub(r"[^a-z0-9]+", " ", s)
        toks = s.split()
        # collapse consecutive duplicates ("investments investments")
        out = []
        for t in toks:
            t = deleet(t)
            if not out or out[-1] != t:
                out.append(t)
        return out

    # -- names --------------------------------------------------------------------
    def name(self, raw):
        """Return (core, full, alt, legal_family, flags, n_unknown_indic)."""
        s = pre(raw).strip()
        flags = 0
        if INDIC_RE.search(s):
            flags |= 1
        s = ID_RE.sub(" ", s)
        alt = ""
        if "|" in s:
            s, extra = s.split("|", 1)
            alt = extra
            flags |= 2
        parts = ALIAS_RE.split(s)
        if len(parts) > 1:
            alt = (alt + " " + " ".join(parts[:-1])).strip()
            s = parts[-1]
            flags |= 4
        s = MS_RE.sub(" ", s)
        s, unk = self._translit_tokens(s)
        # website / social-handle style names -> keep the concatenated label
        web = []
        st = unidecode(s).strip().lower()
        if " " not in st and "." not in st:
            m = DOTLESS_URL_RE.match(st)
            if m:
                web.append(m.group(1))
                s = ""
        for w in s.split():
            wl = unidecode(w).lower().strip(".,;:!?()[]")
            m = URL_RE.match(wl)
            if m:
                web.append(m.group(1).replace("-", ""))
                s = s.replace(w, " ")
            elif (wl.startswith("#") or wl.startswith("@")) and len(wl) > 3 and wl[1:].isalnum():
                web.append(wl[1:])
                s = s.replace(w, " ")
        if web:
            flags |= 8
        toks = self._name_tokens(s) + web
        while toks and toks[0] in HONORIFIC and len(toks) > 1:
            toks = toks[1:]
        # "public limited" is a legal form, "public" alone is a normal word
        fam = []
        core = []
        for i, t in enumerate(toks):
            if t in LEGAL or (t == "public" and i + 1 < len(toks) and toks[i + 1] in ("limited", "ltd")):
                f = LEGAL_FAMILY.get(t, t)
                if not fam or fam[-1] != f:
                    fam.append(f)
            else:
                core.append(t)
        # dangling connectors left by "& Co" / leading "&" decorations
        while len(core) > 1 and core[-1] == "and":
            core.pop()
        while len(core) > 1 and core[0] == "and":
            core.pop(0)
        if not core:
            core = toks
        alt_toks = []
        if alt:
            a, u2 = self._translit_tokens(alt)
            unk += u2
            for w in a.split():
                wl = unidecode(w).lower().strip(".,;:!?()[]")
                m = URL_RE.match(wl)
                if m:
                    alt_toks.append(m.group(1).replace("-", ""))
                else:
                    alt_toks.extend(self._name_tokens(w))
            alt_toks = [t for t in alt_toks if t not in LEGAL]
        return (" ".join(core), " ".join(toks), " ".join(alt_toks), " ".join(sorted(set(fam))), flags, unk)

    # -- addresses ----------------------------------------------------------------
    def address(self, raw):
        """Return list of normalised components (each a list of tokens)."""
        s = pre(raw).strip()
        if not s:
            return []
        comps = []
        for c in s.split(","):
            c = c.strip()
            if not c:
                continue
            if INDIC_RE.search(c):
                key = c.strip(_STRIP)
                if key in self.comp_map:
                    c = self.comp_map[key]
                else:
                    c, _ = self._translit_tokens(c)
            toks = addr_tokens(c)
            if toks:
                comps.append(toks)
        return comps


# ----------------------------------------------------------------------------- addresses
NULLS = {"null", "<null>", "n/a", "na", "none", "nil", "-", "--"}
ORD_WORDS = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
    "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10", "premier": "1", "premiere": "1",
}
ABBR = {
    # English street types (canonical = short form)
    "street": "st", "str": "st", "saint": "st", "sainte": "ste",
    "road": "rd", "avenue": "ave", "av": "ave", "avn": "ave", "aven": "ave",
    "drive": "dr", "drv": "dr", "court": "ct", "crt": "ct", "circle": "cir", "circ": "cir",
    "lane": "ln", "boulevard": "blvd", "bd": "blvd", "boul": "blvd", "bvd": "blvd", "bld": "blvd",
    "highway": "hwy", "hiway": "hwy", "parkway": "pkwy", "pky": "pkwy", "place": "pl", "plz": "plaza",
    "square": "sq", "trail": "trl", "tr": "trl", "crossing": "xing", "terrace": "ter", "terr": "ter",
    "mount": "mt", "mountain": "mtn", "point": "pt", "fort": "ft", "heights": "hts", "junction": "jct",
    "expressway": "expy", "freeway": "fwy", "turnpike": "tpke", "center": "ctr", "centre": "ctr",
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne", "northwest": "nw",
    "southeast": "se", "southwest": "sw", "apartment": "apt", "suite": "ste", "floor": "fl",
    "building": "bldg", "cove": "cv", "creek": "crk", "estates": "est", "estate": "est", "extension": "ext",
    "extn": "ext", "village": "vill", "vil": "vill", "vlg": "vill", "nagar": "ngr", "colony": "col",
    "sector": "sec", "phase": "ph", "stage": "stg", "block": "blk", "opposite": "opp", "near": "nr",
    "behind": "bh", "industrial": "indl", "ind": "indl", "district": "dist", "dt": "dist",
    "taluka": "tq", "taluk": "tq", "tal": "tq", "tehsil": "teh", "post": "po", "office": "off",
    "cross": "crs", "main": "mn", "layout": "lyt", "road no": "rd",
    # French
    "rue": "rue", "r": "rue", "allee": "all", "impasse": "imp", "chemin": "chem", "che": "chem",
    "route": "rte", "residence": "res", "lotissement": "lot", "quai": "qu", "cours": "crs",
    "faubourg": "fbg", "fg": "fbg", "passage": "pass", "hameau": "ham", "lieu": "lieu", "dit": "dit",
    "rond": "rd", "etage": "fl", "batiment": "bldg", "bat": "bldg",
}
# prefix words that are swapped at random between records of the same entity
ADDR_NOISE = {
    "no", "nos", "number", "num", "h", "hno", "hn", "house", "door", "flat", "plot", "shop",
    "survey", "sr", "sy", "kh", "khasra", "unit", "apt", "ste", "pmb", "box",
}
_ORD_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th|er|ere|eme|e)\b")
_POBOX_RE = re.compile(r"\b(?:p\s*o\s*box|post\s*box|pmb|box)\s*#?\s*\d+\w*")
_NUMSIGN_RE = re.compile(r"\b[nN]\s*[°º]")
_ALNUM_RE = re.compile(r"(?<=[0-9])(?=[a-z])|(?<=[a-z])(?=[0-9])")


def addr_tokens(c):
    c = _NUMSIGN_RE.sub(" no ", c)
    c = unidecode(c).lower().strip()
    if c in NULLS:
        return []
    c = _POBOX_RE.sub(" ", c)
    c = _ORD_RE.sub(r"\1", c)
    c = re.sub(r"[^a-z0-9]+", " ", c)
    # split glued letter/digit runs: '1100c' -> '1100 c', 'no50' -> 'no 50', 'a414' -> 'a 414'
    c = _ALNUM_RE.sub(" ", c)
    out = []
    for t in c.split():
        if t in NULLS:
            continue
        if t.isdigit():
            t = t.lstrip("0") or "0"
        else:
            t = ORD_WORDS.get(t, t)
            t = ABBR.get(t, t)
            if t in ADDR_NOISE:
                continue
            # glued ordinal typo like "102rd" / "2rd" -> "102"
            m = re.fullmatch(r"(\d+)(?:st|nd|rd|th)", t)
            if m:
                t = m.group(1).lstrip("0") or "0"
        out.append(t)
    return out
