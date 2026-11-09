"""A6g: the same people in two scripts.

A researcher recorded under a name in Chinese characters (zh), Japanese kanji or kana (ja), Cyrillic (ru) or
Hangul (ko) is compared with the Latin-script profiles that carry the same name once romanized, in either name
order: pinyin for Chinese, every JMnedict reading for Japanese, and for Cyrillic and Hangul a consonant skeleton
that absorbs the usual transliteration variants (Lyubimov / Ljubimov, Seong-cheol / Sungchul). A candidate twin
is confirmed when at least MIN_SHARED of the two profiles' coauthors, romanized the same way, have matching
names: the same person and the same team, recorded twice (counted per coauthor of the script profile, so
that a coauthor with several readings counts once). Only names with at most MAXC Latin candidates are tested. As a chance baseline, each tested profile is also compared with as many decoys: Latin profiles that
share a surname with its candidates but not the name.

    python a6_twins.py zh|ja|ru|ko
"""
import json
import re
import sys
import unicodedata
from array import array
from collections import defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import CACHE, load_csr, log, save_result

MIN_SHARED, MAXC = 2, 5
SCRIPT = sys.argv[1]
LATIN = re.compile(r"^[A-Za-zÀ-ɏ\.\-\s']+$")


def candidate_indexes(latin, names, key_function, needed, needed_s, maximum=MAXC):
    """Keep at most MAXC+1 namesakes per key, while retaining exact ordered decoy pools.

    A key with more than MAXC+1 Latin profiles necessarily has >MAXC non-self candidates,
    even if the focal profile occurs in that key's list. Its overflow flag records that fact;
    the dictionary key remains present for the original decoy-pool membership test.
    Surname pools are never capped: compact int32 arrays preserve every member and its order.
    """
    index, surname_index = defaultdict(list), defaultdict(lambda: array("i"))
    overflow = set()
    for at, raw_index in enumerate(latin, start=1):
        i = int(raw_index)
        ks = key_function(names[i])
        for key in ks & needed:
            bucket = index[key]
            if len(bucket) < maximum + 1:
                bucket.append(i)
            else:
                overflow.add(key)
        for surname in {key[1] for key in ks} & needed_s:
            surname_index[surname].append(i)
        if at % 2_000_000 == 0:
            log(f"candidate index: {at:,}/{len(latin):,} Latin profiles; {len(overflow):,} oversized name keys")
    return index, surname_index, overflow


def usable_candidates(focal_keys, index, overflow, maximum=MAXC):
    """Count every focal profile with a namesake without retaining ineligible large unions."""
    with_any, usable = 0, {}
    for focal, ks in focal_keys.items():
        combined = set()
        oversized = False
        for key in ks:
            if key in overflow:
                oversized = True
                break
            for candidate in index.get(key, ()):
                combined.add(candidate)
                if len(combined) - int(focal in combined) > maximum:
                    oversized = True
                    break
            if oversized:
                break
        if oversized:
            with_any += 1
        else:
            # Retain the original final self-subtraction, including its set iteration behavior.
            candidates = combined - {focal}
            if candidates:
                with_any += 1
                usable[focal] = candidates
    return with_any, usable


def ascii_tokens(name):
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    toks = (t.replace("-", "") for t in re.split(r"[\s\.]+", re.sub(r"[^a-z\s\.\-]", " ", s)))
    return [t for t in toks if t]


def orders(toks):
    """(given, surname), with the surname last or first."""
    return [("".join(toks[:-1]), toks[-1]), ("".join(toks[1:]), toks[0])] if len(toks) >= 2 else []


if SCRIPT == "zh":
    from pypinyin import lazy_pinyin

    FOCAL = re.compile(r"^[一-鿿]{2,4}$")

    def keys(name):
        if FOCAL.match(name):
            py = [p.lower() for p in lazy_pinyin(name)]
            return {("".join(py[1:]), py[0])}                    # the surname is the first character
        return set(orders(ascii_tokens(name)))

elif SCRIPT == "ja":
    import pykakasi

    KK = "一-鿿々〆ヵヶぁ-ゟ゠-ヿ"
    FOCAL = re.compile(rf"^[{KK}]+\s+[{KK}]+$")
    R = json.load(open(CACHE / "jmnedict_readings.json"))
    kks = pykakasi.kakasi()

    def jnorm(t):
        """Collapse Hepburn variants: long vowels (Ohno, Oono, Ōno), n/m before labials (Homma, Honma)."""
        t = re.sub(r"oh(?![aeiouy])", "o", re.sub(r"[^a-z]", "", t))
        for a, b in (("ou", "o"), ("oo", "o"), ("uu", "u"), ("aa", "a"), ("ii", "i"), ("ee", "e"), ("nn", "n")):
            t = t.replace(a, b)
        return re.sub(r"m(?=[bmp])", "n", t)

    @lru_cache(maxsize=None)
    def readings(tok, role):
        out = set(R[role].get(tok, ())) or set(R["given" if role == "surname" else "surname"].get(tok, ()))
        out.add("".join(x["hepburn"] for x in kks.convert(tok)).lower())
        return frozenset(jnorm(r) for r in out) - {""}

    def keys(name):
        if FOCAL.match(name):
            a, b = name.split()
            return {(g, s) for x, y in ((a, b), (b, a)) for g in readings(x, "given") for s in readings(y, "surname")}
        return {(jnorm(g), jnorm(s)) for g, s in orders(ascii_tokens(name))}

elif SCRIPT == "ru":
    FOCAL = re.compile(r"^[Ѐ-ӿ\s\.\-'ʼ’]+$")
    TR = dict(zip("абвгґдеёєжзиіїйклмнопрстуфхцчшщъыьэюяўјљњћђџ",
                  "a b v g g d e e ye zh z i i yi y k l m n o p r s t u f kh ts ch sh shch - y - e yu ya u j lj nj c dj dz".split()))
    TR["ъ"] = TR["ь"] = ""

    def skel(t):
        """Consonant skeleton: Lyubimov, Ljubimov, Liubimov -> lbmv; Ukrainian h and Russian g fall together."""
        t = re.sub(r"[^a-z]", "", t)
        for a, b in (("shch", "s"), ("sch", "s"), ("sh", "s"), ("zh", "z"), ("ch", "c"), ("kh", "h"), ("ts", "c"),
                     ("tz", "c"), ("x", "ks"), ("ck", "k"), ("w", "v"), ("ph", "f"), ("q", "k"), ("g", "h")):
            t = t.replace(a, b)
        if not t:
            return ""
        return re.sub(r"(.)\1+", r"\1", {"y": "i", "j": "i"}.get(t[0], t[0]) + re.sub(r"[aeiouyj]", "", t[1:]))

    def keys(name):
        """(initial, surname skeleton) for every ordered pair of tokens: name order and patronymics vary."""
        if FOCAL.match(name):
            toks = ["".join(TR.get(c, c) for c in re.sub(r"[\-'ʼ’]", "", t)) for t in re.split(r"[\s\.]+", name.lower()) if t]
        else:
            toks = ascii_tokens(name)
        sk = [skel(t) for t in toks]
        return {(sk[i][0], sk[j]) for i in range(len(toks)) for j in range(len(toks))
                if i != j and sk[i] and len(toks[j]) >= 3 and len(sk[j]) >= 2}

elif SCRIPT == "ko":
    FOCAL = re.compile(r"^[가-힣]+(\s+[가-힣]+)?$")
    CHO = "g kk n d tt r m b pp s ss - j jj ch k t p h".split()
    JUNG = "a ae ya yae eo e yeo ye o wa wae oe yo u wo we wi yu eu ui i".split()
    JONG = "- k k k n n n t l k m l l l p l m p p t t ng t t k t p t".split()
    SURNAMES = {
        "김": "kim gim", "이": "lee yi rhee li i rhie", "박": "park pak bak", "최": "choi choe chey",
        "정": "jung jeong chung cheong joung", "강": "kang gang", "조": "cho jo joe", "윤": "yoon yun youn",
        "장": "jang chang", "임": "lim im yim rim", "한": "han", "오": "oh o", "서": "seo suh so", "신": "shin sin",
        "권": "kwon gwon", "황": "hwang", "안": "ahn an", "송": "song", "류": "ryu yoo yu ryoo lyu rhew",
        "유": "yoo yu you ryu", "홍": "hong", "전": "jeon jun chun chon", "고": "ko koh go", "문": "moon mun",
        "양": "yang", "손": "son sohn", "배": "bae pae", "백": "baek paik baik back", "허": "heo huh hur",
        "남": "nam", "심": "shim sim", "노": "noh roh no ro", "하": "ha", "곽": "kwak gwak", "성": "sung seong",
        "차": "cha", "주": "joo ju chu choo", "우": "woo wu u", "구": "koo ku gu", "민": "min", "진": "jin chin",
        "나": "na ra", "지": "ji chi", "엄": "eom um uhm", "채": "chae", "원": "won", "천": "cheon chun",
        "방": "bang pang", "공": "kong gong", "현": "hyun hyeon", "변": "byun byeon pyun", "염": "yeom yum",
        "여": "yeo yu yo", "추": "choo chu", "도": "do doh", "소": "so", "석": "seok suk", "선": "sun seon",
        "설": "seol sul", "마": "ma", "길": "gil kil", "연": "yeon yun youn", "위": "wi wee", "표": "pyo",
        "명": "myung myeong", "기": "ki gi", "반": "ban pan", "왕": "wang", "금": "keum geum kum", "옥": "ok",
        "육": "yook yuk", "인": "in", "맹": "maeng", "제": "je jae", "모": "mo", "탁": "tak", "국": "kook guk kuk",
        "은": "eun", "편": "pyun pyeon", "용": "yong", "예": "ye", "경": "kyung gyeong", "봉": "bong", "어": "eo uh",
        "황보": "hwangbo", "남궁": "namgung namkoong namgoong", "제갈": "jegal", "선우": "sunwoo seonu",
        "독고": "dokgo", "사공": "sagong"}
    REV = defaultdict(set)
    for h, sp in SURNAMES.items():
        for x in sp.split():
            REV[x].add(h)

    def rr(word):
        """Revised Romanization, syllable by syllable."""
        out = []
        for ch in word:
            c = ord(ch) - 0xAC00
            out += [CHO[c // 588], JUNG[c % 588 // 28], JONG[c % 28]]
        return "".join(x for x in out if x != "-")

    def kskel(t):
        """Consonant skeleton: Seong-cheol, Sungchul, Sung Chul -> sngjl; Park, Pak, Bak -> bg."""
        t = re.sub(r"(?<=[aeiou])r(?=[^aeiou]|$)", "", re.sub(r"[^a-z]", "", t))
        for a, b in (("sh", "s"), ("ch", "j"), ("k", "g"), ("t", "d"), ("p", "b"), ("r", "l"), ("f", "b"), ("v", "b"), ("z", "j")):
            t = t.replace(a, b)
        return re.sub(r"(.)\1+", r"\1", re.sub(r"[aeiouwy]", "", t))

    def keys(name):
        if FOCAL.match(name):
            parts = name.split()
            if len(parts) == 2:
                pairs = [(parts[0], parts[1]), (parts[1], parts[0])]
            else:
                w = parts[0]
                s = 2 if len(w) >= 3 and w[:2] in SURNAMES else 1
                pairs = [(w[s:], w[:s])]
            return {(kskel(rr(g)), s if s in SURNAMES else kskel(rr(s))) for g, s in pairs if len(kskel(rr(g))) >= 2}
        return {(kskel(g), sid) for g, s in orders(ascii_tokens(name)) for sid in (REV.get(s) or {kskel(s)})
                if len(kskel(g)) >= 2}

a = pq.read_table(CACHE / "node_attrs.parquet", columns=["node_idx", "name", "community", "degree", "cited"]).to_pandas()
assert (a.node_idx.to_numpy() == np.arange(len(a))).all()
names = a.name.fillna("").tolist()
nm, alive = a.name.fillna(""), a.degree.to_numpy() > 0
focal = np.flatnonzero(nm.str.match(FOCAL.pattern).to_numpy() & alive).tolist()
latin = np.flatnonzero(nm.str.match(LATIN.pattern).to_numpy() & alive)
del nm, alive
log(f"{SCRIPT}: {len(focal):,} profiles in the script; {len(latin):,} Latin-script profiles")
fkeys = {f: keys(names[f]) for f in focal}
needed = set().union(*fkeys.values())
needed_s = {k[1] for k in needed}
index, sindex, overflow = candidate_indexes(latin, names, keys, needed, needed_s)
with_any, usable = usable_candidates(fkeys, index, overflow)
del latin, needed, needed_s, overflow
# Matching needs keys only for tested focal profiles; decoys use index membership, not its candidate lists.
# Keep the complete original ordered surname pools for exactly those keys that can be consulted.
fkeys = {f: fkeys[f] for f in usable}
index = {key for ks in fkeys.values() for key in ks if key in index}
used_surnames = {key[1] for key in index}
sindex = {surname: pool for surname, pool in sindex.items() if surname in used_surnames}
del used_surnames
log(f"{with_any:,} have a romanized namesake; {len(usable):,} with at most {MAXC} (tested)")

offsets, neighbors = load_csr()


@lru_cache(maxsize=500_000)
def name_keys(i):
    return frozenset(keys(names[i]))


def coauthors(i):
    """Name keys of each coauthor that has any."""
    return [k for k in (name_keys(v) for v in neighbors[offsets[i]:offsets[i + 1]].tolist()) if k]


def coauthor_keys(i):
    return set().union(*coauthors(i))


def matched(co, j):
    """How many of these coauthors have a namesake among j's coauthors."""
    ck = coauthor_keys(j)
    return sum(1 for k in co if not k.isdisjoint(ck))


rng = np.random.default_rng(20261024)


def decoys(f, c, ks):
    """As many Latin profiles as candidates, sharing a surname with them but not the name."""
    pools = [sindex[s] for s in {k[1] for k in ks if k in index} if sindex.get(s)]
    out, tries = set(), 0
    while pools and len(out) < len(c) and tries < 50 * len(c):
        tries += 1
        pool = pools[rng.integers(len(pools))]
        j = pool[rng.integers(len(pool))]
        if j != f and j not in c and not (name_keys(j) & ks):
            out.add(j)
    return out


rows = []
for f, c in usable.items():
    co = coauthors(f)
    real = max(((matched(co, j), j) for j in c), key=lambda t: t[0])
    dec = decoys(f, c, fkeys[f])
    null = max((matched(co, j) for j in dec), default=-1)
    rows.append((f, real[1], real[0], len(c), null, len(dec), len(co)))
df = pd.DataFrame(rows, columns=["focal_idx", "twin_idx", "shared", "candidates", "decoy_shared", "decoys", "coauthors"])
df.to_csv(CACHE / f"a6_twins_{SCRIPT}.csv", index=False)
conf = df[df.shared >= MIN_SHARED]
nulls = df[df.decoys > 0]
comm, cited = a["community"], a["cited"]
out = {"script": SCRIPT, "profiles": len(focal), "with_namesake": with_any, "tested": len(usable),
       "confirmed": int(len(conf)), "confirmed_share_of_tested": float(len(conf) / max(len(usable), 1)),
       "confirmed_share_of_profiles": float(len(conf) / max(len(focal), 1)),
       "pass_rate": {t: {"twins": float((df.shared >= t).mean()), "decoys": float((nulls.decoy_shared >= t).mean())}
                     for t in (1, 2, 3, 5, 10)},
       "decoy_tested": int(len(nulls)),
       "confirmed_median_shared": float(conf.shared.median()),
       "confirmed_median_share_of_coauthors": float((conf.shared / conf.coauthors.clip(lower=1)).median()),
       "median_cited": {"script_profile": float(cited[conf.focal_idx].median()), "latin_twin": float(cited[conf.twin_idx].median())},
       "different_community": float((comm[conf.focal_idx].to_numpy() != comm[conf.twin_idx].to_numpy()).mean()),
       "script_communities": comm[focal].value_counts().head(5).to_dict(),
       "twin_communities": comm[conf.twin_idx].value_counts().head(5).to_dict(),
       "examples": [{"script": names[r.focal_idx], "latin": names[r.twin_idx], "shared": int(r.shared)}
                    for r in conf.sort_values("shared", ascending=False).head(25).itertuples()]}
save_result(f"a6_twins_{SCRIPT}", out)
log({k: v for k, v in out.items() if k != "examples"})
for e in out["examples"][:15]:
    print(e)
