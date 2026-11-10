"""A6j: merged profiles from their own name records.

OpenAlex keeps, for every author profile, the name variants under which its works were recorded
(display_name_alternatives). One person's variants are forms of one name ("Paul M. Thompson", "P. M. Thompson",
"Thompson, Paul"); a profile that merges several people with one surname also lists their different given names
("Yi-Xiang Wang" lists "Jiafeng Wang", "Jiajia Wang", "Jian Wang", ...). For the 50,000 highest-degree profiles (as
A6d, from the live API, so after the snapshot) we compare every variant carrying the profile's surname with the
snapshot's display name. Titles ("Dr", "Chair") and tokens with digits or punctuation are dropped. A variant is the same
person's when any of its given names matches the displayed first name (equal, an initial, a prefix of three letters or
more, or a spelling variant) or its given names are the displayed ones without separators; variants made only of
initials are not judged. The rest are counted as distinct given names that cannot be the displayed one.
A profile with two or more such given names is a name-record merge; the revised repair link rule
(common.repair_link_mask) treats it like a profile with a common full name. As a check, the share whose works fell by
more than half in the next snapshot (14 October 2026), the proxy for OpenAlex's own splits used in A6h.

Writes cache/a6_name_records.csv and results/a6_name_records.json.
"""
import json
import re
import subprocess
import time
import unicodedata
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import CACHE, log, save_result

TOP, BATCH, MIN_INCOMPATIBLE = 50_000, 100, 2
TITLES = {"dr", "prof", "professor", "chair", "md", "phd", "mr", "mrs", "ms", "miss", "sir", "jr", "sr", "dds", "msc"}
RAW = CACHE / "a6_name_records_raw.jsonl"


def tokens(name):
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    return [t for t in re.split(r"[\s\-.,'’]+", s) if t and t.isalpha() and t not in TITLES]


def match(a, r):
    """Same name token: equal, an initial or a prefix of three letters or more of the other, or a spelling
    variant (same first letter, 80% similar)."""
    if a == r or (len(a) == 1 and r.startswith(a)) or (len(r) == 1 and a.startswith(r)):
        return True
    if (len(a) >= 3 and r.startswith(a)) or (len(r) >= 3 and a.startswith(r)):    # "Greg" for "Gregory"
        return True
    return len(a) > 2 and len(r) > 2 and a[0] == r[0] and SequenceMatcher(None, a, r).ratio() >= 0.8


def subsequence(short, long_):
    i = 0
    for t in long_:
        if i < len(short) and match(short[i], t):
            i += 1
    return i == len(short)


def given(name_tokens, surname):
    """Given-name tokens of a variant with the profile's surname (first or last), else None."""
    if not name_tokens or surname not in name_tokens:
        return None
    rest = list(name_tokens)
    rest.remove(surname)
    return rest


def expand_initials(g, ref):
    """Tokens that are the reference's initials run together ("gyh" for "Gregory Y. H.") become single letters."""
    initials = "".join(t[0] for t in ref)
    out = []
    for t in g:
        if 1 < len(t) <= len(ref) and t not in ref and subsequence(list(t), list(initials)):
            out.extend(t)
        else:
            out.append(t)
    return out


def compatible(g, ref):
    if not g or "".join(g) == "".join(ref):
        return True
    return any(match(t, ref[0]) for t in g)


def incompatible_names(display, alternatives):
    ref_tokens = tokens(display)
    if len(ref_tokens) < 2:
        return []
    surname = ref_tokens[-1]
    ref = ref_tokens[:-1]
    out = {}                                             # distinct given names, by their letters
    for alt in alternatives or []:
        g = given(tokens(alt), surname)
        if g is None:
            continue
        g = expand_initials(g, ref)
        if all(len(t) < 3 for t in g):                   # initials and two-letter fragments are not judged
            continue
        if not compatible(g, ref):
            out.setdefault("".join(g), " ".join(g))
    return sorted(out.values())


def fetch(aids):
    live = {}
    for i in range(0, len(aids), BATCH):
        ids = "|".join(aids[i:i + BATCH])
        url = (f"https://api.openalex.org/authors?filter=ids.openalex:{ids}"
               f"&select=id,display_name,display_name_alternatives&per_page={BATCH}")
        d = {}
        for attempt in range(4):
            r = subprocess.run(["curl", "-s", "-m", "90", url], capture_output=True, text=True)
            try:
                d = json.loads(r.stdout)
                break
            except json.JSONDecodeError:
                time.sleep(2 + 3 * attempt)
        for rec in d.get("results", []):
            live[rec["id"].rsplit("/", 1)[-1]] = rec
        if i // BATCH % 50 == 0:
            log(f"name records: {min(i + BATCH, len(aids)):,}/{len(aids):,} requested, {len(live):,} resolved")
        time.sleep(0.15)
    return live


if __name__ == "__main__":
    attrs = pq.read_table(CACHE / "node_attrs.parquet", columns=["node_idx", "id", "name", "degree", "works"]).to_pandas()
    top = attrs.sort_values("degree", ascending=False, kind="stable").head(TOP).copy()
    top["aid"] = top["id"].str.extract(r"(A\d+)$")[0]
    if RAW.exists():
        live = {r["id"].rsplit("/", 1)[-1]: r for r in map(json.loads, RAW.read_text().splitlines())}
        log(f"{len(live):,} name records from {RAW.name}")
    else:
        live = fetch(top.aid.tolist())
        RAW.write_text("".join(json.dumps(r) + "\n" for r in live.values()))
    rows = []
    for r in top.itertuples():
        rec = live.get(r.aid)
        bad = incompatible_names(r.name, rec.get("display_name_alternatives")) if rec else []
        rows.append({"node_idx": r.node_idx, "aid": r.aid, "name": r.name, "degree": r.degree, "resolved": rec is not None,
                     "live_display": rec.get("display_name") if rec else None,
                     "alternatives": len(rec.get("display_name_alternatives") or []) if rec else None,
                     "incompatible": len(bad), "examples": "; ".join(bad[:6]),
                     "name_record_merge": len(bad) >= MIN_INCOMPATIBLE})
    df = pd.DataFrame(rows)
    df.to_csv(CACHE / "a6_name_records.csv", index=False)
    new = pd.read_parquet(CACHE / "snapshot_20261014.parquet").set_index("id").works_count
    num = df.aid.str[1:].astype(np.int64)
    after = num.map(new)
    works = top.set_index("node_idx").works.reindex(df.node_idx).to_numpy()
    df["split_proxy"] = (after.to_numpy() < 0.5 * works) & after.notna().to_numpy()
    res = df[df.resolved]
    flagged, other = res[res.name_record_merge], res[~res.name_record_merge]
    examples = {n: df[df.name == n].iloc[0][["incompatible", "examples", "name_record_merge"]].to_dict()
                for n in ("Kai You Wang", "Yi-Xiang Wang", "Yue Wang", "Paul M. Thompson", "Henrik Zetterberg",
                          "Gregory Y.H. Lip", "John P. A. Ioannidis", "Luigi Ferrucci", "Takashi Taniguchi")
                if (df.name == n).any()}
    out = {"top": TOP, "resolved": int(len(res)), "min_incompatible": MIN_INCOMPATIBLE,
           "name_record_merges": int(len(flagged)), "share": float(len(flagged) / len(res)),
           "split_proxy_share": {"name_record_merge": float(flagged.split_proxy.mean()),
                                 "other": float(other.split_proxy.mean())},
           "top1000_share": float(res.head(1000).name_record_merge.mean()),
           "examples": {k: {kk: (bool(vv) if isinstance(vv, (bool, np.bool_)) else (int(vv) if isinstance(vv, (int, np.integer)) else vv))
                            for kk, vv in v.items()} for k, v in examples.items()}}
    save_result("a6_name_records", out)
    log(out)
