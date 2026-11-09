"""A6d: merged profiles from impossible affiliations.

A person holds a handful of affiliations in any one year; a profile that
merges many people with one name lists dozens or hundreds (the 2026 "Jun Wang"
record: 173 institutions in 2022). For the TOP highest-degree mapped authors
and a control sample of ordinary authors (50-300 coauthors) we read OpenAlex's
live author records (affiliations with years, ORCID, name variants) and take
the peak number of distinct institutions in a single year. Profiles already read for the map's network
(cache_min15/a6_affiliations.csv, October 2026) are reused, so only new ones are fetched.
"""
import json
import subprocess
import sys
import time
from collections import Counter

import pandas as pd

from common import CACHE, PAPER, con, log, save_result

TOP, CONTROL, BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 5000, 2000, 100
c = con()
A = f"read_parquet('{CACHE / 'node_attrs.parquet'}')"
top = c.execute(f"SELECT node_idx, id, name, degree, country, cited FROM {A} ORDER BY degree DESC LIMIT {TOP}").fetchdf()
top["group"] = "top"
ctrl = c.execute(f"""SELECT * FROM (SELECT node_idx, id, name, degree, country, cited FROM {A} WHERE degree BETWEEN 50 AND 300)
                     USING SAMPLE reservoir({CONTROL} ROWS) REPEATABLE (5)""").fetchdf()    # sampled after the filter
ctrl["group"] = "control"
df = pd.concat([top, ctrl], ignore_index=True)
df["aid"] = df["id"].str.extract(r"(A\d+)$")[0]
COLS = ["alive", "peak_insts", "peak_year", "peak_countries", "affiliations", "has_orcid", "name_variants"]
prev = PAPER / "cache_min15" / "a6_affiliations.csv"
prev = (pd.read_csv(prev).drop_duplicates("aid").set_index("aid")[COLS] if CACHE != prev.parent and prev.exists()
        else pd.DataFrame(columns=COLS))
todo = df.aid[~df.aid.isin(prev.index)].drop_duplicates().tolist()
log(f"{len(df) - len(todo):,} profiles already read; fetching {len(todo):,}")
live = {}
for i in range(0, len(todo), BATCH):
    ids = "|".join(todo[i:i + BATCH])
    url = (f"https://api.openalex.org/authors?filter=ids.openalex:{ids}"
           f"&select=id,orcid,affiliations,display_name_alternatives,works_count&per_page={BATCH}")
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
    time.sleep(0.15)
log(f"{len(live):,} of {len(todo):,} new profiles resolved")


def peak(rec):
    per_year, countries = Counter(), {}
    for a in rec.get("affiliations") or []:
        cc = (a.get("institution") or {}).get("country_code")
        for y in a.get("years") or []:
            per_year[y] += 1
            countries.setdefault(y, set()).add(cc)
    if not per_year:
        return 0, None, 0
    y, k = per_year.most_common(1)[0]
    return k, y, len(countries[y])


rows = []
for aid in df.aid:
    if aid in prev.index:
        rows.append(tuple(prev.loc[aid]))
        continue
    rec = live.get(aid)
    if rec is None:
        rows.append((False, None, None, None, None, None, None))
        continue
    k, y, ncc = peak(rec)
    rows.append((True, k, y, ncc, len(rec.get("affiliations") or []), bool(rec.get("orcid")),
                 len(rec.get("display_name_alternatives") or [])))
df[COLS] = rows
df.to_csv(CACHE / "a6_affiliations.csv", index=False)
out = {}
for g in ("top", "control"):
    s = df[(df.group == g) & df.alive]
    out[g] = {"n": int((df.group == g).sum()), "resolved": int(len(s)),
              "peak_insts_median": float(s.peak_insts.median()), "peak_insts_p90": float(s.peak_insts.quantile(0.9)),
              "share_peak_ge_10": float((s.peak_insts >= 10).mean()), "share_peak_ge_25": float((s.peak_insts >= 25).mean()),
              "share_peak_ge_50": float((s.peak_insts >= 50).mean()), "share_with_orcid": float(s.has_orcid.mean())}
t = df[(df.group == "top") & df.alive].sort_values("degree", ascending=False)
for k in (100, 1000):
    out[f"top{k}_share_peak_ge_25"] = float((t.head(k).peak_insts >= 25).mean())
save_result("a6_affiliations", out)
log(out)
print(t.head(30)[["name", "country", "degree", "peak_insts", "peak_year", "peak_countries", "affiliations", "has_orcid"]].to_string())
