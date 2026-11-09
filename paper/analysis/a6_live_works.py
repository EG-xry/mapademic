"""A6d follow-up: has OpenAlex split the giant hubs since the snapshot?

The live affiliation check (A6d) reads each profile as it is now. A profile that OpenAlex has broken up since
2026-09-23 looks clean there even though it was merged in our network. For the TOP highest-degree mapped
authors, compare the snapshot's works count with the live one: a profile that lost most of its works has
been split. Profiles already read for the map's network (cache_min15/a6_live_works.csv, October 2026) are reused,
so only new ones are fetched.
"""
import json
import subprocess
import sys
import time

import pandas as pd

from common import CACHE, PAPER, con, log, save_result

TOP, BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 50000, 100
c = con()
df = c.execute(f"""SELECT node_idx, id, name, degree, works, cited FROM read_parquet('{CACHE / 'node_attrs.parquet'}')
                   ORDER BY degree DESC LIMIT {TOP}""").fetchdf()
df["aid"] = df["id"].str.extract(r"(A\d+)$")[0]
prev = PAPER / "cache_min15" / "a6_live_works.csv"
live = ({r.aid: {"works_count": r.works_live, "cited_by_count": r.cited_live}
         for r in pd.read_csv(prev).itertuples() if r.alive} if CACHE != prev.parent and prev.exists() else {})
done = set(pd.read_csv(prev).aid) if CACHE != prev.parent and prev.exists() else set()
todo = [a for a in df.aid if a not in done]
log(f"{len(df) - len(todo):,} profiles already read; fetching {len(todo):,}")
for i in range(0, len(todo), BATCH):
    url = (f"https://api.openalex.org/authors?filter=ids.openalex:{'|'.join(todo[i:i + BATCH])}"
           f"&select=id,works_count,cited_by_count&per_page={BATCH}")
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
df["alive"] = df.aid.isin(live.keys())
df["works_live"] = [live[a]["works_count"] if a in live else None for a in df.aid]
df["cited_live"] = [live[a]["cited_by_count"] if a in live else None for a in df.aid]
df["works_ratio"] = df.works_live / df.works.clip(lower=1)
df.to_csv(CACHE / "a6_live_works.csv", index=False)
s = df[df.alive]
out = {"top": TOP, "resolved": int(len(s)), "share_lost_half": float((s.works_ratio < 0.5).mean()),
       "share_lost_90pct": float((s.works_ratio < 0.1).mean())}
for k in (100, 1000, 5000):
    t = s.head(k)
    out[f"top{k}"] = {"lost_half": float((t.works_ratio < 0.5).mean()), "median_ratio": float(t.works_ratio.median())}
save_result("a6_live_works", out)
log(out)
print(s.head(40)[["name", "degree", "works", "works_live", "cited", "cited_live"]].to_string())
