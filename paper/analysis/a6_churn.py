"""A6c: how stable are OpenAlex author records? The snapshot against the live API.

A random sample of mapped authors (ids from the 2026-09-23 snapshot) is looked
up in the live API, 100 per request (filter=ids.openalex:A|B|...). An id that
no longer resolves has been merged away or deleted; for those that remain we
compare works and citation counts with the snapshot.
"""
import json
import subprocess
import time

import numpy as np

from common import CACHE, ND, con, log, save_result

N, BATCH = 5000, 100
c = con()
sample = c.execute(f"""SELECT regexp_extract(id, '(A\\d+)$', 1) AS aid, display_name, works_count, cited_by_count
                       FROM read_parquet('{ND}/nodes.parquet') USING SAMPLE reservoir({N} ROWS) REPEATABLE (11)""").fetchdf()
live = {}
for i in range(0, len(sample), BATCH):
    ids = "|".join(sample.aid[i:i + BATCH])
    url = (f"https://api.openalex.org/authors?filter=ids.openalex:{ids}"
           f"&select=id,display_name,works_count,cited_by_count&per_page={BATCH}")
    for attempt in range(4):
        r = subprocess.run(["curl", "-s", "-m", "60", url], capture_output=True, text=True)
        try:
            d = json.loads(r.stdout)
            break
        except json.JSONDecodeError:
            time.sleep(2 + 3 * attempt)
    for rec in d.get("results", []):
        live[rec["id"].rsplit("/", 1)[-1]] = rec
    time.sleep(0.15)
log(f"{len(live):,} of {len(sample):,} sampled ids still resolve")
sample["alive"] = sample.aid.isin(live.keys())
s = sample[sample.alive].copy()
s["works_live"] = [live[a]["works_count"] for a in s.aid]
s["cited_live"] = [live[a]["cited_by_count"] for a in s.aid]
s["name_live"] = [live[a]["display_name"] for a in s.aid]
rel_c = (s.cited_live - s.cited_by_count) / s.cited_by_count.clip(lower=1)
rel_w = (s.works_live - s.works_count) / s.works_count.clip(lower=1)
out = {"sample": int(len(sample)), "gone": int((~sample.alive).sum()), "gone_share": float((~sample.alive).mean()),
       "citations_change": {"median": float(rel_c.median()), "share_up_50pct": float((rel_c > 0.5).mean()),
                            "share_down_10pct": float((rel_c < -0.1).mean()), "share_doubled": float((rel_c >= 1).mean())},
       "works_change": {"median": float(rel_w.median()), "share_up_50pct": float((rel_w > 0.5).mean()),
                        "share_down_10pct": float((rel_w < -0.1).mean())},
       "name_changed_share": float((s.name_live != s.display_name).mean())}
sample.to_csv(CACHE / "a6_churn_sample.csv", index=False)
s.assign(rel_cited=rel_c, rel_works=rel_w).to_csv(CACHE / "a6_churn_alive.csv", index=False)
save_result("a6_churn", out)
log(out)
