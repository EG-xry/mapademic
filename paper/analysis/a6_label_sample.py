"""A6f: a stratified sample of hub profiles for hand-labelling (merged vs. one person).

Strata: name commonness (common / not) x busiest-year institutions (<10, 10-24,
25-49, 50+), from the top-50,000 hubs with live affiliation data; 15 per
stratum. Writes paper/label_merges.csv: fill the `label` column with
`merged`, `single` or `unsure`, using the OpenAlex link (works, affiliations,
topics) and a quick web search where needed.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import CACHE, PAPER, PG

PER = 15
aff = pd.read_csv(CACHE / "a6_affiliations.csv")
aff = aff[(aff.group == "top") & aff.alive]
key = pq.read_table(PG / "identity_signals.parquet", columns=["node_idx", "key_n"]).to_pandas()
aff = aff.merge(key, on="node_idx", how="left")
aff["common"] = np.where(aff.key_n >= 200, "common", "not common")
aff["peak_bin"] = pd.cut(aff.peak_insts, [-1, 9, 24, 49, 10**6], labels=["<10", "10-24", "25-49", "50+"])
rng = np.random.default_rng(3)
parts = []
for (c, b), g in aff.groupby(["common", "peak_bin"], observed=True):
    parts.append(g.sample(min(PER, len(g)), random_state=int(rng.integers(1 << 30))))
s = pd.concat(parts).sort_values(["common", "peak_bin", "degree"], ascending=[True, True, False])
s["openalex"] = "https://openalex.org/" + s.aid
s["label"] = ""
s[["name", "country", "degree", "cited", "peak_insts", "peak_year", "peak_countries", "affiliations", "has_orcid",
   "key_n", "common", "peak_bin", "openalex", "label", "node_idx"]].to_csv(PAPER / "label_merges.csv", index=False)
print(len(s), "profiles to label ->", PAPER / "label_merges.csv")
print(s.groupby(["common", "peak_bin"], observed=True).size())
