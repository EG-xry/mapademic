"""A4b: who the hubs are -- the top brokers by A4's betweenness, brokers with few citations (top 0.1% of betweenness
with below-median citations: their scripts, communities and the share of their coauthors in their own community) and
the citation hubs as bridges. Profiles flagged as merged, split or not a person (flags.py) are left out, as in B3."""
import gc

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import CACHE, SCRIPT_COMMUNITIES, flagged_mask, load_csr, name_script, save_result
from b3_hubs import selected_metadata

h = pq.read_table(CACHE / "hubs.parquet").to_pandas()
a = pq.read_table(CACHE / "node_attrs.parquet", columns=["node_idx", "cited", "degree", "community"]).to_pandas()
assert np.array_equal(a.node_idx.to_numpy(), np.arange(len(a)))
comm = a.community.to_numpy(copy=True)
df = h.merge(a, on="node_idx")
df["flagged"] = flagged_mask(len(a))[df.node_idx]
df = df[(df.degree > 0) & ~df.flagged].copy()
del a, h
gc.collect()
df["btw_pct"] = df.btw.rank(pct=True)
df["cited_pct"] = df.cited.rank(pct=True)
df["degree_pct"] = df.degree.rank(pct=True)
out = {"researchers": int(len(df))}
top_cited = df.sort_values("cited", ascending=False).head(1000)
out["top1000_cited"] = {"median_betweenness_pct": float(top_cited.btw_pct.median()),
                        "median_degree_pct": float(top_cited.degree_pct.median()),
                        "median_participation": float(top_cited.participation.median()),
                        "overall_median_participation": float(df.participation.median())}
out["spearman"] = {"betweenness_citations": float(df[["btw", "cited"]].corr("spearman").iloc[0, 1]),
                   "betweenness_degree": float(df[["btw", "degree"]].corr("spearman").iloc[0, 1]),
                   "participation_citations": float(df[["participation", "cited"]].corr("spearman").iloc[0, 1])}
gate = df[(df.btw_pct >= 0.999) & (df.cited_pct <= 0.5)].sort_values("btw", ascending=False).copy()
top25 = df.sort_values("btw", ascending=False).head(25).copy()
# Script/name metadata is needed only for these reported researchers, not all 20M.
metadata = selected_metadata(CACHE / "node_attrs.parquet", set(gate.node_idx) | set(top25.node_idx))
for frame in (gate, top25):
    for column in ("name", "country", "field"):
        frame[column] = metadata[column].reindex(frame.node_idx).to_numpy()
    frame["script"] = name_script(frame.name)
del df, metadata, top_cited
gc.collect()
offsets, neighbors = load_csr()
gate["inside"] = [float((comm[np.asarray(neighbors[offsets[i]:offsets[i + 1]])] == comm[i]).mean()) for i in gate.node_idx]
native = gate[gate.community.isin([SCRIPT_COMMUNITIES[k] for k in ("Japanese script", "Chinese characters", "Korea")])]
out["gatekeepers"] = {"count": int(len(gate)), "scripts": gate.script.value_counts().to_dict(),
                      "scripts_top25": gate.head(25).script.value_counts().to_dict(),
                      "communities": {str(k): int(v) for k, v in gate.community.value_counts().items()},
                      "median_degree": float(gate.degree.median()), "median_participation": float(gate.participation.median()),
                      "in_native_script_communities": {"count": int(len(native)), "median_degree": float(native.degree.median()),
                                                       "median_share_of_coauthors_inside": float(native.inside.median())},
                      "all": gate[["name", "community", "cited", "degree", "inside", "script"]].to_dict("records"),
                      "examples": gate.head(25)[["name", "country", "field", "cited", "degree", "participation", "btw_a", "btw_b"]].to_dict("records")}
cols = ["name", "country", "field", "cited", "degree", "btw", "btw_a", "btw_b", "participation", "script"]
out["top25"] = top25[cols].to_dict("records")
save_result("a4b_hubs", out)
print({k: v for k, v in out.items() if k not in ("gatekeepers", "top25")})
print({k: v for k, v in out["gatekeepers"].items() if k not in ("examples", "all")})
with pd.option_context("display.width", 250):
    print(top25[cols].round(3).to_string())
    print("gatekeepers:")
    print(gate.head(15)[["name", "country", "field", "cited", "degree", "participation", "script"]].to_string())
