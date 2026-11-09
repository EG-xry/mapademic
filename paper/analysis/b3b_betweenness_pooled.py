"""B3b: does the hub comparison depend on taking the smaller of the two betweenness estimates?

A4 estimates betweenness twice, from two independent sets of 1,000 sources, each scaled by n / 1,000, and B3 ranks
by the smaller of the two. Here the alternative is the pooled estimate, their mean, which is the estimate from all
2,000 sources. With B3's eligibility (unflagged profiles with a finite closeness estimate) and B3's random sample of
500,000 for Spearman correlations, this compares the two versions directly (top-k overlap, rank correlation, the top
ten) and recomputes betweenness's top-1,000 overlaps and correlations with degree, strength, closeness and citations.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from b3_hubs import EDGE_BATCH, _add_strength
from common import CACHE, G, flagged_mask, load_csr, log, save_result

TOP = 1000
OTHERS = ("degree", "strength", "closeness", "citations")


def strength_of(path, n):
    left, right = np.zeros(n, np.float64), np.zeros(n, np.float64)
    for batch in pq.ParquetFile(path).iter_batches(batch_size=EDGE_BATCH, columns=["src", "dst", "weight"]):
        _add_strength(batch.column("src").to_numpy(), batch.column("dst").to_numpy(),
                      batch.column("weight").to_numpy().astype(np.float64, copy=False), left, right)
    return left + right


offsets, _ = load_csr()
deg = np.diff(offsets).astype(np.int64)
n = len(deg)
flagged = flagged_mask(n)
hubs = pq.read_table(CACHE / "hubs.parquet", columns=["node_idx", "btw_a", "btw_b", "btw"]).to_pandas().set_index("node_idx")
hubs = hubs.reindex(np.arange(n)).fillna(0)
assert np.allclose(hubs.btw, np.minimum(hubs.btw_a, hubs.btw_b))
close = np.load(CACHE / "closeness_1000.npy")
df = pd.DataFrame({"degree": deg, "strength": strength_of(G / "edges_int32.parquet", n),
                   "closeness": -close, "citations": pq.read_table(CACHE / "node_attrs.parquet", columns=["cited"]).column(0).to_numpy(),
                   "smaller": hubs.btw.to_numpy(), "pooled": ((hubs.btw_a + hubs.btw_b) / 2).to_numpy()})
eligible = df[~flagged & np.isfinite(close)]
log(f"{len(eligible):,} eligible profiles")
sub = eligible.sample(500_000, random_state=1)
top = {k: {m: set(eligible.nlargest(k, m).index) for m in eligible.columns} for k in (10, 100, 1000, 10_000)}
out = {"eligible": len(eligible),
       "pooled_vs_smaller": {"spearman_sample_500k": float(sub[["smaller", "pooled"]].corr("spearman").iloc[0, 1]),
                             **{f"overlap_top{k}": len(top[k]["smaller"] & top[k]["pooled"]) for k in top}},
       "top10": {v: eligible.nlargest(10, v).index.tolist() for v in ("smaller", "pooled")},
       "with_other_measures": {v: {m: {"overlap_top1000": len(top[1000][v] & top[1000][m]),
                                       "spearman_sample_500k": float(sub[[v, m]].corr("spearman").iloc[0, 1])}
                                   for m in OTHERS}
                               for v in ("smaller", "pooled")}}
runs = hubs.loc[eligible.index, ["btw_a", "btw_b"]].to_numpy()
ratio = runs.max(axis=1) / np.maximum(runs.min(axis=1), 1e-12)          # how far the two runs disagree


def profile(index):
    """The top list's two runs and coauthors: a nearby sampled source inflates one run only."""
    at = eligible.index.get_indexer(sorted(index))
    r = ratio[at]
    return {"median_run_ratio": float(np.median(r)), "share_runs_10x_apart": float((r > 10).mean()),
            "median_degree": float(np.median(eligible.degree.to_numpy()[at]))}


out["top1000_profiles"] = {v: profile(top[1000][v]) for v in ("smaller", "pooled")}
out["top1000_profiles"]["eligible_median_degree"] = float(eligible.degree.median())
save_result("b3b_betweenness_pooled", out)
log(out["top1000_profiles"])
log(out["pooled_vs_smaller"])
for v in ("smaller", "pooled"):
    log(v, {m: (r["overlap_top1000"], round(r["spearman_sample_500k"], 3)) for m, r in out["with_other_measures"][v].items()})
