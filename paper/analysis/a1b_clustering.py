"""A1b: clustering. The local clustering coefficient of every researcher (exact), the network's average against
a random network with the same degrees, and clustering by degree. Merged hubs should stand out: the coauthors
of different people rarely work with each other. Compared at matched degree with hubs that look like single
people, using the affiliation evidence of A6d."""
import numpy as np
import pandas as pd
from numba import njit, prange

from common import CACHE, load_csr, log, save_result, suspect_mask


@njit(parallel=True, cache=True)
def triangles(offsets, neighbors, skip, use_skip, nchunks):
    """Links among each node's coauthors; nodes are dealt to threads in strides to balance the hubs."""
    n = offsets.shape[0] - 1
    tri = np.zeros(n, np.int64)
    k = np.zeros(n, np.int64)
    for c in prange(nchunks):
        mark = np.zeros(n, np.bool_)
        for u in range(c, n, nchunks):
            a, b = offsets[u], offsets[u + 1]
            for j in range(a, b):
                if not (use_skip and skip[j]):
                    mark[neighbors[j]] = True
                    k[u] += 1
            if k[u] >= 2:
                t = 0
                for j in range(a, b):
                    if use_skip and skip[j]:
                        continue
                    v = neighbors[j]
                    for l in range(offsets[v], offsets[v + 1]):
                        if mark[neighbors[l]] and not (use_skip and skip[l]):
                            t += 1
                tri[u] = t // 2
            for j in range(a, b):
                mark[neighbors[j]] = False
    return tri, k


offsets, neighbors = load_csr(mmap=False)
skip = np.asarray(suspect_mask(offsets, neighbors))
out = {}
for tag, use_skip in (("all", False), ("flagged", True)):
    tri, k = triangles(offsets, neighbors, skip, use_skip, 28)
    pairs = k * (k - 1) / 2
    c = np.where(k >= 2, tri / np.maximum(pairs, 1), np.nan)
    kf = k.astype(np.float64)
    out[tag] = {"mean_local": float(np.nanmean(c)), "transitivity": float(tri.sum() / pairs.sum()),
                "random_expectation": float(((kf ** 2).mean() - kf.mean()) ** 2 / (len(kf) * kf.mean() ** 3)),
                "share_zero": float(np.nanmean(c == 0))}
    np.save(CACHE / f"clustering_{tag}.npy", c.astype(np.float32))
    log(tag, out[tag])
c = np.load(CACHE / "clustering_all.npy")
deg = np.diff(offsets)
bins = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10**6]
b = np.digitize(deg, bins)
out["by_degree"] = [{"degree_from": bins[i - 1], "nodes": int((b == i).sum()), "median": float(np.nanmedian(c[b == i]))}
                    for i in range(1, len(bins)) if (b == i).any()]
aff = pd.read_csv(CACHE / "a6_affiliations.csv")
top = aff[(aff.group == "top") & aff.alive].copy()
top["clust"] = c[top.node_idx]
top["merged"] = top.peak_insts >= 25
top["single"] = top.peak_insts <= 5
top["dbin"] = pd.qcut(top.degree, 10, duplicates="drop")
rows = []
for dbin, g in top.groupby("dbin", observed=True):
    m, s = g[g.merged].clust, g[g.single].clust
    # AUC: chance a merged hub has lower clustering than a single-looking hub of similar degree
    auc = float((m.to_numpy()[:, None] < s.to_numpy()[None, :]).mean()) if len(m) and len(s) else None
    rows.append({"degree": str(dbin), "merged": int(len(m)), "single": int(len(s)), "median_merged": float(m.median()) if len(m) else None,
                 "median_single": float(s.median()) if len(s) else None, "auc_low_clustering": auc})
out["merged_vs_single"] = rows
save_result("a1b_clustering", out)
for r in rows:
    log(r)
