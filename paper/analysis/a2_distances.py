"""A2: degrees of separation -- the distance distribution, all links vs. flagged links removed.

Full breadth-first searches from K random sources in the largest component
(the same sources for both networks), so each source contributes distances to
every other node: K x 8.4M pairs. The first 100 sources are the original sample,
which the removal experiments (A7, A7b, A9) keep using. Per pair we also record how much removing the
flagged (probable namesake-merge) links lengthens the distance.
"""
import sys

import numpy as np
from numba import njit, prange

from common import CACHE, load_csr, log, save_result, suspect_mask

MAXD = 64
K = int(sys.argv[1]) if len(sys.argv) > 1 else 1000


@njit(cache=True)
def _bfs(offsets, neighbors, skip, use_skip, src, dist, queue):
    n = offsets.shape[0] - 1
    for i in range(n):
        dist[i] = -1
    dist[src] = 0
    queue[0] = src
    head, tail = 0, 1
    while head < tail:
        u = queue[head]
        head += 1
        du = dist[u] + 1
        for j in range(offsets[u], offsets[u + 1]):
            if use_skip and skip[j]:
                continue
            v = neighbors[j]
            if dist[v] < 0:
                dist[v] = du
                queue[tail] = v
                tail += 1
    return tail


@njit(parallel=True, cache=True)
def run(offsets, neighbors, skip, sources, maxd):
    n = offsets.shape[0] - 1
    k = sources.shape[0]
    h_all = np.zeros((k, maxd + 1), np.int64)
    h_clean = np.zeros((k, maxd + 1), np.int64)
    h_diff = np.zeros((k, maxd + 2), np.int64)      # last bin: unreachable once flagged links go
    ecc = np.zeros((k, 2), np.int64)
    for s in prange(k):
        d1 = np.empty(n, np.int16)
        d2 = np.empty(n, np.int16)
        q = np.empty(n, np.int32)
        _bfs(offsets, neighbors, skip, False, sources[s], d1, q)
        _bfs(offsets, neighbors, skip, True, sources[s], d2, q)
        e1 = 0
        e2 = 0
        for i in range(n):
            a = d1[i]
            if a < 0:
                continue
            if a > e1:
                e1 = a
            h_all[s, min(a, maxd)] += 1
            b = d2[i]
            if b < 0:
                h_diff[s, maxd + 1] += 1
            else:
                if b > e2:
                    e2 = b
                h_clean[s, min(b, maxd)] += 1
                h_diff[s, min(b - a, maxd)] += 1
        ecc[s, 0] = e1
        ecc[s, 1] = e2
    return h_all, h_clean, h_diff, ecc


def summarize(h):
    h = h.sum(axis=0).astype(np.float64)
    h[0] = 0                                  # drop each source's distance to itself
    d = np.arange(len(h))
    tot = h.sum()
    cdf = np.cumsum(h) / tot
    return {"pairs": int(tot), "mean": float((d * h).sum() / tot),
            "median": int(np.searchsorted(cdf, 0.5)), "p90_effective_diameter": int(np.searchsorted(cdf, 0.9)),
            "max_seen": int(np.flatnonzero(h)[-1]), "pmf": {int(i): float(h[i] / tot) for i in np.flatnonzero(h)}}


offsets, neighbors = load_csr(mmap=False)
skip = np.asarray(suspect_mask(offsets, neighbors))
lcc = np.load(CACHE / "lcc_all.npy") & np.load(CACHE / "lcc_clean.npy")
rng = np.random.default_rng(20261024)
pool = np.flatnonzero(lcc)
first = rng.choice(pool, min(K, 100), replace=False)
more = rng.choice(np.setdiff1d(pool, first), K - len(first), replace=False)
sources = np.concatenate([first, more]).astype(np.int32)
log(f"{K} sources from {int(lcc.sum()):,} nodes in both largest components")
h_all, h_clean, h_diff, ecc = run(offsets, neighbors, skip, sources, MAXD)
log("done")
per_source_mean_all = (h_all[:, 1:] * np.arange(1, MAXD + 1)).sum(1) / h_all[:, 1:].sum(1)
per_source_mean_clean = (h_clean[:, 1:] * np.arange(1, MAXD + 1)).sum(1) / h_clean[:, 1:].sum(1)
diff = h_diff.sum(axis=0).astype(np.float64)
diff[0] -= K                                   # the sources themselves
out = {"K": K, "seed": 20261024,
       "all_links": summarize(h_all), "flagged_removed": summarize(h_clean),
       "per_source_mean": {"all_mean": float(per_source_mean_all.mean()), "all_sd": float(per_source_mean_all.std(ddof=1)),
                           "all_se": float(per_source_mean_all.std(ddof=1) / np.sqrt(K)),
                           "clean_mean": float(per_source_mean_clean.mean()), "clean_sd": float(per_source_mean_clean.std(ddof=1))},
       "lengthened_by_removal": {"pairs": int(diff.sum()), "share_unchanged": float(diff[0] / diff.sum()),
                                 "share_plus1": float(diff[1] / diff.sum()), "share_plus2_or_more": float(diff[2:MAXD + 1].sum() / diff.sum()),
                                 "share_disconnected": float(diff[MAXD + 1] / diff.sum())},
       "source_eccentricity": {"all_max": int(ecc[:, 0].max()), "clean_max": int(ecc[:, 1].max()),
                               "all_mean": float(ecc[:, 0].mean()), "clean_mean": float(ecc[:, 1].mean())}}
np.save(CACHE / "a2_sources.npy", sources)
save_result("a2_distances", out)
log({k: v for k, v in out.items() if k not in ("all_links", "flagged_removed")})
log("all:", {k: v for k, v in out["all_links"].items() if k != "pmf"})
log("clean:", {k: v for k, v in out["flagged_removed"].items() if k != "pmf"})
