"""A7: how much do merged hubs shorten science? Distances with merges removed, against a matched control.

Variants, from the same random sources as A2:
  all          the network as delivered
  flagged      flagged (region-share) links removed
  merges       flagged links removed + hubs with impossible affiliations removed
               (peak distinct institutions in one year >= PEAK, from the live API, top 50,000 by degree)
  matched      flagged links removed + the merged hubs that have a degree-matched control
  control      flagged links removed + those controls instead: hubs that look like single people
               (peak <= 5), each within 10% of its merged hub's degree
Then the most central researcher (smallest mean distance to everyone), on the network as delivered and on
the cleanest one: mean distance to the sources first, then an exact BFS for the best 50.
"""
import sys

import numpy as np
import pandas as pd
from numba import get_num_threads, njit, prange

from common import CACHE, PG, con, load_csr, log, save_result, repair_link_mask

PEAK = int(sys.argv[1]) if len(sys.argv) > 1 else 25
# "strict": only common names count as merged (initial + surname shared by 200+ records, full name by 10+ mapped
# profiles), the rule the website's flags use; with an uncommon name, 25+ institutions is mostly one clinician with
# trial sites
STRICT = len(sys.argv) > 2 and sys.argv[2] == "strict"
TAG = f"peak{PEAK}" + ("_strict" if STRICT else "")
MAXD = 64


@njit(cache=True)
def _bfs(offsets, neighbors, skip, use_skip, removed, src, dist, queue):
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
            if removed[v] or dist[v] >= 0:
                continue
            dist[v] = du
            queue[tail] = v
            tail += 1
    return tail


@njit(parallel=True, cache=True)
def run(offsets, neighbors, skip, use_skip, removed, sources, maxd):
    n = offsets.shape[0] - 1
    k = sources.shape[0]
    h = np.zeros((k, maxd + 1), np.int64)
    for s in prange(k):
        d = np.empty(n, np.int16)
        q = np.empty(n, np.int32)
        _bfs(offsets, neighbors, skip, use_skip, removed, sources[s], d, q)
        for i in range(n):
            if d[i] > 0:
                h[s, min(d[i], maxd)] += 1
    return h


@njit(parallel=True, cache=True)
def distance_sums(offsets, neighbors, skip, use_skip, removed, sources, nchunks):
    n = offsets.shape[0] - 1
    k = sources.shape[0]
    acc = np.zeros((nchunks, n), np.int32)
    cnt = np.zeros((nchunks, n), np.int32)
    per = (k + nchunks - 1) // nchunks
    for c in prange(nchunks):
        d = np.empty(n, np.int16)
        q = np.empty(n, np.int32)
        for t in range(c * per, min(k, (c + 1) * per)):
            _bfs(offsets, neighbors, skip, use_skip, removed, sources[t], d, q)
            for i in range(n):
                if d[i] >= 0:
                    acc[c, i] += d[i]
                    cnt[c, i] += 1
    return acc.sum(axis=0), cnt.sum(axis=0)


@njit(parallel=True, cache=True)
def exact_mean_distance(offsets, neighbors, skip, use_skip, removed, nodes):
    n = offsets.shape[0] - 1
    out = np.zeros(nodes.shape[0], np.float64)
    for k in prange(nodes.shape[0]):
        d = np.empty(n, np.int16)
        q = np.empty(n, np.int32)
        reached = _bfs(offsets, neighbors, skip, use_skip, removed, nodes[k], d, q)
        tot = 0
        for i in range(n):
            if d[i] > 0:
                tot += d[i]
        out[k] = tot / (reached - 1)
    return out


def summary(h):
    t = h.sum(axis=0).astype(float)
    dd = np.arange(len(t))
    cdf = np.cumsum(t) / t.sum()
    return {"pairs": int(t.sum()), "mean": float((dd * t).sum() / t.sum()), "median": int(np.searchsorted(cdf, 0.5)),
            "p90": int(np.searchsorted(cdf, 0.9)), "share_le_3": float(t[:4].sum() / t.sum()),
            "share_ge_6": float(t[6:].sum() / t.sum())}


offsets, neighbors = load_csr(mmap=False)
n = len(offsets) - 1
deg = np.diff(offsets)
skip = np.asarray(repair_link_mask(offsets, neighbors))
aff = pd.read_csv(CACHE / "a6_affiliations.csv")
top = aff[(aff.group == "top") & aff.alive].copy()
if STRICT:
    # Flags already created these counts. Read only top-hub candidates rather
    # than indexing two complete 20-million-row metadata frames in Python.
    db = con()
    db.execute("SET memory_limit='6GB'")
    selected = top[["node_idx"]]
    db.register("selected", selected)
    counts = db.execute(f"""SELECT s.node_idx, k.key_n, f.full_n FROM selected s
        LEFT JOIN read_parquet('{PG / 'identity_signals.parquet'}') k USING(node_idx)
        LEFT JOIN read_parquet('{CACHE / 'full_name_counts.parquet'}') f USING(node_idx)""").fetchdf().set_index("node_idx")
    db.close()
    common = (counts.key_n.reindex(top.node_idx).to_numpy() >= 200) & (counts.full_n.reindex(top.node_idx).to_numpy() >= 10)
    top = top[common | (top.peak_insts <= 5)]       # merged candidates need a common name; controls need not
    del counts, selected, common
merged = top[top.peak_insts >= PEAK].node_idx.to_numpy()
pool = top[top.peak_insts <= 5].sort_values("degree").reset_index(drop=True)
pdeg = pool.degree.to_numpy()
used, pairs = np.zeros(len(pool), bool), []
for idx, d in top[top.peak_insts >= PEAK].sort_values("degree", ascending=False)[["node_idx", "degree"]].to_numpy():
    lo, hi = np.searchsorted(pdeg, d * 0.9), np.searchsorted(pdeg, d * 1.1, side="right")
    free = np.flatnonzero(~used[lo:hi]) + lo
    if len(free):
        j = free[np.argmin(np.abs(pdeg[free] - d))]
        used[j] = True
        pairs.append((idx, pool.node_idx[j]))
matched, ctrl = np.array(pairs).T
log(f"peak >= {PEAK}: {len(merged):,} merged hubs (total degree {int(deg[merged].sum()):,}); "
    f"{len(ctrl):,} have a matched control (degree {int(deg[matched].sum()):,} vs {int(deg[ctrl].sum()):,})")
np.savez(CACHE / f"a7_sets_{TAG}.npz", merged=merged, matched=matched, control=ctrl)
sources = np.load(CACHE / "a2_sources.npy")[:100]          # the original 100 (A2)
none = np.zeros(n, np.bool_)
rm_merged = none.copy(); rm_merged[merged] = True
rm_matched = none.copy(); rm_matched[matched] = True
rm_ctrl = none.copy(); rm_ctrl[ctrl] = True
keep = ~(rm_merged[sources] | rm_ctrl[sources])
sources = sources[keep]
out = {"PEAK": PEAK, "strict": STRICT, "merged_hubs": int(len(merged)), "merged_degree": int(deg[merged].sum()),
       "matched_hubs": int(len(matched)), "matched_degree": int(deg[matched].sum()),
       "control_hubs": int(len(ctrl)), "control_degree": int(deg[ctrl].sum()), "sources": int(len(sources))}
for tag, use_skip, rm in (("all", False, none), ("flagged", True, none), ("merges", True, rm_merged),
                          ("matched", True, rm_matched), ("control", True, rm_ctrl)):
    out[tag] = summary(run(offsets, neighbors, skip, use_skip, rm, sources, MAXD))
    log(tag, out[tag])
# the center of science: smallest mean distance to everyone, on the network as delivered and on the cleanest one
from common import Names  # noqa: E402
names = Names()
for tag, use_skip, rm in (("all", False, none), ("merges", True, rm_merged)):
    sums, cnt = distance_sums(offsets, neighbors, skip, use_skip, rm, sources, get_num_threads())
    est = np.where(cnt == len(sources), sums / len(sources), np.inf)    # only nodes every source reaches
    best = np.argsort(est)[:50].astype(np.int64)
    exact = exact_mean_distance(offsets, neighbors, skip, use_skip, rm, best)
    order = np.argsort(exact)
    out[f"most_central_{tag}"] = [{"node_idx": int(best[i]), "name": names[int(best[i])], "mean_distance_exact": float(exact[i]),
                                   "mean_distance_sampled": float(est[best[i]]), "degree": int(deg[best[i]])} for i in order[:30]]
    np.save(CACHE / f"closeness_{tag}_{TAG}.npy", est.astype(np.float32))
    log(f"most central ({tag}):", [(r["name"], round(r["mean_distance_exact"], 3)) for r in out[f"most_central_{tag}"][:10]])
save_result(f"a7_remedy_{TAG}", out)
