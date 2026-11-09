"""B2c: are weak ties weak because they are rare or because their teams are large?

Link weight sums 1/(n-1) over a pair's shared works with n authors, so a weak link can be one shared work in a large
team or a few in small ones. From the bipartite graph of researchers and works (a2c_team_cap.build, cached in
cache/bipartite_*.npy), for every link: the number of shared works with at most 50 authors (the works that make
links) and the smallest and largest of their teams. Every link must have at least one such work (checked). Then, as
in B2, the share of links crossing communities and, among links with a known country at both ends, crossing
countries:
  - by number of shared works (1, 2, 3, 4-5, 6-10, 11+);
  - for one-off links, by the size of their one team (2, 3, 4-5, 6-9, 10-19, 20-50);
  - for links formed only in small teams (every shared work with at most 5 authors), by number of shared works;
and B2's neighbourhood overlap for a random two million links, by the same groups, with group sizes and standard
errors (the links are independent uniform draws, with replacement). The shares above count every link, so they
carry no sampling error.
Writes results/b2c_repeat_ties.json.
"""
import numpy as np
import pyarrow.parquet as pq
from numba import njit, prange

from common import CACHE, load_csr, log, save_result

SHARED_EDGES = np.array([1, 2, 3, 4, 6, 11], np.int64)       # 1, 2, 3, 4-5, 6-10, 11+
SHARED_LABELS = ["1", "2", "3", "4-5", "6-10", "11+"]
TEAM_EDGES = np.array([2, 3, 4, 6, 10, 20], np.int64)         # 2, 3, 4-5, 6-9, 10-19, 20-50
TEAM_LABELS = ["2", "3", "4-5", "6-9", "10-19", "20-50"]
SMALL_TEAM, MAX_TEAM, CHUNKS, SAMPLE = 5, 50, 512, 2_000_000


@njit(cache=True)
def _shared(a_off, a_nb, w_size, n, u, v):
    """Shared works of u and v with at most MAX_TEAM authors: (count, smallest team, largest team)."""
    i, ie, j, je = a_off[u], a_off[u + 1], a_off[v], a_off[v + 1]
    count, tmin, tmax = 0, 1 << 30, 0
    if ie - i > 8 * (je - j) or je - j > 8 * (ie - i):            # galloping for a long list against a short one
        if ie - i < je - j:
            i, ie, j, je = j, je, i, ie
        long_ = a_nb[i:ie]
        for t in range(j, je):
            x = a_nb[t]
            p = np.searchsorted(long_, x)
            if p < long_.shape[0] and long_[p] == x:
                s = w_size[x - n]
                if s <= MAX_TEAM:
                    count += 1
                    tmin = min(tmin, s)
                    tmax = max(tmax, s)
        return count, tmin, tmax
    while i < ie and j < je:
        x, y = a_nb[i], a_nb[j]
        if x == y:
            s = w_size[x - n]
            if s <= MAX_TEAM:
                count += 1
                tmin = min(tmin, s)
                tmax = max(tmax, s)
            i += 1
            j += 1
        elif x < y:
            i += 1
        else:
            j += 1
    return count, tmin, tmax


@njit(cache=True)
def _bin(edges, x):
    return np.searchsorted(edges, x, side="right") - 1


@njit(parallel=True, cache=True)
def link_tables(offsets, neighbors, a_off, a_nb, w_size, comm, cc, shared_edges, team_edges):
    """Per chunk of nodes, counts [links, across communities, known countries, across countries] by group."""
    n = offsets.shape[0] - 1
    by_shared = np.zeros((CHUNKS, shared_edges.shape[0], 4), np.int64)
    by_team = np.zeros((CHUNKS, team_edges.shape[0], 4), np.int64)
    small = np.zeros((CHUNKS, shared_edges.shape[0], 4), np.int64)
    bad = np.zeros(CHUNKS, np.int64)
    step = (n + CHUNKS - 1) // CHUNKS
    for c in prange(CHUNKS):
        for u in range(c * step, min(n, (c + 1) * step)):
            for e in range(offsets[u], offsets[u + 1]):
                v = neighbors[e]
                if v <= u:
                    continue
                k, tmin, tmax = _shared(a_off, a_nb, w_size, n, u, v)
                if k == 0:
                    bad[c] += 1
                    continue
                across = 1 if comm[u] != comm[v] else 0
                known = 1 if (cc[u] >= 0 and cc[v] >= 0) else 0
                across_cc = 1 if (known == 1 and cc[u] != cc[v]) else 0
                b = _bin(shared_edges, k)
                by_shared[c, b, 0] += 1
                by_shared[c, b, 1] += across
                by_shared[c, b, 2] += known
                by_shared[c, b, 3] += across_cc
                if k == 1:
                    t = _bin(team_edges, tmin)
                    by_team[c, t, 0] += 1
                    by_team[c, t, 1] += across
                    by_team[c, t, 2] += known
                    by_team[c, t, 3] += across_cc
                if tmax <= SMALL_TEAM:
                    small[c, b, 0] += 1
                    small[c, b, 1] += across
                    small[c, b, 2] += known
                    small[c, b, 3] += across_cc
    return by_shared.sum(axis=0), by_team.sum(axis=0), small.sum(axis=0), bad.sum()


@njit(parallel=True, cache=True)
def sorted_rows(offsets, neighbors):
    out = neighbors.copy()
    for u in prange(offsets.shape[0] - 1):
        out[offsets[u]:offsets[u + 1]].sort()
    return out


@njit(parallel=True, cache=True)
def sample_stats(offsets, sn, a_off, a_nb, w_size, us, vs):
    """For sampled links: shared works, smallest and largest team, and B2's neighbourhood overlap."""
    n = offsets.shape[0] - 1
    m = us.shape[0]
    k = np.zeros(m, np.int64)
    tmin = np.zeros(m, np.int64)
    tmax = np.zeros(m, np.int64)
    ov = np.zeros(m, np.float64)
    for q in prange(m):
        u, v = us[q], vs[q]
        kk, lo, hi = _shared(a_off, a_nb, w_size, n, u, v)
        k[q] = kk
        tmin[q] = lo
        tmax[q] = hi
        i, ie, j, je = offsets[u], offsets[u + 1], offsets[v], offsets[v + 1]
        c = 0
        while i < ie and j < je:
            x, y = sn[i], sn[j]
            if x == y:
                c += 1
                i += 1
                j += 1
            elif x < y:
                i += 1
            else:
                j += 1
        denom = (offsets[u + 1] - offsets[u] - 1) + (offsets[v + 1] - offsets[v] - 1) - c
        ov[q] = c / denom if denom > 0 else 0.0
    return k, tmin, tmax, ov


def table(counts, labels):
    rows = []
    for lab, (links, across, known, across_cc) in zip(labels, counts.tolist()):
        rows.append({"group": lab, "links": int(links), "across_communities": across / links if links else None,
                     "known_country": int(known), "across_countries": across_cc / known if known else None})
    return rows


# read-only memory maps: clean pages the system can drop under pressure instead of swapping
offsets, neighbors = load_csr(mmap=True)
n = len(offsets) - 1
a_off = np.load(CACHE / "bipartite_a_off.npy", mmap_mode="r")
a_nb = np.load(CACHE / "bipartite_a_nb.npy", mmap_mode="r")
w_size = np.load(CACHE / "bipartite_w_size.npy", mmap_mode="r")
assert len(a_off) == n + 1
attrs = pq.read_table(CACHE / "node_attrs.parquet", columns=["node_idx", "community", "country"]).to_pandas()
assert np.array_equal(attrs.node_idx.to_numpy(), np.arange(n))
comm = attrs.community.to_numpy().astype(np.int64)
cc = attrs.country.astype("category").cat.codes.to_numpy().astype(np.int32)
del attrs
log("shared works for every link")
by_shared, by_team, small, bad = link_tables(offsets, neighbors, a_off, a_nb, w_size, comm, cc, SHARED_EDGES, TEAM_EDGES)
links = int(by_shared[:, 0].sum())
log(f"{links:,} links; {int(bad):,} without a shared work of at most {MAX_TEAM} authors")
assert bad == 0, "links without a shared qualifying work: the bipartite graph and the network disagree"
assert links == len(neighbors) // 2
out = {"links": links, "max_team": MAX_TEAM, "small_team": SMALL_TEAM,
       "by_shared_works": table(by_shared, SHARED_LABELS),
       "one_off_by_team_size": table(by_team, TEAM_LABELS),
       "small_team_links_by_shared_works": table(small, SHARED_LABELS)}
log("overlap sample")
rng = np.random.default_rng(20261103)
entries = rng.integers(0, len(neighbors), SAMPLE)
us = (np.searchsorted(offsets, entries, side="right") - 1).astype(np.int64)
vs = neighbors[entries].astype(np.int64)
# B2 saved the CSR rows sorted (cache/ties/sorted_neighbors.npy); check it matches this CSR before reuse
sn = np.load(CACHE / "ties" / "sorted_neighbors.npy", mmap_mode="r")
assert sn.shape == neighbors.shape
for u in rng.choice(n, 2000, replace=False):
    assert np.array_equal(np.sort(neighbors[offsets[u]:offsets[u + 1]]), sn[offsets[u]:offsets[u + 1]])
del neighbors
k, tmin, tmax, ov = sample_stats(offsets, sn, a_off, a_nb, w_size, us, vs)
sb = np.searchsorted(SHARED_EDGES, k, side="right") - 1
tb = np.searchsorted(TEAM_EDGES, tmin, side="right") - 1
out["overlap_sample"] = SAMPLE
out["overlap_by_shared_works"] = {lab: float(ov[sb == i].mean()) for i, lab in enumerate(SHARED_LABELS) if (sb == i).any()}
out["overlap_one_off_by_team_size"] = {lab: float(ov[(k == 1) & (tb == i)].mean()) for i, lab in enumerate(TEAM_LABELS)
                                       if ((k == 1) & (tb == i)).any()}
small_mask = tmax <= SMALL_TEAM
out["overlap_small_team_by_shared_works"] = {lab: float(ov[small_mask & (sb == i)].mean())
                                             for i, lab in enumerate(SHARED_LABELS) if (small_mask & (sb == i)).any()}
# the sampled entries are independent uniform draws of links, so each group mean has standard error sd / sqrt(n)
for key, groups in (("by_shared_works", [(lab, sb == i) for i, lab in enumerate(SHARED_LABELS)]),
                    ("one_off_by_team_size", [(lab, (k == 1) & (tb == i)) for i, lab in enumerate(TEAM_LABELS)]),
                    ("small_team_by_shared_works", [(lab, small_mask & (sb == i)) for i, lab in enumerate(SHARED_LABELS)])):
    out[f"overlap_n_{key}"] = {lab: int(m.sum()) for lab, m in groups if m.any()}
    out[f"overlap_se_{key}"] = {lab: float(ov[m].std(ddof=1) / np.sqrt(m.sum())) for lab, m in groups if m.sum() > 1}
save_result("b2c_repeat_ties", out)
for key in ("by_shared_works", "one_off_by_team_size", "small_team_links_by_shared_works"):
    log(key, [(r["group"], r["links"], round(r["across_communities"] or 0, 3), round(r["across_countries"] or 0, 3)) for r in out[key]])
log("overlap", out["overlap_by_shared_works"], out["overlap_one_off_by_team_size"], out["overlap_small_team_by_shared_works"])
