"""RV ties: is the overlap gradient of B2c built into the projection?

Every shared work with n authors makes its other n - 2 authors common neighbours of the pair, so neighbourhood
overlap can rise with the number of shared works by construction. On the same two million sampled links as B2c (same
seed, same draws), for each link:
  - the plain overlap (Onnela et al. 2007), as in B2c: c / ((k_u - 1) + (k_v - 1) - c), c the common neighbours;
  - the residual overlap: the same after removing, from the common neighbours and from the union, the s retained
    coauthors of the pair's shared works (works with at most 50 authors, the works that make links):
    (c - s) / ((k_u - 1) + (k_v - 1) - c - s);
  - whether every shared work has two authors, so that s = 0 and the overlap holds no built-in part.
Grouped by number of shared works (1, 2, 3, 4-5, 6-10, 11+), for all links, for links formed only in teams of at most
five authors, and for links formed only in two-author works, with group sizes and standard errors (the draws are
independent uniform draws of links, with replacement).
Writes results/rv_ties.json.
"""
import numpy as np
from numba import njit, prange

SHARED_EDGES = np.array([1, 2, 3, 4, 6, 11], np.int64)       # 1, 2, 3, 4-5, 6-10, 11+ (as B2c)
SHARED_LABELS = ["1", "2", "3", "4-5", "6-10", "11+"]
SMALL_TEAM, MAX_TEAM, SAMPLE, SEED = 5, 50, 2_000_000, 20261103


@njit(cache=True)
def shared_works(a_off, a_nb, w_size, n, u, v, out):
    """Shared works of u and v with at most MAX_TEAM authors, written into out as work indices; returns their
    number. Rows of a_nb are sorted work node ids (work index + n)."""
    i, ie, j, je = a_off[u], a_off[u + 1], a_off[v], a_off[v + 1]
    count = 0
    if ie - i > 8 * (je - j) or je - j > 8 * (ie - i):            # galloping, as B2c
        if ie - i < je - j:
            i, ie, j, je = j, je, i, ie
        long_ = a_nb[i:ie]
        for t in range(j, je):
            x = a_nb[t]
            p = np.searchsorted(long_, x)
            if p < long_.shape[0] and long_[p] == x and w_size[x - n] <= MAX_TEAM:
                out[count] = x - n
                count += 1
        return count
    while i < ie and j < je:
        x, y = a_nb[i], a_nb[j]
        if x == y:
            if w_size[x - n] <= MAX_TEAM:
                out[count] = x - n
                count += 1
            i += 1
            j += 1
        elif x < y:
            i += 1
        else:
            j += 1
    return count


@njit(cache=True)
def built_in_common(w_off, w_auth, works, k, u, v):
    """s: the distinct retained authors of the k shared works, other than u and v."""
    total = 0
    for t in range(k):
        w = works[t]
        total += w_off[w + 1] - w_off[w]
    if total == 0:
        return 0
    buf = np.empty(total, np.int64)
    p = 0
    for t in range(k):
        w = works[t]
        for q in range(w_off[w], w_off[w + 1]):
            buf[p] = w_auth[q]
            p += 1
    buf.sort()
    s, last = 0, -1
    for t in range(total):
        a = buf[t]
        if a != last:
            if a != u and a != v:
                s += 1
            last = a
    return s


@njit(parallel=True, cache=True)
def link_stats(offsets, sn, a_off, a_nb, w_off, w_auth, w_size, us, vs):
    """For each link (us[q], vs[q]): shared works, smallest and largest team, common neighbours c, the union of
    other neighbours (k_u - 1) + (k_v - 1) - c, and the built-in common neighbours s."""
    n = offsets.shape[0] - 1
    m = us.shape[0]
    k = np.zeros(m, np.int64)
    tmin = np.zeros(m, np.int64)
    tmax = np.zeros(m, np.int64)
    c = np.zeros(m, np.int64)
    union = np.zeros(m, np.int64)
    s = np.zeros(m, np.int64)
    for q in prange(m):
        u, v = us[q], vs[q]
        cap = min(a_off[u + 1] - a_off[u], a_off[v + 1] - a_off[v])
        works = np.empty(max(cap, 1), np.int64)
        kk = shared_works(a_off, a_nb, w_size, n, u, v, works)
        lo, hi = 1 << 30, 0
        for t in range(kk):
            size = w_size[works[t]]
            lo = min(lo, size)
            hi = max(hi, size)
        k[q], tmin[q], tmax[q] = kk, lo, hi
        i, ie, j, je = offsets[u], offsets[u + 1], offsets[v], offsets[v + 1]
        cc = 0
        while i < ie and j < je:
            x, y = sn[i], sn[j]
            if x == y:
                cc += 1
                i += 1
                j += 1
            elif x < y:
                i += 1
            else:
                j += 1
        c[q] = cc
        union[q] = (offsets[u + 1] - offsets[u] - 1) + (offsets[v + 1] - offsets[v] - 1) - cc
        s[q] = built_in_common(w_off, w_auth, works, kk, u, v)
    return k, tmin, tmax, c, union, s


def overlaps(c, union, s):
    """Plain and residual overlap; 0 where the union is empty."""
    plain = np.divide(c, union, out=np.zeros(len(c)), where=union > 0)
    rest = union - s
    residual = np.divide(c - s, rest, out=np.zeros(len(c)), where=rest > 0)
    return plain, residual


def grouped(values, k, mask):
    """Mean, size and standard error of values by shared-works group, among links in mask."""
    b = np.searchsorted(SHARED_EDGES, k, side="right") - 1
    out = {}
    for i, lab in enumerate(SHARED_LABELS):
        m = mask & (b == i)
        if m.sum() > 1:
            x = values[m]
            out[lab] = {"mean": float(x.mean()), "n": int(m.sum()), "se": float(x.std(ddof=1) / np.sqrt(m.sum()))}
    return out


def sample(offsets, neighbors):
    """B2c's two million link draws, reproduced from its seed."""
    rng = np.random.default_rng(SEED)
    entries = rng.integers(0, len(neighbors), SAMPLE)
    us = (np.searchsorted(offsets, entries, side="right") - 1).astype(np.int64)
    vs = np.asarray(neighbors[entries]).astype(np.int64)
    return us, vs


def main():
    import json

    from common import CACHE, RESULTS, load_csr, log, save_result
    offsets, neighbors = load_csr(mmap=True)
    n = len(offsets) - 1
    a_off = np.load(CACHE / "bipartite_a_off.npy", mmap_mode="r")
    a_nb = np.load(CACHE / "bipartite_a_nb.npy", mmap_mode="r")
    w_off = np.load(CACHE / "bipartite_w_off.npy", mmap_mode="r")
    w_auth = np.load(CACHE / "bipartite_w_auth.npy", mmap_mode="r")
    w_size = np.load(CACHE / "bipartite_w_size.npy", mmap_mode="r")
    assert len(a_off) == n + 1 and len(w_off) == len(w_size) + 1
    us, vs = sample(offsets, neighbors)
    sn = np.load(CACHE / "ties" / "sorted_neighbors.npy", mmap_mode="r")
    assert sn.shape == neighbors.shape
    del neighbors
    log("statistics for the sampled links")
    k, tmin, tmax, c, union, s = link_stats(offsets, sn, a_off, a_nb, w_off, w_auth, w_size, us, vs)
    assert (k > 0).all(), "a sampled link without a shared work of at most 50 authors"
    assert (s <= c).all(), "built-in common neighbours must be common neighbours"
    two = tmax == 2
    assert (s[two] == 0).all(), "two-author works cannot contribute common neighbours"
    plain, residual = overlaps(c, union, s)
    b2c = json.loads((RESULTS / "b2c_repeat_ties.json").read_text())
    every = np.ones(len(k), bool)
    small = tmax <= SMALL_TEAM
    out = {"sample": SAMPLE, "seed": SEED, "max_team": MAX_TEAM, "small_team": SMALL_TEAM,
           "overlap_by_shared_works": grouped(plain, k, every),
           "overlap_small_team_by_shared_works": grouped(plain, k, small),
           "overlap_two_author_by_shared_works": grouped(plain, k, two),
           "residual_overlap_by_shared_works": grouped(residual, k, every),
           "residual_overlap_small_team_by_shared_works": grouped(residual, k, small),
           "built_in_share_of_common": {lab: float(s[m].sum() / max(c[m].sum(), 1)) for lab, m in
                                        ((lab, np.searchsorted(SHARED_EDGES, k, side="right") - 1 == i)
                                         for i, lab in enumerate(SHARED_LABELS))},
           "two_author_links": int(two.sum())}
    for lab, g in out["overlap_by_shared_works"].items():   # same draws and formula as B2c
        assert abs(g["mean"] - b2c["overlap_by_shared_works"][lab]) < 1e-9, (lab, g["mean"])
    save_result("rv_ties", out)
    for key in ("overlap_by_shared_works", "overlap_two_author_by_shared_works", "residual_overlap_by_shared_works",
                "residual_overlap_small_team_by_shared_works"):
        log(key, {lab: (round(g["mean"], 4), g["n"]) for lab, g in out[key].items()})


if __name__ == "__main__":
    main()
