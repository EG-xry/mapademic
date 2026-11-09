"""RV rich club: do hubs link to hubs only through large teams?

B3 finds that the 1,000 best connected unflagged researchers share 3.3 times the links a random network with the same
degrees would give. Co-signing the same works of up to 50 authors links prolific researchers to each other by
construction. Here, for every link, whether the two researchers share at least one link-making work with at most
SMALL authors:
  - among B3's clubs (top 100, 1,000 and 100,000 unflagged researchers by degree), the share of their inside links that
    rest only on works with more than SMALL authors;
  - the rich club again on the subgraph of links with a shared work of at most SMALL authors: clubs ranked by degree
    in that subgraph, their inside links against the same random-network expectation as B3 (Chung-Lu: the sum of
    k_i k_j / 2m over pairs of members) with that subgraph's degrees and links.
Writes results/rv_rich_club.json.
"""
import numpy as np
from numba import njit, prange

SMALL, MAX_TEAM, CHUNKS = 10, 50, 512
CLUBS = (100, 1000, 100_000)


@njit(cache=True)
def smallest_team(a_off, a_nb, w_size, n, u, v):
    """Smallest shared work of u and v among works with at most MAX_TEAM authors (1 << 30 if none)."""
    i, ie, j, je = a_off[u], a_off[u + 1], a_off[v], a_off[v + 1]
    best = 1 << 30
    if ie - i > 8 * (je - j) or je - j > 8 * (ie - i):
        if ie - i < je - j:
            i, ie, j, je = j, je, i, ie
        long_ = a_nb[i:ie]
        for t in range(j, je):
            x = a_nb[t]
            p = np.searchsorted(long_, x)
            if p < long_.shape[0] and long_[p] == x:
                s = w_size[x - n]
                if s <= MAX_TEAM and s < best:
                    best = s
        return best
    while i < ie and j < je:
        x, y = a_nb[i], a_nb[j]
        if x == y:
            s = w_size[x - n]
            if s <= MAX_TEAM and s < best:
                best = s
            i += 1
            j += 1
        elif x < y:
            i += 1
        else:
            j += 1
    return best


@njit(parallel=True, cache=True)
def small_flags(offsets, neighbors, a_off, a_nb, w_size, small):
    """flag[e] for every entry e = (u, v) with v > u: the pair shares a link-making work of at most `small` authors.
    Also counts entries with no link-making shared work (must be zero)."""
    n = offsets.shape[0] - 1
    flag = np.zeros(neighbors.shape[0], np.bool_)
    bad = np.zeros(CHUNKS, np.int64)
    step = (n + CHUNKS - 1) // CHUNKS
    for c in prange(CHUNKS):
        for u in range(c * step, min(n, (c + 1) * step)):
            for e in range(offsets[u], offsets[u + 1]):
                v = neighbors[e]
                if v <= u:
                    continue
                t = smallest_team(a_off, a_nb, w_size, n, u, v)
                if t > MAX_TEAM:
                    bad[c] += 1
                flag[e] = t <= small
    return flag, bad.sum()


@njit(cache=True)
def tally(offsets, neighbors, flag, rank, clubs):
    """Degrees and links of the small-team subgraph; for clubs ranked by `rank`, inside links and inside links
    resting only on larger works."""
    n = offsets.shape[0] - 1
    deg_small = np.zeros(n, np.int64)
    m_small = 0
    inside = np.zeros(clubs.shape[0], np.int64)
    inside_large_only = np.zeros(clubs.shape[0], np.int64)
    for u in range(n):
        for e in range(offsets[u], offsets[u + 1]):
            v = neighbors[e]
            if v <= u:
                continue
            if flag[e]:
                deg_small[u] += 1
                deg_small[v] += 1
                m_small += 1
            hi = max(rank[u], rank[v])
            for c in range(clubs.shape[0]):
                if hi < clubs[c]:
                    inside[c] += 1
                    if not flag[e]:
                        inside_large_only[c] += 1
    return deg_small, m_small, inside, inside_large_only


@njit(cache=True)
def inside_small(offsets, neighbors, flag, rank, clubs):
    """Inside links of the small-team subgraph for clubs ranked by `rank`."""
    n = offsets.shape[0] - 1
    inside = np.zeros(clubs.shape[0], np.int64)
    for u in range(n):
        for e in range(offsets[u], offsets[u + 1]):
            v = neighbors[e]
            if v <= u or not flag[e]:
                continue
            hi = max(rank[u], rank[v])
            for c in range(clubs.shape[0]):
                if hi < clubs[c]:
                    inside[c] += 1
    return inside


def club_rank(degree, flagged, top):
    """Rank among the `top` best connected unflagged researchers (B3's order: stable, flagged last); others top."""
    order = np.argsort(-np.where(flagged, -1, degree), kind="stable")
    rank = np.full(len(degree), top, np.int64)
    rank[order[:top]] = np.arange(min(len(degree), top))
    return rank, order


def chung_lu(k):
    """B3's random-network expectation for links among club members with degrees k: sum over pairs of k_i k_j / 2m,
    written ((sum k)^2 - sum k^2) / (2 * 2m); the caller divides by 2m."""
    k = np.asarray(k, np.float64)
    return (k.sum() ** 2 - (k ** 2).sum()) / 2


def main():
    import json

    from common import CACHE, RESULTS, flagged_mask, load_csr, log, save_result
    offsets, neighbors = load_csr(mmap=True)
    n = len(offsets) - 1
    degree = np.diff(np.asarray(offsets)).astype(np.int64)
    flagged = flagged_mask(n)
    a_off = np.load(CACHE / "bipartite_a_off.npy", mmap_mode="r")
    a_nb = np.load(CACHE / "bipartite_a_nb.npy", mmap_mode="r")
    w_size = np.load(CACHE / "bipartite_w_size.npy", mmap_mode="r")
    clubs = np.array(CLUBS, np.int64)
    log(f"smallest shared team of every link (works of at most {SMALL} authors)")
    flag, bad = small_flags(offsets, neighbors, a_off, a_nb, w_size, SMALL)
    assert bad == 0, "links without a link-making shared work"
    rank, order = club_rank(degree, flagged, int(clubs.max()))
    deg_small, m_small, inside, large_only = tally(offsets, neighbors, flag, rank, clubs)
    links = int(len(neighbors) // 2)
    b3 = json.loads((RESULTS / "b3_hubs.json").read_text())
    reported = {r["club"]: r for r in b3["rich_club"]}
    assert int(inside[list(CLUBS).index(1000)]) == reported[1000]["links_inside"], "club counts disagree with B3"
    by_degree = []
    for c, k in enumerate(CLUBS):
        by_degree.append({"club": k, "links_inside": int(inside[c]), "large_team_only": int(large_only[c]),
                          "share_large_team_only": float(large_only[c] / inside[c]),
                          "ratio_b3": reported.get(k, {}).get("ratio")})
    rank_small, order_small = club_rank(deg_small, flagged, int(clubs.max()))
    inside10 = inside_small(offsets, neighbors, flag, rank_small, clubs)
    small_clubs = []
    for c, k in enumerate(CLUBS):
        members = order_small[:k]
        expected = chung_lu(deg_small[members]) / (2 * m_small)
        small_clubs.append({"club": k, "min_degree": int(deg_small[members].min()), "links_inside": int(inside10[c]),
                            "random_expectation": float(expected), "ratio": float(inside10[c] / expected),
                            "overlap_with_degree_club": int(np.isin(members, order[:k]).sum())})
    out = {"small_team": SMALL, "links": links, "small_team_links": int(m_small),
           "small_team_link_share": float(m_small / links), "clubs_by_degree": by_degree,
           "clubs_by_small_team_degree": small_clubs}
    save_result("rv_rich_club", out)
    log(by_degree)
    log(small_clubs)


if __name__ == "__main__":
    main()
