"""RV bipartite null: how much clustering does projecting works into cliques give by itself?

A1b compares the network's transitivity (0.054) with a random network of the same degrees (about 0.0001). But the
network is the projection of a bipartite graph of researchers and works: every work with m retained researchers
becomes an m-clique. The null here is the bipartite configuration model (Newman, Strogatz and Watts 2001): every
researcher keeps their number of link-making works (at most 50 authors) and every such work its number of retained
researchers, but who works with whom is random. In the sparse limit:
  - triangles come from within-work cliques: T = sum over works of C(m, 3);
  - a researcher's coauthor slots are d = sum over their works of (m - 1). Slots can land on the same person, so the
    expected number of distinct coauthors is k(d) = sum_a c_a (1 - exp(-d a / A)), c_a the researchers with a
    link-making works and A the total of such authorships;
  - transitivity = 3 T / sum C(k, 2), plus the chance closure of a random network with those degrees (as in A1b);
  - a researcher's local clustering = sum over works of C(m - 1, 2) / C(k, 2).
The approximation is checked against the exact projection of stub-shuffled realisations of a 1% random sample of
link-making works.
Writes results/rv_bipartite_null.json.
"""
import numpy as np
from numba import njit, prange

MAX_TEAM = 50
BINS = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10**6]     # A1b's degree bands (Fig. 2c)


@njit(parallel=True, cache=True)
def author_sums(a_off, a_nb, n, m, link):
    """Per researcher: link-making works a, coauthor slots d and within-work closed pairs s3."""
    a = np.zeros(n, np.int64)
    d = np.zeros(n, np.int64)
    s3 = np.zeros(n, np.int64)
    for i in prange(n):
        for t in range(a_off[i], a_off[i + 1]):
            w = a_nb[t] - n
            if link[w]:
                mm = m[w]
                a[i] += 1
                d[i] += mm - 1
                s3[i] += (mm - 1) * (mm - 2) // 2
    return a, d, s3


def distinct_coauthors(d, a):
    """Expected distinct coauthors for coauthor-slot counts d, when each slot is filled by researcher j with
    probability a_j / A (the bipartite configuration model)."""
    d = np.asarray(d, np.float64)
    values, counts = np.unique(a[a > 0], return_counts=True)
    A = float((values * counts).sum())
    ud, inverse = np.unique(d, return_inverse=True)
    k = np.empty(len(ud))
    for lo in range(0, len(ud), 2048):
        block = ud[lo:lo + 2048, None]
        k[lo:lo + 2048] = (counts[None, :] * -np.expm1(-block * values[None, :] / A)).sum(axis=1)
    return k[inverse]


def chance_closure(k, n):
    """A1b's expectation for a random network with degrees k over n nodes: (<k^2> - <k>)^2 / (n <k>^3)."""
    k = np.asarray(k, np.float64)
    m1, m2 = k.sum() / n, (k ** 2).sum() / n
    return float((m2 - m1) ** 2 / (n * m1 ** 3))


def null_statistics(m, link, a_off, a_nb, n):
    """Analytic null: transitivity (with and without the distinct-coauthor correction) and per-researcher local
    clustering and degree."""
    mm = m[link].astype(np.float64)
    triangles = float((mm * (mm - 1) * (mm - 2) / 6).sum())
    a, d, s3 = author_sums(a_off, a_nb, n, m, link)
    k = distinct_coauthors(d, a)
    wedges_raw = float((d.astype(np.float64) * (d - 1) / 2).sum())
    wedges = float((k * (k - 1) / 2).sum())
    local = np.where(k >= 2, np.minimum(s3 / np.maximum(k * (k - 1) / 2, 1e-12), 1.0), np.nan)
    # The chance term is A1b's random-network formula on the null degrees; it is valid only while the largest degree
    # squared stays below twice the links (no structural cutoff), so the cutoff ratio is reported with it.
    chance = chance_closure(k, n)
    return {"triangles": triangles, "wedges_raw": wedges_raw, "wedges": wedges,
            "transitivity_within_raw": 3 * triangles / wedges_raw, "transitivity_within": 3 * triangles / wedges,
            "chance_closure": chance, "cutoff_ratio": float(k.max() ** 2 / k.sum()),
            "transitivity": 3 * triangles / wedges + chance}, local, k


@njit(cache=True)
def projected_transitivity(off, nb):
    """Exact transitivity of a simple graph in CSR form with sorted rows: links among neighbours over pairs."""
    n = off.shape[0] - 1
    mark = np.zeros(n, np.bool_)
    tri, pairs = 0, 0
    for u in range(n):
        k = off[u + 1] - off[u]
        if k < 2:
            continue
        for j in range(off[u], off[u + 1]):
            mark[nb[j]] = True
        t = 0
        for j in range(off[u], off[u + 1]):
            v = nb[j]
            for l in range(off[v], off[v + 1]):
                if mark[nb[l]]:
                    t += 1
        tri += t // 2
        pairs += k * (k - 1) // 2
        for j in range(off[u], off[u + 1]):
            mark[nb[j]] = False
    return tri / pairs


def project(work_of, author, n):
    """CSR (sorted, simple) of the projection of a small bipartite graph given as (work, author) incidences."""
    order = np.lexsort((author, work_of))
    work_of, author = work_of[order], author[order]
    starts = np.flatnonzero(np.r_[True, work_of[1:] != work_of[:-1], True])
    us, vs = [], []
    for s, e in zip(starts[:-1], starts[1:]):
        members = np.unique(author[s:e])
        if len(members) < 2:
            continue
        iu, iv = np.triu_indices(len(members), 1)
        us.append(members[iu])
        vs.append(members[iv])
    u, v = np.concatenate(us), np.concatenate(vs)
    key = np.unique(u.astype(np.int64) * n + v)
    u, v = key // n, key % n
    src, dst = np.r_[u, v], np.r_[v, u]
    order = np.lexsort((dst, src))
    src, dst = src[order], dst[order]
    off = np.zeros(n + 1, np.int64)
    np.add.at(off, src + 1, 1)
    return np.cumsum(off), dst.astype(np.int64)


def small_bipartite(work_of, author, n):
    """a_off/a_nb and m for a small (work, author) incidence list, works numbered 0..W-1 and stored as id + n."""
    works = int(work_of.max()) + 1
    pairs = np.unique(work_of.astype(np.int64) * n + author)          # one incidence per (work, author)
    work_of, author = pairs // n, pairs % n
    m = np.bincount(work_of, minlength=works).astype(np.int64)
    order = np.lexsort((work_of, author))
    a_nb = (work_of[order] + n).astype(np.int64)
    a_off = np.zeros(n + 1, np.int64)
    np.add.at(a_off, author + 1, 1)
    return np.cumsum(a_off), a_nb, m, work_of, author


def validate(w_off, w_auth, link, rng, share=0.01, shuffles=3):
    """Analytic null against exact projections of stub-shuffled samples of link-making works."""
    candidates = np.flatnonzero(link)
    chosen = np.sort(rng.choice(candidates, int(share * len(candidates)), replace=False))
    sizes = (w_off[chosen + 1] - w_off[chosen]).astype(np.int64)
    work_of = np.repeat(np.arange(len(chosen)), sizes)
    author_raw = np.concatenate([np.asarray(w_auth[w_off[w]:w_off[w + 1]]) for w in chosen]).astype(np.int64)
    ids, author = np.unique(author_raw, return_inverse=True)
    n = len(ids)
    off, nb = project(work_of, author, n)
    rows = {"works": int(len(chosen)), "researchers": int(n), "observed_transitivity": float(projected_transitivity(off, nb)),
            "shuffles": []}
    for _ in range(shuffles):
        shuffled = author[rng.permutation(len(author))]
        a_off, a_nb, m, wo, au = small_bipartite(work_of, shuffled, n)
        stats, _, _ = null_statistics(m, np.ones(len(m), bool), a_off, a_nb, n)
        off, nb = project(wo, au, n)
        rows["shuffles"].append({"simulated": float(projected_transitivity(off, nb)),
                                 "analytic": stats["transitivity"], "analytic_within": stats["transitivity_within"],
                                 "analytic_within_raw": stats["transitivity_within_raw"],
                                 "cutoff_ratio": stats["cutoff_ratio"]})
    return rows


def main():
    import json

    from common import CACHE, RESULTS, load_csr, log, save_result
    offsets, _ = load_csr(mmap=True)
    n = len(offsets) - 1
    a_off = np.load(CACHE / "bipartite_a_off.npy", mmap_mode="r")
    a_nb = np.load(CACHE / "bipartite_a_nb.npy", mmap_mode="r")
    w_off = np.load(CACHE / "bipartite_w_off.npy", mmap_mode="r")
    w_auth = np.load(CACHE / "bipartite_w_auth.npy", mmap_mode="r")
    w_size = np.load(CACHE / "bipartite_w_size.npy")
    m = np.diff(np.asarray(w_off)).astype(np.int64)
    link = w_size <= MAX_TEAM
    log(f"{int(link.sum()):,} link-making works of {len(link):,}")
    stats, local, k = null_statistics(m, link, a_off, a_nb, n)
    a1b = json.loads((RESULTS / "a1b_clustering.json").read_text())["all"]
    bands = np.digitize(np.rint(k), BINS)
    by_degree = [{"degree_from": BINS[i - 1], "nodes": int((bands == i).sum()), "median": float(np.nanmedian(local[bands == i]))}
                 for i in range(1, len(BINS)) if (bands == i).any() and np.isfinite(local[bands == i]).any()]
    log("validation on 1% samples of works")
    check = validate(w_off, w_auth, link, np.random.default_rng(20261109))
    out = {"link_making_works": int(link.sum()), "observed_transitivity": a1b["transitivity"],
           "configuration_expectation": a1b["random_expectation"], **stats,
           "observed_over_null": a1b["transitivity"] / stats["transitivity"],
           "null_by_degree": by_degree, "null_mean_local": float(np.nanmean(local)),
           "observed_mean_local": a1b["mean_local"], "validation": check}
    save_result("rv_bipartite_null", out)
    log({key: out[key] for key in ("observed_transitivity", "transitivity_within_raw", "transitivity_within",
                                   "chance_closure", "cutoff_ratio", "transitivity", "observed_over_null",
                                   "null_mean_local", "observed_mean_local")})
    log("validation", check)


if __name__ == "__main__":
    main()
