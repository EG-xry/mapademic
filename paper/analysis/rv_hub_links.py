"""RV hub links: is removing the best connected researchers worse than removing as many links at random?

A9 removes the top 1% and 5% of researchers by degree, which touch 29.6% and 60.9% of all distinct links (A10), and
compares them with removing as many researchers at random, which touches far fewer links. The link-matched baseline
here removes the same shares of distinct links at random (each link kept or dropped as a unit, both directions
together, by a seeded hash of the pair), keeps every researcher, and measures distances from A9's sources (the first
100 of A2's 1,000) as A9 does, with two independent draws per share. The comparison follows Albert, Jeong and
Barabasi (2000), who contrast targeted with random removal.
Writes results/rv_hub_links.json.
"""
import numpy as np
from numba import njit, prange

SEEDS = (20261109, 20261110)


@njit(cache=True)
def uniform_of_pair(lo, hi, seed):
    """A uniform number in [0, 1) from the unordered pair (lo < hi) and a seed (splitmix64)."""
    x = (np.uint64(lo) << np.uint64(32)) | np.uint64(hi)
    x ^= seed
    x += np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(parallel=True, cache=True)
def symmetric_mask(offsets, neighbors, share, seed):
    """skip[e] for every CSR entry: the link of e is dropped, with probability `share`, the same in both directions."""
    n = offsets.shape[0] - 1
    skip = np.zeros(neighbors.shape[0], np.bool_)
    s = np.uint64(seed)
    for u in prange(n):
        for e in range(offsets[u], offsets[u + 1]):
            v = neighbors[e]
            skip[e] = uniform_of_pair(min(u, v), max(u, v), s) < share
    return skip


def main():
    import gc
    import json

    from common import CACHE, RESULTS, load_csr, log, save_result
    from kernels import _bfs, contract, histograms, summary
    offsets, neighbors = load_csr(mmap=True)
    n = len(offsets) - 1
    touching = json.loads((RESULTS / "a10_checks.json").read_text())["touching"]
    shares = {"top_1pct": touching["top_0.01"]["share_of_links"], "top_5pct": touching["top_0.05"]["share_of_links"]}
    giant = int(np.isfinite(np.load(CACHE / "closeness_1000.npy", mmap_mode="r")).sum())
    sources = np.load(CACHE / "a2_sources.npy")[:100]
    saved = np.load(CACHE / "a2c_hist_cap50.npz")
    assert (saved["sources"][:100] == sources).all(), "A2c's saved counts are for other sources"
    before = saved["hist"][:100].astype(np.float64)
    before[:, 0] = 0
    baseline = float((before * np.arange(before.shape[1])).sum() / before.sum())
    a9 = json.loads((RESULTS / "a9_structure.json").read_text())["hub_removal"]
    rep = np.arange(n, dtype=np.int64)
    rows = []
    for label, share in shares.items():
        for seed in SEEDS:
            skip = symmetric_mask(offsets, neighbors, share, seed)
            realized = float(skip.sum() / len(skip))
            off2, nb2 = contract(offsets, neighbors, skip, rep, n)
            del skip
            h = histograms(off2, nb2, sources, 64)
            res = summary(h)
            reach = np.empty(n, np.int16)
            reached = _bfs(off2, nb2, int(sources[0]), reach, np.empty(n, np.int32))
            row = {"matches": label, "share_target": share, "share_removed": realized, "seed": seed,
                   "mean": res["mean"], "median": res["median"], "p90": res["p90"],
                   "giant_share": float(reached / n),
                   "targets_reached_share": float(h[:, 1:].sum() / (len(sources) * (giant - 1)))}
            rows.append(row)
            log(row)
            del off2, nb2, h, reach
            gc.collect()
    out = {"sources": int(len(sources)), "baseline_mean": baseline, "rows": rows,
           "node_removal": [r for r in a9 if r["share"] in (0.01, 0.05)]}
    save_result("rv_hub_links", out)


if __name__ == "__main__":
    main()
