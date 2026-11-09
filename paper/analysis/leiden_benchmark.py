"""Leiden benchmark: leiden.py against igraph's community_leiden on networks with planted communities.

Stochastic block models (igraph Graph.SBM) with planted communities of power-law sizes (exponent 1.5), mean degree
about 15, at 10,000, 100,000 and 1,000,000 nodes and two mixing levels (the expected share of a node's links that
leave its community: 0.2 and 0.4), each unweighted and with lognormal link weights unrelated to the communities.
Our implementation runs as on the real network: ten passes from seed 20261029, resolution 1, beta 0.01. igraph runs
community_leiden with the same objective, resolution and beta until the partition stops changing
(n_iterations=-1), from three seeds; the best modularity is kept. Both partitions are scored with igraph's
modularity(). Also reported: NMI with the planted partition and between the two results, the number of
communities, and how many returned communities are not connected. The networks are reproducible: block sizes, link
weights and the seed of igraph's generator (Python's random module) before each network all come from one NumPy
generator seeded 20261108. Writes results/leiden_benchmark.json.
"""
import random
import time

import igraph as ig
import numpy as np

from common import log, save_result
from leiden import _fill_csr, leiden

SIZES = (10_000, 100_000, 1_000_000)
MIXING = (0.2, 0.4)
MEAN_DEGREE = 15.0
IGRAPH_SEEDS = (1, 2, 3)
OUR_SEED = 20261029


def block_sizes(n, rng):
    """Power-law community sizes (exponent 1.5) between 20 and n/50, summing to n."""
    lo, hi = 20, max(40, n // 50)
    sizes = []
    while sum(sizes) < n:
        u = rng.random()
        s = int((lo ** -0.5 - u * (lo ** -0.5 - hi ** -0.5)) ** -2)        # inverse CDF of p(s) ~ s^-1.5
        sizes.append(min(s, n - sum(sizes)))
    if sizes[-1] < lo:                                                     # fold a small remainder into the last block
        rest = sizes.pop()
        sizes[-1] += rest
    assert sum(sizes) == n
    return sizes


def planted_graph(n, mixing, rng):
    sizes = block_sizes(n, rng)
    p_out = mixing * MEAN_DEGREE / n
    pref = [[p_out] * len(sizes) for _ in sizes]
    for a, s in enumerate(sizes):
        pref[a][a] = min(1.0, (1 - mixing) * MEAN_DEGREE / max(s - 1, 1))
    random.seed(int(rng.integers(2 ** 31)))             # igraph draws from Python's random module: seed it per network
    g = ig.Graph.SBM(pref, sizes, directed=False, allowed_edge_types="simple")
    planted = np.repeat(np.arange(len(sizes)), sizes)
    return g, planted


def csr(n, src, dst, wt):
    deg = np.bincount(src, minlength=n) + np.bincount(dst, minlength=n)
    offsets = np.zeros(n + 1, np.int64)
    np.cumsum(deg, out=offsets[1:])
    nbr = np.empty(offsets[-1], np.int32)
    w = np.empty(offsets[-1], np.float32)
    _fill_csr(offsets, offsets[:-1].copy(), nbr, w, src, dst, wt)
    return offsets, nbr, w


def disconnected(g, memb):
    """Number of communities whose members are not connected by links inside the community."""
    src, dst = np.array(g.get_edgelist()).T
    inside = np.flatnonzero(memb[src] == memb[dst])
    h = ig.Graph(n=g.vcount(), edges=list(zip(src[inside].tolist(), dst[inside].tolist())))
    comp = np.array(h.connected_components().membership)
    pairs = np.unique(np.stack([memb, comp]), axis=1)
    return int((np.bincount(pairs[0]) > 1).sum())


rng = np.random.default_rng(20261108)
cases = []
for n in SIZES:
    for mixing in MIXING:
        g, planted = planted_graph(n, mixing, rng)
        src, dst = (np.array(g.get_edgelist(), np.int32).T if g.ecount() else (np.empty(0, np.int32),) * 2)
        for weighted in (False, True):
            wt = (rng.lognormal(0.0, 1.0, g.ecount()) if weighted else np.ones(g.ecount())).astype(np.float32)
            weights = wt.astype(float).tolist() if weighted else None
            t0 = time.time()
            ours, q_ours_internal = leiden(*csr(n, src, dst, wt), gamma=1.0, beta=0.01, seed=OUR_SEED, passes=10,
                                           log=lambda *a: None)
            t_ours = time.time() - t0
            runs = []
            for s in IGRAPH_SEEDS:
                random.seed(s)
                t0 = time.time()
                p = g.community_leiden(objective_function="modularity", weights=weights, resolution=1.0, beta=0.01,
                                       n_iterations=-1)
                runs.append((g.modularity(p.membership, weights=weights, resolution=1.0), s, np.array(p.membership),
                             time.time() - t0))
            q_best, seed_best, best, _ = max(runs, key=lambda r: r[0])
            q_ours = g.modularity(ours.tolist(), weights=weights, resolution=1.0)
            case = {"nodes": n, "links": g.ecount(), "planted_communities": int(planted.max() + 1), "mixing": mixing,
                    "weights": "lognormal(0, 1)" if weighted else "none",
                    "ours_modularity": q_ours, "ours_modularity_internal": float(q_ours_internal),
                    "igraph_modularity_by_seed": {str(s): q for q, s, _, _ in runs},
                    "igraph_best_seed": seed_best, "igraph_best_modularity": q_best,
                    "relative_difference": (q_ours - q_best) / q_best,
                    "ours_communities": int(ours.max() + 1), "igraph_communities": int(best.max() + 1),
                    "nmi_ours_planted": ig.compare_communities(ours.tolist(), planted.tolist(), method="nmi"),
                    "nmi_igraph_planted": ig.compare_communities(best.tolist(), planted.tolist(), method="nmi"),
                    "nmi_ours_igraph": ig.compare_communities(ours.tolist(), best.tolist(), method="nmi"),
                    "ours_disconnected_communities": disconnected(g, ours),
                    "igraph_disconnected_communities": disconnected(g, best),
                    "seconds_ours": round(t_ours, 1), "seconds_igraph": [round(r[3], 1) for r in runs]}
            cases.append(case)
            log({k: (round(v, 5) if isinstance(v, float) else v) for k, v in case.items()
                 if k not in ("igraph_modularity_by_seed",)})
save_result("leiden_benchmark", {
    "igraph_version": ig.__version__, "our_settings": {"resolution": 1.0, "beta": 0.01, "passes": 10, "seed": OUR_SEED},
    "igraph_settings": {"objective_function": "modularity", "resolution": 1.0, "beta": 0.01, "n_iterations": -1,
                        "seeds": list(IGRAPH_SEEDS), "kept": "best modularity of the three"},
    "generator": {"model": "igraph Graph.SBM, no loops", "community_sizes": "power law, exponent 1.5, 20 to n/50",
                  "mean_degree": MEAN_DEGREE, "mixing": list(MIXING), "seed": 20261108},
    "max_abs_relative_difference": max(abs(c["relative_difference"]) for c in cases),
    "cases": cases})
