"""Leiden passes: on the benchmark's hardest kind of network, does our implementation close its gap to igraph with more
passes? (leiden_benchmark.py runs ours for a fixed ten passes, as on the real network, and igraph to convergence.)

A fresh stochastic block model of that kind, 100,000 nodes at mixing 0.4 with lognormal link weights (the generator of
leiden_benchmark.py, copied here so that importing it does not rerun the benchmark): our implementation's modularity
after each of 40 passes against igraph's best of three runs to convergence, both scored with igraph's modularity().
Writes results/leiden_passes.json.
"""
import random

import igraph as ig
import numpy as np

from common import log, save_result
from leiden import _fill_csr, leiden

N, MIXING, MEAN_DEGREE, PASSES = 100_000, 0.4, 15.0, 40


def block_sizes(n, rng):
    lo, hi = 20, max(40, n // 50)
    sizes = []
    while sum(sizes) < n:
        s = int((lo ** -0.5 - rng.random() * (lo ** -0.5 - hi ** -0.5)) ** -2)
        sizes.append(min(s, n - sum(sizes)))
    if sizes[-1] < lo:
        rest = sizes.pop()
        sizes[-1] += rest
    return sizes


rng = np.random.default_rng(20261109)
random.seed(20261109)
sizes = block_sizes(N, rng)
pref = [[MIXING * MEAN_DEGREE / N] * len(sizes) for _ in sizes]
for a, s in enumerate(sizes):
    pref[a][a] = min(1.0, (1 - MIXING) * MEAN_DEGREE / max(s - 1, 1))
g = ig.Graph.SBM(pref, sizes, directed=False, allowed_edge_types="simple")
src, dst = np.array(g.get_edgelist(), np.int32).T
wt = rng.lognormal(0.0, 1.0, g.ecount()).astype(np.float32)
weights = wt.astype(float).tolist()
deg = np.bincount(src, minlength=N) + np.bincount(dst, minlength=N)
offsets = np.zeros(N + 1, np.int64)
np.cumsum(deg, out=offsets[1:])
nbr, w = np.empty(offsets[-1], np.int32), np.empty(offsets[-1], np.float32)
_fill_csr(offsets, offsets[:-1].copy(), nbr, w, src, dst, wt)
by_pass = {}
leiden(offsets, nbr, w, gamma=1.0, beta=0.01, seed=20261029, passes=PASSES, log=lambda *a: None,
       checkpoint=lambda m, q, p: by_pass.__setitem__(p, g.modularity(m.tolist(), weights=weights, resolution=1.0)))
igraph_runs = []
for s in (1, 2, 3):
    random.seed(s)
    p = g.community_leiden(objective_function="modularity", weights=weights, resolution=1.0, beta=0.01, n_iterations=-1)
    igraph_runs.append(g.modularity(p.membership, weights=weights, resolution=1.0))
best = max(igraph_runs)
out = {"nodes": N, "links": g.ecount(), "mixing": MIXING, "weights": "lognormal(0, 1)", "igraph_runs": igraph_runs,
       "igraph_best": best, "ours_by_pass": {str(k): v for k, v in sorted(by_pass.items())},
       "relative_difference_by_pass": {str(k): (v - best) / best for k, v in sorted(by_pass.items())}}
save_result("leiden_passes", out)
log({k: round(out["relative_difference_by_pass"][str(k)], 5) for k in (1, 5, 10, 20, 30, 40)}, "igraph", igraph_runs)
