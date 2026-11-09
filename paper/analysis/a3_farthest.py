"""A3: the farthest-apart researchers -- exact diameter of the largest component.

Lower bound from repeated double sweeps (BFS from a random node, then from the
farthest node found, ...), then iFUB (Crescenzi et al. 2013) from the middle of
the longest path found: eccentricities of the fringe levels, parallel, until the
bounds meet. Done for the full network and with flagged links removed. Also
prints the farthest pair and the chain between them.
"""
import sys

import numpy as np
from numba import njit, prange

from common import CACHE, Names, load_csr, log, save_result, suspect_mask

SWEEPS = 8
MAX_FRINGE_BFS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000


@njit(cache=True)
def bfs(offsets, neighbors, skip, use_skip, src, dist, queue):
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
    return dist[queue[tail - 1]], queue[tail - 1]


@njit(parallel=True, cache=True)
def eccentricities(offsets, neighbors, skip, use_skip, nodes):
    n = offsets.shape[0] - 1
    out = np.zeros(nodes.shape[0], np.int64)
    far = np.zeros(nodes.shape[0], np.int64)
    for k in prange(nodes.shape[0]):
        d = np.empty(n, np.int16)
        q = np.empty(n, np.int32)
        e, f = bfs(offsets, neighbors, skip, use_skip, nodes[k], d, q)
        out[k] = e
        far[k] = f
    return out, far


@njit(cache=True)
def path_back(offsets, neighbors, skip, use_skip, dist, target):
    """A shortest path from the BFS source to target, walking down the distance labels."""
    L = dist[target]
    path = np.empty(L + 1, np.int64)
    path[L] = target
    cur = target
    for step in range(L, 0, -1):
        for j in range(offsets[cur], offsets[cur + 1]):
            if use_skip and skip[j]:
                continue
            v = neighbors[j]
            if dist[v] == step - 1:
                cur = v
                break
        path[step - 1] = cur
    return path


def diameter(offsets, neighbors, skip, use_skip, lcc, names, rng, tag):
    n = len(offsets) - 1
    dist = np.empty(n, np.int16)
    q = np.empty(n, np.int32)
    best = (-1, None, None)
    for s in range(SWEEPS):
        r = int(rng.choice(np.flatnonzero(lcc)))
        e1, a = bfs(offsets, neighbors, skip, use_skip, r, dist, q)
        e2, b = bfs(offsets, neighbors, skip, use_skip, a, dist, q)
        if e2 > best[0]:
            best = (int(e2), int(a), int(b))
        log(f"[{tag}] sweep {s}: ecc(random)={e1}, ecc(far)={e2}")
    lb, a, b = best
    bfs(offsets, neighbors, skip, use_skip, a, dist, q)
    p = path_back(offsets, neighbors, skip, use_skip, dist, b)
    mid = int(p[len(p) // 2])
    e_mid, _ = bfs(offsets, neighbors, skip, use_skip, mid, dist, q)
    ub = 2 * int(e_mid)
    levels = dist.copy()
    log(f"[{tag}] lower bound {lb}, ecc(middle)={e_mid}, upper bound {ub}")
    i, done_bfs = int(e_mid), 0
    pair = (a, b)
    while ub > lb and i > 0 and done_bfs < MAX_FRINGE_BFS:
        fringe = np.flatnonzero(levels == i).astype(np.int64)
        if done_bfs + len(fringe) > MAX_FRINGE_BFS:
            log(f"[{tag}] fringe level {i} has {len(fringe):,} nodes: over the cap, stopping")
            break
        ecc, far = eccentricities(offsets, neighbors, skip, use_skip, fringe)
        done_bfs += len(fringe)
        k = int(np.argmax(ecc))
        if ecc[k] > lb:
            lb, pair = int(ecc[k]), (int(fringe[k]), int(far[k]))
        log(f"[{tag}] level {i}: {len(fringe)} BFS, max ecc {int(ecc.max())}; lb {lb}")
        if lb > 2 * (i - 1):
            ub = lb
            break
        ub = 2 * (i - 1)
        i -= 1
    a, b = pair
    bfs(offsets, neighbors, skip, use_skip, a, dist, q)
    chain = path_back(offsets, neighbors, skip, use_skip, dist, b)
    return {"diameter_lower": lb, "diameter_upper": ub, "exact": lb == ub, "fringe_bfs": done_bfs,
            "pair": [names[a], names[b]], "pair_idx": [a, b],
            "chain": [names[int(x)] for x in chain], "chain_idx": [int(x) for x in chain]}


offsets, neighbors = load_csr(mmap=False)
skip = np.asarray(suspect_mask(offsets, neighbors))
names = Names()
rng = np.random.default_rng(7)
out = {}
for tag, use_skip, lcc_file in (("all_links", False, "lcc_all.npy"), ("flagged_removed", True, "lcc_clean.npy")):
    out[tag] = diameter(offsets, neighbors, skip, use_skip, np.load(CACHE / lcc_file), names, rng, tag)
    log(tag, {k: v for k, v in out[tag].items() if k != "chain_idx"})
save_result("a3_farthest", out)
