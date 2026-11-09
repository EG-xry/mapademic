"""Leiden communities (Traag, Waltman & van Eck 2019) on the weighted coauthor network, on the CPU.

The communities of the 15-work network came from cuGraph's Leiden, run with the map's layout on a GPU. This is the
same algorithm for networks the GPU did not see, following igraph's community_leiden: modularity at resolution
gamma, fast local moving, a refinement that merges well-connected nodes inside each community at random (with
randomness beta), aggregation on the refined partition, and whole passes repeated from the last partition: ten of
them (igraph repeats until the partition stops changing; here the largest communities keep trading members while
modularity creeps up by under 0.001 a pass). Run as a script, it reads graph/edges_int32.parquet of the network in use and writes
cache/leiden.npy (community of each node index, renumbered by size, largest first) and results/leiden.json.
"""
import sys

import numpy as np
import pyarrow.parquet as pq
from numba import njit


@njit(cache=True)
def _fill_csr(offsets, pos, nbr, w, src, dst, wt):
    for t in range(src.shape[0]):
        u, v = src[t], dst[t]
        nbr[pos[u]] = v
        w[pos[u]] = wt[t]
        pos[u] += 1
        nbr[pos[v]] = u
        w[pos[v]] = wt[t]
        pos[v] += 1


def weighted_csr(edges_path, n):
    """Both directions of every link, with weights, from a (src, dst, weight) parquet of node indices."""
    f = pq.ParquetFile(edges_path)
    deg = np.zeros(n, np.int64)
    for b in f.iter_batches(batch_size=20_000_000, columns=["src", "dst"]):
        deg += np.bincount(b.column(0).to_numpy(), minlength=n)
        deg += np.bincount(b.column(1).to_numpy(), minlength=n)
    offsets = np.zeros(n + 1, np.int64)
    np.cumsum(deg, out=offsets[1:])
    nbr = np.empty(offsets[-1], np.int32)
    w = np.empty(offsets[-1], np.float32)
    pos = offsets[:-1].copy()
    for b in f.iter_batches(batch_size=20_000_000, columns=["src", "dst", "weight"]):
        _fill_csr(offsets, pos, nbr, w, b.column(0).to_numpy().astype(np.int32), b.column(1).to_numpy().astype(np.int32),
                  b.column(2).to_numpy().astype(np.float32))
    return offsets, nbr, w


@njit(cache=True)
def node_strength(offsets, w):
    n = offsets.shape[0] - 1
    k = np.zeros(n, np.float64)
    for u in range(n):
        s = 0.0
        for j in range(offsets[u], offsets[u + 1]):
            s += w[j]
        k[u] = s
    return k


@njit(cache=True)
def modularity(offsets, nbr, w, comm, gamma):
    n = offsets.shape[0] - 1
    inside = np.zeros(n, np.float64)
    tot = np.zeros(n, np.float64)
    two_m = 0.0
    for u in range(n):
        cu = comm[u]
        for j in range(offsets[u], offsets[u + 1]):
            two_m += w[j]
            tot[cu] += w[j]
            if comm[nbr[j]] == cu:
                inside[cu] += w[j]
    q = 0.0
    for c in range(n):
        q += inside[c] / two_m - gamma * (tot[c] / two_m) ** 2
    return q


@njit(cache=True)
def fast_move(offsets, nbr, w, k, comm, gamma, two_m, seed):
    """Queue-based local moving: each node goes to the neighbouring (or an empty) community of largest gain.
    comm holds ids below n and is changed in place; returns the number of moves."""
    np.random.seed(seed)
    n = offsets.shape[0] - 1
    K = np.zeros(n, np.float64)
    size = np.zeros(n, np.int64)
    for u in range(n):
        K[comm[u]] += k[u]
        size[comm[u]] += 1
    empty = np.empty(n, np.int64)
    ne = 0
    for c in range(n):
        if size[c] == 0:
            empty[ne] = c
            ne += 1
    acc = np.zeros(n, np.float64)
    touched = np.empty(n, np.int64)
    queue = np.random.permutation(n).astype(np.int64)
    inq = np.ones(n, np.bool_)
    head, count = 0, n
    moves = 0
    res = gamma / two_m
    while count > 0:
        v = queue[head]
        head = (head + 1) % n
        count -= 1
        inq[v] = False
        cv = comm[v]
        nt = 0
        for j in range(offsets[v], offsets[v + 1]):
            u = nbr[j]
            if u == v:
                continue
            c = comm[u]
            if acc[c] == 0.0:
                touched[nt] = c
                nt += 1
            acc[c] += w[j]
        K[cv] -= k[v]
        size[cv] -= 1
        best, best_gain = cv, acc[cv] - res * k[v] * K[cv]
        for t in range(nt):
            c = touched[t]
            g = acc[c] - res * k[v] * K[c]
            if g > best_gain:
                best, best_gain = c, g
        if best_gain < 0.0 and size[cv] > 0:   # alone in a new community is better
            ne -= 1
            best = empty[ne]
        for t in range(nt):
            acc[touched[t]] = 0.0
        acc[cv] = 0.0
        if size[cv] == 0 and best != cv:
            empty[ne] = cv
            ne += 1
        K[best] += k[v]
        size[best] += 1
        if best != cv:
            comm[v] = best
            moves += 1
            for j in range(offsets[v], offsets[v + 1]):
                u = nbr[j]
                if not inq[u] and comm[u] != best:
                    queue[(head + count) % n] = u
                    count += 1
                    inq[u] = True
    return moves


@njit(cache=True)
def refine(offsets, nbr, w, k, comm, gamma, two_m, beta, seed):
    """Leiden refinement: inside each community, singletons join well-connected subcommunities at random,
    with probability growing as exp(gain / beta) among non-negative gains. Returns subcommunity ids (< n)."""
    np.random.seed(seed)
    n = offsets.shape[0] - 1
    res = gamma / two_m
    KC = np.zeros(n, np.float64)
    for u in range(n):
        KC[comm[u]] += k[u]
    ext = np.zeros(n, np.float64)            # weight from the node to the rest of its community
    for u in range(n):
        for j in range(offsets[u], offsets[u + 1]):
            v = nbr[j]
            if v != u and comm[v] == comm[u]:
                ext[u] += w[j]
    ref = np.arange(n)
    KS = k.copy()
    extS = ext.copy()
    sizeS = np.ones(n, np.int64)
    acc = np.zeros(n, np.float64)
    touched = np.empty(n, np.int64)
    gains = np.empty(n, np.float64)
    order = np.random.permutation(n)
    for v in order:
        cv = comm[v]
        if sizeS[ref[v]] != 1 or ext[v] < res * k[v] * (KC[cv] - k[v]):
            continue
        nt = 0
        for j in range(offsets[v], offsets[v + 1]):
            u = nbr[j]
            if u == v or comm[u] != cv:
                continue
            s = ref[u]
            if acc[s] == 0.0:
                touched[nt] = s
                nt += 1
            acc[s] += w[j]
        best_gain = 0.0
        m = 0
        for t in range(nt):
            s = touched[t]
            if extS[s] >= res * KS[s] * (KC[cv] - KS[s]):
                g = acc[s] - res * k[v] * KS[s]
                if g >= 0.0:
                    touched[m] = s
                    gains[m] = g
                    m += 1
                    if g > best_gain:
                        best_gain = g
            else:
                acc[s] = 0.0
        chosen = -1
        if best_gain > 0.0:
            stay = np.exp(-best_gain / beta)  # staying alone: gain 0
            total = stay
            for t in range(m):
                total += np.exp((gains[t] - best_gain) / beta)
            r = np.random.random() * total
            if r >= stay:
                r -= stay
                chosen = touched[m - 1]
                for t in range(m):
                    r -= np.exp((gains[t] - best_gain) / beta)
                    if r <= 0.0:
                        chosen = touched[t]
                        break
        if chosen >= 0:
            sv = ref[v]
            KS[chosen] += k[v]
            extS[chosen] += ext[v] - 2.0 * acc[chosen]
            sizeS[chosen] += 1
            sizeS[sv] = 0
            ref[v] = chosen
        for t in range(nt):
            acc[touched[t]] = 0.0
        for j in range(offsets[v], offsets[v + 1]):
            u = nbr[j]
            if u != v:
                acc[ref[u]] = 0.0
    return ref


def moved_share(old, new):
    """Share of nodes outside the old community that overlaps most with their new one."""
    key = np.unique(new.astype(np.int64) * (int(old.max()) + 1) + old, return_counts=True)
    best = np.zeros(int(new.max()) + 1, np.int64)
    np.maximum.at(best, key[0] // (int(old.max()) + 1), key[1])
    return 1 - best.sum() / len(new)


@njit(cache=True)
def renumber(labels):
    n = labels.shape[0]
    new = np.full(n, -1, np.int64)
    out = np.empty(n, np.int64)
    m = 0
    for i in range(n):
        if new[labels[i]] < 0:
            new[labels[i]] = m
            m += 1
        out[i] = new[labels[i]]
    return out, m


@njit(cache=True)
def aggregate(offsets, nbr, w, k, group, m):
    """Graph of the m groups (dense ids): summed link weights, self-loops kept, node weights summed."""
    n = offsets.shape[0] - 1
    start = np.zeros(m + 1, np.int64)
    for u in range(n):
        start[group[u] + 1] += 1
    for r in range(m):
        start[r + 1] += start[r]
    pos = start[:-1].copy()
    members = np.empty(n, np.int64)
    for u in range(n):
        members[pos[group[u]]] = u
        pos[group[u]] += 1
    mark = np.full(m, -1, np.int64)
    off = np.zeros(m + 1, np.int64)
    for r in range(m):
        c = 0
        for t in range(start[r], start[r + 1]):
            u = members[t]
            for j in range(offsets[u], offsets[u + 1]):
                g = group[nbr[j]]
                if mark[g] != r:
                    mark[g] = r
                    c += 1
        off[r + 1] = off[r] + c
    nb2 = np.empty(off[-1], np.int32)
    w2 = np.zeros(off[-1], np.float64)
    slot = np.zeros(m, np.int64)
    mark[:] = -1
    kk = np.zeros(m, np.float64)
    for r in range(m):
        c = off[r]
        for t in range(start[r], start[r + 1]):
            u = members[t]
            kk[r] += k[u]
            for j in range(offsets[u], offsets[u + 1]):
                g = group[nbr[j]]
                if mark[g] != r:
                    mark[g] = r
                    slot[g] = c
                    nb2[c] = g
                    c += 1
                w2[slot[g]] += w[j]
    return off, nb2, w2, kk


def leiden_pass(offsets, nbr, w, k, init, gamma, beta, seed, log=print):
    """One run of the Leiden algorithm from the partition init; returns the membership of every node."""
    two_m = k.sum()
    n = len(k)
    node_of = np.arange(n)                      # level node of every original node
    comm, _ = renumber(np.asarray(init, np.int64))
    G = (offsets, nbr, w, k)
    level = 0
    while True:
        moves = fast_move(*G, comm, gamma, two_m, seed + level)
        comm, nc = renumber(comm)
        nn = len(G[3])
        log(f"  level {level}: {nn:,} nodes, {moves:,} moves, {nc:,} communities")
        if nc == nn:
            break
        ref = refine(*G, comm, gamma, two_m, beta, seed + 1000 + level)
        ref, nr = renumber(ref)
        if nr == nn:                            # refinement merged nothing: aggregate the communities themselves
            ref, nr = comm.copy(), nc
        agg_comm = np.empty(nr, np.int64)
        agg_comm[ref] = comm
        G = aggregate(*G, ref, nr)
        node_of = ref[node_of]
        comm = agg_comm
        level += 1
    return comm[node_of]


def leiden(offsets, nbr, w, gamma=1.0, beta=0.01, seed=20261029, passes=10, log=print, checkpoint=None):
    k = node_strength(offsets, w)
    memb = np.arange(len(k))
    q = -1.0
    for p in range(passes):
        new = leiden_pass(offsets, nbr, w, k, memb, gamma, beta, seed + 7919 * p, log)
        q_new = modularity(offsets, nbr, w, new, gamma)
        log(f"pass {p + 1}: modularity {q_new:.6f}, {moved_share(memb, new):.2%} of nodes moved")
        memb, q = new, q_new
        if checkpoint is not None:
            checkpoint(memb, q, p + 1)
    sizes = np.bincount(memb)
    rank = np.empty(len(sizes), np.int64)
    rank[np.argsort(-sizes, kind="stable")] = np.arange(len(sizes))
    return rank[memb], q


if __name__ == "__main__":
    from common import CACHE, G, load_ids, log, save_result
    n = len(load_ids())
    offsets, nbr, w = weighted_csr(G / "edges_int32.parquet", n)
    log(f"weighted CSR: {n:,} nodes, {len(nbr) // 2:,} links")
    gamma = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    memb, q = leiden(offsets, nbr, w, gamma=gamma, log=log,
                     checkpoint=lambda m, q, p: np.save(CACHE / "leiden_checkpoint.npy", m.astype(np.int32)))
    np.save(CACHE / "leiden.npy", memb.astype(np.int32))
    sizes = np.bincount(memb)
    save_result("leiden", {"resolution": gamma, "beta": 0.01, "passes": 10, "modularity": q, "communities": int(len(sizes)),
                           "communities_1000plus": int((sizes >= 1000).sum()),
                           "share_in_1000plus": float(sizes[sizes >= 1000].sum() / n),
                           "largest": sizes[:40].tolist()})
    log(f"modularity {q:.4f}; {len(sizes):,} communities, {(sizes >= 1000).sum()} with 1,000+ members")
