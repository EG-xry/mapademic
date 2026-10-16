"""Shortest coauthor paths on the compressed-sparse-row graph that
pipeline/pathgraph.py writes.

The public path backend searched by fetching coauthors from the OpenAlex API
hop by hop, so long-distance pairs ran out of time or memory on its free
host (Eric, 2026-10-11: Connect "bugs out" for longer depths; three
cross-continent test pairs failed or hung). Here the whole graph (8.4M
authors, 732M directed links) sits in two arrays and a bidirectional
breadth-first search expands whole levels with numpy, always from the side
whose frontier touches fewer links, so a long path costs well under a second.
"""
import numpy as np


def expand(offsets: np.ndarray, neighbors: np.ndarray, frontier: np.ndarray):
    """All (neighbor, from) pairs of a frontier, as two arrays."""
    starts = offsets[frontier].astype(np.int64)
    deg = offsets[frontier + 1].astype(np.int64) - starts
    total = int(deg.sum())
    if total == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    owner = np.repeat(frontier, deg)
    skip = np.repeat(starts - (np.cumsum(deg) - deg), deg)
    return np.asarray(neighbors[skip + np.arange(total)], dtype=np.int64), owner


def shortest_path(offsets: np.ndarray, neighbors: np.ndarray, a: int, b: int,
                  progress=None, max_hops: int = 14) -> list[int] | None:
    """Node indices of one shortest path from a to b, or None if none exists
    within max_hops. Each step expands one whole level of one side, so the
    first level on which the two searches touch gives the shortest length."""
    if a == b:
        return [a]
    n = len(offsets) - 1
    parent = [np.full(n, -1, np.int32), np.full(n, -1, np.int32)]
    parent[0][a], parent[1][b] = a, b
    frontier = [np.array([a], np.int64), np.array([b], np.int64)]
    depth = [0, 0]
    while frontier[0].size and frontier[1].size and depth[0] + depth[1] < max_hops:
        cost = [int((offsets[f + 1] - offsets[f]).sum()) for f in frontier]
        side = 0 if cost[0] <= cost[1] else 1
        nb, owner = expand(offsets, neighbors, frontier[side])
        fresh = parent[side][nb] < 0
        nb, owner = nb[fresh], owner[fresh]
        nb, first = np.unique(nb, return_index=True)
        owner = owner[first]
        parent[side][nb] = owner
        depth[side] += 1
        if progress:
            progress(side, depth[side], len(nb))
        met = nb[parent[1 - side][nb] >= 0]
        if met.size:
            return _join(parent, int(met[0]), a, b)
        frontier[side] = nb
    return None


def _join(parent, meet: int, a: int, b: int) -> list[int]:
    left, x = [], meet
    while x != a:
        left.append(x)
        x = int(parent[0][x])
    left.append(a)
    right, x = [], meet
    while x != b:
        x = int(parent[1][x])
        right.append(x)
    return left[::-1] + right
