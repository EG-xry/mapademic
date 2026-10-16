from collections import deque

import numpy as np
import pytest

from pathserver.search import expand, shortest_path


def csr(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v); adj[v].append(u)
    offsets = np.zeros(n + 1, np.int64)
    offsets[1:] = np.cumsum([len(x) for x in adj])
    return offsets, np.array([v for x in adj for v in x], np.int32), adj


def bfs_len(adj, a, b):
    seen, q = {a: 0}, deque([a])
    while q:
        x = q.popleft()
        if x == b:
            return seen[x]
        for y in adj[x]:
            if y not in seen:
                seen[y] = seen[x] + 1; q.append(y)
    return None


def test_expand_lists_every_neighbor_with_its_source():
    offsets, nbrs, _ = csr(4, [(0, 1), (0, 2), (2, 3)])
    nb, owner = expand(offsets, nbrs, np.array([0, 2]))
    assert sorted(zip(nb.tolist(), owner.tolist())) == [(0, 2), (1, 0), (2, 0), (3, 2)]


def test_path_on_a_chain_and_unreachable_pairs():
    offsets, nbrs, _ = csr(6, [(0, 1), (1, 2), (2, 3), (4, 5)])
    assert shortest_path(offsets, nbrs, 0, 3) == [0, 1, 2, 3]
    assert shortest_path(offsets, nbrs, 3, 0) == [3, 2, 1, 0]
    assert shortest_path(offsets, nbrs, 0, 5) is None
    assert shortest_path(offsets, nbrs, 2, 2) == [2]


@pytest.mark.parametrize("seed", range(8))
def test_matches_plain_bfs_on_random_graphs(seed):
    rng = np.random.default_rng(seed)
    n = 300
    edges = {(int(u), int(v)) for u, v in rng.integers(0, n, size=(500, 2)) if u != v}
    offsets, nbrs, adj = csr(n, sorted(edges))
    for a, b in rng.integers(0, n, size=(40, 2)):
        path = shortest_path(offsets, nbrs, int(a), int(b))
        want = bfs_len(adj, int(a), int(b))
        if want is None:
            assert path is None
            continue
        assert len(path) - 1 == want and path[0] == a and path[-1] == b
        assert all(path[i + 1] in adj[path[i]] for i in range(len(path) - 1))
