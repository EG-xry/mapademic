"""Checks of rv_hub_links: the random link mask is symmetric, hits its target share and removes exactly its links."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rv_hub_links as rh  # noqa: E402
from kernels import contract  # noqa: E402


def random_graph(n=3000, links=30_000, seed=3):
    rng = np.random.default_rng(seed)
    u, v = rng.integers(0, n, links), rng.integers(0, n, links)
    keep = u != v
    key = np.unique(np.minimum(u, v)[keep].astype(np.int64) * n + np.maximum(u, v)[keep])
    a, b = key // n, key % n
    src, dst = np.r_[a, b], np.r_[b, a]
    order = np.lexsort((dst, src))
    src, dst = src[order], dst[order]
    offsets = np.zeros(n + 1, np.int64)
    np.add.at(offsets, src + 1, 1)
    return np.cumsum(offsets), dst.astype(np.int32)


def test_mask_is_symmetric_and_hits_its_share():
    offsets, neighbors = random_graph()
    skip = rh.symmetric_mask(offsets, neighbors, 0.3, 11)
    row = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    dropped = set(zip(row[skip].tolist(), neighbors[skip].tolist()))
    assert all((v, u) in dropped for u, v in dropped)
    assert abs(skip.mean() - 0.3) < 0.02
    other = rh.symmetric_mask(offsets, neighbors, 0.3, 12)
    assert (skip != other).any()                       # seeds give independent draws


def test_contract_drops_exactly_the_masked_links():
    offsets, neighbors = random_graph()
    n = len(offsets) - 1
    skip = rh.symmetric_mask(offsets, neighbors, 0.6, 5)
    off2, nb2 = contract(offsets, neighbors, skip, np.arange(n, dtype=np.int64), n)
    assert off2[-1] == len(neighbors) - int(skip.sum())
