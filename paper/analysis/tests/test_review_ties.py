"""Toy checks of rv_ties: shared works, common neighbours, built-in common neighbours and the two overlaps."""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rv_ties  # noqa: E402

# Six authors, six works (every author retained, so a work's full size is its author count):
# w0 {0,1,2}, w1 {0,1}, w2 {0,3}, w3 {1,3}, w4 {0,4,5}, w5 {1,5}
WORKS = [[0, 1, 2], [0, 1], [0, 3], [1, 3], [0, 4, 5], [1, 5]]
N = 6


def toy():
    w_off = np.cumsum([0] + [len(w) for w in WORKS]).astype(np.int64)
    w_auth = np.array([a for w in WORKS for a in w], np.int64)
    w_size = np.array([len(w) for w in WORKS], np.int64)
    rows = [[] for _ in range(N)]
    for i, w in enumerate(WORKS):
        for a in w:
            rows[a].append(i + N)                     # work node ids follow the researchers
    a_off = np.cumsum([0] + [len(r) for r in rows]).astype(np.int64)
    a_nb = np.array([x for r in rows for x in sorted(r)], np.int64)
    nbrs = [set() for _ in range(N)]
    for w in WORKS:
        for a in w:
            nbrs[a].update(b for b in w if b != a)
    offsets = np.cumsum([0] + [len(s) for s in nbrs]).astype(np.int64)
    sn = np.array([b for s in nbrs for b in sorted(s)], np.int64)
    return offsets, sn, a_off, a_nb, w_off, w_auth, w_size


def test_link_stats_and_overlaps():
    offsets, sn, a_off, a_nb, w_off, w_auth, w_size = toy()
    us, vs = np.array([0, 0, 4], np.int64), np.array([1, 3, 5], np.int64)
    k, tmin, tmax, c, union, s = rv_ties.link_stats(offsets, sn, a_off, a_nb, w_off, w_auth, w_size, us, vs)
    assert k.tolist() == [2, 1, 1]
    assert tmin.tolist() == [2, 2, 3] and tmax.tolist() == [3, 2, 3]
    assert c.tolist() == [3, 1, 1] and union.tolist() == [4, 4, 2] and s.tolist() == [1, 0, 1]
    plain, residual = rv_ties.overlaps(c, union, s)
    assert np.allclose(plain, [0.75, 0.25, 0.5])
    assert np.allclose(residual, [2 / 3, 0.25, 0.0])


def test_large_works_make_no_links():
    offsets, sn, a_off, a_nb, w_off, w_auth, w_size = toy()
    w_size = w_size.copy()
    w_size[0] = rv_ties.MAX_TEAM + 1                    # w0 now counts as a large team: it no longer makes links
    k, tmin, tmax, c, union, s = rv_ties.link_stats(offsets, sn, a_off, a_nb, w_off, w_auth, w_size,
                                                    np.array([0], np.int64), np.array([1], np.int64))
    assert k.tolist() == [1] and tmax.tolist() == [2] and s.tolist() == [0]


def test_grouping_and_sample_match_b2c():
    source = (Path(rv_ties.__file__).with_name("b2c_repeat_ties.py")).read_text()
    assert f"default_rng({rv_ties.SEED})" in source
    edges = re.search(r"SHARED_EDGES = np\.array\((\[[^\]]+\])", source).group(1)
    assert eval(edges) == rv_ties.SHARED_EDGES.tolist()
    g = rv_ties.grouped(np.array([0.1, 0.3, 0.5, 0.7]), np.array([1, 1, 11, 12]), np.ones(4, bool))
    assert set(g) == {"1", "11+"} and abs(g["1"]["mean"] - 0.2) < 1e-12 and g["11+"]["n"] == 2
