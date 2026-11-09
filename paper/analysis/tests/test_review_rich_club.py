"""Toy checks of rv_rich_club: small-team flags, club tallies and the small-team subgraph."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rv_rich_club as rc  # noqa: E402

WORKS = [[0, 1, 2], [0, 1], [0, 3], [1, 3], [0, 4, 5], [1, 5]]
N = 6


def toy(sizes):
    rows = [[] for _ in range(N)]
    for i, w in enumerate(WORKS):
        for a in w:
            rows[a].append(i + N)
    a_off = np.cumsum([0] + [len(r) for r in rows]).astype(np.int64)
    a_nb = np.array([x for r in rows for x in sorted(r)], np.int64)
    nbrs = [set() for _ in range(N)]
    for w in WORKS:
        for a in w:
            nbrs[a].update(b for b in w if b != a)
    offsets = np.cumsum([0] + [len(s) for s in nbrs]).astype(np.int64)
    neighbors = np.array([b for s in nbrs for b in sorted(s)], np.int64)
    return offsets, neighbors, a_off, a_nb, np.array(sizes, np.int64)


def test_flags_tally_and_small_team_clubs():
    # work 4 ({0,4,5}) is a 20-author work: the links 0-4, 0-5 and 4-5 rest only on it
    offsets, neighbors, a_off, a_nb, w_size = toy([3, 2, 2, 2, 20, 2])
    flag, bad = rc.small_flags(offsets, neighbors, a_off, a_nb, w_size, rc.SMALL)
    assert bad == 0
    degree = np.diff(offsets)
    rank, order = rc.club_rank(degree, np.zeros(N, bool), N)
    assert order.tolist() == [0, 1, 5, 2, 3, 4]
    clubs = np.array([2, 6], np.int64)
    deg_small, m_small, inside, large_only = rc.tally(offsets, neighbors, flag, rank, clubs)
    assert deg_small.tolist() == [3, 4, 2, 2, 0, 1] and m_small == 6
    assert inside.tolist() == [1, 9] and large_only.tolist() == [0, 3]
    rank_small, order_small = rc.club_rank(deg_small, np.zeros(N, bool), N)
    assert order_small[:2].tolist() == [1, 0]
    assert rc.inside_small(offsets, neighbors, flag, rank_small, clubs).tolist() == [1, 6]


def test_flagged_researchers_rank_last_and_chung_lu_matches_b3():
    rank, order = rc.club_rank(np.array([9, 8, 7]), np.array([True, False, False]), 3)
    assert order.tolist() == [1, 2, 0]
    k = np.array([2.0, 3.0, 5.0])
    assert rc.chung_lu(k) == (2 * 3 + 2 * 5 + 3 * 5)
