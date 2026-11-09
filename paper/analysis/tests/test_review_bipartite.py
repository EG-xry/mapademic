"""Checks of rv_bipartite_null: exact projected transitivity on a hand graph, and the analytic null against an
exact projection of a random bipartite graph."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rv_bipartite_null as rb  # noqa: E402


def test_projected_transitivity_hand_graph():
    # triangle 0-1-2 plus a pendant 3 on 2: 3 triangle corners over 5 connected pairs
    work_of = np.array([0, 0, 0, 1, 1])
    author = np.array([0, 1, 2, 2, 3])
    off, nb = rb.project(work_of, author, 4)
    assert off.tolist() == [0, 2, 4, 7, 8]
    assert abs(rb.projected_transitivity(off, nb) - 0.6) < 1e-12


def random_bipartite(seed, n, works, exponent, max_size=6):
    rng = np.random.default_rng(seed)
    weight = 1.0 / np.arange(1, n + 1) ** exponent
    weight /= weight.sum()
    sizes = rng.integers(2, max_size, works)
    work_of = np.repeat(np.arange(works), sizes)
    author = np.concatenate([rng.choice(n, s, replace=False, p=weight) for s in sizes])
    return rb.small_bipartite(work_of, author, n)


def test_analytic_null_matches_a_random_bipartite_projection():
    # mild skew: no structural cutoff, so within-work closure plus A1b's chance term should match the projection
    n = 20_000
    a_off, a_nb, m, wo, au = random_bipartite(7, n, 12_000, 0.3)
    stats, local, k = rb.null_statistics(m, np.ones(len(m), bool), a_off, a_nb, n)
    assert stats["cutoff_ratio"] < 1
    off, nb = rb.project(wo, au, n)
    simulated = rb.projected_transitivity(off, nb)
    assert abs(stats["transitivity"] / simulated - 1) < 0.05, (stats["transitivity"], simulated)
    assert stats["wedges"] <= stats["wedges_raw"]
    assert np.nanmax(local) <= 1.0


def test_distinct_coauthor_correction_tracks_wedges_under_heavy_skew():
    # strong skew: hubs repeat coauthors; the corrected wedge count stays near the projection's, the raw one does not
    n = 20_000
    a_off, a_nb, m, wo, au = random_bipartite(7, n, 12_000, 0.8)
    stats, _, _ = rb.null_statistics(m, np.ones(len(m), bool), a_off, a_nb, n)
    off, _ = rb.project(wo, au, n)
    k = np.diff(off)
    wedges = float((k * (k - 1) / 2).sum())
    assert abs(stats["wedges"] / wedges - 1) < 0.1, (stats["wedges"], wedges)
    assert stats["wedges_raw"] / wedges > 1.5


def test_distinct_coauthors_saturates_and_is_linear_when_sparse():
    a = np.array([1] * 1000)
    k = rb.distinct_coauthors(np.array([1, 10, 100_000]), a)
    assert abs(k[0] - 1) < 1e-3 and abs(k[1] - 10) < 0.1 and k[2] <= 1000
