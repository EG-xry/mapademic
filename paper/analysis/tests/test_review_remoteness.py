"""Synthetic checks of rv_remoteness: degree bins, degree prediction and the two between-group fits."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rv_remoteness  # noqa: E402


def test_degree_bins_exact_then_logarithmic():
    b = rv_remoteness.degree_bins(np.array([1, 2, 300, 301, 315, 316, 10_000]))
    assert b[:3].tolist() == [1, 2, 300]
    assert b[3] == b[4] == 301                       # ln(315/300) = 0.0488 < 0.05: the first log bin
    assert b[5] == 302 and b[6] > b[5]               # ln(316/300) = 0.0520: the next one
    assert np.all(np.diff(b) >= 0)


def test_distance_depending_only_on_degree_leaves_no_residual():
    rng = np.random.default_rng(1)
    degree = rng.integers(1, 200, 50_000)
    distance = 6.0 - 0.5 * np.log(degree)
    predicted = rv_remoteness.degree_prediction(degree, distance)
    assert np.allclose(predicted, distance)


def test_additive_and_degree_only_fits():
    p = np.array([2.0, 2.5, 3.0, 3.4])
    s = 1.0 + 2.0 * (p[:, None] + p[None, :])
    alpha, beta, r2 = rv_remoteness.degree_only_fit(s, p)
    assert abs(alpha - 1.0) < 1e-9 and abs(beta - 2.0) < 1e-9 and abs(r2 - 1.0) < 1e-12
    a_true = np.array([1.9, 2.0, 2.4, 2.8])
    a, r2 = rv_remoteness.additive_fit(a_true[:, None] + a_true[None, :])
    assert np.allclose(a, a_true) and abs(r2 - 1.0) < 1e-12
