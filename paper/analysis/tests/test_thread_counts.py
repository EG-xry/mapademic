"""Exact four/eight-thread parity, isolated from the test runner's thread limit."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


def fixture():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    import numba
    import kernels
    import a2c_team_cap as cap

    n = 80
    edges = sorted({tuple(sorted((u, (u + v) % n))) for u in range(n) for v in (1, 7)})
    rows = [[] for _ in range(n)]
    for u, v in edges:
        rows[u].append(v)
        rows[v].append(u)
    offsets = np.concatenate(([0], np.cumsum([len(row) for row in rows]))).astype(np.int64)
    neighbors = np.array([v for row in rows for v in sorted(row)], np.int32)
    target_mask = np.arange(n) % 3 != 0
    sources = np.flatnonzero(target_mask)[:32].astype(np.int64)
    w_auth = np.array(edges, np.int32).reshape(-1)
    w_off = np.arange(0, len(w_auth) + 1, 2, dtype=np.int64)
    w_size = np.full(len(edges), 2, np.int32)
    a_off = np.empty(n + 1, np.int64)
    a_nb = np.empty(len(w_auth), np.int32)
    cap._author_side_into(n, w_off, w_auth, a_off, a_nb)
    labels = np.arange(n) % 3
    results, active = [], []
    for threads in (4, 8):
        numba.set_num_threads(threads)
        results.append((kernels.histograms(offsets, neighbors, sources, 64),
                        *kernels.distance_sums(offsets, neighbors, sources, threads),
                        *kernels.histograms_distance_sums(offsets, neighbors, sources, 64, threads),
                        kernels.exact_mean(offsets, neighbors, sources),
                        *kernels.group_distances(offsets, neighbors, sources, labels, 3),
                        *cap.run_with_targets(a_off, a_nb, w_off, w_auth, w_size, sources, 50, 64, target_mask)))
        active.append(numba.get_num_threads())
    assert active == [4, 8]
    assert all(np.array_equal(a, b) for a, b in zip(*results)), 'thread-count-dependent numerical result'
    print(json.dumps({'active_threads': active, 'threading_layer': numba.threading_layer(),
                      'output_arrays_exact': len(results[0]), 'researchers': n, 'sources': len(sources),
                      'kernels': ['histograms', 'distance_sums', 'histograms_distance_sums', 'exact_mean',
                                  'group_distances', 'run_with_targets']}, sort_keys=True))


class ThreadCounts(unittest.TestCase):
    def test_four_and_eight_threads_give_exact_counts_and_means(self):
        environment = {**os.environ, 'NUMBA_NUM_THREADS': '8', 'OMP_NUM_THREADS': '1',
                       'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'}
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--fixture'],
                                env=environment, text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        evidence = json.loads(result.stdout)
        self.assertEqual(evidence['active_threads'], [4, 8])
        self.assertEqual(evidence['output_arrays_exact'], 11)


if __name__ == '__main__':
    if '--fixture' in sys.argv:
        fixture()
    else:
        unittest.main()
