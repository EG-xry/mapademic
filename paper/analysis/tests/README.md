# Synthetic regression checks

Run from the repository root using the paper environment:

```sh
NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .tmp/venv-paper/bin/python paper/analysis/tests/run_tests.py
```

The checks use small synthetic networks, temporary Parquet files and independent
reference calculations. They do not read the full researcher network or contact
OpenAlex. They cover equivalence of bounded-memory aggregations, conditional
denominators, metadata alignment, cached-source identity, restart behavior and
the deposit's graph-reconstruction rules. Check the individual test files for
the exact cases. The runner supports both `unittest.TestCase` classes and
zero-argument `test_*` functions; pytest is not required.

These regression checks are separate from the completed-run audit:

```sh
.tmp/venv-paper/bin/python paper/analysis/run_final_validation.py
```

That audit reads the real outputs, checks checksums and reconstructs the repaired
edge count. Run it after the analysis runner and final manuscript build finish.
The wrapper always requires the manuscript, holds the shared runner lock and
uses the owned-process memory watchdog. It records a separate validation receipt
without changing the parent runner state. Visual review of every figure and
manuscript page remains necessary. Its orchestration fixtures use temporary
Python subprocesses only; they never invoke the production validator.
