"""Small independent parity check for selected A8 work-ratio/name metadata."""
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from repair_inputs import hub_metadata


def test_selected_metadata_matches_original_full_series_reindex():
    old = pd.DataFrame({"id": ["https://openalex.org/A3", "A1", "https://openalex.org/A2", "A4"],
                        "works_count": [30, 0, 10, 99]})
    new = pd.DataFrame({"id": [1, 3, 4], "works_count": [2, 12, 98]})
    key = pd.DataFrame({"node_idx": [0, 1, 2, 3], "key_n": [200, 20, 900, 300]})
    full = pd.DataFrame({"node_idx": [0, 1, 2, 3], "full_n": [10, 1, 15, 4]})
    target = pd.DataFrame({"node_idx": [2, 0, 1], "id": [3, 1, 2]})
    with tempfile.TemporaryDirectory() as directory:
        paths = [Path(directory) / f"{i}.parquet" for i in range(4)]
        for frame, path in zip((old, new, key, full), paths):
            frame.to_parquet(path, index=False)
        c = duckdb.connect()
        result = hub_metadata(c, target, *paths)
        c.close()
    previous = pd.Series(old.works_count.to_numpy(), index=old.id.str.extract(r"A(\d+)$")[0].astype("int64"))
    expected_new = new.set_index("id").works_count.reindex(target.id).to_numpy()
    np.testing.assert_array_equal(result.index.to_numpy(), target.node_idx)
    np.testing.assert_array_equal(result.old_works, previous.reindex(target.id).to_numpy())
    np.testing.assert_allclose(result.new_works, expected_new, equal_nan=True)
    np.testing.assert_array_equal(result.key_n, key.set_index("node_idx").key_n.reindex(target.node_idx))
    np.testing.assert_array_equal(result.full_n, full.set_index("node_idx").full_n.reindex(target.node_idx))
    with np.errstate(divide="ignore", invalid="ignore"):
        np.testing.assert_allclose(result.new_works / result.old_works,
            expected_new / previous.reindex(target.id).to_numpy(), equal_nan=True)
