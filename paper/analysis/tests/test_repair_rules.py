"""Revised repair rules (2026-11-03): the tiered same-name join rule, the repair column of the deposited flagged
links, and the validator reading only the links the repaired network removes."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ANALYSIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS))
import export_deposit as deposit  # noqa: E402
import validate_min5 as validate  # noqa: E402
from common import split_keep  # noqa: E402


def test_split_rule_tiers():
    group = np.array([5, 6, 6, 20, 21, 2])
    shared = np.array([3, 3, 8, 10, 15, 2])
    small = np.array([15, 15, 15, 20, 15, 2])
    assert split_keep(group, shared, small).tolist() == [True, False, True, True, False, False]


def test_flagged_links_carry_repair_column_and_validator_filters_it():
    # triangle 0-1-2: suspect links (0,1) and (1,2); the repair removes only (0,1)
    offsets = np.array([0, 2, 4, 6], np.int64)
    neighbors = np.array([1, 2, 0, 2, 0, 1], np.int32)
    mask = np.array([True, False, True, True, False, True])
    repair = np.array([True, False, True, False, False, False])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "flagged_links.parquet"
        assert deposit.export_flagged_links(offsets, neighbors, mask, path, batch_size=4, repair=repair) == (2, 4)
        table = pq.read_table(path)
        assert table.column_names == ["src", "dst", "repair"]
        rows = sorted(zip(table.column("src").to_pylist(), table.column("dst").to_pylist(), table.column("repair").to_pylist()))
        assert rows == [(0, 1, True), (1, 2, False)]
        assert validate.keys_from_parquet(path).tolist() == sorted([int(validate.pack(np.array([0]), np.array([1]))[0]),
                                                                     int(validate.pack(np.array([1]), np.array([2]))[0])])
        assert validate.keys_from_parquet(path, repair_only=True).tolist() == [int(validate.pack(np.array([0]), np.array([1]))[0])]


def test_name_record_merges():
    from a6_name_records import incompatible_names
    assert incompatible_names("Kai You Wang", ["Anzhen Wang", "Kai Wang", "Wang K", "Yong-Guang Wang"]) == ["anzhen", "yong guang"]
    assert incompatible_names("Paul M. Thompson", ["P. M. Thompson", "PM Thompson", "Thompson, Paul M.",
                                                   "Paul Michael Thompson"]) == []
    assert incompatible_names("Gregory Y.H. Lip", ["GYH Lip", "Greg Lip", "Gregory Yoke Hong Lip"]) == []
    assert incompatible_names("Mohammad Smith", ["Mohammed Smith", "M. Smith"]) == []
