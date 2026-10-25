"""scripts/export_flags.py: flagged profiles reach Connect's author numbering, and the name filters hold."""
import json

import numpy as np
import pandas as pd
import pytest

from scripts import export_flags


@pytest.fixture
def release(tmp_path, monkeypatch):
    """A five-author release: pathgraph ids by node index, paths ids by a shuffled author number."""
    rd = tmp_path / "data" / "r1"
    (rd / "pathgraph").mkdir(parents=True)
    (rd / "paths").mkdir()
    ids = np.array([5_000_000_010, 5_000_000_020, 5_000_000_030, 5_000_000_040, 5_000_000_050], np.uint64)
    np.save(rd / "pathgraph" / "ids.npy", ids)
    by_number = (ids[[3, 0, 4, 1, 2]] - 5_000_000_000).astype("<u4")      # author number k holds node [3,0,4,1,2][k]
    by_number.tofile(rd / "paths" / "ids-x.bin")
    (rd / "paths" / "manifest.json").write_text(json.dumps({"id_base": 5_000_000_000, "files": {"ids": "ids-x.bin"}}))
    pd.DataFrame({"node_idx": range(5), "key_n": [900, 900, 3, 5, 900]}).to_parquet(
        rd / "pathgraph" / "identity_signals.parquet")
    cache = tmp_path / "cache"
    cache.mkdir()
    pd.DataFrame({"node_idx": range(5), "name": ["Wei Wang", "Wei Wang", "Eugene Braunwald", "Rare Person", "Wei Wang"]}
                 ).to_parquet(cache / "node_attrs.parquet")
    pd.DataFrame({"node_idx": [0, 2, 3], "group": "top", "alive": True, "peak_insts": [984, 41, 30],
                  "peak_year": [2026, 1997, 2012]}).to_csv(cache / "a6_affiliations.csv", index=False)
    pd.DataFrame({"node_idx": [1, 3], "alive": True, "works": [11318, 897], "works_live": [277, 114],
                  "works_ratio": [277 / 11318, 114 / 897]}).to_csv(cache / "a6_live_works.csv", index=False)
    np.save(cache / "nonperson_idx.npy", np.array([4]))
    monkeypatch.setattr(export_flags, "ROOT", tmp_path)
    monkeypatch.setattr(export_flags, "COMMON_FULL_NAME", 3)               # three "Wei Wang" in this tiny release
    return cache


def test_author_numbers_follow_the_paths_renumbering(release, tmp_path):
    numbers, ids = export_flags.author_numbers(tmp_path / "data" / "r1", np.array([0, 3, 2]))
    assert numbers.tolist() == [1, 0, 4]
    assert ids.tolist() == [10, 40, 30]


def test_build_keeps_common_names_and_lets_later_kinds_win(release):
    out = export_flags.build("r1", release)["flags"]
    # node 0 (Wei Wang, 984 institutions): merged; node 2 (Braunwald, rare name) and node 3 (rare full name) are not
    assert out["merged"] == {"n": [1], "id": [10], "peak": [984], "year": [2026]}
    # node 1 (common name, lost most works): split; node 3's lost half (rare name) is a consolidation, not a split
    assert out["split"] == {"n": [3], "id": [20], "then": [11318], "now": [277]}
    assert out["notPerson"] == {"n": [2], "id": [50]}
