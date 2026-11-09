"""Small independent references for the bounded-memory manuscript checks."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from a10_checks import demographic_checks, edge_checks
from a6_institutions import auc_pairwise, features


def test_chunked_edges_match_vector_reference():
    community = np.array([0, 0, 1, 1, 2, 3], dtype=np.int32)
    src = np.array([0, 0, 1, 2, 3, 0], dtype=np.int32)
    dst = np.array([1, 2, 3, 3, 4, 4], dtype=np.int32)
    weight = np.array([1, 0.5, 2, 3, 0.2, 0.7], dtype=np.float32)
    masks = {"one": np.array([1, 1, 0, 0, 0, 0], bool), "two": np.array([0, 0, 1, 0, 1, 0], bool)}
    table = pa.table({"src": src, "dst": dst, "weight": weight})
    w = weight.astype(np.float64)
    strength = np.bincount(src, w, minlength=6) + np.bincount(dst, w, minlength=6)
    same = community[src] == community[dst]
    inside = np.bincount(community[src[same]], w[same], minlength=4)
    sc = np.bincount(community, strength, minlength=4)
    expected_q = (inside / w.sum() - (sc / (2 * w.sum())) ** 2).sum()
    for size in (1, 2, 9):
        result = edge_checks(table.to_batches(max_chunksize=size), community, masks)
        assert result["links"] == 6
        assert result["touching"] == {"one": 4, "two": 4}
        assert np.isclose(result["modularity"], expected_q)
        assert np.isclose(result["weight_inside_share"], inside.sum() / w.sum())


def test_demographics_conditional_denominators_and_empty_batches():
    rows = pd.DataFrame({"field": ["Social Sciences"] * 4 + ["Medicine"],
        "name": ["Alice Brown", "张三", "Other Name", "李四", "Bob Jones"],
        "country": ["US", None, "US", "CN", "US"], "institution": ["I1", None, "I2", "I3", None],
        "citations": [0, 0, 10, 10, 0], "fallback": [False, True, False, True, True]})
    expected = {"n": 4, "uncited": 0.5, "fallback": 2, "fallback_share": 0.5,
        "fallback_uncited": 0.5, "without_fallback_n": 2, "without_fallback_uncited": 0.5,
        "uncited_native_script": 0.5, "latin_with_institution_uncited": 0.5,
        "all_fallback": 3, "all_fallback_social_sciences": 2 / 3}
    table = pa.Table.from_pandas(rows, preserve_index=False)
    for size in (1, 2, 9):
        result = demographic_checks(table.to_batches(max_chunksize=size))
        assert result["country_known"] == 0.8
        assert result["social_sciences"] == expected


def test_institution_features_deduplicate_years_and_rank_auc_handles_ties():
    assert features([("I1", [2000, 2001, 2002]), ("I1", [2001]),
                     ("I2", [2001, 2002, 2003]), ("I3", [2002])]) == (3, 2, 2, 0.75)
    assert features([]) == (0, 0, 0, 0.0)
    a, b = np.array([1, 2, 2, 4]), np.array([0, 2, 3])
    expected = ((a[:, None] > b).mean() + 0.5 * (a[:, None] == b).mean())
    assert np.isclose(auc_pairwise(a, b), expected)
    assert auc_pairwise(np.array([]), b) is None


def test_snapshot_join_metrics_match_independent_pandas_reference():
    import tempfile
    import duckdb
    from a6_churn_snapshot import compare_snapshot
    old = pd.DataFrame({"id": [f"https://openalex.org/A{i}" for i in range(1, 7)],
                        "works_count": [20, 10, 4, 8, 0, 12], "cited_by_count": [10, 0, 5, 100, 0, 8]})
    new = pd.DataFrame({"id": [1, 2, 3, 4, 5], "works_count": [1, 10, 10, 7, 3],
                        "cited_by_count": [3, 4, 5, 80, 0]})
    attrs = pd.DataFrame({"id": old.id, "node_idx": np.arange(6)})
    identity = pd.DataFrame({"node_idx": np.arange(6), "key_n": [200, 1, 200, 1, 1, 1]})
    with tempfile.TemporaryDirectory() as directory:
        paths = [Path(directory) / f"{name}.parquet" for name in ("nodes", "snapshot", "attrs", "identity")]
        for frame, path in zip((old, new, attrs, identity), paths):
            frame.to_parquet(path, index=False)
        c = duckdb.connect()
        result = compare_snapshot(c, *paths)
        c.close()
    merged = old.assign(id=np.arange(1, 7)).merge(new, on="id", how="left", suffixes=("", "_new"))
    alive = merged.works_count_new.notna()
    rw = (merged.works_count_new - merged.works_count) / merged.works_count.clip(lower=1)
    rc = (merged.cited_by_count_new - merged.cited_by_count) / merged.cited_by_count.clip(lower=1)
    assert result["mapped"] == 6 and result["gone"] == 1 and result["gone_share"] == 1 / 6
    assert result["works"] == {"down_10pct": (rw[alive] < -0.1).mean(), "down_50pct": (rw[alive] < -0.5).mean(),
                               "up_10pct": (rw[alive] > 0.1).mean(), "up_50pct": (rw[alive] > 0.5).mean(),
                               "unchanged": (rw[alive] == 0).mean()}
    assert result["citations"] == {"down_10pct": (rc[alive] < -0.1).mean(), "up_50pct": (rc[alive] > 0.5).mean()}
    for label, group in (("common", identity.key_n >= 200), ("other", identity.key_n < 200)):
        assert result["by_name"][label] == {"gone_share": (~alive[group]).mean(),
                                           "works_down_10pct": (rw[alive & group] < -0.1).mean()}


def test_snapshot_release_guard_refuses_future_bucket_and_checks_header():
    import io
    from a6_institutions import release_state
    class Bucket:
        def __init__(self, release):
            self.release = release
        def open_input_file(self, path):
            return io.BytesIO((f"RELEASE NOTES\n\nRELEASE {self.release}\n" if path.endswith("RELEASE_NOTES.txt")
                               else '{"files": []}').encode())
    state = release_state(Bucket("2026-10-14"))
    assert state["release"] == "2026-10-14" and len(state["manifest_sha256"]) == 64
    try:
        release_state(Bucket("2026-11-14"))
    except RuntimeError as error:
        assert "refusing to mix snapshots" in str(error)
    else:
        raise AssertionError("future snapshot was accepted")
