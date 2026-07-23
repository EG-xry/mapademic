import json

import duckdb
import numpy as np
import pytest

from scripts.export_ids_bin import _id_to_u64, export_ids


def write_web(path, rows):
    """rows: (id, xw, yw, community, cited, is_ring)"""
    vals = ", ".join(
        f"('{i}', 'n{i}', {x}, {y}, {comm}, 20, {c}, 'I', 'Biology', {str(ring).upper()})"
        for i, x, y, comm, c, ring in rows
    )
    duckdb.sql(
        f"COPY (SELECT id, display_name, xw, yw, community, works_count, cited_by_count, "
        f"institution, field, is_ring FROM (VALUES {vals}) "
        f"t(id, display_name, xw, yw, community, works_count, cited_by_count, institution, field, is_ring))"
        f" TO '{path}' (FORMAT PARQUET)"
    )


def _load_ids(out_dir, n):
    return np.fromfile(out_dir / "ids.bin", dtype="<u8")[:n]


def test_id_to_u64_bare_and_full_url():
    assert _id_to_u64("A0") == 0
    assert _id_to_u64("A5101914242") == 5101914242
    assert _id_to_u64("https://openalex.org/A5101914242") == 5101914242


def test_id_to_u64_rejects_malformed():
    with pytest.raises(ValueError):
        _id_to_u64("not-an-id")


def test_file_size_and_meta_merge(tmp_path):
    web = tmp_path / "web.parquet"
    rows = [(f"A{i}", 0.1 * (i % 9), 0.1 * (i % 9), 1, 100 + i, False) for i in range(37)]
    write_web(web, rows)
    out = tmp_path / "vector_experiment"
    out.mkdir()
    # pre-existing meta.json, as export_points_bin.py would have written
    (out / "meta.json").write_text(json.dumps({"count": 37}))
    meta = export_ids(str(web), out)
    assert meta["ids"]["count"] == 37
    assert (out / "ids.bin").stat().st_size == 37 * 8
    written_meta = json.loads((out / "meta.json").read_text())
    assert written_meta["count"] == 37   # pre-existing key preserved
    assert written_meta["ids"] == meta["ids"]


def test_ordering_and_values_match_ids(tmp_path):
    web = tmp_path / "web.parquet"
    # distinct cited so ordering is deterministic: highest cited first
    rows = [(f"A{i}", 0.0, 0.0, 1, i, False) for i in range(5)]
    write_web(web, rows)
    out = tmp_path / "vector_experiment"
    meta = export_ids(str(web), out)
    ids = _load_ids(out, meta["ids"]["count"])
    assert list(ids) == [4, 3, 2, 1, 0]   # A4 has highest cited_by_count -> first


def test_ring_rows_excluded(tmp_path):
    web = tmp_path / "web.parquet"
    rows = [
        ("A0", 0.1, 0.1, 1, 100, False),
        ("A1", 0.2, 0.2, 1, 200, True),   # is_ring -> dust, dropped
        ("A2", 0.3, 0.3, 1, 300, False),
    ]
    write_web(web, rows)
    out = tmp_path / "vector_experiment"
    meta = export_ids(str(web), out)
    assert meta["ids"]["count"] == 2


def test_sample_flag_limits_and_keeps_top(tmp_path):
    web = tmp_path / "web.parquet"
    rows = [(f"A{i}", 0.0, 0.0, 1, i, False) for i in range(20)]
    write_web(web, rows)
    out = tmp_path / "vector_experiment"
    meta = export_ids(str(web), out, sample=5)
    assert meta["ids"]["count"] == 5
    assert (out / "ids.bin").stat().st_size == 5 * 8
    ids = _load_ids(out, 5)
    assert list(ids) == [19, 18, 17, 16, 15]   # top 5 by cited_by_count desc


def test_alignment_with_export_points_ordering(tmp_path):
    """Same source + filter + ORDER BY as export_points_bin -- row i here must
    be the same author as row i of positions.bin for the viewer's fallback
    hover to resolve the right id."""
    from scripts.export_points_bin import export_points

    web = tmp_path / "web.parquet"
    rows = [
        ("A0", 0.1, 0.1, 1, 100, False),
        ("A1", 0.2, 0.2, 1, 200, True),   # dropped by both exporters
        ("A2", 0.3, 0.3, 1, 300, False),
        ("A3", 0.4, 0.4, 1, 50, False),
    ]
    write_web(web, rows)
    out = tmp_path / "vector_experiment"
    points_meta = export_points(str(web), out)
    ids_meta = export_ids(str(web), out)
    assert ids_meta["ids"]["count"] == points_meta["count"]
    ids = _load_ids(out, ids_meta["ids"]["count"])
    # cited desc: A2(300), A0(100), A3(50)
    assert list(ids) == [2, 0, 3]
