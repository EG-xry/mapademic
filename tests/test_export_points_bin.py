import json
import math

import duckdb
import numpy as np

from pipeline.palette import community_rgb
from scripts.export_points_bin import GREY_RGB, MIN_MEMBERS, QUANT, SIZE_SCALE, export_points


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


def write_web_with_big_community(path, big_community=7, big_n=1000, cx=0.9, cy=0.5):
    """One community with >= MIN_MEMBERS members (colored) plus a small
    sub-threshold community and a ring/dust row (both grey)."""
    duckdb.sql(
        f"COPY ("
        f"  SELECT 'B' || range::VARCHAR id, 'n' display_name, {cx} xw, {cy} yw,"
        f"         {big_community} community, 20 works_count, (1000 - range) cited_by_count,"
        f"         'i' institution, 'Medicine' field, FALSE is_ring FROM range({big_n})"
        f"  UNION ALL"
        f"  SELECT 'S0' id, 'small' display_name, 0.2 xw, 0.2 yw,"
        f"         999 community, 20 works_count, 5000 cited_by_count,"
        f"         'i' institution, 'Chemistry' field, FALSE is_ring"
        f"  UNION ALL"
        f"  SELECT 'R0' id, 'ring' display_name, 0.8 xw, 0.1 yw,"
        f"         555555 community, 20 works_count, 9999 cited_by_count,"
        f"         'i' institution, NULL field, TRUE is_ring"
        f") TO '{path}' (FORMAT PARQUET)"
    )
    return big_community, big_n, cx, cy


def _load_positions(out_dir, n):
    return np.fromfile(out_dir / "positions.bin", dtype="<u2").reshape(n, 2)


def _load_attrs(out_dir, n):
    return np.fromfile(out_dir / "attrs.bin", dtype=np.uint8).reshape(n, 4)


def test_file_sizes_match_count(tmp_path):
    web = tmp_path / "web.parquet"
    rows = [(f"A{i}", 0.1 * (i % 9), 0.1 * (i % 9), 1, 100 + i, False) for i in range(37)]
    write_web(web, rows)
    out = tmp_path / "vector_experiment"
    meta = export_points(str(web), out)
    assert meta["count"] == 37
    assert (out / "positions.bin").stat().st_size == 37 * 4
    assert (out / "attrs.bin").stat().st_size == 37 * 4
    written_meta = json.loads((out / "meta.json").read_text())
    assert written_meta == meta


def test_quantization_roundtrip(tmp_path):
    web = tmp_path / "web.parquet"
    coords = [(0.0, 0.0), (1.0, 1.0), (0.123456, 0.987654), (0.5, 0.5)]
    rows = [(f"A{i}", x, y, 1, 10, False) for i, (x, y) in enumerate(coords)]
    write_web(web, rows)
    out = tmp_path / "vector_experiment"
    meta = export_points(str(web), out)
    pos = _load_positions(out, meta["count"])
    # ordering is cited_by_count DESC, id ASC; all cited equal here so id order (A0..A3)
    for (x, y), (xq, yq) in zip(coords, pos):
        assert abs(xq / QUANT - x) <= 1 / QUANT + 1e-12
        assert abs(yq / QUANT - y) <= 1 / QUANT + 1e-12


def test_size_clamps_and_scales(tmp_path):
    web = tmp_path / "web.parquet"
    huge_cited = int((SIZE_SCALE * 3) ** 2)  # sqrt(huge)/SIZE_SCALE = 3 -> clamps to 255
    rows = [
        ("A0", 0.1, 0.1, 1, 0, False),                       # cited 0 -> s = 0
        ("A1", 0.2, 0.2, 1, huge_cited, False),               # way past cap -> s clamps to 255
    ]
    write_web(web, rows)
    out = tmp_path / "vector_experiment"
    meta = export_points(str(web), out)
    attrs = _load_attrs(out, meta["count"])
    ids_order = {"A0": 1, "A1": 0}  # A1 has higher cited_by_count -> comes first
    assert attrs[ids_order["A1"], 3] == 255
    assert attrs[ids_order["A0"], 3] == 0


def test_ordering_by_cited_desc(tmp_path):
    web = tmp_path / "web.parquet"
    # distinct x per row so we can recover which row landed in which slot
    rows = [(f"A{i}", i / 10.0, 0.0, 1, i, False) for i in range(5)]
    write_web(web, rows)
    out = tmp_path / "vector_experiment"
    meta = export_points(str(web), out)
    pos = _load_positions(out, meta["count"])
    xs = [round(xq / QUANT, 1) for xq, _ in pos]
    assert xs == [0.4, 0.3, 0.2, 0.1, 0.0]  # highest cited_by_count (A4) first


def test_colors_big_community_matches_palette_small_grey_ring_excluded(tmp_path):
    web = tmp_path / "web.parquet"
    big_community, big_n, cx, cy = write_web_with_big_community(web)
    out = tmp_path / "vector_experiment"
    meta = export_points(str(web), out)
    # R0 is is_ring -> dropped entirely; order: cited_by_count DESC -> S0 (5000), then B0..B999 (1000..1)
    assert meta["count"] == big_n + 1
    attrs = _load_attrs(out, meta["count"])
    expected_big_rgb = community_rgb(big_community, big_n, cx, cy, min_members=MIN_MEMBERS)
    expected_big_u8 = np.clip(np.round(np.array(expected_big_rgb) * 255), 0, 255).astype(np.uint8)
    grey_u8 = np.clip(np.round(np.array(GREY_RGB) * 255), 0, 255).astype(np.uint8)

    small_row = attrs[0, :3]    # S0, community 999 with 1 member -> below MIN_MEMBERS -> grey
    big_row = attrs[1, :3]      # B0, community `big_community` with big_n members -> colored

    assert np.array_equal(small_row, grey_u8)
    assert np.array_equal(big_row, expected_big_u8)


def test_ring_rows_excluded(tmp_path):
    web = tmp_path / "web.parquet"
    rows = [
        ("A0", 0.1, 0.1, 1, 100, False),
        ("A1", 0.2, 0.2, 1, 200, True),   # is_ring -> dust, dropped
        ("A2", 0.3, 0.3, 1, 300, False),
    ]
    write_web(web, rows)
    out = tmp_path / "vector_experiment"
    meta = export_points(str(web), out)
    assert meta["count"] == 2


def test_sample_flag_limits_and_keeps_top(tmp_path):
    web = tmp_path / "web.parquet"
    rows = [(f"A{i}", 0.0, 0.0, 1, i, False) for i in range(20)]
    write_web(web, rows)
    out = tmp_path / "vector_experiment"
    meta = export_points(str(web), out, sample=5)
    assert meta["count"] == 5
    assert (out / "positions.bin").stat().st_size == 5 * 4
    assert (out / "attrs.bin").stat().st_size == 5 * 4
