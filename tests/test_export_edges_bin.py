import json

import duckdb
import numpy as np

from pipeline.palette import community_rgb
from scripts.export_edges_bin import QUANT, SIZE_SCALE, WEIGHT_CAP, export_edges
from scripts.export_points_bin import GREY_RGB, MIN_MEMBERS


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


def write_edges(path, rows):
    """rows: (src, dst, weight)"""
    vals = ", ".join(f"('{s}', '{d}', {w})" for s, d, w in rows)
    duckdb.sql(
        f"COPY (SELECT src, dst, weight FROM (VALUES {vals}) t(src, dst, weight))"
        f" TO '{path}' (FORMAT PARQUET)"
    )


def _load_pos(out_dir, n):
    return np.fromfile(out_dir / "edges_pos.bin", dtype="<u2").reshape(n, 4)


def _load_attr(out_dir, n):
    return np.fromfile(out_dir / "edges_attr.bin", dtype=np.uint8).reshape(n, 8)


def _rgb_u8(rgb):
    return np.clip(np.round(np.array(rgb, dtype=np.float64) * 255), 0, 255).astype(np.uint8)


def test_file_sizes_match_count(tmp_path):
    web = tmp_path / "web.parquet"
    edges = tmp_path / "edges.parquet"
    nodes = [(f"A{i}", 0.1 * (i % 9), 0.1 * (i % 9), 1, 100, False) for i in range(10)]
    write_web(web, nodes)
    edge_rows = [(f"A{i}", f"A{i + 1}", 2 + i) for i in range(9)]
    write_edges(edges, edge_rows)
    out = tmp_path / "vector_experiment"
    meta = export_edges(str(edges), str(web), out)
    assert meta["count"] == 9
    assert (out / "edges_pos.bin").stat().st_size == 9 * 8
    assert (out / "edges_attr.bin").stat().st_size == 9 * 8
    written_meta = json.loads((out / "edges_meta.json").read_text())
    assert written_meta == meta


def test_ring_edges_excluded(tmp_path):
    web = tmp_path / "web.parquet"
    edges = tmp_path / "edges.parquet"
    nodes = [
        ("A0", 0.1, 0.1, 1, 100, False),
        ("A1", 0.2, 0.2, 1, 100, False),
        ("A2", 0.3, 0.3, 1, 100, True),  # is_ring -> any edge touching it is dropped
    ]
    write_web(web, nodes)
    edge_rows = [
        ("A0", "A1", 5),   # kept
        ("A1", "A2", 5),   # dropped, A2 is ring
        ("A2", "A0", 5),   # dropped, A2 is ring
    ]
    write_edges(edges, edge_rows)
    out = tmp_path / "vector_experiment"
    meta = export_edges(str(edges), str(web), out)
    assert meta["count"] == 1


def test_weight_filter(tmp_path):
    web = tmp_path / "web.parquet"
    edges = tmp_path / "edges.parquet"
    nodes = [
        ("A0", 0.1, 0.1, 1, 100, False),
        ("A1", 0.2, 0.2, 1, 100, False),
        ("A2", 0.3, 0.3, 1, 100, False),
        ("A3", 0.4, 0.4, 1, 100, False),
    ]
    write_web(web, nodes)
    edge_rows = [
        ("A0", "A1", 1),   # below weight >= 2 threshold -> dropped
        ("A2", "A3", 2),   # kept
    ]
    write_edges(edges, edge_rows)
    out = tmp_path / "vector_experiment"
    meta = export_edges(str(edges), str(web), out)
    assert meta["count"] == 1


def test_ordering_weight_desc(tmp_path):
    web = tmp_path / "web.parquet"
    edges = tmp_path / "edges.parquet"
    # distinct x per node so the source endpoint's x identifies the row
    nodes = [(f"P{i}", i / 10.0, 0.0, 1, 10, False) for i in range(5)]
    write_web(web, nodes)
    edge_rows = [
        ("P0", "P1", 3),
        ("P1", "P2", 10),
        ("P2", "P3", 3),  # ties P0-P1 at weight 3 -> src tiebreak: 'P0' < 'P2'
        ("P3", "P4", 7),
    ]
    write_edges(edges, edge_rows)
    out = tmp_path / "vector_experiment"
    meta = export_edges(str(edges), str(web), out)
    pos = _load_pos(out, meta["count"])
    x0s = [round(x0 / QUANT, 1) for x0, _, _, _ in pos]
    # weight DESC: P1-P2(10), P3-P4(7), then tie at 3 broken by src asc: P0-P1, P2-P3
    assert x0s == [0.1, 0.3, 0.0, 0.2]


def test_colors_big_vs_tail_community_per_endpoint(tmp_path):
    web = tmp_path / "web.parquet"
    edges = tmp_path / "edges.parquet"
    big_community, big_n, cx, cy = write_web_with_big_community(web)
    write_edges(edges, [("S0", "B0", 5)])  # src = tail community, dst = big community
    out = tmp_path / "vector_experiment"
    meta = export_edges(str(edges), str(web), out)
    assert meta["count"] == 1
    attrs = _load_attr(out, meta["count"])

    expected_big_u8 = _rgb_u8(community_rgb(big_community, big_n, cx, cy, min_members=MIN_MEMBERS))
    expected_grey_u8 = _rgb_u8(GREY_RGB)

    src_rgb = attrs[0, 0:3]  # S0 (tail community) -> grey
    dst_rgb = attrs[0, 3:6]  # B0 (big community) -> colored

    assert np.array_equal(src_rgb, expected_grey_u8)
    assert np.array_equal(dst_rgb, expected_big_u8)


def test_quantization_roundtrip(tmp_path):
    web = tmp_path / "web.parquet"
    edges = tmp_path / "edges.parquet"
    nodes = [
        ("A0", 0.0, 0.0, 1, 10, False),
        ("A1", 1.0, 1.0, 1, 10, False),
        ("A2", 0.123456, 0.987654, 1, 10, False),
        ("A3", 0.5, 0.5, 1, 10, False),
    ]
    write_web(web, nodes)
    write_edges(edges, [("A0", "A1", 5), ("A2", "A3", 5)])
    out = tmp_path / "vector_experiment"
    meta = export_edges(str(edges), str(web), out)
    pos = _load_pos(out, meta["count"])
    coords = {"A0": (0.0, 0.0), "A1": (1.0, 1.0), "A2": (0.123456, 0.987654), "A3": (0.5, 0.5)}
    expected = [
        (*coords["A0"], *coords["A1"]),
        (*coords["A2"], *coords["A3"]),
    ]
    for (x0, y0, x1, y1), (ex0, ey0, ex1, ey1) in zip(pos, expected):
        assert abs(x0 / QUANT - ex0) <= 1 / QUANT + 1e-12
        assert abs(y0 / QUANT - ey0) <= 1 / QUANT + 1e-12
        assert abs(x1 / QUANT - ex1) <= 1 / QUANT + 1e-12
        assert abs(y1 / QUANT - ey1) <= 1 / QUANT + 1e-12


def test_weight_and_size_attrs(tmp_path):
    web = tmp_path / "web.parquet"
    edges = tmp_path / "edges.parquet"
    huge_cited = int((SIZE_SCALE * 3) ** 2)  # sqrt(huge)/SIZE_SCALE = 3 -> clamps to 255
    nodes = [
        ("A0", 0.1, 0.1, 1, 0, False),
        ("A1", 0.2, 0.2, 1, huge_cited, False),
    ]
    write_web(web, nodes)
    write_edges(edges, [("A0", "A1", 999)])  # weight way past cap -> clamps to 255
    out = tmp_path / "vector_experiment"
    meta = export_edges(str(edges), str(web), out)
    attrs = _load_attr(out, meta["count"])
    assert attrs[0, 6] == 255  # weight clamps to WEIGHT_CAP -> scaled to 255
    assert attrs[0, 7] == 255  # size uses max of the two endpoints -> A1 clamps to 255
    assert WEIGHT_CAP == 20


def test_sample_flag_limits_and_keeps_top(tmp_path):
    web = tmp_path / "web.parquet"
    edges = tmp_path / "edges.parquet"
    nodes = [(f"A{i}", 0.0, 0.0, 1, 10, False) for i in range(21)]
    write_web(web, nodes)
    edge_rows = [(f"A{i}", f"A{i + 1}", i + 2) for i in range(20)]
    write_edges(edges, edge_rows)
    out = tmp_path / "vector_experiment"
    meta = export_edges(str(edges), str(web), out, sample=5)
    assert meta["count"] == 5
    assert (out / "edges_pos.bin").stat().st_size == 5 * 8
    assert (out / "edges_attr.bin").stat().st_size == 5 * 8
