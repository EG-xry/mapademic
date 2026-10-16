import math

import duckdb
import pytest

from pipeline.warm_start import build_init_positions


def write(path, sql):
    duckdb.sql(f"COPY ({sql}) TO '{path}' (FORMAT PARQUET)")
    return str(path)


@pytest.fixture
def graph(tmp_path):
    """Five nodes: A,B,C laid out before; D is new with neighbours A and B
    (weights 3 and 1); E is new and only touches D (two hops from a seed)."""
    nodes = write(tmp_path / "nodes_int32.parquet",
                  "SELECT * FROM (VALUES (0, 'A'), (1, 'B'), (2, 'C'), (3, 'D'), (4, 'E'))"
                  " t(node_idx, id)")
    edges = write(tmp_path / "edges_int32.parquet",
                  "SELECT src::INTEGER AS src, dst::INTEGER AS dst, weight::FLOAT AS weight"
                  " FROM (VALUES (0, 3, 3.0), (3, 1, 1.0), (3, 4, 2.0), (0, 1, 1.0))"
                  " t(src, dst, weight)")
    prev = write(tmp_path / "coords.parquet",
                 "SELECT * FROM (VALUES ('A', 0.0, 0.0), ('B', 100.0, 0.0), ('C', 0.0, 50.0),"
                 " ('Z', 9.0, 9.0)) t(id, x, y)")
    return nodes, edges, prev


def positions(path):
    return {v: (x, y) for v, x, y in duckdb.sql(
        f"SELECT vertex, x, y FROM '{path}' ORDER BY vertex").fetchall()}


def test_carried_over_nodes_keep_their_previous_position(graph, tmp_path):
    out = tmp_path / "init_pos.parquet"
    build_init_positions(*graph, str(out))
    pos = positions(out)
    assert pos[0] == (0.0, 0.0) and pos[1] == (100.0, 0.0) and pos[2] == (0.0, 50.0)


def test_new_nodes_start_at_their_coauthors_weighted_mean(graph, tmp_path):
    out = tmp_path / "init_pos.parquet"
    stats = build_init_positions(*graph, str(out))
    pos = positions(out)
    # D: weights A=3, B=1 -> x = (3*0 + 1*100) / 4 = 25, within jitter
    assert pos[3][0] == pytest.approx(25.0, abs=0.01)
    assert pos[3][1] == pytest.approx(0.0, abs=0.01)
    # E reaches a position through D on the second hop
    assert pos[4][0] == pytest.approx(pos[3][0], abs=0.01)
    assert stats == {"carried": 3, "from_neighbours": 2, "scattered": 0}


def test_every_vertex_gets_a_finite_position(graph, tmp_path):
    out = tmp_path / "init_pos.parquet"
    build_init_positions(*graph, str(out))
    rows = duckdb.sql(f"SELECT vertex, x, y FROM '{out}'").fetchall()
    assert sorted(v for v, _, _ in rows) == [0, 1, 2, 3, 4]
    assert all(math.isfinite(x) and math.isfinite(y) for _, x, y in rows)
    types = dict(duckdb.sql(f"DESCRIBE SELECT * FROM '{out}'").fetchall()[i][:2] for i in range(3))
    assert types == {"vertex": "INTEGER", "x": "FLOAT", "y": "FLOAT"}   # FA2 pos_list format


def test_nodes_without_any_positioned_neighbour_are_scattered_deterministically(tmp_path):
    nodes = write(tmp_path / "n.parquet",
                  "SELECT * FROM (VALUES (0, 'A'), (1, 'B'), (2, 'Q')) t(node_idx, id)")
    edges = write(tmp_path / "e.parquet",
                  "SELECT 0::INTEGER AS src, 1::INTEGER AS dst, 1.0::FLOAT AS weight")
    prev = write(tmp_path / "c.parquet",
                 "SELECT * FROM (VALUES ('A', -10.0, 0.0), ('B', 10.0, 0.0)) t(id, x, y)")
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    stats = build_init_positions(nodes, edges, prev, str(a))
    build_init_positions(nodes, edges, prev, str(b))
    assert stats["scattered"] == 1
    assert positions(a) == positions(b)
    qx, qy = positions(a)[2]
    assert math.hypot(qx, qy) <= 10.0 + 1e-6   # inside the previous median radius


def test_coincident_new_nodes_are_pulled_apart(tmp_path):
    # two new nodes with the same single neighbour must not start on one point
    nodes = write(tmp_path / "n.parquet",
                  "SELECT * FROM (VALUES (0, 'A'), (1, 'N1'), (2, 'N2')) t(node_idx, id)")
    edges = write(tmp_path / "e.parquet",
                  "SELECT * FROM (VALUES (0::INTEGER, 1::INTEGER, 1.0::FLOAT),"
                  " (0::INTEGER, 2::INTEGER, 1.0::FLOAT)) t(src, dst, weight)")
    prev = write(tmp_path / "c.parquet", "SELECT 'A' AS id, 5.0 AS x, 5.0 AS y")
    out = tmp_path / "p.parquet"
    build_init_positions(nodes, edges, prev, str(out))
    pos = positions(out)
    assert pos[1] != pos[2]
