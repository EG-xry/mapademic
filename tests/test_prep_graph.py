import duckdb
import pytest

from pipeline.prep_graph import prep_graph

A, B, C = "https://openalex.org/A1", "https://openalex.org/A2", "https://openalex.org/A3"


@pytest.fixture
def graph_inputs(tmp_path):
    con = duckdb.connect()
    con.execute(
        f"""
        COPY (SELECT * FROM (VALUES
            ('{B}', 'Bee', 20, 200, 'UCSD', 'Neuroscience'),
            ('{A}', 'Aye', 30, 300, NULL, NULL),
            ('{C}', 'Sea', 15, 50, NULL, 'Biology')
        ) t(id, display_name, works_count, cited_by_count, institution, field))
        TO '{tmp_path / "nodes.parquet"}' (FORMAT PARQUET)
        """
    )
    con.execute(
        f"""
        COPY (SELECT * FROM (VALUES
            ('{A}', '{B}', 2.0),
            ('{B}', '{C}', 0.25)
        ) t(src, dst, weight))
        TO '{tmp_path / "edges.parquet"}' (FORMAT PARQUET)
        """
    )
    return tmp_path


def test_renumbers_nodes_deterministically_by_id(graph_inputs):
    out = graph_inputs / "graph"
    n_nodes, n_edges = prep_graph(
        str(graph_inputs / "nodes.parquet"),
        str(graph_inputs / "edges.parquet"),
        out,
    )
    assert (n_nodes, n_edges) == (3, 2)
    rows = duckdb.sql(
        f"SELECT node_idx, id, display_name FROM '{out / 'nodes_int32.parquet'}' ORDER BY node_idx"
    ).fetchall()
    assert rows == [(0, A, "Aye"), (1, B, "Bee"), (2, C, "Sea")]  # ORDER BY id


def test_edges_remapped_to_int32_with_float_weight(graph_inputs):
    out = graph_inputs / "graph"
    prep_graph(
        str(graph_inputs / "nodes.parquet"),
        str(graph_inputs / "edges.parquet"),
        out,
    )
    schema = {
        name: typ
        for name, typ, *_ in duckdb.sql(
            f"DESCRIBE SELECT * FROM '{out / 'edges_int32.parquet'}'"
        ).fetchall()
    }
    assert schema == {"src": "INTEGER", "dst": "INTEGER", "weight": "FLOAT"}
    rows = duckdb.sql(
        f"SELECT src, dst, weight FROM '{out / 'edges_int32.parquet'}' ORDER BY src"
    ).fetchall()
    assert rows == [(0, 1, 2.0), (1, 2, 0.25)]


def test_min_weight_prunes_edges_not_nodes(graph_inputs):
    out = graph_inputs / "graph"
    n_nodes, n_edges = prep_graph(
        str(graph_inputs / "nodes.parquet"),
        str(graph_inputs / "edges.parquet"),
        out,
        min_weight=0.34,
    )
    assert (n_nodes, n_edges) == (3, 1)  # weak B-C edge dropped, node C kept
    rows = duckdb.sql(
        f"SELECT src, dst FROM '{out / 'edges_int32.parquet'}'"
    ).fetchall()
    assert rows == [(0, 1)]
