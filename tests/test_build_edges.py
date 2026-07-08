import duckdb
import pytest

from pipeline.build_edges import build_edges
from tests.conftest import write_extract_output

A, B, C = "https://openalex.org/A1", "https://openalex.org/A2", "https://openalex.org/A3"


def write_nodes(tmp_path, ids):
    path = tmp_path / "nodes.parquet"
    con = duckdb.connect()
    con.execute("CREATE TABLE n (id VARCHAR)")
    con.executemany("INSERT INTO n VALUES (?)", [(i,) for i in ids])
    con.execute(f"COPY n TO '{path}' (FORMAT PARQUET)")
    return str(path)


def write_works(tmp_path, rows):
    path = tmp_path / "works_authorships.parquet"
    write_extract_output(path, rows)
    return str(path)


def edges_of(out_path):
    rows = duckdb.sql(
        f"SELECT src, dst, weight FROM '{out_path}' ORDER BY src, dst"
    ).fetchall()
    return {(r[0], r[1]): pytest.approx(r[2]) for r in rows}


def test_pair_explosion_and_weight(tmp_path):
    works = write_works(tmp_path, [("W1", 3, [A, B, C])])
    nodes = write_nodes(tmp_path, [A, B, C])
    out = tmp_path / "edges.parquet"
    n = build_edges(works, nodes, str(out))
    assert n == 3
    assert edges_of(out) == {(A, B): 0.5, (A, C): 0.5, (B, C): 0.5}


def test_weights_sum_across_shared_works(tmp_path):
    works = write_works(tmp_path, [("W1", 2, [A, B]), ("W2", 2, [A, B])])
    nodes = write_nodes(tmp_path, [A, B])
    out = tmp_path / "edges.parquet"
    build_edges(works, nodes, str(out))
    assert edges_of(out) == {(A, B): 2.0}


def test_hyperauthorship_cutoff_excludes_work(tmp_path):
    big = [f"https://openalex.org/AX{i:02d}" for i in range(51)]
    works = write_works(tmp_path, [("W1", 51, big)])
    nodes = write_nodes(tmp_path, big)
    out = tmp_path / "edges.parquet"
    assert build_edges(works, nodes, str(out)) == 0


def test_cutoff_boundary_50_authors_included(tmp_path):
    big = [f"https://openalex.org/AX{i:02d}" for i in range(50)]
    works = write_works(tmp_path, [("W1", 50, big)])
    nodes = write_nodes(tmp_path, big)
    out = tmp_path / "edges.parquet"
    assert build_edges(works, nodes, str(out)) == 50 * 49 // 2


def test_unselected_authors_drop_from_pairs_but_count_in_weight(tmp_path):
    # C is not in nodes: no edges touch C, but n_authors=3 so weight is 1/2
    works = write_works(tmp_path, [("W1", 3, [A, B, C])])
    nodes = write_nodes(tmp_path, [A, B])
    out = tmp_path / "edges.parquet"
    build_edges(works, nodes, str(out))
    assert edges_of(out) == {(A, B): 0.5}


def test_single_author_work_yields_no_edges(tmp_path):
    works = write_works(tmp_path, [("W1", 1, [A])])
    nodes = write_nodes(tmp_path, [A])
    out = tmp_path / "edges.parquet"
    assert build_edges(works, nodes, str(out)) == 0


def test_duplicate_author_in_list_no_self_edge_or_double_count(tmp_path):
    # OpenAlex data noise: same author listed twice on one work
    works = write_works(tmp_path, [("W1", 3, [A, A, B])])
    nodes = write_nodes(tmp_path, [A, B])
    out = tmp_path / "edges.parquet"
    build_edges(works, nodes, str(out))
    edges = edges_of(out)
    assert set(edges) == {(A, B)}
    assert edges[(A, B)] == pytest.approx(0.5)  # n_authors=3 per snapshot count
