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


def test_duplicate_work_id_across_partitions_counted_once(tmp_path):
    # Snapshot re-release landing mid-extract can leave the same work_id in
    # two partition extract files; it must be counted once, not twice.
    works_dir = tmp_path / "works_authorships"
    works_dir.mkdir()
    write_extract_output(works_dir / "part_0000.parquet", [("W1", 2, [A, B])])
    write_extract_output(works_dir / "part_0001.parquet", [("W1", 2, [A, B])])
    nodes = write_nodes(tmp_path, [A, B])
    out = tmp_path / "edges.parquet"
    build_edges(str(works_dir / "*.parquet"), nodes, str(out))
    assert edges_of(out) == {(A, B): 1.0}


def test_xpac_and_dataset_works_add_no_weight(tmp_path):
    # the 2026-07 build's worst pair drew 447,612 of weight from per-shot
    # dataset deposits; only the ordinary article may count here
    works = write_works(tmp_path, [
        ("W1", 2, [A, B]),
        ("W2", 2, [A, B], {"is_xpac": True}),
        ("W3", 2, [A, B], {"type": "dataset"}),
        ("W4", 2, [A, B], {"is_paratext": True}),
    ])
    nodes = write_nodes(tmp_path, [A, B])
    out = tmp_path / "edges.parquet"
    build_edges(works, nodes, str(out))
    assert edges_of(out) == {(A, B): 1.0}


def test_all_works_restores_the_unfiltered_weights(tmp_path):
    works = write_works(tmp_path, [
        ("W1", 2, [A, B]),
        ("W2", 2, [A, B], {"is_xpac": True}),
    ])
    nodes = write_nodes(tmp_path, [A, B])
    out = tmp_path / "edges.parquet"
    build_edges(works, nodes, str(out), all_works=True)
    assert edges_of(out) == {(A, B): 2.0}


def test_edges_keep_full_openalex_ids(tmp_path):
    # the extract stores numeric keys; edges.parquet must still carry the
    # string ids every downstream stage joins on
    works = write_works(tmp_path, [("W1", 2, [A, B])])
    nodes = write_nodes(tmp_path, [A, B])
    out = tmp_path / "edges.parquet"
    build_edges(works, nodes, str(out))
    types = dict(duckdb.sql(f"DESCRIBE SELECT * FROM '{out}'").fetchall()[i][:2]
                 for i in range(3))
    assert types == {"src": "VARCHAR", "dst": "VARCHAR", "weight": "DOUBLE"}


def brute_force(works, node_ids, max_authors=50):
    """Reference: the weight definition written out directly."""
    from collections import defaultdict
    out = defaultdict(float)
    for _wid, n, aids in works:
        if not 2 <= n <= max_authors:
            continue
        kept = sorted({a for a in aids if a in node_ids}, key=lambda a: int(a[1:]))
        for i in range(len(kept)):
            for j in range(i + 1, len(kept)):
                out[(kept[i], kept[j])] += 1.0 / (n - 1)
    return out


def test_matches_brute_force_on_a_random_graph(tmp_path):
    import random
    rng = random.Random(3)
    authors = [f"A{i}" for i in range(1, 80)]
    node_ids = set(rng.sample(authors, 55))
    works = []
    for w in range(1, 400):
        k = rng.choice([1, 2, 2, 3, 4, 6, 12, 51])
        aids = [rng.choice(authors) for _ in range(k)]   # repeats happen, as in OpenAlex
        works.append((f"W{w}", k, aids))
    path = write_works(tmp_path, works)
    nodes = write_nodes(tmp_path, sorted(node_ids))
    out = tmp_path / "edges.parquet"
    n = build_edges(path, nodes, str(out), buckets=4)
    expected = brute_force(works, node_ids)
    assert n == len(expected)
    got = {frozenset((s, d)): w for s, d, w in
           duckdb.sql(f"SELECT src, dst, weight FROM '{out}'").fetchall()}
    assert set(got) == {frozenset(p) for p in expected}
    for p, w in expected.items():
        assert got[frozenset(p)] == pytest.approx(w)


def test_bucket_count_does_not_change_the_result(tmp_path):
    works = write_works(tmp_path, [
        ("W1", 3, [A, B, C]), ("W2", 2, [A, C]), ("W3", 4, [A, B, C, "https://openalex.org/A9"]),
    ])
    nodes = write_nodes(tmp_path, [A, B, C])
    one, three = tmp_path / "e1.parquet", tmp_path / "e3.parquet"
    build_edges(works, nodes, str(one), buckets=1)
    build_edges(works, nodes, str(three), buckets=3)
    assert edges_of(one) == edges_of(three)


def test_no_scratch_left_behind(tmp_path):
    works = write_works(tmp_path, [("W1", 2, [A, B])])
    nodes = write_nodes(tmp_path, [A, B])
    out = tmp_path / "edges.parquet"
    build_edges(works, nodes, str(out), buckets=2)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "edges.parquet", "nodes.parquet", "works_authorships.parquet"]
