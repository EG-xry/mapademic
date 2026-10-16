import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.subcommunities import build_subcommunities, fold_minor, leiden


def clique_edges(nodes):
    return [(a, b) for i, a in enumerate(nodes) for b in nodes[i + 1:]]


def two_groups(n_each=30, extra=()):
    a = list(range(n_each)); b = list(range(n_each, 2 * n_each))
    edges = clique_edges(a) + clique_edges(b) + [(0, n_each)] + list(extra)
    src, dst = np.array([e[0] for e in edges]), np.array([e[1] for e in edges])
    return src, dst, np.ones(len(edges))


def test_leiden_separates_two_dense_groups_deterministically():
    src, dst, w = two_groups()
    m1, m2 = leiden(60, src, dst, w), leiden(60, src, dst, w)
    assert (m1 == m2).all()
    assert len(set(m1[:30])) == 1 and len(set(m1[30:])) == 1 and m1[0] != m1[30]


def test_minor_groups_fold_into_the_major_group_they_link_to():
    # a 3-node triangle hanging off group B is too small for its own area
    tri = [(60, 61), (61, 62), (60, 62), (60, 45), (61, 46)]
    src, dst, w = two_groups(extra=tri)
    membership = np.concatenate([np.zeros(30, int), np.ones(30, int), np.full(3, 2)])
    sub, links = fold_minor(membership, src, dst, w, major_share=0.1, major_min=5)
    assert set(sub[60:]) == {sub[30]}            # joined B, its only link
    assert sub[0] != sub[30]
    assert links == {(0, 1): 1.0}                # the single A-B link


def test_fold_minor_declines_to_split_one_dominant_group():
    src, dst, w = two_groups(n_each=30)
    membership = np.concatenate([np.zeros(57, int), np.ones(3, int)])
    sub, links = fold_minor(membership, src, dst, w, major_share=0.1, major_min=5)
    assert sub is None and links == {}


def test_sub_zero_is_the_largest_area():
    edges = clique_edges(list(range(20))) + clique_edges(list(range(20, 60))) + [(0, 20)]
    src, dst = np.array([e[0] for e in edges]), np.array([e[1] for e in edges])
    membership = np.concatenate([np.zeros(20, int), np.ones(40, int)])
    sub, _ = fold_minor(membership, src, dst, np.ones(len(edges)), major_share=0.1, major_min=5)
    assert sub[20] == 0 and sub[0] == 1


def test_build_writes_members_and_links(tmp_path):
    src, dst, w = two_groups()
    ids = [f"A{i}" for i in range(60)]
    pq.write_table(pa.table({"node_idx": pa.array(range(60), pa.int32()), "id": ids}), tmp_path / "nodes.parquet")
    pq.write_table(pa.table({"src": pa.array(src, pa.int32()), "dst": pa.array(dst, pa.int32()),
                             "weight": pa.array(w, pa.float32())}), tmp_path / "edges.parquet")
    pq.write_table(pa.table({"id": ids, "community": pa.array([7] * 60, pa.int32()),
                             "institution": ["Inst"] * 60, "cited_by_count": pa.array([50] * 60, pa.int32())}),
                   tmp_path / "coords.parquet")
    meta = build_subcommunities(str(tmp_path / "coords.parquet"), str(tmp_path / "nodes.parquet"),
                                str(tmp_path / "edges.parquet"), str(tmp_path / "subs.parquet"),
                                str(tmp_path / "links.parquet"), split_min=10, major_min=5)
    assert meta["communities"]["7"]["subs"] == 2
    rows = duckdb.sql(f"SELECT community, sub, count(*) FROM '{tmp_path / 'subs.parquet'}' GROUP BY 1, 2 ORDER BY 2").fetchall()
    assert rows == [(7, 0, 30), (7, 1, 30)]
    assert duckdb.sql(f"SELECT community, a, b, weight FROM '{tmp_path / 'links.parquet'}'").fetchall() == [(7, 0, 1, 1.0)]


def test_sparse_communities_are_not_split(tmp_path):
    src, dst, w = two_groups()
    ids = [f"A{i}" for i in range(60)]
    pq.write_table(pa.table({"node_idx": pa.array(range(60), pa.int32()), "id": ids}), tmp_path / "nodes.parquet")
    pq.write_table(pa.table({"src": pa.array(src, pa.int32()), "dst": pa.array(dst, pa.int32()),
                             "weight": pa.array(w, pa.float32())}), tmp_path / "edges.parquet")
    # no institutions, no citations: a near-empty community (palette.sparse_communities)
    pq.write_table(pa.table({"id": ids, "community": pa.array([7] * 60, pa.int32()),
                             "institution": pa.array([None] * 60, pa.string()),
                             "cited_by_count": pa.array([0] * 60, pa.int32())}), tmp_path / "coords.parquet")
    meta = build_subcommunities(str(tmp_path / "coords.parquet"), str(tmp_path / "nodes.parquet"),
                                str(tmp_path / "edges.parquet"), str(tmp_path / "subs.parquet"),
                                str(tmp_path / "links.parquet"), split_min=10, major_min=5, sparse_min=10)
    assert meta["faded"] == [7] and meta["communities"] == {}
    assert duckdb.sql(f"SELECT count(*) FROM '{tmp_path / 'subs.parquet'}'").fetchone()[0] == 0
