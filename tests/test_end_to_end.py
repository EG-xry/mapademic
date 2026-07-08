"""extract -> filter -> edges over one coherent fixture, as the real run composes them."""
import duckdb

from pipeline.build_edges import build_edges
from pipeline.extract_works import connect, extract_partition, partitions
from pipeline.filter_authors import filter_authors
from tests.conftest import write_works_partition

HINTON = "https://openalex.org/A_hinton"
COLLAB = "https://openalex.org/A_collab"
ONEHIT = "https://openalex.org/A_onehit"

EMPTY_INST = "CAST([] AS STRUCT(display_name VARCHAR)[])"
EMPTY_TOPICS = "CAST([] AS STRUCT(display_name VARCHAR, field STRUCT(display_name VARCHAR))[])"


def test_extract_filter_edges(tmp_path):
    # authors snapshot fixture
    authors_part = tmp_path / "authors" / "updated_date=2026-01-01"
    authors_part.mkdir(parents=True)
    duckdb.sql(
        f"""
        COPY (
            SELECT * FROM (VALUES
                ('{HINTON}', 'G. Hinton', 400, 900000, {EMPTY_INST}, {EMPTY_TOPICS}),
                ('{COLLAB}', 'Co Author', 12, 300, {EMPTY_INST}, {EMPTY_TOPICS}),
                ('{ONEHIT}', 'One Hit', 1, 5, {EMPTY_INST}, {EMPTY_TOPICS})
            ) t(id, display_name, works_count, cited_by_count,
                last_known_institutions, topics)
        ) TO '{authors_part / "part_0000.parquet"}' (FORMAT PARQUET)
        """
    )
    # works snapshot fixture
    works_src = tmp_path / "works"
    write_works_partition(
        works_src / "updated_date=2026-01-01",
        [("W1", 3, [HINTON, COLLAB, ONEHIT])],
    )

    # extract
    extracts = tmp_path / "works_authorships"
    extracts.mkdir()
    con = connect(str(works_src))
    for part in partitions(con, str(works_src)):
        extract_partition(con, str(works_src), part, extracts)

    # filter
    nodes_out = tmp_path / "nodes.parquet"
    kept = filter_authors(
        str(tmp_path / "authors" / "*" / "*.parquet"), str(nodes_out), min_works=5
    )
    assert kept == 2

    # edges
    edges_out = tmp_path / "edges.parquet"
    n = build_edges(str(extracts / "*.parquet"), str(nodes_out), str(edges_out))
    assert n == 1

    src, dst, weight = duckdb.sql(f"SELECT * FROM '{edges_out}'").fetchone()
    assert {src, dst} == {HINTON, COLLAB}   # ONEHIT filtered out upstream
    assert weight == 0.5                     # n_authors=3 counts ONEHIT
