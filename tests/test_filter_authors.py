import duckdb
import pytest

from pipeline.filter_authors import filter_authors

EMPTY_INST = "CAST([] AS STRUCT(display_name VARCHAR)[])"
EMPTY_TOPICS = "CAST([] AS STRUCT(display_name VARCHAR, field STRUCT(display_name VARCHAR))[])"


@pytest.fixture
def authors_glob(tmp_path):
    part = tmp_path / "authors" / "updated_date=2026-01-01"
    part.mkdir(parents=True)
    duckdb.sql(
        f"""
        COPY (
            SELECT * FROM (VALUES
                ('https://openalex.org/A1', 'Low Works', 2, 20,
                 {EMPTY_INST}, {EMPTY_TOPICS}),
                ('https://openalex.org/A2', 'Busy Bee', 5, 50,
                 [{{'display_name': 'UCSD'}}],
                 [{{'display_name': 'ULM',
                    'field': {{'display_name': 'Neuroscience'}}}}]),
                ('https://openalex.org/A3', 'Prolific', 100, 1000,
                 {EMPTY_INST}, {EMPTY_TOPICS})
            ) t(id, display_name, works_count, cited_by_count,
                last_known_institutions, topics)
        ) TO '{part / "part_0000.parquet"}' (FORMAT PARQUET)
        """
    )
    return str(tmp_path / "authors" / "*" / "*.parquet")


def test_threshold_drops_low_works_count_authors(authors_glob, tmp_path):
    out = tmp_path / "nodes.parquet"
    kept = filter_authors(authors_glob, str(out), min_works=5)
    ids = {r[0] for r in duckdb.sql(f"SELECT id FROM '{out}'").fetchall()}
    assert kept == 2
    assert ids == {"https://openalex.org/A2", "https://openalex.org/A3"}


def test_extracts_first_institution_and_field(authors_glob, tmp_path):
    out = tmp_path / "nodes.parquet"
    filter_authors(authors_glob, str(out), min_works=5)
    row = duckdb.sql(
        f"SELECT institution, field FROM '{out}' WHERE id LIKE '%A2'"
    ).fetchone()
    assert row == ("UCSD", "Neuroscience")


def test_missing_institution_and_topics_yield_nulls(authors_glob, tmp_path):
    out = tmp_path / "nodes.parquet"
    filter_authors(authors_glob, str(out), min_works=5)
    row = duckdb.sql(
        f"SELECT institution, field FROM '{out}' WHERE id LIKE '%A3'"
    ).fetchone()
    assert row == (None, None)
