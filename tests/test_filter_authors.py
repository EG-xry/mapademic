import duckdb
import pytest

from pipeline.filter_authors import filter_authors
from tests.conftest import write_extract_output

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


@pytest.fixture
def works_glob(tmp_path):
    """A2 has 5 qualifying works; A3's 100 snapshot works are all xpac
    deposits; A1 has 2 articles plus 4 datasets."""
    a1, a2, a3 = (f"https://openalex.org/A{i}" for i in (1, 2, 3))
    rows = [(f"W{i}", 1, [a2]) for i in range(1, 6)]
    rows += [(f"W{i}", 1, [a3], {"is_xpac": True}) for i in range(10, 16)]
    rows += [("W20", 1, [a1]), ("W21", 1, [a1])]
    rows += [(f"W{i}", 1, [a1], {"type": "dataset"}) for i in range(30, 34)]
    path = tmp_path / "works.parquet"
    write_extract_output(path, rows)
    return str(path)


def test_threshold_counts_qualifying_works_from_the_extract(authors_glob, works_glob, tmp_path):
    out = tmp_path / "nodes.parquet"
    kept = filter_authors(authors_glob, str(out), min_works=5, works_glob=works_glob)
    rows = duckdb.sql(f"SELECT id, works_count, works_kept FROM '{out}'").fetchall()
    assert kept == 1
    # works_count stays the snapshot figure (tooltips); works_kept drives selection
    assert rows == [("https://openalex.org/A2", 5, 5)]


def test_all_works_counts_every_extract_row(authors_glob, works_glob, tmp_path):
    out = tmp_path / "nodes.parquet"
    filter_authors(authors_glob, str(out), min_works=5, works_glob=works_glob,
                   all_works=True)
    got = dict(duckdb.sql(f"SELECT id, works_kept FROM '{out}'").fetchall())
    assert got == {"https://openalex.org/A1": 6, "https://openalex.org/A2": 5,
                   "https://openalex.org/A3": 6}


def test_repeated_author_on_one_work_counts_once(authors_glob, tmp_path):
    a2 = "https://openalex.org/A2"
    path = tmp_path / "w.parquet"
    write_extract_output(path, [("W1", 2, [a2, a2])])
    out = tmp_path / "nodes.parquet"
    filter_authors(authors_glob, str(out), min_works=1, works_glob=str(path))
    assert duckdb.sql(f"SELECT works_kept FROM '{out}'").fetchall() == [(1,)]


def test_snapshot_mode_leaves_works_kept_null(authors_glob, tmp_path):
    out = tmp_path / "nodes.parquet"
    filter_authors(authors_glob, str(out), min_works=5)
    assert {r[0] for r in duckdb.sql(f"SELECT works_kept FROM '{out}'").fetchall()} == {None}


@pytest.fixture
def junk_fixture(tmp_path):
    """Authors that are not people: placeholders and uncited bulk records."""
    part = tmp_path / "junk_authors" / "updated_date=2026-01-01"
    part.mkdir(parents=True)
    people = [
        (1, "Et. Al.", 900), (2, "ほか", 50), (3, "-", 10), (4, ":unav", 0),
        (5, "Mehmet Ali Kanat", 12),          # contains "et al" but is a person
        (6, "Kankakee Daily Journal", 0),     # 0 citations, 1000+ works
        (7, "Prolific Person", 30000),        # 1000+ works, cited
        (8, "Quiet Scholar", 0),              # uncited, but few works: kept
    ]
    duckdb.sql(
        "COPY (SELECT * FROM (VALUES "
        + ", ".join(f"('https://openalex.org/A{i}', '{n}', 20, {c}, {EMPTY_INST}, {EMPTY_TOPICS})"
                    for i, n, c in people)
        + f") t(id, display_name, works_count, cited_by_count, last_known_institutions, topics))"
          f" TO '{part / 'part_0000.parquet'}' (FORMAT PARQUET)"
    )
    rows = []
    for i, _n, _c in people:
        count = 1000 if i in (6, 7) else 20
        rows += [(f"W{i}{j:05d}", 1, [f"https://openalex.org/A{i}"]) for j in range(count)]
    works = tmp_path / "junk_works.parquet"
    write_extract_output(works, rows)
    return str(tmp_path / "junk_authors" / "*" / "*.parquet"), str(works)


def test_placeholder_and_bulk_record_authors_are_dropped(junk_fixture, tmp_path):
    authors, works = junk_fixture
    out = tmp_path / "nodes.parquet"
    filter_authors(authors, str(out), min_works=15, works_glob=works)
    names = {r[0] for r in duckdb.sql(f"SELECT display_name FROM '{out}'").fetchall()}
    assert names == {"Mehmet Ali Kanat", "Prolific Person", "Quiet Scholar"}


def test_keep_placeholders_restores_them(junk_fixture, tmp_path):
    authors, works = junk_fixture
    out = tmp_path / "nodes.parquet"
    kept = filter_authors(authors, str(out), min_works=15, works_glob=works,
                          keep_placeholders=True)
    assert kept == 8
