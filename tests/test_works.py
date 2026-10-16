import duckdb
import pytest

from pipeline.works import (
    EXCLUDED_TYPES, check_extract_schema, id_key_sql, qualifying_works_sql,
)
from tests.conftest import write_extract_output


def ids_of(path, **kw):
    q = qualifying_works_sql(str(path), **kw)
    return sorted(r[0] for r in duckdb.sql(f"SELECT work_id FROM ({q})").fetchall())


def test_id_key_takes_trailing_digits():
    got = duckdb.sql(
        f"SELECT {id_key_sql(repr('https://openalex.org/A5023888391'))}"
    ).fetchone()[0]
    assert got == 5023888391


def test_filter_drops_xpac_paratext_and_excluded_types(tmp_path):
    path = tmp_path / "w.parquet"
    write_extract_output(path, [
        ("W1", 2, ["A1", "A2"]),
        ("W2", 2, ["A1", "A2"], {"is_xpac": True}),
        ("W3", 2, ["A1", "A2"], {"is_paratext": True}),
        ("W4", 2, ["A1", "A2"], {"type": "dataset"}),
        ("W5", 2, ["A1", "A2"], {"type": "preprint"}),
        ("W6", 2, ["A1", "A2"], {"is_retracted": True}),
    ])
    # retracted articles stay: the collaboration still happened
    assert ids_of(path) == [1, 5, 6]


def test_all_works_disables_the_filter(tmp_path):
    path = tmp_path / "w.parquet"
    write_extract_output(path, [
        ("W1", 2, ["A1", "A2"]),
        ("W2", 2, ["A1", "A2"], {"is_xpac": True}),
        ("W4", 2, ["A1", "A2"], {"type": "dataset"}),
    ])
    assert ids_of(path, all_works=True) == [1, 2, 4]


def test_null_flags_and_type_count_as_ordinary_works(tmp_path):
    path = tmp_path / "w.parquet"
    write_extract_output(path, [
        ("W1", 2, ["A1", "A2"], {"is_xpac": None, "is_paratext": None, "type": None}),
    ])
    assert ids_of(path) == [1]


def test_author_count_bounds(tmp_path):
    path = tmp_path / "w.parquet"
    write_extract_output(path, [
        ("W1", 1, ["A1"]),
        ("W2", 2, ["A1", "A2"]),
        ("W3", 3, ["A1", "A2", "A3"]),
    ])
    assert ids_of(path, min_authors=2, max_authors=2) == [2]


def test_duplicate_work_rows_collapse(tmp_path):
    d = tmp_path / "x"
    d.mkdir()
    write_extract_output(d / "a.parquet", [("W1", 2, ["A1", "A2"])])
    write_extract_output(d / "b.parquet", [("W1", 2, ["A1", "A2"])])
    assert ids_of(d / "*.parquet") == [1]


def test_every_excluded_type_is_a_real_openalex_type():
    # the work-types vocabulary as of the 2026-09-23 release
    known = {
        "article", "book", "book-chapter", "book-review", "conference-abstract",
        "conference-paper", "data-paper", "dataset", "dissertation", "editorial",
        "erratum", "letter", "libguides", "other", "paratext", "peer-review",
        "preprint", "reference-entry", "report", "retraction", "review",
        "software", "software-paper", "standard", "supplementary-materials",
    }
    assert set(EXCLUDED_TYPES) <= known


def test_legacy_extract_is_refused_with_a_hint(tmp_path):
    path = tmp_path / "old.parquet"
    duckdb.sql(
        f"COPY (SELECT 'W1' AS work_id, 2 AS n_authors, ['A1', 'A2'] AS author_ids)"
        f" TO '{path}' (FORMAT PARQUET)"
    )
    with pytest.raises(SystemExit, match="predates the work-quality filter"):
        check_extract_schema(duckdb.connect(), str(path))
