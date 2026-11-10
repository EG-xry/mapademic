import argparse
import json

import duckdb
import pyarrow.fs as pafs
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipeline import extract_works
from tests.conftest import write_works_partition

A = [f"https://openalex.org/A{i}" for i in range(4)]
PART1, PART2 = "updated_date=2026-01-01", "updated_date=2026-02-01"


def make_src(tmp_path):
    src = tmp_path / "works"
    write_works_partition(
        src / PART1,
        [("https://openalex.org/W1", 2, [A[1], A[2]]),
         ("https://openalex.org/W2", 1, [A[1]],
          {"type": "dataset", "is_xpac": True, "publication_year": 2019,
           "cited_by_count": 7})],
    )
    write_works_partition(
        src / PART2,
        [("https://openalex.org/W3", 3, [A[1], A[2], A[3]])],
    )
    return src


def extract_one(src, out, partition=PART1, expected_rows=None):
    con = duckdb.connect()
    return extract_works.extract_file(
        con, pafs.LocalFileSystem(), str(src / partition / "part_0000.parquet"),
        out, expected_rows)


def test_extract_file_writes_integer_ids(tmp_path):
    src = make_src(tmp_path)
    out = tmp_path / "out" / PART1 / "part_0000.parquet"
    assert extract_one(src, out) is True
    rows = duckdb.sql(
        f"SELECT work_id, n_authors, author_ids FROM '{out}' ORDER BY work_id"
    ).fetchall()
    assert rows == [(1, 2, [1, 2]), (2, 1, [1])]


def test_extract_carries_the_quality_filter_columns(tmp_path):
    src = make_src(tmp_path)
    out = tmp_path / "out.parquet"
    extract_one(src, out)
    row = duckdb.sql(
        f"SELECT year, type, is_xpac, is_paratext, is_retracted, cited_by_count "
        f"FROM '{out}' WHERE work_id = 2"
    ).fetchone()
    assert row == (2019, "dataset", True, False, False, 7)


def test_existing_checkpoint_is_skipped(tmp_path):
    src = make_src(tmp_path)
    out = tmp_path / "out.parquet"
    assert extract_one(src, out) is True
    assert extract_one(src, out) is False


def test_extract_retains_deposited_authorship_evidence_not_profile_orcid(tmp_path):
    src = tmp_path / "enriched.parquet"
    pq.write_table(pa.Table.from_pylist([{
        "id": "https://openalex.org/W1", "authors_count": 2, "publication_year": 2022,
        "type": "article", "is_xpac": False, "is_paratext": False, "is_retracted": False,
        "cited_by_count": 5, "doi": "https://doi.org/10.1234/test", "title": "A real collaboration",
        "primary_topic": {"field": {"id": "https://openalex.org/fields/33"}},
        "authorships": [{"author": {"id": A[1], "orcid": "wrong-profile-value"},
                         "raw_author_name": "Paul Miller", "raw_orcid": "0000-0003-0443-3957",
                         "institutions": [{"id": "https://openalex.org/I1"}]},
                        {"author": {"id": A[2], "orcid": None}, "raw_author_name": "Another Person",
                         "raw_orcid": None, "institutions": []}],
    }]), src)
    out = tmp_path / "out.parquet"
    extract_works.extract_file(duckdb.connect(), pafs.LocalFileSystem(), str(src), out, 1)
    row = pq.read_table(out).to_pylist()[0]
    assert row["doi"] == "https://doi.org/10.1234/test"
    assert row["title"] == "A real collaboration"
    assert row["authorship_evidence"][0] == {
        "author_id": 1, "raw_name": "Paul Miller", "raw_orcid": "0000-0003-0443-3957",
        "institution_ids": ["https://openalex.org/I1"], "field_id": "https://openalex.org/fields/33",
        "institution_names": [], "field_name": None,
    }
    assert row["authorship_evidence"][1]["raw_orcid"] is None


def test_sparse_snapshot_still_extracts_without_inventing_identity_evidence(tmp_path):
    src = make_src(tmp_path)
    out = tmp_path / "out.parquet"
    extract_one(src, out)
    evidence = pq.read_table(out).to_pylist()[0]["authorship_evidence"]
    assert all(e["raw_name"] is None and e["raw_orcid"] is None for e in evidence)


def test_row_count_mismatch_fails_without_a_checkpoint(tmp_path):
    # the manifest says 5 rows, the file holds 2: a truncated or
    # mixed-release source must not become a trusted checkpoint
    src = make_src(tmp_path)
    out = tmp_path / "out" / "x.parquet"
    with pytest.raises(RuntimeError, match="expected 5 rows"):
        extract_one(src, out, expected_rows=5)
    assert not out.exists()
    assert not list((tmp_path / "out").glob("*"))


def run_stage(tmp_path, src, manifest_rows, jobs=2):
    """Drive the stage's run() against a local source and a hand-built manifest."""
    manifest = {"date": "2026-09-23", "entities": [{"entity": "works", "files": [
        {"url": f"s3://openalex/data/parquet/works/{part}/part_0000.parquet",
         "meta": {"content_length": 1, "record_count": rows}}
        for part, rows in manifest_rows.items()]}]}
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest))
    out = tmp_path / "extract"
    args = argparse.Namespace(src=str(src), out=str(out), manifest=str(mpath),
                              force=False, jobs=jobs)
    return extract_works.run(args), out


def test_run_extracts_every_manifest_file(tmp_path):
    src = make_src(tmp_path)
    rc, out = run_stage(tmp_path, src, {PART1: 2, PART2: 1})
    assert rc == 0
    assert sorted(p.relative_to(out).as_posix() for p in out.glob("*/*.parquet")) == [
        f"{PART1}/part_0000.parquet", f"{PART2}/part_0000.parquet"]
    assert (out / "RELEASE").read_text().strip() == "2026-09-23"
    assert not list(out.rglob("*.tmp"))


def test_run_fails_on_a_file_whose_count_disagrees(tmp_path):
    src = make_src(tmp_path)
    with pytest.raises(RuntimeError, match="expected 9 rows"):
        run_stage(tmp_path, src, {PART1: 2, PART2: 9})


def test_run_refuses_files_from_another_release(tmp_path):
    src = make_src(tmp_path)
    run_stage(tmp_path, src, {PART1: 2, PART2: 1})
    # the next release drops PART2: its old extract must not linger unnoticed
    with pytest.raises(SystemExit, match="not in release"):
        run_stage(tmp_path, src, {PART1: 2})
