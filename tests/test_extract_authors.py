import argparse
import json

import duckdb

from pipeline import extract_authors
from pipeline.filter_authors import filter_authors

PART = "updated_date=2026-04-08"


def write_snapshot_authors(path):
    """Authors shaped like the real snapshot: extra top-level columns and
    extra fields inside the nested institution/topic structs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    inst = ("{'id': 'I1', 'ror': 'r', 'display_name': '%s', 'country_code': 'US', "
            "'type': 'education', 'lineage': ['I1']}")
    topic = ("{'id': 'T1', 'display_name': '%s', 'count': 3, "
             "'subfield': {'id': 'S1', 'display_name': 'Sub'}, "
             "'field': {'id': 'F1', 'display_name': '%s'}}")
    duckdb.sql(f"""
        COPY (SELECT * FROM (VALUES
            ('https://openalex.org/A1', 'Ada', 'ORCID1', 40, 900,
             [{inst % 'UCSD'}, {inst % 'MIT'}],
             [{topic % ('Deep Learning', 'Computer Science')},
              {topic % ('Statistics', 'Mathematics')}]),
            ('https://openalex.org/A2', 'Bo', NULL, 3, 1,
             NULL, NULL)
        ) t(id, display_name, orcid, works_count, cited_by_count,
            last_known_institutions, topics))
        TO '{path}' (FORMAT PARQUET)
    """)


def run_stage(tmp_path):
    src = tmp_path / "authors"
    write_snapshot_authors(src / PART / "part_0000.parquet")
    manifest = {"date": "2026-09-23", "entities": [{"entity": "authors", "files": [
        {"url": f"s3://openalex/data/parquet/authors/{PART}/part_0000.parquet",
         "meta": {"content_length": 1, "record_count": 2}}]}]}
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest))
    out = tmp_path / "authors_extract"
    args = argparse.Namespace(src=str(src), out=str(out), manifest=str(mpath),
                              force=False, jobs=1)
    assert extract_authors.run(args) == 0
    return out


def test_extract_keeps_only_the_fields_the_map_reads(tmp_path):
    out = run_stage(tmp_path)
    cols = {r[0]: r[1] for r in duckdb.sql(
        f"DESCRIBE SELECT * FROM read_parquet('{out}/*/*.parquet',"
        f" hive_partitioning=false)").fetchall()}
    assert set(cols) == {"id", "display_name", "works_count", "cited_by_count",
                         "last_known_institutions", "topics"}
    assert cols["last_known_institutions"] == "STRUCT(display_name VARCHAR)[]"
    assert cols["topics"] == ("STRUCT(display_name VARCHAR, "
                              "field STRUCT(display_name VARCHAR))[]")


def test_extract_keeps_the_first_institution_and_topic(tmp_path):
    out = run_stage(tmp_path)
    rows = duckdb.sql(
        f"SELECT id, len(last_known_institutions), len(topics), "
        f"last_known_institutions[1].display_name, topics[1].display_name "
        f"FROM '{out}/*/*.parquet' ORDER BY id").fetchall()
    assert rows[0] == ("https://openalex.org/A1", 1, 1, "UCSD", "Deep Learning")
    assert rows[1][0] == "https://openalex.org/A2"
    assert rows[1][3:] == (None, None)


def test_filter_reads_the_extract_like_the_mirror(tmp_path):
    out = run_stage(tmp_path)
    nodes = tmp_path / "nodes.parquet"
    assert filter_authors(f"{out}/*/*.parquet", str(nodes), min_works=5) == 1
    row = duckdb.sql(f"SELECT id, institution, field FROM '{nodes}'").fetchone()
    assert row == ("https://openalex.org/A1", "UCSD", "Computer Science")
