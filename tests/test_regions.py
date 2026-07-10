import json

import duckdb
import pytest

from pipeline.regions import build_regions


def make_fixture(tmp_path):
    web = tmp_path / "coords_web.parquet"
    # community 1: 3 authors clustered near (0.3, 0.3); community 2: 2 authors
    rows = [
        ("A1", 0.30, 0.30, 1), ("A2", 0.31, 0.30, 1), ("A3", 0.30, 0.31, 1),
        ("B1", 0.70, 0.70, 2), ("B2", 0.71, 0.70, 2),
    ]
    vals = ", ".join(
        f"('{i}', 'N', {x}, {y}, {c}, 20, 100, 'I', 'Biology')" for i, x, y, c in rows
    )
    duckdb.sql(f"COPY (SELECT * FROM (VALUES {vals}) t(id, display_name, xw, yw,"
               f" community, works_count, cited_by_count, institution, field))"
               f" TO '{web}' (FORMAT PARQUET)")
    authors = tmp_path / "authors" / "updated_date=2026-01-01"
    authors.mkdir(parents=True)
    # 'Genome Editing' distinctive to community 1; 'Deep Learning' to 2;
    # 'Statistics' ubiquitous (appears in both -> low distinctiveness)
    topic_rows = [
        ("A1", "Genome Editing"), ("A2", "Genome Editing"), ("A3", "Statistics"),
        ("B1", "Deep Learning"), ("B2", "Statistics"),
    ]
    tvals = ", ".join(
        f"('{i}', [{{'display_name': '{t}'}}])" for i, t in topic_rows
    )
    duckdb.sql(f"COPY (SELECT * FROM (VALUES {tvals}) t(id, topics))"
               f" TO '{authors / 'part_0000.parquet'}' (FORMAT PARQUET)")
    return str(web), str(tmp_path / "authors" / "*" / "*.parquet")


def test_distinctive_names_and_geometry(tmp_path):
    web, authors = make_fixture(tmp_path)
    out = tmp_path / "regions.json"
    n = build_regions(web, authors, str(out), top_n=10, keep=10)
    regions = json.loads(out.read_text())
    assert n == len(regions) == 2
    by_comm = {r["community"]: r for r in regions}
    assert by_comm[1]["name"] == "Genome Editing"     # not ubiquitous Statistics
    assert by_comm[2]["name"] == "Deep Learning"
    assert by_comm[1]["members"] == 3
    assert by_comm[1]["xw"] == pytest.approx(0.3033, abs=1e-3)
    assert by_comm[1]["rank"] == 1                     # biggest first
    assert by_comm[1]["zmin"] == 2 and by_comm[2]["zmin"] == 2


def test_missing_authors_glob_fails_clearly(tmp_path):
    web, _ = make_fixture(tmp_path)
    with pytest.raises(SystemExit, match="authors snapshot"):
        build_regions(web, str(tmp_path / "nope" / "*.parquet"),
                      str(tmp_path / "r.json"))
