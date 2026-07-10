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


def make_many_communities_fixture(tmp_path, n_comms=35):
    """n_comms communities of strictly descending size (community k has
    n_comms + 1 - k members) so ranks are deterministic and at least one
    community lands at rank > 30."""
    web = tmp_path / "coords_web.parquet"
    web_rows = []
    topic_rows = []
    for k in range(1, n_comms + 1):
        size = n_comms + 1 - k          # community 1 biggest, community n_comms smallest
        for j in range(size):
            aid = f"A{k}_{j}"
            # each community clustered at its own x; y fixed so yw/spread are checkable
            web_rows.append((aid, 0.01 * k, 0.5, k))
            topic_rows.append((aid, f"Topic{k}"))   # each community's own distinctive topic
    vals = ", ".join(
        f"('{i}', 'N', {x}, {y}, {c}, 20, 100, 'I', 'Biology')"
        for i, x, y, c in web_rows
    )
    duckdb.sql(f"COPY (SELECT * FROM (VALUES {vals}) t(id, display_name, xw, yw,"
               f" community, works_count, cited_by_count, institution, field))"
               f" TO '{web}' (FORMAT PARQUET)")
    authors = tmp_path / "authors" / "updated_date=2026-01-01"
    authors.mkdir(parents=True)
    tvals = ", ".join(
        f"('{i}', [{{'display_name': '{t}'}}])" for i, t in topic_rows
    )
    duckdb.sql(f"COPY (SELECT * FROM (VALUES {tvals}) t(id, topics))"
               f" TO '{authors / 'part_0000.parquet'}' (FORMAT PARQUET)")
    return str(web), str(tmp_path / "authors" / "*" / "*.parquet")


def test_zoom_bands_and_full_field_set(tmp_path):
    web, authors = make_many_communities_fixture(tmp_path, n_comms=35)
    out = tmp_path / "regions.json"
    n = build_regions(web, authors, str(out), top_n=100, keep=100)
    regions = json.loads(out.read_text())
    assert n == len(regions) == 35
    by_rank = {r["rank"]: r for r in regions}

    # a rank>30 region falls in the deep zoom band
    deep = by_rank[31]
    assert deep["zmin"] == 4 and deep["zmax"] == 6

    # a rank<=30 region stays in the shallow band
    assert by_rank[1]["zmin"] == 2 and by_rank[1]["zmax"] == 4

    # full field set present and correct for one region (rank 1 = community 1)
    r = by_rank[1]
    assert set(r) == {"name", "xw", "yw", "spread", "members", "rank",
                      "community", "zmin", "zmax"}
    assert r["community"] == 1
    assert r["name"] == "Topic1"
    assert r["members"] == 35
    assert r["yw"] == pytest.approx(0.5, abs=1e-9)
    assert r["xw"] == pytest.approx(0.01, abs=1e-9)
    assert r["spread"] >= 0.0
