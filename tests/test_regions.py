import json

import duckdb
import pytest

from pipeline.regions import (FALLBACK_TOPIC, build_regions, country_label, field_label,
                              sub_label)

TOPIC_SUBFIELD = {"Heart Failure": "Cardiology and Cardiovascular Medicine",
                  "Breast Cancer": "Oncology", "Deep Learning": "Artificial Intelligence",
                  "Compilers": "Software", "Statistics": "Statistics and Probability",
                  FALLBACK_TOPIC: "Law"}
INSTITUTION_COUNTRY = {"Kyoto U": "JP", "Osaka U": "JP", "MIT": "US", "Oxford": "GB",
                       "ETH": "CH", "Tsinghua": "CN", "Sorbonne": "FR", "Charite": "DE"}


def write_fixture(tmp_path, people):
    """people: list of dicts with id, community, field, topic, institution and
    optional sub/xw/yw/cited. Writes coords_web, authors, institutions and
    topics tables; returns the four paths/globs."""
    web = tmp_path / "coords_web.parquet"
    rows = ", ".join(
        f"('{p['id']}', 'N', {p.get('xw', 0.3)}, {p.get('yw', 0.3)}, {p['community']},"
        f" CAST({p.get('sub', -1)} AS SMALLINT), 20, {p.get('cited', 100)},"
        f" {('NULL' if p.get('institution') is None else repr(p['institution']))}, '{p['field']}', FALSE)"
        for p in people)
    duckdb.sql(f"COPY (SELECT * FROM (VALUES {rows}) t(id, display_name, xw, yw, community, sub,"
               f" works_count, cited_by_count, institution, field, is_ring)) TO '{web}' (FORMAT PARQUET)")
    authors = tmp_path / "authors" / "updated_date=2026-01-01"
    authors.mkdir(parents=True)
    tvals = ", ".join(f"('{p['id']}', [{{'display_name': '{p['topic']}'}}])" for p in people)
    duckdb.sql(f"COPY (SELECT * FROM (VALUES {tvals}) t(id, topics))"
               f" TO '{authors / 'part_0000.parquet'}' (FORMAT PARQUET)")
    inst = tmp_path / "institutions" / "d=1"
    inst.mkdir(parents=True)
    ivals = ", ".join(f"('{n}', '{c}')" for n, c in INSTITUTION_COUNTRY.items())
    duckdb.sql(f"COPY (SELECT * FROM (VALUES {ivals}) t(display_name, country_code))"
               f" TO '{inst / 'part_0000.parquet'}' (FORMAT PARQUET)")
    top = tmp_path / "topics" / "d=1"
    top.mkdir(parents=True)
    tv = ", ".join(f"('{t}', {{'display_name': '{s}'}})" for t, s in TOPIC_SUBFIELD.items())
    duckdb.sql(f"COPY (SELECT * FROM (VALUES {tv}) t(display_name, subfield))"
               f" TO '{top / 'part_0000.parquet'}' (FORMAT PARQUET)")
    return (str(web), str(tmp_path / "authors" / "*" / "*.parquet"),
            str(tmp_path / "institutions" / "*" / "*.parquet"), str(tmp_path / "topics" / "*" / "*.parquet"))


def people(prefix, community, n, field, topic, institutions, **extra):
    return [{"id": f"{prefix}{i}", "community": community, "field": field, "topic": topic,
             "institution": institutions[i % len(institutions)], **extra} for i in range(n)]


def build(tmp_path, ppl, **kw):
    web, authors, inst, topics = write_fixture(tmp_path, ppl)
    out = tmp_path / "regions.json"
    build_regions(web, authors, str(out), inst, topics, **kw)
    return json.loads(out.read_text())


def test_field_first_then_country_only_when_the_country_explains_the_cluster(tmp_path):
    # Japan is ~5% of the map but nearly all of community 2; community 1 is
    # US-heavy, yet no more than the US is across the whole map
    intl = ["MIT", "MIT", "Oxford", "ETH", "Tsinghua", "Sorbonne", "Charite"]
    ppl = (people("I", 1, 500, "Computer Science", "Deep Learning", intl)
           + people("J", 2, 40, "Medicine", "Heart Failure", ["Kyoto U", "Osaka U"]))
    regions = build(tmp_path, ppl, top_n=10, keep=10)
    by = {r["community"]: r for r in regions}
    assert by[2]["name"] == "Medicine (Japan)" and by[2]["kind"] == "national"
    assert by[1]["name"] == "Computer Science" and by[1]["kind"] == "field"


def test_fallback_topic_records_do_not_count_toward_the_field(tmp_path):
    # 40 of 70 members carry OpenAlex's fallback topic (field Social Sciences);
    # the rest are engineers, which is what the community actually is
    ppl = (people("F", 1, 40, "Social Sciences", FALLBACK_TOPIC, ["MIT", "Oxford"])
           + people("E", 1, 30, "Engineering", "Compilers", ["MIT", "Oxford"]))
    regions = build(tmp_path, ppl, top_n=10, keep=10)
    assert regions[0]["name"] == "Engineering"


def test_near_empty_communities_get_no_name(tmp_path):
    ppl = (people("A", 1, 30, "Medicine", "Heart Failure", ["MIT"])
           + people("Z", 2, 20, "Social Sciences", FALLBACK_TOPIC, [None], cited=0))
    regions = build(tmp_path, ppl, top_n=10, keep=10, fade_min=10)
    assert [r["community"] for r in regions] == [1]


def test_repeated_names_are_made_unique_with_a_topic(tmp_path):
    ppl = (people("A", 1, 40, "Medicine", "Heart Failure", ["MIT", "Oxford", "ETH", "Sorbonne", "Charite"])
           + people("B", 2, 30, "Medicine", "Breast Cancer", ["MIT", "Oxford", "ETH", "Sorbonne", "Charite"]))
    names = [r["name"] for r in build(tmp_path, ppl, top_n=10, keep=10)]
    assert names == ["Medicine", "Medicine: Breast Cancer"]


def test_sub_areas_named_by_subfield_and_by_a_country_other_than_the_regions(tmp_path):
    us = ["MIT"]
    ppl = (people("C", 1, 40, "Medicine", "Heart Failure", us, sub=0, xw=0.30)
           + people("O", 1, 40, "Medicine", "Breast Cancer", us, sub=1, xw=0.32)
           + people("U", 1, 15, "Medicine", "Heart Failure", ["Oxford"], sub=2, xw=0.34)   # the region's own mix,
           + people("V", 1, 15, "Medicine", "Breast Cancer", ["Oxford"], sub=2, xw=0.34)   # but British
           + people("X", 1, 30, "Medicine", "Statistics", us, sub=3, xw=0.36))
    regions = build(tmp_path, ppl, top_n=10, keep=10)
    subs = {r["sub"]: r for r in regions if r.get("level") == 2}
    assert subs[0]["name"] == "Cardiology"
    assert subs[1]["name"] == "Oncology"
    assert subs[2]["name"] == "Medicine (UK)"      # region's own country (US) never named
    assert subs[2]["members"] == 30 and subs[2]["xw"] == pytest.approx(0.34)
    assert subs[0]["community"] == 1 and 3.5 <= subs[0]["zmin"] <= 8


def test_zoom_bands_and_level_one_fields(tmp_path):
    ppl = []
    for k in range(1, 36):
        ppl += people(f"P{k}_", k, 36 - k, "Medicine", "Heart Failure", ["MIT"], xw=0.01 * k, yw=0.5)
    regions = build(tmp_path, ppl, top_n=100, keep=100)
    by_rank = {r["rank"]: r for r in regions}
    assert by_rank[31]["zmin"] == 4 and by_rank[31]["zmax"] == 6
    assert by_rank[1]["zmin"] == 2 and by_rank[1]["zmax"] == 4
    r = by_rank[1]
    assert set(r) == {"name", "kind", "xw", "yw", "spread", "members", "rank", "community", "zmin", "zmax"}
    assert r["community"] == 1 and r["members"] == 35
    assert r["xw"] == pytest.approx(0.01) and r["yw"] == pytest.approx(0.5)


def test_missing_inputs_fail_clearly(tmp_path):
    web, authors, inst, topics = write_fixture(tmp_path, people("A", 1, 3, "Medicine", "Heart Failure", ["MIT"]))
    with pytest.raises(SystemExit, match="authors snapshot"):
        build_regions(web, str(tmp_path / "nope" / "*.parquet"), str(tmp_path / "r.json"), inst, topics)
    with pytest.raises(SystemExit, match="institutions table"):
        build_regions(web, authors, str(tmp_path / "r.json"), str(tmp_path / "nope" / "*.parquet"), topics)


def test_defaults_follow_mapademic_data(tmp_path, monkeypatch):
    import argparse

    from pipeline import regions
    web, *_ = write_fixture(tmp_path, people("A", 1, 3, "Medicine", "Heart Failure", ["MIT"]))
    snap = tmp_path / "data" / "snapshot"
    snap.mkdir(parents=True)
    for name in ("authors", "institutions", "topics"):
        (tmp_path / name).rename(snap / name)
    monkeypatch.setenv("MAPADEMIC_DATA", str(tmp_path / "data"))
    p = argparse.ArgumentParser()
    regions.add_parser(p)
    args = p.parse_args(["--web", web, "--out", str(tmp_path / "r.json"), "--top-n", "10", "--keep", "10"])
    assert args.authors is None and args.institutions is None
    assert regions.run(args) == 0
    assert json.loads((tmp_path / "r.json").read_text())[0]["name"] == "Medicine"


def test_label_rules():
    assert field_label([("Medicine", 0.57), ("Biochemistry, Genetics and Molecular Biology", 0.14)]) == "Medicine"
    assert field_label([("Engineering", 0.25), ("Medicine", 0.18)]) == "Multidisciplinary"
    assert field_label([("Agricultural and Biological Sciences", 0.38), ("Environmental Science", 0.27)]) \
        == "Biology, Agriculture & Environmental Science"
    assert country_label([("US", 0.50, 2.5), ("GB", 0.10, 2.4)]) is None          # US-led but not national
    assert country_label([("ES", 0.59, 23.0), ("MX", 0.10, 14.6), ("AR", 0.07, 14.4)]) == "Spain, Mexico, Argentina"
    assert country_label([("NG", 0.10, 30.0), ("TH", 0.10, 27.0)]) is None        # spread over countries
    assert sub_label([("Dentistry", 0.03, 15.0)], [("US", 0.8)], "US", "Medicine", None) == "Dentistry"
    assert sub_label([], [("SE", 0.38), ("DK", 0.26), ("NO", 0.16)], "US", "Medicine", None) \
        == "Medicine (Sweden, Denmark, Norway)"
    assert sub_label([], [("US", 0.9)], "US", "Medicine", "Health Policy") == "Health Policy"
