import json

import duckdb

import pipeline.build_index as build_index
from pipeline.build_index import (
    build_id_shards,
    build_label_tiles,
    build_search_shards,
    normalize,
)


def write_web(path, rows):
    """rows: (id, name, xw, yw, cited)"""
    vals = ", ".join(
        f"('{i}', '{n}', {x}, {y}, 1, 20, {c}, 'I', 'Biology')"
        for i, n, x, y, c in rows
    )
    duckdb.sql(f"COPY (SELECT id, display_name, xw, yw, community, works_count, cited_by_count, institution, field, FALSE AS is_ring FROM (VALUES {vals}) t(id, display_name, xw, yw,"
               f" community, works_count, cited_by_count, institution, field))"
               f" TO '{path}' (FORMAT PARQUET)")


def test_normalize():
    assert normalize("Géraldine  O'Brien-Smith") == "geraldine o brien smith"
    assert normalize("李明") == "李明"          # non-latin kept verbatim (lowercased)


def test_label_tiles_ranked_and_capped(tmp_path):
    web = tmp_path / "w.parquet"
    rows = [(f"A{i}", f"Name{i}", 0.25, 0.25, 1000 - i) for i in range(60)]
    rows.append(("B0", "Far Away", 0.75, 0.75, 5))
    write_web(web, rows)
    out = tmp_path / "index"
    n = build_label_tiles(str(web), out)
    assert n > 0
    z6 = json.loads((out / "labels/6/16/47.json").read_text())  # 0.25 -> tile 16, y-flip of 16 at z6 = 47
    assert len(z6["l"]) == 50                       # capped at z6
    assert z6["l"][0][0] == "Name0"                 # highest cited first
    z9 = json.loads((out / "labels/9/128/383.json").read_text())
    assert len(z9["l"]) == 60                       # cap 200 not hit
    far = json.loads((out / "labels/6/48/15.json").read_text())
    assert far["l"][0][0] == "Far Away"


def test_label_tiles_200_cap(tmp_path):
    web = tmp_path / "w.parquet"
    # 250 researchers in one z8/z9 tile -> capped at 200 at z8, not capped at z9 (cap 4000)
    rows = [(f"C{i}", f"Name{i}", 0.25, 0.25, 5000 - i) for i in range(250)]
    write_web(web, rows)
    out = tmp_path / "index"
    build_label_tiles(str(web), out)
    z8 = json.loads((out / "labels/8/64/191.json").read_text())  # 0.25*256=64; y-flip 64@z8=191
    assert len(z8["l"]) == 200                      # capped at z8
    assert z8["l"][0][0] == "Name0"                 # highest cited first
    z9 = json.loads((out / "labels/9/128/383.json").read_text())
    assert len(z9["l"]) == 250                       # not capped at z9 (cap 4000)


def test_search_shards_cover_everyone(tmp_path):
    web = tmp_path / "w.parquet"
    write_web(web, [("A1", "Alice Zhang", 0.3, 0.3, 10),
                    ("A2", "alan turing", 0.4, 0.4, 99),
                    ("A3", "李明", 0.5, 0.5, 5)])
    out = tmp_path / "index"
    n = build_search_shards(str(web), out)
    assert n == 3
    al = json.loads((out / "search/al.json").read_text())
    assert [e[1] for e in al] == ["alan turing", "Alice Zhang"]  # cited desc
    other = json.loads((out / "search/_.json").read_text())
    assert other[0][1] == "李明"


def test_z9_cap_and_ring_excluded_from_labels(tmp_path):
    web = tmp_path / "web.parquet"
    duckdb.sql(
        "COPY (SELECT 'A' || range::VARCHAR id, 'name' || range::VARCHAR display_name,"
        " 0.5 xw, 0.5 yw, 1 community, 20 works_count,"
        " (2000000 - range)::INT cited_by_count, 'i' institution, 'Medicine' field,"
        " (range = 0) is_ring FROM range(1200))"
        f" TO '{web}' (FORMAT PARQUET)")
    n = build_label_tiles(str(web), tmp_path / "index")
    z9 = json.loads((tmp_path / "index" / "labels" / "9" / "256" / "255.json").read_text())
    assert len(z9["l"]) == 1199          # cap raised to 4000; all non-ring rows fit
    names = {e[0] for e in z9["l"]}
    assert "name0" not in names          # the ring node (highest cited) is excluded


def test_id_shards_bucket_math_and_content(tmp_path):
    web = tmp_path / "w.parquet"
    write_web(web, [
        ("https://openalex.org/A5108093963", "Geoffrey Hinton", 0.111111, 0.222222, 100),
        ("https://openalex.org/A5072532913", "Noam Chomsky", 0.333333, 0.444444, 200),
        ("https://openalex.org/AXXX", "No Digits Here", 0.5, 0.5, 1),
    ])
    out = tmp_path / "index"
    n = build_id_shards(str(web), out)
    assert n == 3
    hinton = json.loads((out / "ids/963.json").read_text())  # 5108093963 % 1000
    assert hinton["https://openalex.org/A5108093963"] == [0.111111, 0.222222]
    chomsky = json.loads((out / "ids/913.json").read_text())  # 5072532913 % 1000
    assert chomsky["https://openalex.org/A5072532913"] == [0.333333, 0.444444]
    zero = json.loads((out / "ids/0.json").read_text())  # no digits -> bucket 0
    assert zero["https://openalex.org/AXXX"] == [0.5, 0.5]


def test_id_shards_include_ring_nodes(tmp_path):
    web = tmp_path / "web.parquet"
    duckdb.sql(
        "COPY (SELECT * FROM (VALUES "
        "('https://openalex.org/A1', 'Ring Node', 0.1, 0.1, 1, 20, 5, 'i', 'Biology', TRUE), "
        "('https://openalex.org/A2', 'Normal Node', 0.2, 0.2, 1, 20, 5, 'i', 'Biology', FALSE)"
        ") t(id, display_name, xw, yw, community, works_count, cited_by_count, institution, field, is_ring))"
        f" TO '{web}' (FORMAT PARQUET)")
    out = tmp_path / "index"
    n = build_id_shards(str(web), out)
    assert n == 2                        # ring node counted, unlike label tiles
    ring_bucket = json.loads((out / "ids/1.json").read_text())
    assert "https://openalex.org/A1" in ring_bucket


def test_id_shards_clears_stale_dir(tmp_path):
    web = tmp_path / "w.parquet"
    write_web(web, [("https://openalex.org/A1", "One", 0.1, 0.1, 1)])
    out = tmp_path / "index"
    ids_dir = out / "ids"
    ids_dir.mkdir(parents=True)
    (ids_dir / "stale.json").write_text("{}")
    build_id_shards(str(web), out)
    assert not (ids_dir / "stale.json").exists()


def test_search_shards_hot_split(tmp_path, monkeypatch):
    monkeypatch.setattr(build_index, "SHARD_SPLIT_BYTES", 2000)
    rows = []
    for i in range(20):
        rows.append((f"C{i}", f"Abcresearcher Number{i:03d}", 0.1, 0.1, 100 - i))
    for i in range(20):
        rows.append((f"D{i}", f"Abdresearcher Number{i:03d}", 0.1, 0.1, 100 - i))
    rows.append(("E0", "Ab", 0.2, 0.2, 5))  # exactly-2-char normalized name
    web = tmp_path / "w.parquet"
    write_web(web, rows)
    out = tmp_path / "index"
    n = build_search_shards(str(web), out)
    assert n == len(rows)
    assert (out / "search/abc.json").exists()
    assert (out / "search/abd.json").exists()
    assert (out / "search/ab.json").exists()   # 2-char shard always written
    ab = json.loads((out / "search/ab.json").read_text())
    assert [e[1] for e in ab] == ["Ab"]         # only the exact-2-char name stays
    abc = json.loads((out / "search/abc.json").read_text())
    assert len(abc) == 20
    assert all(e[0].startswith("abc") for e in abc)
    abd = json.loads((out / "search/abd.json").read_text())
    assert len(abd) == 20
    assert all(e[0].startswith("abd") for e in abd)


def test_search_shards_hot_split_parent_written_even_when_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(build_index, "SHARD_SPLIT_BYTES", 2000)
    rows = [(f"C{i}", f"Abcresearcher Number{i:03d}", 0.1, 0.1, 100 - i) for i in range(30)]
    web = tmp_path / "w.parquet"
    write_web(web, rows)
    out = tmp_path / "index"
    build_search_shards(str(web), out)
    assert (out / "search/abc.json").exists()
    ab = json.loads((out / "search/ab.json").read_text())
    assert ab == []                             # no exact-2-char names, but still written


def test_search_shards_not_split_below_threshold(tmp_path):
    web = tmp_path / "w.parquet"
    write_web(web, [("A1", "Alice Zhang", 0.3, 0.3, 10)])
    out = tmp_path / "index"
    build_search_shards(str(web), out)
    assert not (out / "search/ali.json").exists()  # small bucket: no 3-char split
