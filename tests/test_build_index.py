import json

import duckdb
import pytest

from pipeline.build_index import build_label_tiles, build_search_shards, normalize


def write_web(path, rows):
    """rows: (id, name, xw, yw, cited)"""
    vals = ", ".join(
        f"('{i}', '{n}', {x}, {y}, 1, 20, {c}, 'I', 'Biology')"
        for i, n, x, y, c in rows
    )
    duckdb.sql(f"COPY (SELECT * FROM (VALUES {vals}) t(id, display_name, xw, yw,"
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
