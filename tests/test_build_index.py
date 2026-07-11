import json
from collections import Counter

import duckdb

import pipeline.build_index as build_index
from pipeline.build_index import (
    build_community_boundaries,
    build_community_shards,
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


def write_web_communities(path, rows):
    """rows: (id, name, xw, yw, community, cited, is_ring)"""
    vals = ", ".join(
        f"('{i}', '{n}', {x}, {y}, {comm}, 20, {c}, 'I', 'Biology', {str(ring).upper()})"
        for i, n, x, y, comm, c, ring in rows
    )
    duckdb.sql(
        f"COPY (SELECT id, display_name, xw, yw, community, works_count, cited_by_count, "
        f"institution, field, is_ring FROM (VALUES {vals}) "
        f"t(id, display_name, xw, yw, community, works_count, cited_by_count, institution, field, is_ring))"
        f" TO '{path}' (FORMAT PARQUET)"
    )


def assert_search_shards_lossless(out_dir, rows):
    """Union of unique ids across all search shards equals the input ids;
    each id appears once (single-token names, or first token == last token)
    or twice (dual first-token/last-token bucketing)."""
    seen = []
    for p in (out_dir / "search").glob("*.json"):
        for e in json.loads(p.read_text()):
            seen.append(e[2])                    # author id
    counts = Counter(seen)
    assert set(counts) == {r[0] for r in rows}
    assert all(1 <= c <= 2 for c in counts.values())


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
    assert len(z6["l"][0]) == 6                      # [display_name, id, xw, yw, cited, community]
    assert z6["l"][0][5] == 1                        # community round-trips (write_web hardcodes 1)
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


def test_label_tiles_community_round_trips(tmp_path):
    web = tmp_path / "web.parquet"
    duckdb.sql(
        "COPY (SELECT * FROM (VALUES "
        "('A1', 'Alice', 0.25, 0.25, 7, 20, 100, 'I', 'Biology', FALSE), "
        "('A2', 'Bob', 0.25, 0.25, 3, 20, 50, 'I', 'Biology', FALSE)"
        ") t(id, display_name, xw, yw, community, works_count, cited_by_count, institution, field, is_ring))"
        f" TO '{web}' (FORMAT PARQUET)")
    out = tmp_path / "index"
    build_label_tiles(str(web), out)
    z6 = json.loads((out / "labels/6/16/47.json").read_text())
    by_id = {e[1]: e for e in z6["l"]}
    assert by_id["A1"][5] == 7                       # known fixture row: Alice's community
    assert by_id["A2"][5] == 3                       # known fixture row: Bob's community


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
    assert_search_shards_lossless(out, rows)


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
    assert_search_shards_lossless(out, rows)


def test_search_shards_not_split_below_threshold(tmp_path):
    web = tmp_path / "w.parquet"
    write_web(web, [("A1", "Alice Zhang", 0.3, 0.3, 10)])
    out = tmp_path / "index"
    build_search_shards(str(web), out)
    assert not (out / "search/ali.json").exists()  # small bucket: no 3-char split


def test_search_shards_recursive_split_to_max_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(build_index, "SHARD_SPLIT_BYTES", 500)
    rows = [(f"C{i}", f"Abcderesearcher Number{i:03d}", 0.1, 0.1, 500 - i) for i in range(40)]
    rows.append(("E2", "Ab", 0.2, 0.2, 5))      # exactly-2-char normalized name
    rows.append(("E3", "Abc", 0.2, 0.2, 5))     # exactly-3-char
    rows.append(("E4", "Abcd", 0.2, 0.2, 5))    # exactly-4-char
    web = tmp_path / "w.parquet"
    write_web(web, rows)
    out = tmp_path / "index"
    n = build_search_shards(str(web), out)
    assert n == len(rows)
    # every prefix level 2..5 written; exactly-K-char names stay at level K
    ab = json.loads((out / "search/ab.json").read_text())
    assert [e[1] for e in ab] == ["Ab"]
    abc = json.loads((out / "search/abc.json").read_text())
    assert [e[1] for e in abc] == ["Abc"]
    abcd = json.loads((out / "search/abcd.json").read_text())
    assert [e[1] for e in abcd] == ["Abcd"]
    abcde = json.loads((out / "search/abcde.json").read_text())
    assert len(abcde) == 40                     # oversized but capped at MAX_PREFIX_LEN
    assert not (out / "search/abcder.json").exists()
    assert_search_shards_lossless(out, rows)


def test_search_shards_catchall_codepoint_split(tmp_path, monkeypatch):
    monkeypatch.setattr(build_index, "SHARD_SPLIT_BYTES", 500)
    rows = [(f"L{i}", f"李研究者{i:03d}号", 0.1, 0.1, 100 - i) for i in range(20)]
    rows.append(("W0", "王五", 0.2, 0.2, 50))
    rows.append(("X0", "!!!", 0.3, 0.3, 1))     # normalizes to empty -> stays in _
    web = tmp_path / "w.parquet"
    write_web(web, rows)
    out = tmp_path / "index"
    n = build_search_shards(str(web), out)
    assert n == len(rows)
    li = json.loads((out / f"search/_{ord('李') % build_index.CATCHALL_BUCKETS}.json").read_text())
    assert len(li) == 20
    wang = json.loads((out / f"search/_{ord('王') % build_index.CATCHALL_BUCKETS}.json").read_text())
    assert [e[2] for e in wang] == ["W0"]
    catch = json.loads((out / "search/_.json").read_text())  # parent always written
    assert [e[2] for e in catch] == ["X0"]      # empty-norm entry stays in _
    assert_search_shards_lossless(out, rows)


def test_search_shards_catchall_not_split_below_threshold(tmp_path):
    web = tmp_path / "w.parquet"
    rows = [("A3", "李明", 0.5, 0.5, 5)]
    write_web(web, rows)
    out = tmp_path / "index"
    build_search_shards(str(web), out)
    assert (out / "search/_.json").exists()
    assert not (out / f"search/_{ord('李') % build_index.CATCHALL_BUCKETS}.json").exists()
    assert_search_shards_lossless(out, rows)


def test_search_shards_dual_token_family(tmp_path):
    rows = [("S1", "Terrence J. Sejnowski", 0.1, 0.1, 994)]
    web = tmp_path / "w.parquet"
    write_web(web, rows)
    out = tmp_path / "index"
    n = build_search_shards(str(web), out)
    assert n == 1
    te = json.loads((out / "search/te.json").read_text())
    se = json.loads((out / "search/se.json").read_text())
    assert [e[1] for e in te] == ["Terrence J. Sejnowski"]
    assert [e[1] for e in se] == ["Terrence J. Sejnowski"]
    assert te[0] == se[0]                        # same entry: full spaced norm, not rotated
    assert te[0][0] == "terrence j sejnowski"
    assert_search_shards_lossless(out, rows)


def test_search_shards_single_token_not_duplicated(tmp_path):
    rows = [("A1", "Prince", 0.1, 0.1, 10)]
    web = tmp_path / "w.parquet"
    write_web(web, rows)
    out = tmp_path / "index"
    n = build_search_shards(str(web), out)
    assert n == 1
    pr = json.loads((out / "search/pr.json").read_text())
    assert len(pr) == 1
    assert_search_shards_lossless(out, rows)


def test_search_shards_first_equals_last_not_duplicated(tmp_path):
    rows = [("A1", "Anna Anna", 0.1, 0.1, 10)]
    web = tmp_path / "w.parquet"
    write_web(web, rows)
    out = tmp_path / "index"
    build_search_shards(str(web), out)
    an = json.loads((out / "search/an.json").read_text())
    assert len(an) == 1
    assert_search_shards_lossless(out, rows)


def test_search_shards_initial_last_token_skipped(tmp_path):
    rows = [("Z1", "Zhang Y", 0.1, 0.1, 50)]
    web = tmp_path / "w.parquet"
    write_web(web, rows)
    out = tmp_path / "index"
    n = build_search_shards(str(web), out)
    assert n == 1
    zh = json.loads((out / "search/zh.json").read_text())
    assert [e[2] for e in zh] == ["Z1"]           # findable via the first-token family
    assert not (out / "search/_.json").exists()   # last token "y" (1 char) never routes to catch-all
    assert_search_shards_lossless(out, rows)       # id appears exactly once


def test_search_shards_recursive_split_surname_family(tmp_path, monkeypatch):
    monkeypatch.setattr(build_index, "SHARD_SPLIT_BYTES", 500)
    rows = [(f"S{i}", f"Xt{i} Sejnowski{i:03d}", 0.1, 0.1, 500 - i) for i in range(40)]
    rows.append(("SE0", "Yy Se", 0.2, 0.2, 5))   # last token exactly "se" (2 chars): stays at parent
    web = tmp_path / "w.parquet"
    write_web(web, rows)
    out = tmp_path / "index"
    n = build_search_shards(str(web), out)
    assert n == len(rows)
    se = json.loads((out / "search/se.json").read_text())
    assert [e[2] for e in se] == ["SE0"]         # only the exact-2-char surname stays at the parent
    sejno = json.loads((out / "search/sejno.json").read_text())
    assert len(sejno) == 40                      # split key came from the surname, capped at MAX_PREFIX_LEN
    assert all("sejnowski" in e[0] for e in sejno)
    assert_search_shards_lossless(out, rows)


def test_search_shards_catchall_last_token_nonascii(tmp_path, monkeypatch):
    monkeypatch.setattr(build_index, "SHARD_SPLIT_BYTES", 500)
    rows = [(f"J{i}", f"John{i} 李{i:03d}", 0.1, 0.1, 100 - i) for i in range(20)]
    web = tmp_path / "w.parquet"
    write_web(web, rows)
    out = tmp_path / "index"
    n = build_search_shards(str(web), out)
    assert n == len(rows)
    li = json.loads((out / f"search/_{ord('李') % build_index.CATCHALL_BUCKETS}.json").read_text())
    assert len(li) == 20
    assert {e[2] for e in li} == {f"J{i}" for i in range(20)}
    assert_search_shards_lossless(out, rows)


def test_smooth_majority_kills_speckle_and_keeps_ties():
    import numpy as np

    from pipeline.build_index import _smooth_majority

    # 3x3 grid: a lone community-9 speckle surrounded by community-0.
    arr = np.zeros((3, 3), dtype=np.int32)
    arr[1, 1] = 9
    out = _smooth_majority(arr)
    assert out[1, 1] == 0                        # speckle absorbed by majority neighbor

    # Three distinct single-count neighbors (1, 2, and the center's own 5)
    # tie for best_count=1 -> center is kept unchanged.
    tie = np.array([
        [1, 2, -1],
        [-1, 5, -1],
        [-1, -1, -1],
    ], dtype=np.int32)
    out_tie = _smooth_majority(tie)
    assert out_tie[1, 1] == 5                     # tie among 1, 2, 5 -> center kept
    assert out_tie[2, 0] == -1                    # empty cells stay empty


def test_boundaries_straight_line_between_two_communities(tmp_path):
    web = tmp_path / "web.parquet"
    rows = []
    aid = 0
    for cy in range(4):
        for cx in range(4):
            comm = 0 if cx < 2 else 1
            x = (cx + 0.5) / 4
            y = (cy + 0.5) / 4
            rows.append((f"A{aid}", f"N{aid}", x, y, comm, 10, False))
            aid += 1
    write_web_communities(web, rows)
    out = tmp_path / "boundaries.json"
    n = build_community_boundaries(str(web), str(out), grid=4, min_cell=1)
    assert n == 1                                 # one merged run, not four separate segments
    geo = json.loads(out.read_text())
    assert geo["type"] == "FeatureCollection"
    assert len(geo["features"]) == 1
    geom = geo["features"][0]["geometry"]
    assert geom["type"] == "MultiLineString"
    assert geom["coordinates"] == [[[0.5, 0.0], [0.5, 1.0]]]


def test_boundaries_speckle_under_min_cell_produces_none(tmp_path):
    web = tmp_path / "web.parquet"
    rows = [
        ("A0", "N0", 0.15, 0.15, 0, 10, False),   # cell (0,0): 3 authors, community 0
        ("A1", "N1", 0.15, 0.15, 0, 9, False),
        ("A2", "N2", 0.15, 0.15, 0, 8, False),
        ("A3", "N3", 0.4, 0.15, 1, 5, False),     # cell (1,0): 1 author, community 1 -> under min_cell
    ]
    write_web_communities(web, rows)
    out = tmp_path / "boundaries.json"
    n = build_community_boundaries(str(web), str(out), grid=4, min_cell=3)
    assert n == 0
    geo = json.loads(out.read_text())
    assert geo["features"][0]["geometry"]["coordinates"] == []


def test_boundaries_fragment_filter_drops_small_blob_keeps_long_boundary(tmp_path):
    web = tmp_path / "web.parquet"
    grid = 20
    rows = []
    aid = 0
    # Long straight boundary: populate the two columns adjacent to the split
    # (cx=9 community 0, cx=10 community 1) across every row, so run-length
    # merging produces one full-height segment.
    for cy in range(grid):
        for cx, comm in ((9, 0), (10, 1)):
            x = (cx + 0.5) / grid
            y = (cy + 0.5) / grid
            rows.append((f"A{aid}", f"N{aid}", x, y, comm, 10, False))
            aid += 1
    # Small isolated blob: one community-2 cell deep inside community 0's
    # territory, boxed in by its 4 orthogonal neighbors so it forms a closed
    # 4-edge ring (perimeter 4 * 1/grid = 0.2, under the 0.25 min_len below).
    blob = (3, 3)
    neighbors = [(2, 3), (4, 3), (3, 2), (3, 4)]
    for cx, cy in [blob] + neighbors:
        comm = 2 if (cx, cy) == blob else 0
        x = (cx + 0.5) / grid
        y = (cy + 0.5) / grid
        rows.append((f"A{aid}", f"N{aid}", x, y, comm, 10, False))
        aid += 1
    write_web_communities(web, rows)
    out = tmp_path / "boundaries.json"
    # smooth_passes=0 isolates the fragment-filter behavior from majority-
    # filter erosion (a lone cell would otherwise be absorbed by smoothing
    # before ever reaching the fragment filter).
    n = build_community_boundaries(
        str(web), str(out), grid=grid, min_cell=1, smooth_passes=0, min_len=0.25
    )
    assert n == 1                                  # blob ring dropped, long boundary kept
    geo = json.loads(out.read_text())
    lines = geo["features"][0]["geometry"]["coordinates"]
    assert len(lines) == 1
    assert lines[0] == [[0.5, 0.0], [0.5, 1.0]]


def test_chaikin_smooth_curves_staircase_and_keeps_endpoints():
    from pipeline.build_index import _chaikin_smooth

    staircase = [[0, 0], [1, 0], [1, 1], [2, 1], [2, 2]]
    out = _chaikin_smooth(staircase, iterations=2)
    assert len(out) > len(staircase)
    assert out[0] == [0, 0]
    assert out[-1] == [2, 2]
    xs = [p[0] for p in staircase]
    ys = [p[1] for p in staircase]
    # loose hull sanity: every output point stays within the input's bbox
    # (a strict superset of its convex hull)
    assert all(min(xs) <= p[0] <= max(xs) for p in out)
    assert all(min(ys) <= p[1] <= max(ys) for p in out)


def test_community_shards_top_n_ordering_and_min_members(tmp_path):
    web = tmp_path / "web.parquet"
    rows = []
    for i in range(5):
        rows.append((f"A{i}", f"Name{i}", 0.1 + i * 0.01, 0.1, 0, 100 - i, False))
    rows.append(("B0", "Beta0", 0.5, 0.5, 1, 999, False))
    rows.append(("B1", "Beta1", 0.5, 0.5, 1, 998, False))
    write_web_communities(web, rows)
    out = tmp_path / "index"
    n = build_community_shards(str(web), out, top_n=3, min_members=3)
    assert n == 1                                 # only community 0 has >= 3 members
    assert not (out / "communities" / "1.json").exists()
    c0 = json.loads((out / "communities" / "0.json").read_text())
    assert len(c0) == 3                           # top_n cap
    assert [e[4] for e in c0] == [100, 99, 98]     # cited desc
    assert [e[0] for e in c0] == ["Name0", "Name1", "Name2"]


def test_community_shards_clears_stale_dir(tmp_path):
    web = tmp_path / "web.parquet"
    rows = [(f"A{i}", f"Name{i}", 0.1, 0.1, 0, 10, False) for i in range(3)]
    write_web_communities(web, rows)
    out = tmp_path / "index"
    communities_dir = out / "communities"
    communities_dir.mkdir(parents=True)
    (communities_dir / "stale.json").write_text("[]")
    build_community_shards(str(web), out, min_members=1)
    assert not (communities_dir / "stale.json").exists()
    assert (communities_dir / "0.json").exists()
