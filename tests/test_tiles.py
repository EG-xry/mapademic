import json

import duckdb
import numpy as np
import pytest
from PIL import Image

from pipeline.palette import group_key, load_community_stats
from pipeline import sizing
from pipeline.tiles import (MAXZ, PIX, TILE, aggregate_maxz, legend_inputs,
                            load_level9, load_markers, reduce_level,
                            render_zoom, write_legend)

# MAXZ and PIX are imported from pipeline.tiles; values follow whatever
# native max zoom the pipeline currently targets.


def write_web(path, rows):
    """rows: (id, xw, yw, community, field)"""
    vals = ", ".join(
        f"('{i}', 'N', {xw}, {yw}, {c}, 20, 100, 'I', "
        + (f"'{f}'" if f else "NULL") + ")"
        for i, xw, yw, c, f in rows
    )
    duckdb.sql(
        f"COPY (SELECT * FROM (VALUES {vals}) t(id, display_name, xw, yw,"
        f" community, works_count, cited_by_count, institution, field))"
        f" TO '{path}' (FORMAT PARQUET)"
    )


@pytest.fixture
def tiny_web(tmp_path):
    p = tmp_path / "coords_web.parquet"
    # two points in the same z9 pixel (dominant test), one lone point far away
    eps = 0.2 / PIX
    write_web(p, [
        ("a", 0.25, 0.25, 1, "Biology"),
        ("b", 0.25 + eps, 0.25 + eps, 1, "Biology"),
        ("c", 0.25, 0.25, 2, "Chemistry"),   # same pixel, minority
        ("d", 0.75, 0.75, 3, "Computer Science"),
    ])
    return str(p)


def test_aggregate_dominant_and_counts(tiny_web, tmp_path):
    out = str(tmp_path / "pixels_z9.parquet")
    con = duckdb.connect()
    aggregate_maxz(tiny_web, out, con)
    rows = duckdb.sql(
        f"SELECT px, py, cnt, gkey FROM '{out}' ORDER BY px"
    ).fetchall()
    assert len(rows) == 2                      # two occupied pixels
    dense = rows[0]
    assert dense[2] == 3                       # a+b+c share one pixel
    assert dense[3] == group_key(1, -1)        # dominant community wins
    assert rows[1][2] == 1


def test_aggregate_schema_has_no_field(tmp_path):
    web = tmp_path / "web.parquet"
    duckdb.sql(
        "COPY (SELECT 'A1' id, 'n' display_name, 0.25 xw, 0.25 yw,"
        " 1 community, 20 works_count, 10 cited_by_count, 'i' institution,"
        " 'Medicine' field, FALSE is_ring)"
        f" TO '{web}' (FORMAT PARQUET)")
    out = tmp_path / "px.parquet"
    aggregate_maxz(str(web), str(out), duckdb.connect())
    cols = [r[0] for r in duckdb.sql(f"DESCRIBE SELECT * FROM '{out}'").fetchall()]
    assert cols == ["px", "py", "cnt", "gkey"]


def test_pyramid_counts_conserved(tiny_web, tmp_path):
    out = str(tmp_path / "pixels_z9.parquet")
    aggregate_maxz(tiny_web, out, duckdb.connect())
    pal = {group_key(1, -1): (1.0, 0.0, 0.0), group_key(2, -1): (0.0, 1.0, 0.0), group_key(3, -1): (0.0, 0.0, 1.0)}
    level = load_level9(out, pal)
    total = level["cnt"].sum()
    for _ in range(MAXZ):                      # 9 reductions -> zoom 0
        level = reduce_level(level)
        assert level["cnt"].sum() == total
    # 2 distinct pixels (128px apart at z0), same tile
    assert len(level["cnt"]) == 2


def test_reduce_level_empty_returns_empty():
    empty = {
        "px": np.empty(0, np.int64), "py": np.empty(0, np.int64),
        "cnt": np.empty(0, np.int64), "rgb": np.empty((0, 3), np.float32),
    }
    out = reduce_level(empty)                   # must not crash on idx[0]
    assert len(out["px"]) == 0 and len(out["cnt"]) == 0
    assert out["rgb"].shape == (0, 3)
    assert out["px"].dtype == np.int64 and out["rgb"].dtype == np.float32


def test_render_writes_expected_tiles_and_is_idempotent(tiny_web, tmp_path):
    pixels = str(tmp_path / "pixels_z9.parquet")
    aggregate_maxz(tiny_web, pixels, duckdb.connect())
    pal = {group_key(1, -1): (0.2, 0.8, 0.4), group_key(2, -1): (0.9, 0.1, 0.1), group_key(3, -1): (0.1, 0.1, 0.9)}
    level = load_level9(pixels, pal)
    for _ in range(MAXZ - 1):                  # reduce to zoom 1 (2x2 tiles)
        level = reduce_level(level)
    out = tmp_path / "tiles"
    n = render_zoom(level, 1, out, bloom=False)
    assert n == 2                              # points at (.25,.25) and (.75,.75)
    # XYZ y-flip: yw=0.75 (upper area) -> tile row 0; yw=0.25 -> row 1
    assert (out / "1/0/1.png").exists() and (out / "1/1/0.png").exists()
    img = np.asarray(Image.open(out / "1/1/0.png"))
    assert img.shape == (256, 256, 3) and img.max() > 0
    assert render_zoom(level, 1, out, bloom=False) == 0   # resume: all skipped
    # GOLDEN pixel: the dense pixel (cnt=3, dominant community 1) is the
    # max-count pixel -> rank 1.0 -> brightness == BRIGHT_CEIL, so its RGB equals
    # the palette color scaled by the ceiling. Mirror the render's float32 path.
    from pipeline.tiles import BRIGHT_CEIL
    base = np.array(pal[group_key(1, -1)], dtype=np.float32)
    expected = tuple(
        (np.clip(base * np.float32(BRIGHT_CEIL), 0, 1) * 255).astype(np.uint8)
    )
    dense = np.asarray(Image.open(out / "1/0/1.png"))
    ys, xs = np.nonzero(dense.sum(axis=2))
    assert tuple(dense[ys[0], xs[0]]) == expected


def test_render_bloom_lights_up_neighbors(tmp_path):
    # single occupied pixel, well inside a tile (avoid edge clipping)
    level = {
        "px": np.array([128], dtype=np.int64),
        "py": np.array([128], dtype=np.int64),
        "cnt": np.array([100], dtype=np.int64),
        "rgb": np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
    }
    out = tmp_path / "tiles"
    n = render_zoom(level, MAXZ, out, bloom=True)
    assert n == 1
    ntiles = 1 << MAXZ
    ty = (ntiles - 1) - (128 // TILE)   # XYZ y-flip for tile row
    img = np.asarray(Image.open(out / str(MAXZ) / "0" / f"{ty}.png"))
    # center pixel: ix = 128, iy = (TILE-1) - 128 = 127
    cx, cy = 128, 127
    center = int(img[cy, cx].sum())
    assert center > 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbor = int(img[cy + dy, cx + dx].sum())
            assert neighbor > 0, f"neighbor ({dx},{dy}) should be lit by bloom"
            assert neighbor < center, f"neighbor ({dx},{dy}) should be dimmer than center"


def test_styled_pixels_deterministic(tiny_web, tmp_path):
    pixels = str(tmp_path / "pixels_z9.parquet")
    aggregate_maxz(tiny_web, pixels, duckdb.connect())
    pal = {group_key(1, -1): (1.0, 0.0, 0.0), group_key(2, -1): (0.0, 1.0, 0.0), group_key(3, -1): (0.0, 0.0, 1.0)}
    level = load_level9(pixels, pal)
    for _ in range(MAXZ - 1):
        level = reduce_level(level)
    a, b = tmp_path / "ta", tmp_path / "tb"
    render_zoom(level, 1, a, bloom=False)
    render_zoom(level, 1, b, bloom=False)
    ia = np.asarray(Image.open(a / "1/1/0.png"))
    ib = np.asarray(Image.open(b / "1/1/0.png"))
    assert np.array_equal(ia, ib)              # golden-by-self: bytes-level stability


def write_cited(path, rows):
    """rows: (id, xw, yw, cited, is_ring)"""
    vals = ", ".join(f"('{i}', 'n', {x}, {y}, 1, 20, {c}, 'i', 'Medicine', {str(r).upper()})"
                     for i, x, y, c, r in rows)
    duckdb.sql(f"COPY (SELECT * FROM (VALUES {vals}) t(id, display_name, xw, yw, community,"
               f" works_count, cited_by_count, institution, field, is_ring))"
               f" TO '{path}' (FORMAT PARQUET)")


def test_markers_keep_only_authors_whose_disc_outgrows_their_pixel(tmp_path):
    web = tmp_path / "web.parquet"
    write_cited(web, [("TOP", 0.5, 0.5, 300_000, False), ("MID", 0.25, 0.25, 2_000, False),
                      ("LOW", 0.75, 0.75, 3, False), ("RING", 0.1, 0.1, 300_000, True)])
    m = load_markers(duckdb.connect(), str(web), {1: (1.0, 0.0, 0.0)})
    assert sorted(m["t"].round(3).tolist()) == sorted(
        sizing.size_t([300_000, 2_000]).round(3).tolist())   # LOW too small, RING excluded
    assert m["xw"].dtype == np.float64                          # sub-pixel centres for anti-aliasing


def one_marker(t, xw=128.5 / PIX, yw=128.5 / PIX):
    return {"xw": np.array([xw]), "yw": np.array([yw]), "t": np.array([t]),
            "rgb": np.array([[0.0, 1.0, 0.0]], np.float32)}


def dim_level(px=128, py=128):
    return {"px": np.array([px]), "py": np.array([py]), "cnt": np.array([1]),
            "rgb": np.array([[0.1, 0.1, 0.1]], np.float32)}


def lit(out, z, x=0, yu=0, thresh=100):
    ntiles = 1 << z
    img = np.asarray(Image.open(out / str(z) / str(x) / f"{ntiles - 1 - yu}.png"))
    return int((img[:, :, 1] > thresh).sum())


def test_top_cited_author_is_a_disc_of_the_max_radius_at_z10(tmp_path):
    render_zoom(dim_level(), MAXZ, tmp_path, bloom=False, markers=one_marker(1.0))
    r = sizing.radius_px(MAXZ, 1.0)                     # 4.5 px
    assert abs(lit(tmp_path, MAXZ) - np.pi * r * r) < 0.25 * np.pi * r * r


def test_disc_area_grows_with_citations(tmp_path):
    areas = []
    for t in (0.3, 0.6, 1.0):
        out = tmp_path / str(t)
        render_zoom(dim_level(), MAXZ, out, bloom=False, markers=one_marker(t))
        areas.append(lit(out, MAXZ, thresh=20))
    assert areas[0] < areas[1] < areas[2]


def test_disc_shrinks_one_zoom_out_and_vanishes_below_disc_zoom(tmp_path):
    m = one_marker(1.0, xw=100.5 / PIX, yw=100.5 / PIX)   # inside tile 0/0 at z8, z9 and z10
    sizes = {}
    for z in (MAXZ, MAXZ - 1, sizing.DISC_MIN_ZOOM - 1):
        shift = MAXZ - z
        level = dim_level(px=int(m["xw"][0] * PIX) >> shift, py=int(m["yw"][0] * PIX) >> shift)
        out = tmp_path / f"z{z}"
        render_zoom(level, z, out, bloom=False, markers=m)
        sizes[z] = lit(out, z)
    assert sizes[MAXZ - 1] < sizes[MAXZ]
    assert sizes[sizing.DISC_MIN_ZOOM - 1] == 0          # density picture only


def test_disc_on_a_tile_edge_is_finished_in_the_neighbouring_tile(tmp_path):
    # centre 1.5 px left of the tile 0/1 border; the neighbour has no
    # occupied pixel of its own and must still be rendered with the overhang
    m = one_marker(1.0, xw=(TILE - 1.5) / PIX, yw=128.5 / PIX)
    render_zoom(dim_level(px=TILE - 2), MAXZ, tmp_path, bloom=False, markers=m)
    assert lit(tmp_path, MAXZ, x=0) > 0 and lit(tmp_path, MAXZ, x=1) > 0


def test_legend_json(tmp_path):
    web = tmp_path / "web.parquet"
    duckdb.sql(
        "COPY (SELECT 'A' || range::VARCHAR id, 'n' display_name, 0.5 xw,"
        " 0.5 yw, 1 community, 20 works_count, 10 cited_by_count,"
        " 'i' institution, 'Medicine' field, FALSE is_ring FROM range(1500))"
        f" TO '{web}' (FORMAT PARQUET)")
    regions = tmp_path / "regions.json"
    regions.write_text(json.dumps([{"community": 1, "name": "Cardiology"}]))
    out = tmp_path / "legend.json"
    stats = load_community_stats(duckdb.connect(), str(web))
    assert stats == [(1, "Medicine", 1500, 0.5, 0.5)]   # 5-tuple: adds centroid
    write_legend(stats, str(out), str(regions), min_members=1000)
    entries = json.loads(out.read_text())
    assert entries == [{"community": 1, "name": "Cardiology", "kind": "field",
                        "field": "Medicine", "members": 1500,
                        "color": entries[0]["color"]}]
    assert entries[0]["color"].startswith("#") and len(entries[0]["color"]) == 7


def test_sub_groups_colour_their_own_pixels(tmp_path):
    # one community, two sub-groups on different pixels -> two colour keys
    web = tmp_path / "web.parquet"
    duckdb.sql(
        "COPY (SELECT 'A' || range::VARCHAR id, 'n' display_name,"
        " CASE WHEN range < 5 THEN 0.25 ELSE 0.75 END xw, 0.5 yw, 1 community,"
        " CAST(CASE WHEN range < 5 THEN 0 ELSE 1 END AS SMALLINT) sub,"
        " 20 works_count, 10 cited_by_count, 'i' institution, 'Medicine' field, FALSE is_ring"
        " FROM range(8))"
        f" TO '{web}' (FORMAT PARQUET)")
    out = tmp_path / "px.parquet"
    aggregate_maxz(str(web), str(out), duckdb.connect())
    keys = {r[0] for r in duckdb.sql(f"SELECT gkey FROM '{out}'").fetchall()}
    assert keys == {group_key(1, 0), group_key(1, 1)}


def test_legend_lists_sub_groups_and_leaves_faded_communities_out(tmp_path):
    web = tmp_path / "web.parquet"
    # community 1: 1500 members in two sub-groups; community 2: 1200 near-empty records
    duckdb.sql(
        "COPY (SELECT 'A' || range::VARCHAR id, 'n' display_name, 0.4 xw, 0.5 yw, 1 community,"
        " CAST(CASE WHEN range < 1000 THEN 0 ELSE 1 END AS SMALLINT) sub, 20 works_count,"
        " 300 cited_by_count, 'i' institution, 'Medicine' field, FALSE is_ring FROM range(1500)"
        " UNION ALL SELECT 'B' || range::VARCHAR, 'n', 0.8, 0.5, 2, CAST(-1 AS SMALLINT), 3, 0,"
        " NULL, 'Social Sciences', FALSE FROM range(1200))"
        f" TO '{web}' (FORMAT PARQUET)")
    regions = tmp_path / "regions.json"
    regions.write_text(json.dumps([
        {"community": 1, "name": "Medicine (Italy)", "kind": "national"},
        {"level": 2, "community": 1, "sub": 0, "name": "Cardiology", "xw": 0.4, "yw": 0.5},
        {"level": 2, "community": 1, "sub": 1, "name": "Oncology", "xw": 0.41, "yw": 0.5}]))
    con = duckdb.connect()
    stats = load_community_stats(con, str(web))
    pal, faded, subs = legend_inputs(con, str(web), stats)
    assert faded == {2} and subs == {1: [(0, 1000), (1, 500)]}
    out = tmp_path / "legend.json"
    write_legend(stats, str(out), str(regions), palette=pal, faded=faded, subs=subs)
    entries = json.loads(out.read_text())
    assert [e["community"] for e in entries] == [1]
    e = entries[0]
    assert e["kind"] == "national" and e["name"] == "Medicine (Italy)"
    assert [(x["sub"], x["name"], x["members"]) for x in e["subs"]] == [(0, "Cardiology", 1000), (1, "Oncology", 500)]
    assert e["subs"][0]["color"] == e["color"]          # largest sub-group keeps the community colour
    assert e["subs"][1]["color"] != e["color"]


def test_dust_is_left_out_of_the_tiles(tmp_path):
    web = tmp_path / "web.parquet"
    duckdb.sql(
        "COPY (SELECT 'A' || range::VARCHAR id, 'n' display_name,"
        " CASE WHEN range < 5 THEN 0.25 ELSE 0.75 END xw, 0.5 yw, 1 community, CAST(-1 AS SMALLINT) sub,"
        " 20 works_count, 10 cited_by_count, 'i' institution, 'Medicine' field, FALSE is_ring,"
        " range >= 5 AS is_dust FROM range(8))"
        f" TO '{web}' (FORMAT PARQUET)")
    out = tmp_path / "px.parquet"
    assert aggregate_maxz(str(web), str(out), duckdb.connect()) == 1    # only the non-dust pixel
