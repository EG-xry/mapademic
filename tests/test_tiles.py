import json

import duckdb
import numpy as np
import pytest
from PIL import Image

from pipeline.palette import load_community_stats
from pipeline.tiles import (EDGE_MAXZ, MAXZ, PIX, TILE, aggregate_maxz,
                            load_level9, load_splats, reduce_level,
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
        f"SELECT px, py, cnt, community FROM '{out}' ORDER BY px"
    ).fetchall()
    assert len(rows) == 2                      # two occupied pixels
    dense = rows[0]
    assert dense[2] == 3                       # a+b+c share one pixel
    assert dense[3] == 1                       # dominant community wins
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
    assert cols == ["px", "py", "cnt", "community"]


def test_pyramid_counts_conserved(tiny_web, tmp_path):
    out = str(tmp_path / "pixels_z9.parquet")
    aggregate_maxz(tiny_web, out, duckdb.connect())
    pal = {1: (1.0, 0.0, 0.0), 2: (0.0, 1.0, 0.0), 3: (0.0, 0.0, 1.0)}
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
    pal = {1: (0.2, 0.8, 0.4), 2: (0.9, 0.1, 0.1), 3: (0.1, 0.1, 0.9)}
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
    base = np.array(pal[1], dtype=np.float32)
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
    pal = {1: (1.0, 0.0, 0.0), 2: (0.0, 1.0, 0.0), 3: (0.0, 0.0, 1.0)}
    level = load_level9(pixels, pal)
    for _ in range(MAXZ - 1):
        level = reduce_level(level)
    a, b = tmp_path / "ta", tmp_path / "tb"
    render_zoom(level, 1, a, bloom=False)
    render_zoom(level, 1, b, bloom=False)
    ia = np.asarray(Image.open(a / "1/1/0.png"))
    ib = np.asarray(Image.open(b / "1/1/0.png"))
    assert np.array_equal(ia, ib)              # golden-by-self: bytes-level stability


def test_splats_only_above_threshold(tmp_path):
    con = duckdb.connect()
    web = tmp_path / "web.parquet"
    duckdb.sql(
        "COPY (SELECT * FROM (VALUES"
        " ('BIG', 'n', 0.5, 0.5, 1, 20, 100000, 'i', 'Medicine', FALSE),"
        " ('SML', 'n', 0.25, 0.25, 1, 20, 100, 'i', 'Medicine', FALSE))"
        " t(id, display_name, xw, yw, community, works_count, cited_by_count,"
        " institution, field, is_ring))"
        f" TO '{web}' (FORMAT PARQUET)")
    pal = {1: (1.0, 0.0, 0.0)}
    s = load_splats(con, str(web), pal)
    assert len(s["px"]) == 1                      # only BIG (SML is below the 5k floor)
    assert s["px"][0] == PIX // 2 and s["py"][0] == PIX // 2
    assert s["rad0"][0] == 2                      # 100k cited -> tier (60_000, 2)


def test_render_zoom_draws_splat_disc(tmp_path):
    level = {"px": np.array([128]), "py": np.array([128]),
             "cnt": np.array([1]), "rgb": np.array([[0.1, 0.1, 0.1]], np.float32)}
    splats = {"px": np.array([128 + 3]), "py": np.array([128]),
              "rgb": np.array([[0.0, 1.0, 0.0]], np.float32),
              "rad0": np.array([2])}
    render_zoom(level, MAXZ, tmp_path, bloom=False, splats=splats)
    # MAXZ: level px are already zoom-MAXZ pixel coords; tile 0/0 holds px<256.
    # XYZ y-flip means tile row (yu=0) is written as ty = ntiles - 1.
    ntiles = 1 << MAXZ
    img = np.asarray(Image.open(tmp_path / str(MAXZ) / "0" / f"{ntiles - 1}.png"))
    ys, xs = np.nonzero(img[:, :, 1] > 200)       # bright green disc pixels
    assert len(ys) >= 5                           # radius-2 disc, not 1 pixel


def test_splat_tier_radii_resolved_at_load(tmp_path):
    con = duckdb.connect()
    web = tmp_path / "web.parquet"
    duckdb.sql(
        "COPY (SELECT * FROM (VALUES"
        " ('T1', 'n', 0.5, 0.5, 1, 20, 6000, 'i', 'Medicine', FALSE),"
        " ('T3', 'n', 0.25, 0.25, 1, 20, 250000, 'i', 'Medicine', FALSE))"
        " t(id, display_name, xw, yw, community, works_count, cited_by_count,"
        " institution, field, is_ring))"
        f" TO '{web}' (FORMAT PARQUET)")
    pal = {1: (1.0, 0.0, 0.0)}
    s = load_splats(con, str(web), pal)
    rad_by_px = dict(zip(s["px"].tolist(), s["rad0"].tolist()))
    assert rad_by_px[PIX // 2] == 1                # 6k cited -> tier (5_000, 1)
    assert rad_by_px[PIX // 4] == 3                # 250k cited -> tier (200_000, 3)


def test_6k_splat_visible_at_z10_not_z9(tmp_path):
    # tier (5_000, 1): rad0 == 1. At z10 (shift 0) radius stays 1; at z9
    # (shift 1) radius drops to 0 -> no splat drawn at all.
    level = {"px": np.array([128]), "py": np.array([128]),
             "cnt": np.array([1]), "rgb": np.array([[0.1, 0.1, 0.1]], np.float32)}
    splats = {"px": np.array([128]), "py": np.array([128]),
              "rgb": np.array([[0.0, 1.0, 0.0]], np.float32),
              "rad0": np.array([1])}
    out10 = tmp_path / "z10"
    render_zoom(level, MAXZ, out10, bloom=False, splats=splats)
    ntiles10 = 1 << MAXZ
    img10 = np.asarray(Image.open(out10 / str(MAXZ) / "0" / f"{ntiles10 - 1}.png"))
    assert (img10[:, :, 1] > 200).sum() == 5       # radius-1 disc == 5 px

    out9 = tmp_path / "z9"
    render_zoom(level, MAXZ - 1, out9, bloom=False, splats=splats)
    ntiles9 = 1 << (MAXZ - 1)
    img9 = np.asarray(Image.open(out9 / str(MAXZ - 1) / "0" / f"{ntiles9 - 1}.png"))
    assert (img9[:, :, 1] > 200).sum() == 0        # no splat at z9


def test_250k_splat_shrinks_from_r3_to_r2_at_z9(tmp_path):
    # tier (200_000, 3): rad0 == 3. At z10 radius stays 3 (29 px disc); at
    # z9 radius drops by one to 2 (13 px disc), still visible.
    level = {"px": np.array([128]), "py": np.array([128]),
             "cnt": np.array([1]), "rgb": np.array([[0.1, 0.1, 0.1]], np.float32)}
    splats = {"px": np.array([128]), "py": np.array([128]),
              "rgb": np.array([[0.0, 1.0, 0.0]], np.float32),
              "rad0": np.array([3])}
    out10 = tmp_path / "z10"
    render_zoom(level, MAXZ, out10, bloom=False, splats=splats)
    ntiles10 = 1 << MAXZ
    img10 = np.asarray(Image.open(out10 / str(MAXZ) / "0" / f"{ntiles10 - 1}.png"))
    assert (img10[:, :, 1] > 200).sum() == 29      # radius-3 disc == 29 px

    out9 = tmp_path / "z9"
    render_zoom(level, MAXZ - 1, out9, bloom=False, splats=splats)
    ntiles9 = 1 << (MAXZ - 1)
    img9 = np.asarray(Image.open(out9 / str(MAXZ - 1) / "0" / f"{ntiles9 - 1}.png"))
    assert (img9[:, :, 1] > 200).sum() == 13       # radius-2 disc == 13 px


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
    assert entries == [{"community": 1, "name": "Cardiology",
                        "field": "Medicine", "members": 1500,
                        "color": entries[0]["color"]}]
    assert entries[0]["color"].startswith("#") and len(entries[0]["color"]) == 7


def test_render_zoom_bakes_faint_edges_at_edge_maxz(tmp_path):
    # edges are stored at native MAXZ pixel resolution and get right-shifted
    # by (MAXZ - z) at render time, so scale x2 to land on the same
    # zoom-EDGE_MAXZ pixels the old MAXZ-scale test targeted (shift == 0).
    level = {"px": np.array([10]), "py": np.array([10]),
             "cnt": np.array([1]), "rgb": np.array([[1.0, 0, 0]], np.float32)}
    edges = {"x0": np.array([20]), "y0": np.array([20]),
             "x1": np.array([220]), "y1": np.array([20])}
    render_zoom(level, EDGE_MAXZ, tmp_path, bloom=False, edges=edges)
    # EDGE_MAXZ: yu = py // TILE = 0, ntiles = 1 << EDGE_MAXZ, ty = (ntiles-1) - yu
    ntiles = 1 << EDGE_MAXZ
    img = np.asarray(Image.open(tmp_path / str(EDGE_MAXZ) / "0" / f"{ntiles - 1}.png")).astype(int)
    mid = img[(TILE - 1) - 10, 60]              # a pixel along the edge
    assert 3 <= mid.max() <= 40                 # faint but present
    assert img[(TILE - 1) - 10, 10, 0] > 200    # node still bright red


def test_no_regular_edges_at_maxz(tmp_path):
    # z10 stays edge-free: regular edges are capped at EDGE_MAXZ (9), so
    # passing an edges dict at z == MAXZ must draw nothing.
    level = {"px": np.array([10]), "py": np.array([10]),
             "cnt": np.array([1]), "rgb": np.array([[1.0, 0, 0]], np.float32)}
    edges = {"x0": np.array([10]), "y0": np.array([10]),
             "x1": np.array([110]), "y1": np.array([10])}
    render_zoom(level, MAXZ, tmp_path, bloom=False, edges=edges)
    ntiles = 1 << MAXZ
    img = np.asarray(Image.open(tmp_path / str(MAXZ) / "0" / f"{ntiles - 1}.png")).astype(int)
    mid = img[(TILE - 1) - 10, 60]              # a pixel along where the edge would be
    assert mid.max() == 0                        # no regular edges drawn at MAXZ
    assert img[(TILE - 1) - 10, 10, 0] > 200      # node still bright red


def test_no_edges_below_z8(tmp_path):
    level = {"px": np.array([0]), "py": np.array([0]),
             "cnt": np.array([1]), "rgb": np.array([[1.0, 0, 0]], np.float32)}
    edges = {"x0": np.array([0]), "y0": np.array([0]),
             "x1": np.array([200 << 2]), "y1": np.array([0])}
    render_zoom(level, 7, tmp_path, bloom=False, edges=edges)
    ntiles = 1 << 7
    img = np.asarray(Image.open(tmp_path / "7" / "0" / f"{ntiles - 1}.png")).astype(int)
    assert img[TILE - 1, 30].max() == 0         # nothing drawn along the line


def test_render_zoom_draws_w1_edges_at_maxz(tmp_path):
    level = {"px": np.array([10]), "py": np.array([10]),
             "cnt": np.array([1]), "rgb": np.array([[1.0, 0, 0]], np.float32)}
    edges_w1 = {"x0": np.array([10]), "y0": np.array([10]),
                "x1": np.array([110]), "y1": np.array([10])}
    render_zoom(level, MAXZ, tmp_path, bloom=False, edges_w1=edges_w1)
    ntiles = 1 << MAXZ
    img = np.asarray(Image.open(tmp_path / str(MAXZ) / "0" / f"{ntiles - 1}.png")).astype(int)
    mid = img[(TILE - 1) - 10, 60]              # a pixel along the w1 edge
    assert 1 <= mid.max() <= 15                 # faint (W1_ALPHA=0.05 < EDGE_ALPHA=0.08)
    assert img[(TILE - 1) - 10, 10, 0] > 200    # node still bright red


def test_w1_edges_not_drawn_below_maxz(tmp_path):
    level = {"px": np.array([10]), "py": np.array([10]),
             "cnt": np.array([1]), "rgb": np.array([[1.0, 0, 0]], np.float32)}
    edges_w1 = {"x0": np.array([10]), "y0": np.array([10]),
                "x1": np.array([110]), "y1": np.array([10])}
    render_zoom(level, MAXZ - 1, tmp_path, bloom=False, edges_w1=edges_w1)
    ntiles = 1 << (MAXZ - 1)
    img = np.asarray(Image.open(tmp_path / str(MAXZ - 1) / "0" / f"{ntiles - 1}.png")).astype(int)
    mid = img[(TILE - 1) - 5, 30]                # where the shifted edge midpoint would land
    assert mid.max() == 0                        # w1 edges draw only at z == MAXZ


def test_edge_with_midpoint_in_other_tile_still_renders(tmp_path):
    # Edge (10,10)-(700,10) post-shift at EDGE_MAXZ (raw coords x2, since
    # edges are native-MAXZ-scale and get >>1 at z == EDGE_MAXZ): midpoint
    # x=355 lies in tile (1,0), but the segment crosses tile (0,0). Pins
    # neighbor-bucket union correctness: a midpoint-only bucket lookup would
    # miss this edge for tile (0,0).
    level = {"px": np.array([10]), "py": np.array([10]),
             "cnt": np.array([1]), "rgb": np.array([[1.0, 0, 0]], np.float32)}
    edges = {"x0": np.array([20]), "y0": np.array([20]),
             "x1": np.array([1400]), "y1": np.array([20])}
    render_zoom(level, EDGE_MAXZ, tmp_path, bloom=False, edges=edges)
    ntiles = 1 << EDGE_MAXZ
    img = np.asarray(Image.open(tmp_path / str(EDGE_MAXZ) / "0" / f"{ntiles - 1}.png")).astype(int)
    assert img[(TILE - 1) - 10, 200].max() >= 3   # edge drawn inside tile (0,0)


def test_regular_edges_share_single_glow_ceiling(tmp_path):
    # 4 overlapping regular-edge copies on the SAME pixels (4*0.08=0.32
    # accumulated alpha) get clipped to the 0.25 ceiling before EDGE_RGB is
    # applied. This replaces the old cross-layer (edges + w1) ceiling test,
    # which is no longer reachable now that regular edges (z8-9) and w1
    # edges (z == MAXZ only) never render at the same zoom.
    from pipeline.tiles import EDGE_RGB
    level = {"px": np.array([200]), "py": np.array([200]),
             "cnt": np.array([1]), "rgb": np.array([[1.0, 0, 0]], np.float32)}
    edges = {"x0": np.array([20] * 4), "y0": np.array([20] * 4),
             "x1": np.array([220] * 4), "y1": np.array([20] * 4)}
    render_zoom(level, EDGE_MAXZ, tmp_path, bloom=False, edges=edges)
    ntiles = 1 << EDGE_MAXZ
    img = np.asarray(Image.open(tmp_path / str(EDGE_MAXZ) / "0" / f"{ntiles - 1}.png"))
    expected = tuple((np.clip(np.float32(0.25) * EDGE_RGB, 0, 1) * 255)
                     .astype(np.uint8))
    assert tuple(img[(TILE - 1) - 10, 60]) == expected
    # combined brightness never exceeds the single ceiling on any channel
    line = img[(TILE - 1) - 10, 10:111].astype(int)
    assert (line <= np.array(expected)[None, :]).all()
