import json

import duckdb
import numpy as np
import pytest
from PIL import Image

from pipeline.palette import load_community_stats
from pipeline.tiles import (MAXZ, PIX, TILE, aggregate_z9, load_level9,
                            load_splats, reduce_level, render_zoom,
                            write_legend)

# MAXZ == 9, PIX == 2**9 * 256 == 131072


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
    aggregate_z9(tiny_web, out, con)
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
    aggregate_z9(str(web), str(out), duckdb.connect())
    cols = [r[0] for r in duckdb.sql(f"DESCRIBE SELECT * FROM '{out}'").fetchall()]
    assert cols == ["px", "py", "cnt", "community"]


def test_pyramid_counts_conserved(tiny_web, tmp_path):
    out = str(tmp_path / "pixels_z9.parquet")
    aggregate_z9(tiny_web, out, duckdb.connect())
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
    aggregate_z9(tiny_web, pixels, duckdb.connect())
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
    aggregate_z9(tiny_web, pixels, duckdb.connect())
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


def test_splats_drawn_at_z9_only_above_threshold(tmp_path):
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
    s = load_splats(con, str(web), pal, min_cited=60000)
    assert len(s["px"]) == 1                      # only BIG
    assert s["px"][0] == PIX // 2 and s["py"][0] == PIX // 2


def test_render_zoom_draws_splat_disc(tmp_path):
    level = {"px": np.array([128]), "py": np.array([128]),
             "cnt": np.array([1]), "rgb": np.array([[0.1, 0.1, 0.1]], np.float32)}
    splats = {"px": np.array([128 + 3]), "py": np.array([128]),
              "rgb": np.array([[0.0, 1.0, 0.0]], np.float32)}
    render_zoom(level, 9, tmp_path, bloom=False, splats=splats)
    # z9: level px are already zoom-9 pixel coords; tile 0/0 holds px<256.
    # XYZ y-flip means tile row (yu=0) is written as ty = ntiles - 1.
    ntiles = 1 << 9
    img = np.asarray(Image.open(tmp_path / "9" / "0" / f"{ntiles - 1}.png"))
    ys, xs = np.nonzero(img[:, :, 1] > 200)       # bright green disc pixels
    assert len(ys) >= 5                           # radius-2 disc, not 1 pixel


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
    write_legend(stats, str(out), str(regions), min_members=1000)
    entries = json.loads(out.read_text())
    assert entries == [{"community": 1, "name": "Cardiology",
                        "field": "Medicine", "members": 1500,
                        "color": entries[0]["color"]}]
    assert entries[0]["color"].startswith("#") and len(entries[0]["color"]) == 7


def test_render_zoom_bakes_faint_edges_at_z9(tmp_path):
    level = {"px": np.array([10]), "py": np.array([10]),
             "cnt": np.array([1]), "rgb": np.array([[1.0, 0, 0]], np.float32)}
    edges = {"x0": np.array([10]), "y0": np.array([10]),
             "x1": np.array([110]), "y1": np.array([10])}
    render_zoom(level, 9, tmp_path, bloom=False, edges=edges)
    # z9: yu = py // TILE = 0, ntiles = 1 << 9 = 512, ty = (ntiles-1) - yu = 511
    # (brief's literal "255.png" assumed ntiles=256; corrected per Task 6 brief's
    # own instruction to recompute the y-flip when a literal is mathematically
    # wrong -- see test_render_zoom_draws_splat_disc which already derives the
    # same z9/yu=0 tile as ntiles - 1, not 255).
    img = np.asarray(Image.open(tmp_path / "9" / "0" / "511.png")).astype(int)
    mid = img[(TILE - 1) - 10, 60]              # a pixel along the edge
    assert 3 <= mid.max() <= 40                 # faint but present
    assert img[(TILE - 1) - 10, 10, 0] > 200    # node still bright red


def test_no_edges_below_z8(tmp_path):
    level = {"px": np.array([0]), "py": np.array([0]),
             "cnt": np.array([1]), "rgb": np.array([[1.0, 0, 0]], np.float32)}
    edges = {"x0": np.array([0]), "y0": np.array([0]),
             "x1": np.array([200 << 2]), "y1": np.array([0])}
    render_zoom(level, 7, tmp_path, bloom=False, edges=edges)
    ntiles = 1 << 7
    img = np.asarray(Image.open(tmp_path / "7" / "0" / f"{ntiles - 1}.png")).astype(int)
    assert img[TILE - 1, 30].max() == 0         # nothing drawn along the line


def test_edge_with_midpoint_in_other_tile_still_renders(tmp_path):
    # Edge (10,10)-(700,10) at z9: midpoint x=355 lies in tile (1,0), but the
    # segment crosses tile (0,0). Pins neighbor-bucket union correctness: a
    # midpoint-only bucket lookup would miss this edge for tile (0,0).
    level = {"px": np.array([10]), "py": np.array([10]),
             "cnt": np.array([1]), "rgb": np.array([[1.0, 0, 0]], np.float32)}
    edges = {"x0": np.array([10]), "y0": np.array([10]),
             "x1": np.array([700]), "y1": np.array([10])}
    render_zoom(level, 9, tmp_path, bloom=False, edges=edges)
    ntiles = 1 << 9
    img = np.asarray(Image.open(tmp_path / "9" / "0" / f"{ntiles - 1}.png")).astype(int)
    assert img[(TILE - 1) - 10, 200].max() >= 3   # edge drawn inside tile (0,0)
