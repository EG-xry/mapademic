import duckdb
import numpy as np
import pytest
from PIL import Image

from pipeline.tiles import (MAXZ, PIX, aggregate_z9, load_level9,
                            reduce_level, render_zoom)

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
        f"SELECT px, py, cnt, field, community FROM '{out}' ORDER BY px"
    ).fetchall()
    assert len(rows) == 2                      # two occupied pixels
    dense = rows[0]
    assert dense[2] == 3                       # a+b+c share one pixel
    assert dense[3] == "Biology" and dense[4] == 1   # dominant wins
    assert rows[1][2] == 1


def test_pyramid_counts_conserved(tiny_web, tmp_path):
    out = str(tmp_path / "pixels_z9.parquet")
    aggregate_z9(tiny_web, out, duckdb.connect())
    level = load_level9(out)
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
    level = load_level9(pixels)
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
    # GOLDEN pixel: the dense pixel (cnt=3, dominant Biology/community 1) has
    # rank 2/2 -> brightness 1.0, so its RGB equals the raw palette color.
    from pipeline.palette import field_community_rgb
    expected = tuple(int(c * 255) for c in field_community_rgb("Biology", 1))  # trunc, matching astype(uint8)
    dense = np.asarray(Image.open(out / "1/0/1.png"))
    ys, xs = np.nonzero(dense.sum(axis=2))
    assert tuple(dense[ys[0], xs[0]]) == expected


def test_styled_pixels_deterministic(tiny_web, tmp_path):
    pixels = str(tmp_path / "pixels_z9.parquet")
    aggregate_z9(tiny_web, pixels, duckdb.connect())
    level = load_level9(pixels)
    for _ in range(MAXZ - 1):
        level = reduce_level(level)
    a, b = tmp_path / "ta", tmp_path / "tb"
    render_zoom(level, 1, a, bloom=False)
    render_zoom(level, 1, b, bloom=False)
    ia = np.asarray(Image.open(a / "1/1/0.png"))
    ib = np.asarray(Image.open(b / "1/1/0.png"))
    assert np.array_equal(ia, ib)              # golden-by-self: bytes-level stability
