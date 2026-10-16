import re
from pathlib import Path

import numpy as np
import pytest

from pipeline import sizing
from pipeline.sizing import CAP, KNEE, radius_px, size_byte, size_t

DEV_HTML = Path(__file__).resolve().parent.parent / "web" / "dev.html"


def test_t_runs_from_zero_to_one_and_saturates_at_the_cap():
    assert size_t(0) == 0.0
    assert size_t(CAP) == pytest.approx(1.0)
    assert size_t(10 * CAP) == 1.0
    assert size_t(-5) == 0.0                         # defensive: bad data never shrinks below 0


def test_t_is_log_scaled_above_the_knee():
    t = size_t(np.array([0, 10, 100, 1_000, 10_000, 100_000]))
    assert np.all(np.diff(t) > 0)
    # above the knee each decade adds about the same amount
    steps = np.diff(size_t(np.array([1_000, 10_000, 100_000])))
    assert steps[0] == pytest.approx(steps[1], rel=0.1)
    # below it, a decade adds much less: lightly cited authors stay small
    assert size_t(10) - size_t(1) < 0.25 * steps[0]


def test_radius_at_zoom_10_runs_from_0_6_to_7_6_px():
    assert radius_px(10, 0.0) == pytest.approx(0.6)
    assert radius_px(10, 1.0) == pytest.approx(7.6)
    assert radius_px(10, size_t(KNEE * 10)) < radius_px(10, size_t(CAP / 2))


def test_gamma_keeps_mid_cited_authors_small():
    # the median author (358 citations) gets well under half the top bonus
    bonus = radius_px(10, size_t(358)) - radius_px(10, 0.0)
    assert bonus < 0.2 * (radius_px(10, 1.0) - radius_px(10, 0.0))


def test_min_t_drawn_inverts_the_radius():
    from pipeline.sizing import MIN_DRAW_PX, min_t_drawn
    t = min_t_drawn(10)
    assert radius_px(10, t) == pytest.approx(MIN_DRAW_PX)
    assert min_t_drawn(12) == 0.0            # base dot alone is already big enough


def test_base_dot_halves_per_zoom_and_bonus_shrinks_more_slowly():
    assert radius_px(9, 0.0) == pytest.approx(0.3)
    bonus10 = radius_px(10, 1.0) - radius_px(10, 0.0)
    bonus9 = radius_px(9, 1.0) - radius_px(9, 0.0)
    assert bonus9 == pytest.approx(bonus10 / np.sqrt(2))


def test_size_byte_quantizes_t():
    b = size_byte(np.array([0, KNEE, CAP, 10 * CAP]))
    assert b.dtype == np.uint8
    assert b[0] == 0 and b[2] == 255 and b[3] == 255
    assert 0 < b[1] < 255


def test_viewer_mirror_uses_the_same_constants():
    # web/dev.html re-implements t and r for the deep-zoom points; a drift
    # would make points jump in size at the raster hand-off
    html = DEV_HTML.read_text()
    for name, value in [("SIZE_KNEE", sizing.KNEE), ("SIZE_CAP", sizing.CAP),
                        ("SIZE_BASE_PX", sizing.BASE_PX), ("SIZE_BONUS_PX", sizing.BONUS_PX),
                        ("SIZE_GAMMA", sizing.GAMMA)]:
        m = re.search(rf"const {name} = ([0-9_.]+);", html)
        assert m, f"{name} missing from web/dev.html"
        assert float(m.group(1).replace("_", "")) == value, name


def test_viewer_knows_the_raster_tile_zoom():
    # MapLibre zoom counts 512px tiles; the raster tiles are 256px, so the
    # viewer's sizes must read tile zoom = map zoom + 1, capped at the deepest
    # tiles -- otherwise points are half the raster's size at the hand-off
    from pipeline.tiles import MAXZ, TILE
    html = DEV_HTML.read_text()
    source = re.search(r"dots: \{ type: \"raster\".*?tileSize: (\d+), minzoom: 0, maxzoom: (\d+)", html, re.S)
    assert source, "raster source missing from web/dev.html"
    assert int(source.group(1)) == TILE and int(source.group(2)) == MAXZ
    offset = re.search(r"const TILE_ZOOM_OFFSET = (\d+);", html)
    assert offset and int(offset.group(1)) == np.log2(512 / TILE)
    deepest = re.search(r"const RASTER_MAX_TILE_ZOOM = (\d+);", html)
    assert deepest and int(deepest.group(1)) == MAXZ
