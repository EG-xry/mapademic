"""Aggregate web coords at zoom 9, pyramid-reduce, write styled PNG tiles."""
import os
from pathlib import Path

import duckdb
import numpy as np
from PIL import Image

from pipeline.config import apply_resource_limits, data_dir
from pipeline.palette import field_community_rgb

MAXZ = 9
TILE = 256
PIX = (2 ** MAXZ) * TILE  # 131072 virtual pixels per side at zoom 9


def aggregate_z9(webcoords_path: str, out_path: str, con) -> int:
    """Per-pixel count + dominant (field, community). Returns occupied pixels."""
    apply_resource_limits(con)
    return con.execute(
        f"""
        COPY (
            WITH binned AS (
                SELECT least({PIX - 1}, CAST(floor(CAST(xw AS DOUBLE) * {PIX}) AS INT)) AS px,
                       least({PIX - 1}, CAST(floor(CAST(yw AS DOUBLE) * {PIX}) AS INT)) AS py,
                       field, community
                FROM read_parquet('{webcoords_path}')
            ),
            grouped AS (
                SELECT px, py, field, community, count(*) AS c
                FROM binned GROUP BY px, py, field, community
            ),
            ranked AS (
                SELECT px, py, field, community, c,
                       sum(c) OVER (PARTITION BY px, py) AS cnt,
                       row_number() OVER (PARTITION BY px, py
                                          ORDER BY c DESC, community,
                                                   field NULLS LAST) AS rn
                FROM grouped
            )
            SELECT px, py, CAST(cnt AS BIGINT) AS cnt, field, community
            FROM ranked WHERE rn = 1
        ) TO '{out_path}' (FORMAT PARQUET)
        """
    ).fetchone()[0]


def load_level9(pixels_path: str) -> dict:
    con = duckdb.connect()
    # palette per DISTINCT (field, community) pair (~216k), then broadcast -
    # avoids a python loop over all 8.6M pixels
    pairs = con.execute(
        f"""SELECT field, community, row_number() OVER () - 1 AS pi
            FROM (SELECT DISTINCT field, community
                  FROM read_parquet('{pixels_path}'))"""
    ).fetchall()
    pair_rgb = np.empty((len(pairs), 3), dtype=np.float32)
    for field, community, pi in pairs:
        pair_rgb[pi] = field_community_rgb(
            None if field is None else str(field), int(community)
        )
    con.execute("CREATE TEMP TABLE pal (field VARCHAR, community INT, pi INT)")
    con.executemany("INSERT INTO pal VALUES (?, ?, ?)", pairs)
    t = con.execute(
        f"""SELECT p.px, p.py, p.cnt, pal.pi
            FROM read_parquet('{pixels_path}') p
            JOIN pal ON pal.community = p.community
                    AND pal.field IS NOT DISTINCT FROM p.field
            ORDER BY p.py, p.px"""
    ).fetchnumpy()
    return {
        "px": t["px"].astype(np.int64),
        "py": t["py"].astype(np.int64),
        "cnt": t["cnt"].astype(np.int64),
        "rgb": pair_rgb[t["pi"].astype(np.int64)],
    }


def reduce_level(level: dict) -> dict:
    """2x2 -> 1: counts summed, rgb of the heaviest child (dominant approx)."""
    if len(level["px"]) == 0:
        return {
            "px": np.empty(0, np.int64), "py": np.empty(0, np.int64),
            "cnt": np.empty(0, np.int64), "rgb": np.empty((0, 3), np.float32),
        }
    px, py = level["px"] >> 1, level["py"] >> 1
    key = px * 2**31 + py  # unique combined key (px, py < 2**17)
    order = np.argsort(key, kind="stable")
    key_s, cnt_s, rgb_s = key[order], level["cnt"][order], level["rgb"][order]
    px_s, py_s = px[order], py[order]
    boundaries = np.flatnonzero(np.diff(key_s)) + 1
    groups = np.split(np.arange(len(key_s)), boundaries)
    n = len(groups)
    out = {
        "px": np.empty(n, np.int64), "py": np.empty(n, np.int64),
        "cnt": np.empty(n, np.int64), "rgb": np.empty((n, 3), np.float32),
    }
    for gi, idx in enumerate(groups):
        out["px"][gi] = px_s[idx[0]]
        out["py"][gi] = py_s[idx[0]]
        c = cnt_s[idx]
        out["cnt"][gi] = c.sum()
        out["rgb"][gi] = rgb_s[idx[np.argmax(c)]]
    return out


# Styling constants (tuned at the Task 4 QA gate).
BRIGHT_FLOOR = 0.05   # sparse outer/halo pixels nearly vanish -> no bright ring,
BRIGHT_CEIL = 1.0     #   darker gaps between clusters -> structure reads
BRIGHT_GAMMA = 1.25   # >1 dims low/mid density, keeps dense cores bright (contrast)


def _brightness(cnt: np.ndarray) -> np.ndarray:
    vals = np.log1p(cnt.astype(np.float64))
    ranks = np.searchsorted(np.sort(vals), vals, side="right") / len(vals)
    shaped = ranks ** BRIGHT_GAMMA
    return (BRIGHT_FLOOR + (BRIGHT_CEIL - BRIGHT_FLOOR) * shaped).astype(np.float32)


BLOOM = np.array([[0.06, 0.12, 0.06], [0.12, 0.0, 0.12], [0.06, 0.12, 0.06]],
                 dtype=np.float32)


def render_zoom(level: dict, z: int, out_dir: Path, bloom: bool) -> int:
    """Write XYZ PNG tiles for one zoom; skip existing. Returns tiles written."""
    ntiles = 1 << z
    px, py = level["px"], level["py"]  # already in zoom-z pixel space
    bright = _brightness(level["cnt"])
    color = level["rgb"] * bright[:, None]
    tx, ty_up = px // TILE, py // TILE
    written = 0
    for t in np.unique(tx * ntiles + ty_up):
        x, yu = int(t // ntiles), int(t % ntiles)
        ty = (ntiles - 1) - yu                       # XYZ y-flip, here only
        path = out_dir / str(z) / str(x) / f"{ty}.png"
        if path.exists():
            continue
        sel = (tx == x) & (ty_up == yu)
        img = np.zeros((TILE, TILE, 3), dtype=np.float32)
        ix = px[sel] - x * TILE
        iy_up = py[sel] - yu * TILE
        iy = (TILE - 1) - iy_up                      # flip rows inside the tile
        img[iy, ix] = color[sel]
        if bloom:
            base = img.copy()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    w = BLOOM[dy + 1, dx + 1]
                    if w:
                        img[max(0, dy):TILE + min(0, dy) or TILE,
                            max(0, dx):TILE + min(0, dx) or TILE] += \
                            w * base[max(0, -dy):TILE + min(0, -dy) or TILE,
                                     max(0, -dx):TILE + min(0, -dx) or TILE]
        arr = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / (path.name + ".tmp")
        Image.fromarray(arr).save(tmp, optimize=True, format="PNG")
        os.replace(tmp, path)                        # atomic: no truncated PNG on crash
        written += 1
    return written


def _parse_zooms(spec: str) -> list[int]:
    lo, _, hi = spec.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def add_parser(parser) -> None:
    parser.add_argument("--zooms", default="0-9", help="e.g. 0-5 for the QA gate")
    parser.add_argument("--web", default=None)
    parser.add_argument("--out", default=None)


def run(args) -> int:
    web = args.web or str(data_dir() / "coords_web.parquet")
    out = Path(args.out) if args.out else data_dir() / "tiles"
    pixels = str(data_dir() / "pixels_z9.parquet")
    if not Path(pixels).exists():
        n = aggregate_z9(web, pixels, duckdb.connect())
        print(f"aggregated {n:,} occupied z9 pixels", flush=True)
    zooms = sorted(_parse_zooms(args.zooms), reverse=True)  # deep -> shallow
    level = load_level9(pixels)
    for z in range(MAXZ, -1, -1):
        if z in zooms:
            w = render_zoom(level, z, out, bloom=(z >= 8))
            print(f"zoom {z}: {w} tiles written", flush=True)
        if z > 0:
            level = reduce_level(level)
    return 0
