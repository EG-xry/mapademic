"""Aggregate web coords at zoom 9, pyramid-reduce, write styled PNG tiles."""
import json
import os
from pathlib import Path

import duckdb
import numpy as np
from PIL import Image

from pipeline.config import apply_resource_limits, data_dir
from pipeline.palette import community_rgb, load_community_stats

MAXZ = 9
TILE = 256
PIX = (2 ** MAXZ) * TILE  # 131072 virtual pixels per side at zoom 9

SPLAT_RADIUS = {8: 1, 9: 2}   # citation-star disc radius in px at each zoom


def aggregate_z9(webcoords_path: str, out_path: str, con) -> int:
    """Per-pixel count + dominant community. Returns occupied pixels."""
    apply_resource_limits(con)
    return con.execute(
        f"""
        COPY (
            WITH binned AS (
                SELECT least({PIX - 1}, CAST(floor(CAST(xw AS DOUBLE) * {PIX}) AS INT)) AS px,
                       least({PIX - 1}, CAST(floor(CAST(yw AS DOUBLE) * {PIX}) AS INT)) AS py,
                       community
                FROM read_parquet('{webcoords_path}')
            ),
            grouped AS (
                SELECT px, py, community, count(*) AS c
                FROM binned GROUP BY px, py, community
            ),
            ranked AS (
                SELECT px, py, community, c,
                       sum(c) OVER (PARTITION BY px, py) AS cnt,
                       row_number() OVER (PARTITION BY px, py
                                          ORDER BY c DESC, community) AS rn
                FROM grouped
            )
            SELECT px, py, CAST(cnt AS BIGINT) AS cnt, community
            FROM ranked WHERE rn = 1
        ) TO '{out_path}' (FORMAT PARQUET)
        """
    ).fetchone()[0]


def load_level9(pixels_path: str, palette: dict[int, tuple]) -> dict:
    con = duckdb.connect()
    t = con.execute(
        f"SELECT px, py, cnt, community FROM read_parquet('{pixels_path}')"
        " ORDER BY py, px").fetchnumpy()
    comm = t["community"].astype(np.int64)
    uniq, inv = np.unique(comm, return_inverse=True)
    lut = np.array([palette.get(int(c), (0.35, 0.35, 0.35)) for c in uniq],
                   dtype=np.float32)
    return {"px": t["px"].astype(np.int64), "py": t["py"].astype(np.int64),
            "cnt": t["cnt"].astype(np.int64), "rgb": lut[inv]}


def load_splats(con, web_path: str, palette: dict, min_cited: int) -> dict:
    t = con.execute(
        f"""SELECT least({PIX - 1}, CAST(floor(CAST(xw AS DOUBLE) * {PIX}) AS INT)) px,
                   least({PIX - 1}, CAST(floor(CAST(yw AS DOUBLE) * {PIX}) AS INT)) py,
                   community
            FROM read_parquet('{web_path}')
            WHERE cited_by_count >= {int(min_cited)} AND NOT is_ring"""
    ).fetchnumpy()
    comm = t["community"].astype(np.int64)
    rgb = np.array([palette.get(int(c), (0.6, 0.6, 0.6)) for c in comm],
                   dtype=np.float32)
    return {"px": t["px"].astype(np.int64), "py": t["py"].astype(np.int64),
            "rgb": rgb}


def write_legend(stats: list[tuple[int, str | None, int]], out_path: str,
                 regions_path: str | None, min_members: int = 1000) -> int:
    names = {}
    if regions_path and Path(regions_path).exists():
        names = {r["community"]: r["name"]
                 for r in json.loads(Path(regions_path).read_text())
                 if "community" in r}
    entries = []
    for c, f, n in stats:              # stats already ordered members DESC, community
        if n < min_members:
            continue
        r, g, b = community_rgb(c, f, n, min_members)
        entries.append({
            "community": c, "name": names.get(c),
            "field": f, "members": n,
            "color": "#%02x%02x%02x" % (int(r*255), int(g*255), int(b*255)),
        })
    tmp = str(out_path) + ".tmp"
    Path(tmp).write_text(json.dumps(entries, ensure_ascii=False, indent=1))
    os.replace(tmp, out_path)
    return len(entries)


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


def render_zoom(level: dict, z: int, out_dir: Path, bloom: bool,
                splats: dict | None = None) -> int:
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
        if splats is not None and z in SPLAT_RADIUS:
            shift = MAXZ - z
            spx, spy = splats["px"] >> shift, splats["py"] >> shift
            ssel = (spx // TILE == x) & (spy // TILE == yu)
            rad = SPLAT_RADIUS[z]
            for sx, sy_up, srgb in zip(spx[ssel] - x * TILE,
                                       spy[ssel] - yu * TILE,
                                       splats["rgb"][ssel]):
                sy = (TILE - 1) - sy_up
                for dy in range(-rad, rad + 1):
                    for dx in range(-rad, rad + 1):
                        if dx*dx + dy*dy > rad*rad:
                            continue
                        yy, xx = sy + dy, sx + dx
                        if 0 <= yy < TILE and 0 <= xx < TILE:
                            img[yy, xx] = np.maximum(img[yy, xx], srgb)
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
    parser.add_argument("--force-aggregate", action="store_true",
                         help="re-aggregate pixels_z9.parquet even if it looks fresh")
    parser.add_argument("--splat-min-cited", type=int, default=60000,
                         help="cited_by_count threshold for z8-9 citation splats (~p99.9)")


def run(args) -> int:
    web = args.web or str(data_dir() / "coords_web.parquet")
    out = Path(args.out) if args.out else data_dir() / "tiles"
    pixels = str(data_dir() / "pixels_z9.parquet")
    stale = (
        Path(pixels).exists()
        and Path(web).exists()
        and os.path.getmtime(web) > os.path.getmtime(pixels)
    )
    if Path(pixels).exists() and not stale:
        cols = {r[0] for r in duckdb.connect().execute(
            f"DESCRIBE SELECT * FROM read_parquet('{pixels}')").fetchall()}
        if "field" in cols:            # old schema cached before Task 4
            stale = True
    if args.force_aggregate or not Path(pixels).exists() or stale:
        n = aggregate_z9(web, pixels, duckdb.connect())
        print(f"aggregated {n:,} occupied z9 pixels", flush=True)
    else:
        print("reusing cached pixels_z9.parquet (delete it or re-run webcoords to force)",
              flush=True)
    con = duckdb.connect()
    stats = load_community_stats(con, web)          # single parquet scan feeds both
    pal = {c: community_rgb(c, f, n) for c, f, n in stats}
    zooms = sorted(_parse_zooms(args.zooms), reverse=True)  # deep -> shallow
    level = load_level9(pixels, pal)
    splats = (load_splats(con, web, pal, args.splat_min_cited)
              if any(z in SPLAT_RADIUS for z in zooms) else None)
    index_dir = data_dir() / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    n_legend = write_legend(stats, str(index_dir / "legend.json"),
                            str(data_dir() / "regions.json"))
    print(f"legend: {n_legend} communities written", flush=True)
    for z in range(MAXZ, -1, -1):
        if z in zooms:
            w = render_zoom(level, z, out, bloom=(z >= 8), splats=splats)
            print(f"zoom {z}: {w} tiles written", flush=True)
        if z > 0:
            level = reduce_level(level)
    return 0
