"""Aggregate web coords at zoom MAXZ, pyramid-reduce, write styled PNG tiles."""
import json
import os
from pathlib import Path

import duckdb
import numpy as np
from PIL import Image

from pipeline import sizing
from pipeline.config import apply_resource_limits, data_dir
from pipeline.palette import (build_group_palette, community_rgb, group_key, group_key_expr,
                              load_community_stats, not_dust_sql, sparse_communities)

MAXZ = 10
TILE = 256
PIX = (2 ** MAXZ) * TILE  # 262144 virtual pixels per side at zoom MAXZ

PIXELS_NAME = f"pixels_z{MAXZ}.parquet"


def aggregate_maxz(webcoords_path: str, out_path: str, con) -> int:
    """Per-pixel count + dominant colour group (community, sub-group), dust
    left out (the viewer draws it as its own layer). Returns occupied pixels."""
    apply_resource_limits(con)
    gkey = group_key_expr(con, webcoords_path)
    return con.execute(
        f"""
        COPY (
            WITH binned AS (
                SELECT least({PIX - 1}, CAST(floor(CAST(xw AS DOUBLE) * {PIX}) AS INT)) AS px,
                       least({PIX - 1}, CAST(floor(CAST(yw AS DOUBLE) * {PIX}) AS INT)) AS py,
                       {gkey} AS gkey
                FROM read_parquet('{webcoords_path}') WHERE {not_dust_sql(con, webcoords_path)}
            ),
            grouped AS (
                SELECT px, py, gkey, count(*) AS c
                FROM binned GROUP BY px, py, gkey
            ),
            ranked AS (
                SELECT px, py, gkey, c,
                       sum(c) OVER (PARTITION BY px, py) AS cnt,
                       row_number() OVER (PARTITION BY px, py
                                          ORDER BY c DESC, gkey) AS rn
                FROM grouped
            )
            SELECT px, py, CAST(cnt AS BIGINT) AS cnt, gkey
            FROM ranked WHERE rn = 1
        ) TO '{out_path}' (FORMAT PARQUET)
        """
    ).fetchone()[0]


def load_level9(pixels_path: str, palette: dict[int, tuple]) -> dict:
    con = duckdb.connect()
    t = con.execute(
        f"SELECT px, py, cnt, gkey FROM read_parquet('{pixels_path}')"
        " ORDER BY py, px").fetchnumpy()
    comm = t["gkey"].astype(np.int64)
    uniq, inv = np.unique(comm, return_inverse=True)
    lut = np.array([palette.get(int(c), (0.35, 0.35, 0.35)) for c in uniq],
                   dtype=np.float32)
    return {"px": t["px"].astype(np.int64), "py": t["py"].astype(np.int64),
            "cnt": t["cnt"].astype(np.int64), "rgb": lut[inv]}


def load_markers(con, web_path: str, palette: dict) -> dict:
    """Authors drawn as citation-sized discs (pipeline.sizing) at z8+: every
    non-ring author whose disc at MAXZ outgrows its own density pixel. Centres
    stay in world units (float) so discs are anti-aliased at sub-pixel
    precision at every zoom."""
    t_min = sizing.min_t_drawn(MAXZ)
    t = con.execute(
        f"""SELECT CAST(xw AS DOUBLE) AS xw, CAST(yw AS DOUBLE) AS yw,
                   cited_by_count, {group_key_expr(con, web_path)} AS gkey
            FROM read_parquet('{web_path}') WHERE {not_dust_sql(con, web_path)}"""
    ).fetchnumpy()
    st = sizing.size_t(t["cited_by_count"])
    keep = st >= t_min
    comm = t["gkey"][keep].astype(np.int64)
    uniq, inv = np.unique(comm, return_inverse=True)
    lut = np.array([palette.get(int(c), (0.6, 0.6, 0.6)) for c in uniq], dtype=np.float32)
    return {"xw": np.asarray(t["xw"][keep], np.float64), "yw": np.asarray(t["yw"][keep], np.float64),
            "t": st[keep], "rgb": lut[inv].reshape(-1, 3)}


def _bucket_markers(markers: dict, z: int) -> tuple[dict, dict]:
    """Per-zoom marker geometry plus {(tile x, tile y_up): indices} for every
    tile a disc (with its anti-aliased rim) touches - at most 2x2 tiles, since
    a radius is far below a tile."""
    scale = float((1 << z) * TILE)
    r = sizing.radius_px(z, markers["t"])
    keep = np.flatnonzero(r >= sizing.MIN_DRAW_PX)
    geo = {"sx": markers["xw"][keep] * scale, "su": markers["yw"][keep] * scale,
           "r": r[keep],
           "color": (markers["rgb"][keep] * sizing.brightness(markers["t"][keep])[:, None]
                     ).astype(np.float32)}
    if not len(keep):
        return geo, {}
    ext = geo["r"] + 0.5
    lo_x = np.floor((geo["sx"] - ext) / TILE).astype(np.int64)
    hi_x = np.floor((geo["sx"] + ext) / TILE).astype(np.int64)
    lo_y = np.floor((geo["su"] - ext) / TILE).astype(np.int64)
    hi_y = np.floor((geo["su"] + ext) / TILE).astype(np.int64)
    idx = np.arange(len(keep))
    keys, owners = [], []
    for tx in (lo_x, hi_x):
        for ty in (lo_y, hi_y):
            keys.append(tx * (1 << 20) + ty)
            owners.append(idx)
    keys, owners = np.concatenate(keys), np.concatenate(owners)
    pairs = np.unique(np.stack([keys, owners], axis=1), axis=0)   # lo == hi duplicates
    bounds = np.flatnonzero(np.diff(pairs[:, 0])) + 1
    buckets = {}
    for chunk in np.split(pairs, bounds):
        k = int(chunk[0, 0])
        buckets[(k >> 20, k & ((1 << 20) - 1))] = chunk[:, 1]
    return geo, buckets


def _draw_discs(img: np.ndarray, lx: np.ndarray, ly: np.ndarray,
                r: np.ndarray, color: np.ndarray) -> None:
    """Anti-aliased discs, max-blended into img (rows down). lx/ly are
    continuous tile-local coordinates; pixel (col, row) has its centre at
    (col + 0.5, row + 0.5)."""
    reach = int(np.ceil(r.max() + 0.5))
    off = np.arange(-reach, reach + 1)
    ox, oy = (a.ravel() for a in np.meshgrid(off, off))
    cx = np.floor(lx).astype(np.int64)[:, None] + ox[None, :]
    cy = np.floor(ly).astype(np.int64)[:, None] + oy[None, :]
    a = np.clip(r[:, None] + 0.5 - np.hypot(cx + 0.5 - lx[:, None], cy + 0.5 - ly[:, None]),
                0.0, 1.0)
    ok = (a > 0) & (cx >= 0) & (cx < TILE) & (cy >= 0) & (cy < TILE)
    vals = (color[:, None, :] * a[:, :, None].astype(np.float32))[ok]
    np.maximum.at(img, (cy[ok], cx[ok]), vals)


def write_legend(stats: list[tuple[int, str | None, int, float, float]], out_path: str,
                 regions_path: str | None, min_members: int = 1000,
                 palette: dict | None = None, faded: set[int] = frozenset(),
                 subs: dict[int, list[tuple[int, int]]] | None = None) -> int:
    """index/legend.json: one entry per coloured community (faded ones are
    dust, so absent), with its kind (international field or national network)
    and, for split communities, its sub-groups' names, sizes and shades."""
    regions = json.loads(Path(regions_path).read_text()) if regions_path and Path(regions_path).exists() else []
    top = {r["community"]: r for r in regions if "community" in r and r.get("level", 1) == 1}
    sub_regions = {(r["community"], r["sub"]): r for r in regions if r.get("level") == 2}
    hexc = lambda rgb: "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in rgb)
    entries = []
    for c, f, n, cx, cy in stats:       # stats already ordered members DESC, community
        if n < min_members or c in faded:
            continue
        rgb = (palette or {}).get(group_key(c, -1)) or community_rgb(c, n, cx, cy, min_members)
        e = {"community": c, "name": top.get(c, {}).get("name"),
             "kind": top.get(c, {}).get("kind", "field"),
             "field": f, "members": n, "color": hexc(rgb)}
        if subs and c in subs:
            e["subs"] = [{"sub": s_, "members": m,
                          "name": sub_regions.get((c, s_), {}).get("name"),
                          "xw": sub_regions.get((c, s_), {}).get("xw"),
                          "yw": sub_regions.get((c, s_), {}).get("yw"),
                          "zmin": sub_regions.get((c, s_), {}).get("zmin"),
                          "color": hexc((palette or {}).get(group_key(c, s_), rgb))}
                         for s_, m in subs[c]]
        entries.append(e)
    tmp = str(out_path) + ".tmp"
    Path(tmp).write_text(json.dumps(entries, ensure_ascii=False, indent=1))
    os.replace(tmp, out_path)
    return len(entries)


def legend_inputs(con, web_path: str, stats) -> tuple[dict, set[int], dict]:
    """(group palette, faded communities, {community: [(sub, members)]})."""
    pal = build_group_palette(con, web_path, stats=stats)
    faded = sparse_communities(con, web_path)
    subs: dict[int, list] = {}
    if "sub" in group_key_expr(con, web_path):
        for c, s_, m in con.execute(f"""SELECT community, sub, count(*) FROM read_parquet('{web_path}')
                                        WHERE sub >= 0 GROUP BY 1, 2 ORDER BY 1, 2""").fetchall():
            subs.setdefault(int(c), []).append((int(s_), int(m)))
    return pal, faded, subs


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

EDGE_ALPHA = 0.08
EDGE_RGB = np.array([0.45, 0.50, 0.60], dtype=np.float32)  # cool grey
EDGE_MINZ = 8
EDGE_MAXZ = 9  # regular edges cap out here; z10 (MAXZ) stays edge-free, see
               # per-author coauthor rays in the viewer for max-zoom detail
W1_ALPHA = 0.05  # weight-1 edges, drawn only at z == MAXZ


def load_edges(path: str) -> dict:
    t = duckdb.connect().execute(
        f"SELECT x0, y0, x1, y1 FROM read_parquet('{path}')").fetchnumpy()
    return {k: t[k].astype(np.int64) for k in ("x0", "y0", "x1", "y1")}


def _bucket_edges(edges: dict, z: int) -> tuple[dict, int]:
    """Group edge indices by the shifted-z tile of their midpoint, each edge
    stored exactly ONCE (no replication into neighbours: at 14M+ edges the
    replicated variant peaked at tens of GB). Returns (buckets, radius); at
    render time a tile unions its neighbour buckets within `radius`, which is
    derived from HALF the max edge length at this zoom - any point on a
    segment is at most len/2 from the midpoint. Still avoids rescanning all
    edges for each of up to 1024x1024 tiles at z10."""
    shift = MAXZ - z
    ex0, ey0 = edges["x0"] >> shift, edges["y0"] >> shift
    ex1, ey1 = edges["x1"] >> shift, edges["y1"] >> shift
    n = len(ex0)
    if n == 0:
        return {}, 0
    max_len = int(max(np.abs(ex1 - ex0).max(), np.abs(ey1 - ey0).max()))
    radius = (max_len // 2) // TILE + 1
    mx, my = ((ex0 + ex1) // 2) // TILE, ((ey0 + ey1) // 2) // TILE
    key = mx * (1 << 20) + my
    order = np.argsort(key, kind="stable")     # idx array == order (idx was arange)
    key_s = key[order]
    mx_s, my_s = mx[order], my[order]
    boundaries = np.flatnonzero(np.diff(key_s)) + 1
    starts = np.concatenate(([0], boundaries))
    groups = np.split(order, boundaries)
    buckets = {(int(mx_s[s]), int(my_s[s])): g for s, g in zip(starts, groups)}
    return buckets, radius


def _accumulate_edges(acc: np.ndarray, edges: dict, z: int, x: int, yu: int,
                      buckets: dict, radius: int, alpha: float) -> None:
    """Rasterize edges near tile (x, yu) additively into the shared alpha
    buffer `acc` (alpha per sample; no clip or color here). Both edge sets add
    into ONE buffer so the 0.25 glow ceiling is shared for future co-occurring
    layers. Today at most one layer draws per zoom. render_zoom clips the
    combined buffer once and applies EDGE_RGB once."""
    hits = [buckets[(bx, by)]
            for bx in range(x - radius, x + radius + 1)
            for by in range(yu - radius, yu + radius + 1)
            if (bx, by) in buckets]
    idxs = np.concatenate(hits) if hits else None
    if idxs is None or not len(idxs):
        return
    shift = MAXZ - z
    ex0, ey0 = edges["x0"][idxs] >> shift, edges["y0"][idxs] >> shift
    ex1, ey1 = edges["x1"][idxs] >> shift, edges["y1"][idxs] >> shift
    # edges whose bbox intersects this tile
    tx0, ty0 = x * TILE, yu * TILE
    esel = ((np.minimum(ex0, ex1) < tx0 + TILE)
            & (np.maximum(ex0, ex1) >= tx0)
            & (np.minimum(ey0, ey1) < ty0 + TILE)
            & (np.maximum(ey0, ey1) >= ty0))
    for a0, b0, a1, b1 in zip(ex0[esel], ey0[esel], ex1[esel], ey1[esel]):
        ns = max(2, int(max(abs(a1 - a0), abs(b1 - b0))) + 1)
        xs = np.linspace(a0, a1, ns).round().astype(np.int64) - tx0
        ys_up = np.linspace(b0, b1, ns).round().astype(np.int64) - ty0
        keep = (xs >= 0) & (xs < TILE) & (ys_up >= 0) & (ys_up < TILE)
        np.add.at(acc, ((TILE - 1) - ys_up[keep], xs[keep]), alpha)


def render_zoom(level: dict, z: int, out_dir: Path, bloom: bool,
                markers: dict | None = None, edges: dict | None = None,
                edges_w1: dict | None = None) -> int:
    """Write XYZ PNG tiles for one zoom; skip existing. Returns tiles written."""
    ntiles = 1 << z
    px, py = level["px"], level["py"]  # already in zoom-z pixel space
    bright = _brightness(level["cnt"])
    color = level["rgb"] * bright[:, None]
    tx, ty_up = px // TILE, py // TILE
    edge_buckets, edge_radius = ((None, 0)
                                 if edges is None or z < EDGE_MINZ or z > EDGE_MAXZ
                                 else _bucket_edges(edges, z))
    w1_buckets, w1_radius = ((None, 0) if edges_w1 is None or z != MAXZ
                             else _bucket_edges(edges_w1, z))
    mgeo, mbuckets = ((None, {}) if markers is None or z < sizing.DISC_MIN_ZOOM
                      else _bucket_markers(markers, z))
    # pixels sorted by tile once, so each tile is a slice instead of a scan of
    # every pixel (8.5M at z10 x ~500k tiles was hours of comparisons)
    pix_key = tx * ntiles + ty_up
    pix_order = np.argsort(pix_key, kind="stable")
    pix_key_sorted = pix_key[pix_order]
    tiles_todo = np.unique(np.concatenate([
        pix_key,
        np.array([bx * ntiles + by for bx, by in mbuckets
                  if 0 <= bx < ntiles and 0 <= by < ntiles], dtype=np.int64)]))
    written = 0
    for t in tiles_todo:
        x, yu = int(t // ntiles), int(t % ntiles)
        ty = (ntiles - 1) - yu                       # XYZ y-flip, here only
        path = out_dir / str(z) / str(x) / f"{ty}.png"
        if path.exists():
            continue
        lo, hi = np.searchsorted(pix_key_sorted, [t, t + 1])
        sel = pix_order[lo:hi]
        img = np.zeros((TILE, TILE, 3), dtype=np.float32)
        if w1_buckets is not None or edge_buckets is not None:
            acc = np.zeros((TILE, TILE), dtype=np.float32)
            if w1_buckets is not None:
                _accumulate_edges(acc, edges_w1, z, x, yu, w1_buckets, w1_radius,
                                  W1_ALPHA)
            if edge_buckets is not None:
                _accumulate_edges(acc, edges, z, x, yu, edge_buckets, edge_radius,
                                  EDGE_ALPHA)
            # shared-acc structure for future co-occurring layers; today at most one draws
            img += np.clip(acc, 0, 0.25)[:, :, None] * EDGE_RGB
        ix = px[sel] - x * TILE
        iy_up = py[sel] - yu * TILE
        iy = (TILE - 1) - iy_up                      # flip rows inside the tile
        img[iy, ix] = color[sel]
        hit = mbuckets.get((x, yu))
        if hit is not None:   # citation-sized discs (pipeline.sizing), rows flip like pixels
            _draw_discs(img, mgeo["sx"][hit] - x * TILE, TILE - (mgeo["su"][hit] - yu * TILE),
                        mgeo["r"][hit], mgeo["color"][hit])
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
    parser.add_argument("--zooms", default=f"0-{MAXZ}", help="e.g. 0-5 for the QA gate")
    parser.add_argument("--web", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--force-aggregate", action="store_true",
                         help=f"re-aggregate {PIXELS_NAME} even if it looks fresh")
    parser.add_argument("--edges", nargs="?", const="__default__", default=None,
                         help=f"bake faint coauthor edges into z{EDGE_MINZ}-z{EDGE_MAXZ}"
                              " tiles; bare flag uses data/edges_px.parquet;"
                              " already-rendered tiles are skipped, so delete the"
                              f" z{EDGE_MINZ}-z{EDGE_MAXZ} tile dirs first to re-bake")
    parser.add_argument("--edges-w1", nargs="?", const="__default__", default=None,
                         help=f"bake faint weight-1 coauthor edges into z{MAXZ} tiles"
                              " only; bare flag uses data/edges_px_w1.parquet;"
                              " already-rendered tiles are skipped, so delete the"
                              f" z{MAXZ} tile dir first to re-bake")


def run(args) -> int:
    web = args.web or str(data_dir() / "coords_web.parquet")
    out = Path(args.out) if args.out else data_dir() / "tiles"
    pixels = str(data_dir() / PIXELS_NAME)
    stale = (
        Path(pixels).exists()
        and Path(web).exists()
        and os.path.getmtime(web) > os.path.getmtime(pixels)
    )
    if Path(pixels).exists() and not stale:
        cols = {r[0] for r in duckdb.connect().execute(
            f"DESCRIBE SELECT * FROM read_parquet('{pixels}')").fetchall()}
        if "field" in cols or "gkey" not in cols:   # cached before Task 4 / sub-groups
            stale = True
    if args.force_aggregate or not Path(pixels).exists() or stale:
        n = aggregate_maxz(web, pixels, duckdb.connect())
        print(f"aggregated {n:,} occupied z{MAXZ} pixels", flush=True)
    else:
        print(f"reusing cached {PIXELS_NAME} (delete it or re-run webcoords to force)",
              flush=True)
    con = duckdb.connect()
    stats = load_community_stats(con, web)          # single parquet scan feeds both
    pal, faded, subs = legend_inputs(con, web, stats)
    zooms = sorted(_parse_zooms(args.zooms), reverse=True)  # deep -> shallow
    level = load_level9(pixels, pal)
    markers = (load_markers(con, web, pal)
               if any(z >= sizing.DISC_MIN_ZOOM for z in zooms) else None)
    edges = None
    if args.edges:
        # Check if any zoom in the requested range can draw edges
        if any(EDGE_MINZ <= z <= EDGE_MAXZ for z in zooms):
            epath = (str(data_dir() / "edges_px.parquet")
                     if args.edges == "__default__" else args.edges)
            edges = load_edges(epath)
            print(f"baking {len(edges['x0']):,} edges into z{EDGE_MINZ}-z{EDGE_MAXZ} tiles",
                  flush=True)
        else:
            zoom_range = f"{min(zooms)}-{max(zooms)}"
            print(f"note: --edges has no effect for zooms {zoom_range} (edges bake at z{EDGE_MINZ}-z{EDGE_MAXZ})",
                  flush=True)
    edges_w1 = None
    if args.edges_w1:
        # Check if z10 (MAXZ) is in the requested zooms
        if MAXZ in zooms:
            w1path = (str(data_dir() / "edges_px_w1.parquet")
                      if args.edges_w1 == "__default__" else args.edges_w1)
            edges_w1 = load_edges(w1path)
            print(f"baking {len(edges_w1['x0']):,} weight-1 edges into z{MAXZ} tiles", flush=True)
        else:
            zoom_range = f"{min(zooms)}-{max(zooms)}"
            print(f"note: --edges-w1 has no effect for zooms {zoom_range} (edges-w1 bake at z{MAXZ} only)",
                  flush=True)
    index_dir = data_dir() / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    n_legend = write_legend(stats, str(index_dir / "legend.json"),
                            str(data_dir() / "regions.json"), palette=pal, faded=faded, subs=subs)
    print(f"legend: {n_legend} communities written", flush=True)
    for z in range(MAXZ, -1, -1):
        if z in zooms:
            w = render_zoom(level, z, out, bloom=(z >= MAXZ - 1), markers=markers,
                            edges=edges, edges_w1=edges_w1)
            print(f"zoom {z}: {w} tiles written", flush=True)
        if z > 0:
            level = reduce_level(level)
    return 0
