"""Label/hit tiles (zooms 6-9) and prefix search shards. Zero backend.

Label tile entry shape: [display_name, id, xw, yw, cited, community]
(community as int). Backward-compat: consumers must index fields
positionally 0..4 as before; field 5 (community) is additive.

Search shard fallback order (viewer): try prefix shards from
min(len(normalized concatenated name), 5) chars down to 2 chars, then the
codepoint shard `_<ord(first char of normalized name) mod 128>`, then `_`.
That lookup is unchanged. What changed is which shards an author's entry
lives in: each author is indexed under BOTH the first token and the last
token of their normalized name (skipped for single-token names, when
first token == last token, and when the last token is under 2 chars, e.g.
an initial). Both insertions use the same entry (the full
spaced norm, not a rotated one) -- an entry is just findable via either
token's shard family. An author's id can therefore appear in up to two
shard files; the viewer dedupes by id client-side.
(The docstring doubles as argparse help, so no percent signs here.)
"""
import json
import math
import os
import shutil
import unicodedata
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np

from pipeline.config import apply_resource_limits, data_dir
from pipeline.palette import load_community_stats

LABEL_ZOOMS = {6: 50, 7: 50, 8: 200, 9: 4000}   # zoom -> per-tile capacity
SHARD_SPLIT_BYTES = 4_000_000                   # shards larger than this split by one more prefix char
MAX_PREFIX_LEN = 5                              # deepest prefix-shard key length
CATCHALL_BUCKETS = 128                          # codepoint-modulus for the `_` catch-all split (JS mirror: web/dev.html)


def normalize(name: str) -> str:
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = "".join(c if (c.isalnum() or c == " ") else " " for c in s)
    return " ".join(s.split())


def build_label_tiles(web: str, out_dir: Path) -> int:
    labels_dir = out_dir / "labels"
    if labels_dir.exists():
        shutil.rmtree(labels_dir)
    con = duckdb.connect()
    apply_resource_limits(con)
    written = 0
    for z, cap in LABEL_ZOOMS.items():
        ntiles = 1 << z
        rows = con.execute(
            f"""
            SELECT tx, ty_up, display_name, id, xw, yw, cited_by_count, community FROM (
                SELECT least({ntiles - 1}, CAST(floor(CAST(xw AS DOUBLE) * {ntiles}) AS INT)) AS tx,
                       least({ntiles - 1}, CAST(floor(CAST(yw AS DOUBLE) * {ntiles}) AS INT)) AS ty_up,
                       display_name, id, CAST(xw AS DOUBLE) AS xw, CAST(yw AS DOUBLE) AS yw, cited_by_count, community,
                       row_number() OVER (
                           PARTITION BY least({ntiles - 1}, CAST(floor(CAST(xw AS DOUBLE) * {ntiles}) AS INT)),
                                        least({ntiles - 1}, CAST(floor(CAST(yw AS DOUBLE) * {ntiles}) AS INT))
                           ORDER BY cited_by_count DESC, id
                       ) AS rn
                FROM read_parquet('{web}') WHERE NOT is_ring
            ) WHERE rn <= {cap}
            ORDER BY tx, ty_up, cited_by_count DESC, id
            """
        ).fetchall()
        tiles = defaultdict(list)
        for tx, ty_up, name, aid, xw, yw, cited, community in rows:
            tiles[(tx, ty_up)].append(
                [name, aid, round(xw, 6), round(yw, 6), int(cited), int(community)]
            )
        for (tx, ty_up), entries in tiles.items():
            ty = (ntiles - 1) - ty_up               # XYZ y-flip
            p = out_dir / "labels" / str(z) / str(tx) / f"{ty}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.parent / (p.name + ".tmp")
            tmp.write_text(json.dumps({"l": entries}, ensure_ascii=False))
            os.replace(tmp, p)
            written += 1
    return written


def _write_json_atomic(path: Path, obj) -> None:
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False))
    os.replace(tmp, path)


def _shard_bytes(entries) -> int:
    return len(json.dumps(entries, ensure_ascii=False).encode("utf-8"))


def _bucket_key(token: str) -> str:
    """2-char ascii-alpha head of a name token, else the catch-all key.

    Shared by both the first-token and the last-token insertion so the two
    families are bucketed by identical rules.
    """
    head = token[:2]
    if len(head) == 2 and head.isascii() and head.isalpha():
        return head
    return "_"


def _split_prefix_shard(key: str, items: list, out: dict) -> None:
    """Recursively split an alpha-ascii prefix shard by one more prefix char.

    items: list of (sort_token, entry) pairs. sort_token is whichever name
    token (first or last) routed this entry into the shard family -- the
    split key is derived from that token, not from the entry's full norm,
    so a last-token (surname) family recurses on the surname's own chars.

    A split parent is always written (possibly small) so the viewer fallback
    chain is deterministic. Entries whose sort_token is exactly len(key)
    chars, or whose next-char prefix is not alpha-ascii, stay at the parent
    level.
    """
    k = len(key)
    if _shard_bytes([e for _, e in items]) <= SHARD_SPLIT_BYTES or k >= MAX_PREFIX_LEN:
        out[key] = [e for _, e in items]
        return
    sub = defaultdict(list)
    parent = []
    for token, e in items:
        child = token[:k + 1]
        if len(token) > k and child.isascii() and child.isalpha():
            sub[child].append((token, e))
        else:
            parent.append((token, e))
    out[key] = [e for _, e in parent]           # always written: viewer fallback
    for child_key, child_items in sub.items():
        _split_prefix_shard(child_key, child_items, out)


def _split_catchall_shard(items: list, out: dict) -> None:
    """Split the `_` catch-all by first-char codepoint when oversized.

    items: list of (sort_token, entry) pairs (see _split_prefix_shard).
    Entries go to `_<ord(first char of sort_token) % CATCHALL_BUCKETS>` (JS
    mirror: "_" + (norm.codePointAt(0) % 128)); entries with an empty
    sort_token stay in `_`, which is always written.
    """
    if _shard_bytes([e for _, e in items]) <= SHARD_SPLIT_BYTES:
        out["_"] = [e for _, e in items]
        return
    sub = defaultdict(list)
    parent = []
    for token, e in items:
        if token:
            sub[f"_{ord(token[0]) % CATCHALL_BUCKETS}"].append((token, e))
        else:
            parent.append((token, e))
    out["_"] = [e for _, e in parent]            # always written: viewer fallback
    for child_key, child_items in sub.items():
        out[child_key] = [e for _, e in child_items]


def build_search_shards(web: str, out_dir: Path) -> int:
    search_dir = out_dir / "search"
    if search_dir.exists():
        shutil.rmtree(search_dir)
    con = duckdb.connect()
    apply_resource_limits(con)
    shards = defaultdict(list)
    rows = con.execute(
        f"""SELECT display_name, id, CAST(xw AS DOUBLE) AS xw, CAST(yw AS DOUBLE) AS yw, cited_by_count
            FROM read_parquet('{web}') ORDER BY cited_by_count DESC, id"""
    ).fetchall()
    for name, aid, xw, yw, cited in rows:
        norm = normalize(name or "")
        entry = [norm, name, aid, round(xw, 6), round(yw, 6), int(cited)]
        tokens = norm.split(" ")
        first_tok, last_tok = tokens[0], tokens[-1]
        shards[_bucket_key(first_tok)].append((first_tok, entry))
        if len(tokens) > 1 and first_tok != last_tok and len(last_tok) >= 2:
            shards[_bucket_key(last_tok)].append((last_tok, entry))
    final = {}
    for key, items in shards.items():
        if key == "_":
            _split_catchall_shard(items, final)
        else:
            _split_prefix_shard(key, items, final)
    sdir = out_dir / "search"
    sdir.mkdir(parents=True, exist_ok=True)
    for key, entries in final.items():
        _write_json_atomic(sdir / f"{key}.json", entries)
    return len(rows)


def build_id_shards(web: str, out_dir: Path) -> int:
    ids_dir = out_dir / "ids"
    if ids_dir.exists():
        shutil.rmtree(ids_dir)
    con = duckdb.connect()
    apply_resource_limits(con)
    buckets = defaultdict(dict)
    rows = con.execute(
        f"""SELECT id, CAST(xw AS DOUBLE) AS xw, CAST(yw AS DOUBLE) AS yw
            FROM read_parquet('{web}')"""
    ).fetchall()
    for aid, xw, yw in rows:
        digits = "".join(filter(str.isdigit, aid))
        bucket = int(digits) % 1000 if digits else 0
        buckets[bucket][aid] = [round(xw, 6), round(yw, 6)]
    ids_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for bucket, entries in buckets.items():
        _write_json_atomic(ids_dir / f"{bucket}.json", entries)
        total += len(entries)
    return total


BOUNDARIES_MAX_BYTES = 2_500_000   # serialized-size target for boundaries.json


def _smooth_majority(arr: np.ndarray) -> np.ndarray:
    """One 3x3 majority (mode) filter pass over non-empty cells (-1 = empty).

    For each non-empty cell, the winner is the value that appears most often
    among the 9 cells in its neighborhood (out-of-bounds and empty cells are
    excluded as candidates). A genuine tie between two different values keeps
    the cell's original (center) value unchanged.
    """
    h, w = arr.shape
    padded = np.pad(arr, 1, mode="constant", constant_values=-1)
    offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    center_idx = offsets.index((0, 0))
    stack = np.stack(
        [padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy, dx in offsets]
    )
    valid = stack != -1
    counts = np.zeros(stack.shape, dtype=np.int16)
    for i in range(len(offsets)):
        counts[i] = ((stack == stack[i]) & valid & valid[i]).sum(axis=0)
    counts = np.where(valid, counts, -1)              # invalid candidates can't win
    best_count = counts.max(axis=0)
    is_best = (counts == best_count) & valid
    first_best_idx = np.argmax(is_best, axis=0)        # first candidate hitting best_count
    best_value = np.take_along_axis(stack, first_best_idx[None, :, :], axis=0)[0]
    masked = np.where(is_best, stack, -2)              # -2: sentinel, never a real value
    agrees = (masked == best_value[None, :, :]) | (masked == -2)
    tie = ~agrees.all(axis=0)                          # >1 distinct value at best_count
    smoothed = np.where(tie, stack[center_idx], best_value)
    return np.where(arr != -1, smoothed, -1).astype(arr.dtype)


def _boundary_lines(arr: np.ndarray, grid: int) -> list:
    """Shared edges between 4-adjacent cells of differing non-empty
    communities, in world coords, run-length merged along each row/column."""
    lines = []
    diff_h = (arr[:-1, :] != -1) & (arr[1:, :] != -1) & (arr[:-1, :] != arr[1:, :])
    for cx in range(grid - 1):
        bx = round((cx + 1) / grid, 6)
        col = diff_h[cx].tolist()
        run_start = None
        for cy in range(grid + 1):
            active = cy < grid and col[cy]
            if active:
                if run_start is None:
                    run_start = cy
            elif run_start is not None:
                lines.append([[bx, round(run_start / grid, 6)], [bx, round(cy / grid, 6)]])
                run_start = None
    diff_v = (arr[:, :-1] != -1) & (arr[:, 1:] != -1) & (arr[:, :-1] != arr[:, 1:])
    for cy in range(grid - 1):
        by = round((cy + 1) / grid, 6)
        row = diff_v[:, cy].tolist()
        run_start = None
        for cx in range(grid + 1):
            active = cx < grid and row[cx]
            if active:
                if run_start is None:
                    run_start = cx
            elif run_start is not None:
                lines.append([[round(run_start / grid, 6), by], [round(cx / grid, 6), by]])
                run_start = None
    return lines


def _assemble_polylines(segments: list) -> list:
    """Chain 2-point segments that share an endpoint into connected
    polylines. Simple dict-based endpoint join: greedily extends a chain's
    head/tail with any unused segment touching that point. Choice among
    3+-way junctions is arbitrary but deterministic; good enough for
    boundary-line fragment filtering, not a general planar-graph solver."""
    endpoint_map = defaultdict(list)
    segs = [(tuple(a), tuple(b)) for a, b in segments]
    for i, (p0, p1) in enumerate(segs):
        endpoint_map[p0].append(i)
        endpoint_map[p1].append(i)
    used = [False] * len(segs)
    polylines = []
    for i in range(len(segs)):
        if used[i]:
            continue
        used[i] = True
        chain = [segs[i][0], segs[i][1]]
        while True:
            last = chain[-1]
            nxt = next((j for j in endpoint_map[last] if not used[j]), None)
            if nxt is None:
                break
            a, b = segs[nxt]
            chain.append(b if a == last else a)
            used[nxt] = True
        while True:
            first = chain[0]
            nxt = next((j for j in endpoint_map[first] if not used[j]), None)
            if nxt is None:
                break
            a, b = segs[nxt]
            chain.insert(0, a if b == first else b)
            used[nxt] = True
        polylines.append([list(p) for p in chain])
    return polylines


def _polyline_length(points: list) -> float:
    """Total euclidean length of a polyline (world-coord units)."""
    return sum(
        math.hypot(x1 - x0, y1 - y0)
        for (x0, y0), (x1, y1) in zip(points, points[1:])
    )


def _chaikin_smooth(points: list, iterations: int = 2) -> list:
    """Chaikin corner-cutting: each interior segment is cut at its 1/4 and
    3/4 points, replacing sharp staircase corners with a smooth curve. The
    first and last points are always kept exactly (open polyline). A
    2-point polyline has no corner to cut and is returned unchanged."""
    pts = [tuple(p) for p in points]
    for _ in range(iterations):
        if len(pts) < 3:
            break
        new_pts = [pts[0]]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            new_pts.append((0.75 * x0 + 0.25 * x1, 0.75 * y0 + 0.25 * y1))
            new_pts.append((0.25 * x0 + 0.75 * x1, 0.25 * y0 + 0.75 * y1))
        new_pts.append(pts[-1])
        pts = new_pts
    return [[round(x, 6), round(y, 6)] for x, y in pts]


def build_community_boundaries(web: str, out_path: str, grid: int = 512, min_cell: int = 6,
                                smooth_passes: int = 3, min_len: float = 0.02) -> int:
    """Curved community-boundary polylines: majority-community raster over a
    grid x grid grid (cells with < min_cell authors are empty), smooth_passes
    rounds of the 3x3 majority-filter speckle cleanup, then boundary edges
    between differing neighbors run-length merged, chained into connected
    polylines by shared endpoint, fragments shorter than min_len (world
    units) dropped, and survivors passed through 2 iterations of Chaikin
    corner-cutting so the raster staircase reads as an organic curve.
    Returns the number of surviving polylines."""
    con = duckdb.connect()
    apply_resource_limits(con)
    rows = con.execute(
        f"""
        WITH pts AS (
            SELECT least({grid - 1}, CAST(floor(CAST(xw AS DOUBLE) * {grid}) AS INT)) AS cx,
                   least({grid - 1}, CAST(floor(CAST(yw AS DOUBLE) * {grid}) AS INT)) AS cy,
                   community
            FROM read_parquet('{web}') WHERE NOT is_ring
        ),
        counts AS (
            SELECT cx, cy, community, count(*) AS c FROM pts GROUP BY 1, 2, 3
        ),
        totals AS (
            SELECT cx, cy, sum(c) AS n FROM counts GROUP BY 1, 2
        ),
        ranked AS (
            SELECT cx, cy, community, c,
                   row_number() OVER (PARTITION BY cx, cy ORDER BY c DESC, community ASC) AS rn
            FROM counts
        )
        SELECT r.cx, r.cy, r.community
        FROM ranked r JOIN totals t ON t.cx = r.cx AND t.cy = r.cy
        WHERE r.rn = 1 AND t.n >= {min_cell}
        """
    ).fetchall()
    arr = np.full((grid, grid), -1, dtype=np.int32)
    for cx, cy, community in rows:
        arr[cx, cy] = community
    for _ in range(smooth_passes):
        arr = _smooth_majority(arr)
    segments = _boundary_lines(arr, grid)
    polylines = _assemble_polylines(segments)
    polylines = [p for p in polylines if _polyline_length(p) >= min_len]
    polylines = [_chaikin_smooth(p) for p in polylines]
    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {},
            "geometry": {"type": "MultiLineString", "coordinates": polylines},
        }],
    }
    _write_json_atomic(Path(out_path), geojson)
    size = len(json.dumps(geojson, ensure_ascii=False).encode("utf-8"))
    if size > BOUNDARIES_MAX_BYTES:
        print(f"note: boundaries.json is {size:,} bytes (target < {BOUNDARIES_MAX_BYTES:,}); "
              f"consider raising min_cell (currently {min_cell})")
    return len(polylines)


def build_community_shards(web: str, out_dir: Path, top_n: int = 20000, min_members: int = 1000) -> int:
    """Per-community top-N-by-citations roster, for communities with enough
    members to also appear in the legend (see palette.load_community_stats)."""
    communities_dir = out_dir / "communities"
    if communities_dir.exists():
        shutil.rmtree(communities_dir)
    communities_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    apply_resource_limits(con)
    stats = load_community_stats(con, web)
    qualifying = [c for c, _f, n, _cx, _cy in stats if n >= min_members]
    if not qualifying:
        return 0
    ids_sql = ", ".join(str(c) for c in qualifying)
    rows = con.execute(
        f"""
        SELECT community, display_name, id, CAST(xw AS DOUBLE) AS xw, CAST(yw AS DOUBLE) AS yw, cited_by_count
        FROM (
            SELECT community, display_name, id, xw, yw, cited_by_count,
                   row_number() OVER (PARTITION BY community ORDER BY cited_by_count DESC, id) AS rn
            FROM read_parquet('{web}')
            WHERE NOT is_ring AND community IN ({ids_sql})
        ) WHERE rn <= {top_n}
        ORDER BY community, cited_by_count DESC, id
        """
    ).fetchall()
    shards = defaultdict(list)
    for community, name, aid, xw, yw, cited in rows:
        shards[community].append([name, aid, round(xw, 6), round(yw, 6), int(cited)])
    for community, entries in shards.items():
        _write_json_atomic(communities_dir / f"{community}.json", entries)
    return len(shards)


def add_parser(parser) -> None:
    parser.add_argument("--web", default=None)
    parser.add_argument("--out", default=None)


def run(args) -> int:
    web = args.web or str(data_dir() / "coords_web.parquet")
    out = Path(args.out) if args.out else data_dir() / "index"
    t = build_label_tiles(web, out)
    s = build_search_shards(web, out)
    i = build_id_shards(web, out)
    b = build_community_boundaries(web, str(out / "boundaries.json"))
    m = build_community_shards(web, out)
    print(f"{t:,} label tiles, {s:,} searchable names, {i:,} id-shard entries, "
          f"{b:,} boundary polylines, {m:,} community shards -> {out}")
    return 0
