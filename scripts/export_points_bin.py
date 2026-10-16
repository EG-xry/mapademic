"""Binary point-cloud export for a WebGL vector-points experiment (R15a).

Standalone experiment script, not a pipeline stage: reads coords_web.parquet
and writes two flat little-endian arrays under data/vector_experiment/ for
GPU-friendly progressive loading -- positions.bin (x,y uint16, 4 bytes/point)
and attrs.bin (r,g,b,size uint8, 4 bytes/point) -- plus meta.json. data/index
and data/tiles are untouched.

R16: rows with is_ring=true (re-scattered dust authors with no retained
coauthor links -- see pipeline/webcoords.py's ring_pred) are dropped from
the export entirely. They stay searchable via data/index but are no longer
drawn as interactive point-cloud dots.

Colors reuse pipeline.palette (same min_members=1000 rule as the production
legend); points whose community falls below that threshold share one fixed
grey rather than each getting its own tiny per-community jitter, so the
tail reads as a single consistent dust color (close to the viewer's
#565660).
"""
import argparse
import json
import time
from pathlib import Path

import duckdb
import numpy as np

from pipeline.config import apply_resource_limits, data_dir
from pipeline.palette import (build_group_palette, community_rgb, group_key, group_key_expr,
                              group_rgb_u8, load_community_stats, not_dust_sql, sparse_communities)
from pipeline.sizing import size_byte

QUANT = 65535       # xw,yw uint16 quantization scale (round(v * QUANT))
SIZE_SCALE = 700    # sqrt(cited_by_count) scale cap; only edges_attr.bin's s still uses it
MIN_MEMBERS = 1000  # same threshold as palette.build_community_palette

# Canonical dust/tail grey: community_rgb() for a fixed sub-threshold
# community (id 0, 0 members) rather than each community's own jittered
# grey, so every below-threshold point -- ring/dust included -- renders as
# the same color. Close to the viewer's #565660 dust family.
GREY_RGB = community_rgb(0, 0, 0.5, 0.5, min_members=MIN_MEMBERS)


def _rgb_to_u8(rgb) -> np.ndarray:
    return np.clip(np.round(np.array(rgb, dtype=np.float64) * 255), 0, 255).astype(np.uint8)


def _color_palette(con, web_path: str) -> dict:
    """palette.group_key -> RGB in [0,1]: community colours, sub-group shades,
    and the shared GREY_RGB for tail (< MIN_MEMBERS) and faded communities."""
    stats = load_community_stats(con, web_path)
    pal = build_group_palette(con, web_path, min_members=MIN_MEMBERS, stats=stats)
    faded = sparse_communities(con, web_path, MIN_MEMBERS)
    for community, _field, members, _cx, _cy in stats:
        if members < MIN_MEMBERS or community in faded:
            pal[group_key(community, -1)] = GREY_RGB
    return pal


def export_points(web_path: str, out_dir: Path, sample: int | None = None) -> dict:
    """Core export: builds positions.bin, attrs.bin, meta.json under out_dir.

    Deterministic ordering: cited_by_count DESC, id. --sample takes the
    first N rows of that ordering (highest-cited first).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    apply_resource_limits(con)

    pal = _color_palette(con, web_path)

    limit_clause = f"LIMIT {int(sample)}" if sample else ""
    cols = con.execute(
        f"""SELECT CAST(xw AS DOUBLE) AS xw, CAST(yw AS DOUBLE) AS yw,
                   cited_by_count, {group_key_expr(con, web_path)} AS gkey, community
            FROM read_parquet('{web_path}')
            WHERE {not_dust_sql(con, web_path)}
            ORDER BY cited_by_count DESC, id
            {limit_clause}"""
    ).fetchnumpy()

    n = cols["xw"].shape[0]

    x_q = np.clip(np.round(cols["xw"] * QUANT), 0, QUANT).astype("<u2")
    y_q = np.clip(np.round(cols["yw"] * QUANT), 0, QUANT).astype("<u2")
    positions = np.empty((n, 2), dtype="<u2")
    positions[:, 0] = x_q
    positions[:, 1] = y_q

    rgb = group_rgb_u8(pal, cols["gkey"], default=GREY_RGB)  # (n, 3) uint8

    # R33: the shared log size rule (pipeline/sizing.py), read by the viewer's
    # point-cloud shader with the same formula as the raster discs
    s = size_byte(cols["cited_by_count"])

    attrs = np.empty((n, 4), dtype=np.uint8)
    attrs[:, 0:3] = rgb
    attrs[:, 3] = s

    positions_path = out_dir / "positions.bin"
    attrs_path = out_dir / "attrs.bin"
    positions.tofile(positions_path)
    attrs.tofile(attrs_path)
    groups = _write_community_bins(con, web_path, out_dir / "communities",
                                   cols["community"], positions, attrs)
    dust_n = _write_dust_bin(con, web_path, out_dir / "dust.bin", pal)

    meta = {
        "count": int(n),
        "files": {"positions": "positions.bin", "attrs": "attrs.bin"},
        "record_layout": {
            "positions.bin": "little-endian uint16 x, uint16 y per point (4 bytes/point)",
            "attrs.bin": "uint8 r, uint8 g, uint8 b, uint8 s per point (4 bytes/point)",
        },
        "quantization": QUANT,
        "size": "pipeline.sizing.size_byte: round(255 * log1p(cited/100) / log1p(1500)), clipped",
        "generated_from": "coords_web.parquet",
        "filters": "WHERE NOT is_ring (dust authors excluded; kept searchable via data/index, not in the point cloud)",
        "communities": {"dir": "communities", "layout": "<community>.bin: the community's positions "
                        "records then its attrs records, cited-desc", "counts": groups},
        "dust": {"file": "dust.bin", "count": dust_n,
                 "layout": "positions records then attrs records: the outer ring, small and "
                           "near-empty communities (coords_web is_dust), cited-desc"},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _write_community_bins(con, web_path: str, out_dir: Path, community: np.ndarray,
                          positions: np.ndarray, attrs: np.ndarray) -> dict:
    """communities/<c>.bin for every legend community: all of its members, so
    the viewer can draw a filtered group whole (the JSON community shards
    stop at the top 20,000 authors, a sliver of a 1.4M-member group).
    Positions records then attrs records, both in the export's cited-desc
    order; 8 bytes per author."""
    import shutil
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    faded = sparse_communities(con, web_path, MIN_MEMBERS)
    legend = [c for c, _f, n, _x, _y in load_community_stats(con, web_path)
              if n >= MIN_MEMBERS and c not in faded]
    community = np.asarray(community, dtype=np.int64)
    order = np.argsort(community, kind="stable")             # keeps cited-desc within a community
    sorted_c = community[order]
    counts = {}
    for c in legend:
        lo, hi = np.searchsorted(sorted_c, [c, c + 1])
        idx = order[lo:hi]
        with open(out_dir / f"{c}.bin", "wb") as f:
            positions[idx].tofile(f)
            attrs[idx].tofile(f)
        counts[str(c)] = int(len(idx))
    return counts


def _write_dust_bin(con, web_path: str, path: Path, pal: dict) -> int:
    """dust.bin: every dust author (coords_web is_dust: the outer ring, small
    and near-empty communities), which tiles and the point cloud leave out,
    for the viewer's switchable dust layer. Same 8-byte records as
    communities/<c>.bin."""
    cols = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{web_path}')").fetchall()}
    if "is_dust" not in cols:
        path.unlink(missing_ok=True)
        return 0
    d = con.execute(
        f"""SELECT CAST(xw AS DOUBLE) AS xw, CAST(yw AS DOUBLE) AS yw, cited_by_count,
                   {group_key_expr(con, web_path)} AS gkey
            FROM read_parquet('{web_path}') WHERE is_dust
            ORDER BY cited_by_count DESC, id"""
    ).fetchnumpy()
    n = len(d["xw"])
    pos = np.empty((n, 2), dtype="<u2")
    pos[:, 0] = np.clip(np.round(d["xw"] * QUANT), 0, QUANT)
    pos[:, 1] = np.clip(np.round(d["yw"] * QUANT), 0, QUANT)
    attrs = np.empty((n, 4), dtype=np.uint8)
    attrs[:, 0:3] = group_rgb_u8(pal, d["gkey"], default=GREY_RGB)
    attrs[:, 3] = size_byte(d["cited_by_count"])
    with open(path, "wb") as f:
        pos.tofile(f)
        attrs.tofile(f)
    return n


def add_parser(parser) -> None:
    parser.add_argument("--web", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--sample", type=int, default=None, help="export only the first N points (for tests)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_parser(parser)
    args = parser.parse_args()
    web = args.web or str(data_dir() / "coords_web.parquet")
    out = Path(args.out) if args.out else data_dir() / "vector_experiment"
    t0 = time.time()
    meta = export_points(web, out, sample=args.sample)
    elapsed = time.time() - t0
    pos_size = (out / "positions.bin").stat().st_size
    attrs_size = (out / "attrs.bin").stat().st_size
    print(f"{meta['count']:,} points -> {out} in {elapsed:.1f}s "
          f"(positions.bin {pos_size:,}B, attrs.bin {attrs_size:,}B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
