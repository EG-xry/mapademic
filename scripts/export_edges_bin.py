"""Binary edge export for a WebGL/GPU connection-lines experiment (R22a).

Standalone experiment script, not a pipeline stage: reads edges.parquet
JOINed to coords_web.parquet for both endpoints and writes two flat
little-endian arrays under data/vector_experiment/ for GPU-friendly
progressive loading -- edges_pos.bin (x0,y0,x1,y1 uint16, 8 bytes/edge)
and edges_attr.bin (r0,g0,b0,r1,g1,b1,w,s uint8, 8 bytes/edge) -- plus
edges_fine.bin (x0,y0,x1,y1 int8, 4 bytes/edge: each coordinate's rounding
remainder in 1/254 of a uint16 step; R37, Eric: "connection lines not
perfectly placed upon dots" -- uint16 alone is 1/65535 of the map, up to 4px
off at zoom 10 and 16px at 12, while edges_pos + edges_fine/254 is under half
a pixel at zoom 14; the viewer still draws edges_pos alone if it is absent) --
plus
edges_meta.json (named distinctly from export_points_bin's meta.json,
since both scripts share the vector_experiment/ directory). data/index
and data/tiles are untouched.

Edges touching a ring/dust node (is_ring=true, see pipeline/webcoords.py's
ring_pred) are excluded, same as the point-cloud export. Only the
repeat-collaboration set (weight >= 2) is exported; no edge-length filter
is applied here.

Colors reuse pipeline.palette via scripts.export_points_bin's community
color LUT (same min_members=1000 rule as the legend and the point cloud):
each endpoint gets its own community color, or the shared tail grey if its
community is below threshold.

Strongest-first tier files (edges_{pos,attr}_1m.bin, _3m.bin) are written
as literal prefixes of the full pair, since the export is already ordered
heaviest-first; web/dev.html's EDGE_TIERS fetches the smallest tier covering
its Line count slider (1M is the default first download).
"""
import argparse
import json
import time
from pathlib import Path

import duckdb
import numpy as np

from pipeline.config import apply_resource_limits, data_dir
from pipeline.palette import group_key_expr, group_rgb_u8
from scripts.export_points_bin import GREY_RGB, SIZE_SCALE, _color_palette

QUANT = 65535        # xw,yw uint16 quantization scale (round(v * QUANT))
FINE = 254           # edges_fine.bin: v * QUANT = edges_pos + edges_fine / FINE (254, not 256:
                     # a remainder of +-0.5 step then fits int8 without clipping)
MIN_WEIGHT = 2        # repeat-collaboration threshold
WEIGHT_CAP = 20        # weight clamp before scaling to uint8
# name -> edge count; names are the file suffixes web/dev.html's EDGE_TIERS fetch
TIERS = {"1m": 1_000_000, "3m": 3_000_000}


def export_edges(edges_path: str, web_path: str, out_dir: Path, sample: int | None = None,
                 tiers: dict[str, int] | None = None) -> dict:
    """Core export: builds edges_pos.bin, edges_attr.bin, meta.json under out_dir.

    Deterministic ordering: weight DESC, src, dst. --sample takes the
    first N rows of that ordering (heaviest edges first).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    apply_resource_limits(con)

    pal = _color_palette(con, web_path)
    gkey = group_key_expr(con, web_path)

    limit_clause = f"LIMIT {int(sample)}" if sample else ""
    cols = con.execute(
        f"""SELECT e.weight AS weight,
                   CAST(p0.xw AS DOUBLE) AS xw0, CAST(p0.yw AS DOUBLE) AS yw0,
                   CAST(p1.xw AS DOUBLE) AS xw1, CAST(p1.yw AS DOUBLE) AS yw1,
                   p0.gkey AS gkey0, p1.gkey AS gkey1,
                   p0.cited_by_count AS cited0, p1.cited_by_count AS cited1
            FROM read_parquet('{edges_path}') e
            JOIN (SELECT id, xw, yw, {gkey} AS gkey, cited_by_count
                  FROM read_parquet('{web_path}') WHERE NOT is_ring) p0
              ON p0.id = e.src
            JOIN (SELECT id, xw, yw, {gkey} AS gkey, cited_by_count
                  FROM read_parquet('{web_path}') WHERE NOT is_ring) p1
              ON p1.id = e.dst
            WHERE e.weight >= {MIN_WEIGHT}
            ORDER BY e.weight DESC, e.src, e.dst
            {limit_clause}"""
    ).fetchnumpy()

    n = cols["weight"].shape[0]

    def quant(v):
        return np.clip(np.round(v * QUANT), 0, QUANT).astype("<u2")

    positions = np.empty((n, 4), dtype="<u2")
    fines = np.empty((n, 4), dtype="i1")
    for k, c in enumerate(("xw0", "yw0", "xw1", "yw1")):
        positions[:, k] = quant(cols[c])
        rest = cols[c] * QUANT - positions[:, k]            # in [-0.5, 0.5] uint16 steps
        fines[:, k] = np.clip(np.round(rest * FINE), -128, 127)

    rgb0 = group_rgb_u8(pal, cols["gkey0"], default=GREY_RGB)  # (n, 3) uint8
    rgb1 = group_rgb_u8(pal, cols["gkey1"], default=GREY_RGB)

    weight = cols["weight"].astype(np.float64)
    w = np.clip(np.round(np.clip(weight, 0, WEIGHT_CAP) / WEIGHT_CAP * 255), 0, 255).astype(np.uint8)

    cited0 = cols["cited0"].astype(np.float64)
    cited1 = cols["cited1"].astype(np.float64)
    s0 = np.sqrt(np.maximum(cited0, 0)) / SIZE_SCALE * 255
    s1 = np.sqrt(np.maximum(cited1, 0)) / SIZE_SCALE * 255
    s = np.clip(np.round(np.maximum(s0, s1)), 0, 255).astype(np.uint8)

    attrs = np.empty((n, 8), dtype=np.uint8)
    attrs[:, 0:3] = rgb0
    attrs[:, 3:6] = rgb1
    attrs[:, 6] = w
    attrs[:, 7] = s

    pos_path = out_dir / "edges_pos.bin"
    attr_path = out_dir / "edges_attr.bin"
    positions.tofile(pos_path)
    attrs.tofile(attr_path)
    fines.tofile(out_dir / "edges_fine.bin")

    files = {"pos": "edges_pos.bin", "attr": "edges_attr.bin", "fine": "edges_fine.bin"}
    layout = {
        "edges_pos.bin": "little-endian uint16 x0,y0,x1,y1 per edge (8 bytes/edge)",
        "edges_attr.bin": "uint8 r0,g0,b0,r1,g1,b1,w,s per edge (8 bytes/edge)",
        "edges_fine.bin": "int8 x0,y0,x1,y1 per edge (4 bytes/edge): coordinate = (edges_pos + edges_fine / 254) / 65535",
    }

    subsets = {}
    for name, size in (TIERS if tiers is None else tiers).items():
        k = min(n, size)
        positions[:k].tofile(out_dir / f"edges_pos_{name}.bin")
        attrs[:k].tofile(out_dir / f"edges_attr_{name}.bin")
        fines[:k].tofile(out_dir / f"edges_fine_{name}.bin")
        files[f"pos_{name}"] = f"edges_pos_{name}.bin"
        files[f"attr_{name}"] = f"edges_attr_{name}.bin"
        files[f"fine_{name}"] = f"edges_fine_{name}.bin"
        subsets[name] = int(k)
    subsets["full"] = int(n)

    meta = {
        "count": int(n),
        "files": files,
        "layout": layout,
        "quantization": QUANT,
        "fine": FINE,
        "size_scale": SIZE_SCALE,
        "weight_cap": WEIGHT_CAP,
        "generated_from": "edges.parquet + coords_web.parquet",
        "filters": (
            f"WHERE e.weight >= {MIN_WEIGHT} (repeat-collaboration set, no length filter) "
            "AND NOT is_ring on both endpoints (dust edges excluded)"
        ),
        "subsets": subsets,
    }
    (out_dir / "edges_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def add_parser(parser) -> None:
    parser.add_argument("--edges", default=None)
    parser.add_argument("--web", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--sample", type=int, default=None, help="export only the first N edges (for tests)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_parser(parser)
    args = parser.parse_args()
    edges = args.edges or str(data_dir() / "edges.parquet")
    web = args.web or str(data_dir() / "coords_web.parquet")
    out = Path(args.out) if args.out else data_dir() / "vector_experiment"
    t0 = time.time()
    meta = export_edges(edges, web, out, sample=args.sample)
    elapsed = time.time() - t0
    pos_size = (out / "edges_pos.bin").stat().st_size
    attr_size = (out / "edges_attr.bin").stat().st_size
    msg = (
        f"{meta['count']:,} edges -> {out} in {elapsed:.1f}s "
        f"(edges_pos.bin {pos_size:,}B, edges_attr.bin {attr_size:,}B)"
    )
    msg += "; tiers " + ", ".join(f"{k} {v:,}" for k, v in meta["subsets"].items())
    print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
