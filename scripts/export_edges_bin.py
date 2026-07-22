"""Binary edge export for a WebGL/GPU connection-lines experiment (R22a).

Standalone experiment script, not a pipeline stage: reads edges.parquet
JOINed to coords_web.parquet for both endpoints and writes two flat
little-endian arrays under data/vector_experiment/ for GPU-friendly
progressive loading -- edges_pos.bin (x0,y0,x1,y1 uint16, 8 bytes/edge)
and edges_attr.bin (r0,g0,b0,r1,g1,b1,w,s uint8, 8 bytes/edge) -- plus
edges_meta.json (named distinctly from export_points_bin's meta.json,
since both scripts share the vector_experiment/ directory). data/index
and data/tiles are untouched.

Edges touching a ring/dust node (is_ring=true, see pipeline/webcoords.py's
ring_pred) are excluded, same as the point-cloud export. Only the
repeat-collaboration set (weight >= 2) is exported; no edge-length filter
is applied here (unlike pipeline/edge_px.py's tile-baking stage).

Colors reuse pipeline.palette via scripts.export_points_bin's community
color LUT (same min_members=1000 rule as the legend and the point cloud):
each endpoint gets its own community color, or the shared tail grey if its
community is below threshold.

If the exported pair exceeds 300MB combined, a capped companion pair
(edges_pos_top.bin/edges_attr_top.bin) holding the top 3,000,000 edges by
weight is also written, since the export is already ordered heaviest-first.
"""
import argparse
import json
import time
from pathlib import Path

import duckdb
import numpy as np

from pipeline.config import apply_resource_limits, data_dir
from scripts.export_points_bin import SIZE_SCALE, _build_color_lut

QUANT = 65535        # xw,yw uint16 quantization scale (round(v * QUANT))
MIN_WEIGHT = 2        # repeat-collaboration threshold
WEIGHT_CAP = 20        # weight clamp before scaling to uint8
TOP_CAP_BYTES = 300 * 1024 * 1024   # combined-size trigger for capped companion
TOP_CAP_EDGES = 3_000_000            # size of the capped companion pair


def export_edges(edges_path: str, web_path: str, out_dir: Path, sample: int | None = None) -> dict:
    """Core export: builds edges_pos.bin, edges_attr.bin, meta.json under out_dir.

    Deterministic ordering: weight DESC, src, dst. --sample takes the
    first N rows of that ordering (heaviest edges first).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    apply_resource_limits(con)

    lut = _build_color_lut(con, web_path)

    limit_clause = f"LIMIT {int(sample)}" if sample else ""
    cols = con.execute(
        f"""SELECT e.weight AS weight,
                   CAST(p0.xw AS DOUBLE) AS xw0, CAST(p0.yw AS DOUBLE) AS yw0,
                   CAST(p1.xw AS DOUBLE) AS xw1, CAST(p1.yw AS DOUBLE) AS yw1,
                   p0.community AS community0, p1.community AS community1,
                   p0.cited_by_count AS cited0, p1.cited_by_count AS cited1
            FROM read_parquet('{edges_path}') e
            JOIN (SELECT id, xw, yw, community, cited_by_count
                  FROM read_parquet('{web_path}') WHERE NOT is_ring) p0
              ON p0.id = e.src
            JOIN (SELECT id, xw, yw, community, cited_by_count
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
    positions[:, 0] = quant(cols["xw0"])
    positions[:, 1] = quant(cols["yw0"])
    positions[:, 2] = quant(cols["xw1"])
    positions[:, 3] = quant(cols["yw1"])

    community0 = np.clip(cols["community0"].astype(np.int64), 0, lut.shape[0] - 1)
    community1 = np.clip(cols["community1"].astype(np.int64), 0, lut.shape[0] - 1)
    rgb0 = lut[community0]  # (n, 3) uint8
    rgb1 = lut[community1]  # (n, 3) uint8

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

    files = {"pos": "edges_pos.bin", "attr": "edges_attr.bin"}
    layout = {
        "edges_pos.bin": "little-endian uint16 x0,y0,x1,y1 per edge (8 bytes/edge)",
        "edges_attr.bin": "uint8 r0,g0,b0,r1,g1,b1,w,s per edge (8 bytes/edge)",
    }

    total_bytes = pos_path.stat().st_size + attr_path.stat().st_size
    capped = total_bytes > TOP_CAP_BYTES and n > TOP_CAP_EDGES
    if capped:
        top_pos_path = out_dir / "edges_pos_top.bin"
        top_attr_path = out_dir / "edges_attr_top.bin"
        positions[:TOP_CAP_EDGES].tofile(top_pos_path)
        attrs[:TOP_CAP_EDGES].tofile(top_attr_path)
        files["pos_top"] = "edges_pos_top.bin"
        files["attr_top"] = "edges_attr_top.bin"
        layout["edges_pos_top.bin"] = (
            f"top {TOP_CAP_EDGES:,} edges by weight, same layout as edges_pos.bin (viewer default)"
        )
        layout["edges_attr_top.bin"] = (
            f"top {TOP_CAP_EDGES:,} edges by weight, same layout as edges_attr.bin (viewer default)"
        )

    meta = {
        "count": int(n),
        "files": files,
        "layout": layout,
        "quantization": QUANT,
        "size_scale": SIZE_SCALE,
        "weight_cap": WEIGHT_CAP,
        "generated_from": "edges.parquet + coords_web.parquet",
        "filters": (
            f"WHERE e.weight >= {MIN_WEIGHT} (repeat-collaboration set, no length filter) "
            "AND NOT is_ring on both endpoints (dust edges excluded)"
        ),
    }
    if capped:
        meta["top_cap_edges"] = TOP_CAP_EDGES
        meta["top_cap_note"] = (
            "combined edges_pos.bin+edges_attr.bin exceeded 300MB; "
            f"capped companion pair holds the top {TOP_CAP_EDGES:,} edges by weight "
            "(viewer defaults to this pair)"
        )
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
    if "pos_top" in meta["files"]:
        top_pos_size = (out / "edges_pos_top.bin").stat().st_size
        top_attr_size = (out / "edges_attr_top.bin").stat().st_size
        msg += (
            f"; capped top {meta['top_cap_edges']:,} -> edges_pos_top.bin {top_pos_size:,}B, "
            f"edges_attr_top.bin {top_attr_size:,}B"
        )
    print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
