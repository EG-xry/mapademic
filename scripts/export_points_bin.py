"""Binary point-cloud export for a WebGL vector-points experiment (R15a).

Standalone experiment script, not a pipeline stage: reads coords_web.parquet
(all rows, including ring/dust) and writes two flat little-endian arrays
under data/vector_experiment/ for GPU-friendly progressive loading --
positions.bin (x,y uint16, 4 bytes/point) and attrs.bin (r,g,b,size uint8,
4 bytes/point) -- plus meta.json. data/index and data/tiles are untouched.

Colors reuse pipeline.palette (same min_members=1000 rule as the production
legend); points whose community falls below that threshold -- including
every ring/dust singleton community -- share one fixed grey rather than
each getting its own tiny per-community jitter, so the tail reads as a
single consistent dust color (close to the viewer's #565660).
"""
import argparse
import json
import time
from pathlib import Path

import duckdb
import numpy as np

from pipeline.config import apply_resource_limits, data_dir
from pipeline.palette import community_rgb, load_community_stats

QUANT = 65535       # xw,yw uint16 quantization scale (round(v * QUANT))
SIZE_SCALE = 700    # sqrt(cited_by_count) scale cap (700 ~= sqrt(~490k))
MIN_MEMBERS = 1000  # same threshold as palette.build_community_palette

# Canonical dust/tail grey: community_rgb() for a fixed sub-threshold
# community (id 0, 0 members) rather than each community's own jittered
# grey, so every below-threshold point -- ring/dust included -- renders as
# the same color. Close to the viewer's #565660 dust family.
GREY_RGB = community_rgb(0, 0, 0.5, 0.5, min_members=MIN_MEMBERS)


def _rgb_to_u8(rgb) -> np.ndarray:
    return np.clip(np.round(np.array(rgb, dtype=np.float64) * 255), 0, 255).astype(np.uint8)


def _build_color_lut(con, web_path: str) -> np.ndarray:
    """Dense community-id -> (r,g,b) uint8 lookup table.

    Covers every community id present in the parquet (ring/dust rows are
    each their own singleton community). Communities with fewer than
    MIN_MEMBERS members map to the shared GREY_RGB.
    """
    stats = load_community_stats(con, web_path)
    max_id = max((c for c, *_ in stats), default=-1)
    lut = np.empty((max_id + 1, 3), dtype=np.uint8)
    lut[:] = _rgb_to_u8(GREY_RGB)
    for community, _field, members, cx, cy in stats:
        if members >= MIN_MEMBERS:
            rgb = community_rgb(community, members, cx, cy, min_members=MIN_MEMBERS)
            lut[community] = _rgb_to_u8(rgb)
    return lut


def export_points(web_path: str, out_dir: Path, sample: int | None = None) -> dict:
    """Core export: builds positions.bin, attrs.bin, meta.json under out_dir.

    Deterministic ordering: cited_by_count DESC, id. --sample takes the
    first N rows of that ordering (highest-cited first).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    apply_resource_limits(con)

    lut = _build_color_lut(con, web_path)

    limit_clause = f"LIMIT {int(sample)}" if sample else ""
    cols = con.execute(
        f"""SELECT CAST(xw AS DOUBLE) AS xw, CAST(yw AS DOUBLE) AS yw,
                   cited_by_count, community
            FROM read_parquet('{web_path}')
            ORDER BY cited_by_count DESC, id
            {limit_clause}"""
    ).fetchnumpy()

    n = cols["xw"].shape[0]

    x_q = np.clip(np.round(cols["xw"] * QUANT), 0, QUANT).astype("<u2")
    y_q = np.clip(np.round(cols["yw"] * QUANT), 0, QUANT).astype("<u2")
    positions = np.empty((n, 2), dtype="<u2")
    positions[:, 0] = x_q
    positions[:, 1] = y_q

    community = np.clip(cols["community"].astype(np.int64), 0, lut.shape[0] - 1)
    rgb = lut[community]  # (n, 3) uint8

    cited = cols["cited_by_count"].astype(np.float64)
    s = np.clip(np.round(np.sqrt(np.maximum(cited, 0)) / SIZE_SCALE * 255), 0, 255).astype(np.uint8)

    attrs = np.empty((n, 4), dtype=np.uint8)
    attrs[:, 0:3] = rgb
    attrs[:, 3] = s

    positions_path = out_dir / "positions.bin"
    attrs_path = out_dir / "attrs.bin"
    positions.tofile(positions_path)
    attrs.tofile(attrs_path)

    meta = {
        "count": int(n),
        "files": {"positions": "positions.bin", "attrs": "attrs.bin"},
        "record_layout": {
            "positions.bin": "little-endian uint16 x, uint16 y per point (4 bytes/point)",
            "attrs.bin": "uint8 r, uint8 g, uint8 b, uint8 s per point (4 bytes/point)",
        },
        "quantization": QUANT,
        "size_scale": SIZE_SCALE,
        "generated_from": "coords_web.parquet",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


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
