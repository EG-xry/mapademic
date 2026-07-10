"""asinh radial compression: layout coords -> unit-square web coords."""
import json
from pathlib import Path

import duckdb

from pipeline.config import apply_resource_limits, data_dir

MARGIN = 0.02  # points land in [MARGIN, 1-MARGIN]


def build_webcoords(coords_path: str, out_path: str) -> dict:
    con = duckdb.connect()
    apply_resource_limits(con)
    # median center per controller decision - mean is halo-sensitive
    cx, cy = con.execute(
        f"SELECT median(x), median(y) FROM read_parquet('{coords_path}')"
    ).fetchone()
    # median radius as the asinh scale; guard degenerate all-at-center inputs
    s = con.execute(
        f"""
        SELECT greatest(median(sqrt((x - {cx})*(x - {cx}) + (y - {cy})*(y - {cy}))), 1e-9)
        FROM read_parquet('{coords_path}')
        """
    ).fetchone()[0]
    # asinh(v) = ln(v + sqrt(v*v + 1)); DuckDB lacks asinh
    r2max = con.execute(
        f"""
        SELECT max(ln(r/{s} + sqrt((r/{s})*(r/{s}) + 1))) FROM (
            SELECT sqrt((x - {cx})*(x - {cx}) + (y - {cy})*(y - {cy})) AS r
            FROM read_parquet('{coords_path}')
        )
        """
    ).fetchone()[0] or 1e-9
    half = 0.5 - MARGIN
    n = con.execute(
        f"""
        COPY (
            WITH polar AS (
                SELECT *,
                       sqrt((x - {cx})*(x - {cx}) + (y - {cy})*(y - {cy})) AS r,
                       atan2(y - {cy}, x - {cx}) AS theta
                FROM read_parquet('{coords_path}')
            ),
            compressed AS (
                SELECT *,
                       CASE WHEN r = 0 THEN 0.0
                            ELSE ln(r/{s} + sqrt((r/{s})*(r/{s}) + 1)) / {r2max}
                       END AS ru
                FROM polar
            )
            SELECT id, display_name,
                   0.5 + {half} * ru * cos(theta) AS xw,
                   0.5 + {half} * ru * sin(theta) AS yw,
                   community, works_count, cited_by_count, institution, field
            FROM compressed
        ) TO '{out_path}' (FORMAT PARQUET)
        """
    ).fetchone()[0]
    stats = {"n": n, "cx": cx, "cy": cy, "s": s, "r2max": r2max}
    Path(out_path + ".meta.json").write_text(json.dumps(stats))
    return stats


def add_parser(parser) -> None:
    parser.add_argument("--coords", default=None)
    parser.add_argument("--out", default=None)


def run(args) -> int:
    coords = args.coords or str(data_dir() / "coords.parquet")
    out = args.out or str(data_dir() / "coords_web.parquet")
    stats = build_webcoords(coords, out)
    print(f"{stats['n']:,} points -> {out} (s={stats['s']:.1f}, r2max={stats['r2max']:.3f})")
    return 0
