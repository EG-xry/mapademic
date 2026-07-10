"""asinh radial compression: layout coords -> unit-square web coords."""
import json
from pathlib import Path

import duckdb

from pipeline.config import apply_resource_limits, data_dir

MARGIN = 0.02  # points land in [MARGIN, 1-MARGIN]


def build_webcoords(coords_path: str, out_path: str, spread: float = 1.35,
                    ring_comm_max: int = 100) -> dict:
    con = duckdb.connect()
    apply_resource_limits(con)
    # median center per controller decision - mean is halo-sensitive
    cx, cy = con.execute(
        f"SELECT median(x), median(y) FROM read_parquet('{coords_path}')"
    ).fetchone()
    # median radius as the asinh scale; guard degenerate all-at-center inputs
    s = con.execute(
        f"""SELECT greatest(median(sqrt((x-{cx})*(x-{cx}) + (y-{cy})*(y-{cy}))), 1e-9)
            FROM read_parquet('{coords_path}')"""
    ).fetchone()[0]
    con.execute(
        f"""CREATE TEMP TABLE polar AS
            SELECT *,
                   sqrt((x-{cx})*(x-{cx}) + (y-{cy})*(y-{cy})) / {s} AS rs,
                   atan2(y-{cy}, x-{cx}) AS theta,
                   count(*) OVER (PARTITION BY community) AS comm_n
            FROM read_parquet('{coords_path}')"""
    )
    # asinh(v) = ln(v + sqrt(v*v + 1)); DuckDB lacks asinh
    con.execute(
        """CREATE TEMP TABLE au AS
           SELECT *, ln(rs + sqrt(rs*rs + 1)) AS a FROM polar"""
    )
    amax = con.execute("SELECT max(a) FROM au").fetchone()[0] or 1e-9
    # ring rule: small disconnected community parked out near the max radius
    ring_pred = (f"(comm_n < {int(ring_comm_max)} AND a > 0.9 * {amax})"
                 if ring_comm_max > 0 else "FALSE")
    a2max = con.execute(
        f"SELECT max(a) FROM au WHERE NOT {ring_pred}"
    ).fetchone()[0] or 1e-9
    half = 0.5 - MARGIN
    # hash-based uniforms in [0,1): DuckDB hash() is a stable UBIGINT
    con.execute(
        f"""CREATE TEMP TABLE placed AS
            SELECT id, display_name, community, works_count, cited_by_count,
                   institution, field, {ring_pred} AS is_ring,
                   CASE WHEN {ring_pred}
                        THEN (0.86 + 0.12 * ((hash(id) % 100000) / 100000.0))
                        ELSE least(a / {a2max}, 1.0) END AS ru,
                   CASE WHEN {ring_pred}
                        THEN 2 * pi() * ((hash(id || '/t') % 100000) / 100000.0)
                        ELSE theta END AS th
            FROM au"""
    )
    n = con.execute(
        f"""COPY (
              SELECT id, display_name,
                     0.5 + {half} * ru * cos(th) AS xw,
                     0.5 + {half} * ru * sin(th) AS yw,
                     community, works_count, cited_by_count, institution,
                     field, is_ring
              FROM placed
            ) TO '{out_path}' (FORMAT PARQUET)"""
    ).fetchone()[0]
    ring_n = con.execute("SELECT count(*) FROM placed WHERE is_ring").fetchone()[0]
    stats = {"n": n, "cx": cx, "cy": cy, "s": s, "r2max": a2max,
             "ring_n": ring_n, "spread": spread}
    Path(out_path + ".meta.json").write_text(json.dumps(stats))
    return stats


def add_parser(parser) -> None:
    parser.add_argument("--coords", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--spread", type=float, default=1.35)
    parser.add_argument("--ring-comm-max", type=int, default=100)


def run(args) -> int:
    coords = args.coords or str(data_dir() / "coords.parquet")
    out = args.out or str(data_dir() / "coords_web.parquet")
    stats = build_webcoords(coords, out, spread=args.spread,
                            ring_comm_max=args.ring_comm_max)
    print(f"{stats['n']:,} points -> {out} (s={stats['s']:.1f}, "
          f"r2max={stats['r2max']:.3f}, ring={stats['ring_n']:,}, "
          f"spread={stats['spread']})")
    return 0
