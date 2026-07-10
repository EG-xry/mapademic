"""Precompute drawable coauthor edges in max-zoom (z10) pixel space (for tile baking)."""
import duckdb

from pipeline.config import apply_resource_limits, data_dir
from pipeline.tiles import PIX


def build_edge_px(edges_path: str, web_path: str, out_path: str,
                  min_weight: int = 2, max_weight: int | None = None,
                  max_len_px: int = 1536) -> int:
    con = duckdb.connect()
    apply_resource_limits(con)
    max_weight_clause = f"AND e.weight < {int(max_weight)}\n" if max_weight is not None else ""
    n = con.execute(
        f"""COPY (
              SELECT p0.px AS x0, p0.py AS y0, p1.px AS x1, p1.py AS y1
              FROM read_parquet('{edges_path}') e
              JOIN (SELECT id,
                           least({PIX-1}, CAST(floor(CAST(xw AS DOUBLE)*{PIX}) AS INT)) px,
                           least({PIX-1}, CAST(floor(CAST(yw AS DOUBLE)*{PIX}) AS INT)) py
                    FROM read_parquet('{web_path}') WHERE NOT is_ring) p0
                ON p0.id = e.src
              JOIN (SELECT id,
                           least({PIX-1}, CAST(floor(CAST(xw AS DOUBLE)*{PIX}) AS INT)) px,
                           least({PIX-1}, CAST(floor(CAST(yw AS DOUBLE)*{PIX}) AS INT)) py
                    FROM read_parquet('{web_path}') WHERE NOT is_ring) p1
                ON p1.id = e.dst
              WHERE e.weight >= {int(min_weight)}
                {max_weight_clause}
                AND sqrt((p0.px - p1.px)**2 + (p0.py - p1.py)**2) <= {int(max_len_px)}
            ) TO '{out_path}' (FORMAT PARQUET)"""
    ).fetchone()[0]
    return n


def add_parser(parser) -> None:
    parser.add_argument("--edges", default=None)
    parser.add_argument("--web", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--min-weight", type=int, default=2)
    parser.add_argument("--max-weight", type=int, default=None)
    parser.add_argument("--max-len-px", type=int, default=1536)


def run(args) -> int:
    edges = args.edges or str(data_dir() / "edges.parquet")
    web = args.web or str(data_dir() / "coords_web.parquet")
    out = args.out or str(data_dir() / "edges_px.parquet")
    n = build_edge_px(edges, web, out, args.min_weight, args.max_weight, args.max_len_px)
    print(f"{n:,} drawable edges -> {out}")
    return 0
