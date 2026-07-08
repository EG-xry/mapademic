"""Renumber author ids to int32 and emit GPU-ready graph files for layout."""
from pathlib import Path

import duckdb

from pipeline.config import apply_resource_limits, data_dir


def prep_graph(
    nodes_path: str, edges_path: str, out_dir: Path, min_weight: float = 0.0
) -> tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    apply_resource_limits(con)
    con.execute(f"SET temp_directory='{out_dir / '.duckdb_tmp'}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute(
        f"""
        CREATE VIEW numbered AS
        SELECT CAST(row_number() OVER (ORDER BY id) - 1 AS INTEGER) AS node_idx, *
        FROM read_parquet('{nodes_path}')
        """
    )
    n_nodes = con.execute(
        f"""
        COPY (SELECT * FROM numbered ORDER BY node_idx)
        TO '{out_dir / "nodes_int32.parquet"}' (FORMAT PARQUET)
        """
    ).fetchone()[0]
    n_edges = con.execute(
        f"""
        COPY (
            SELECT na.node_idx AS src, nb.node_idx AS dst,
                   CAST(e.weight AS FLOAT) AS weight
            FROM read_parquet('{edges_path}') e
            JOIN numbered na ON na.id = e.src
            JOIN numbered nb ON nb.id = e.dst
            WHERE e.weight >= {float(min_weight)}
        ) TO '{out_dir / "edges_int32.parquet"}' (FORMAT PARQUET)
        """
    ).fetchone()[0]
    return n_nodes, n_edges


def add_parser(parser) -> None:
    parser.add_argument("--min-weight", type=float, default=0.0)
    parser.add_argument("--nodes", default=None, help="override nodes.parquet path")
    parser.add_argument("--edges", default=None, help="override edges.parquet path")
    parser.add_argument("--out", default=None, help="override output dir")


def run(args) -> int:
    nodes = args.nodes or str(data_dir() / "nodes.parquet")
    edges = args.edges or str(data_dir() / "edges.parquet")
    out = Path(args.out) if args.out else data_dir() / "graph"
    n_nodes, n_edges = prep_graph(nodes, edges, out, min_weight=args.min_weight)
    print(f"{n_nodes:,} nodes, {n_edges:,} edges (min_weight={args.min_weight}) -> {out}")
    return 0
