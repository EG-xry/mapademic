"""Build weighted coauthorship edges from authorship extracts and selected nodes."""
import duckdb

from pipeline.config import data_dir

# Spec-locked: works with more than 50 authors are excluded from edge building
# (a 3,000-author CERN paper would otherwise contribute ~4.5M clique edges).
MAX_AUTHORS = 50


def build_edges(
    works_glob: str, nodes_path: str, out_path: str, max_authors: int = MAX_AUTHORS
) -> int:
    con = duckdb.connect()
    con.execute(
        f"""
        CREATE TABLE edges AS
        WITH exploded AS (
            SELECT work_id, unnest(author_ids) AS author_id, n_authors
            FROM read_parquet('{works_glob}')
            WHERE n_authors BETWEEN 2 AND {int(max_authors)}
        ),
        -- DISTINCT guards against the same author appearing twice on one work
        kept AS (
            SELECT DISTINCT e.work_id, e.author_id, e.n_authors
            FROM exploded e
            JOIN read_parquet('{nodes_path}') n ON n.id = e.author_id
        ),
        pairs AS (
            SELECT
                a.author_id AS src,
                b.author_id AS dst,
                1.0 / (a.n_authors - 1) AS w
            FROM kept a
            JOIN kept b
              ON a.work_id = b.work_id AND a.author_id < b.author_id
        )
        SELECT src, dst, sum(w) AS weight
        FROM pairs
        GROUP BY src, dst
        """
    )
    con.execute(f"COPY edges TO '{out_path}' (FORMAT PARQUET)")
    return con.execute("SELECT count(*) FROM edges").fetchone()[0]


def add_parser(parser) -> None:
    parser.add_argument("--max-authors", type=int, default=MAX_AUTHORS)
    parser.add_argument("--works", default=None, help="override extracts glob")
    parser.add_argument("--nodes", default=None, help="override nodes.parquet path")
    parser.add_argument("--out", default=None, help="override output path")


def run(args) -> int:
    works = args.works or str(data_dir() / "works_authorships" / "*.parquet")
    nodes = args.nodes or str(data_dir() / "nodes.parquet")
    out = args.out or str(data_dir() / "edges.parquet")
    n = build_edges(works, nodes, out, max_authors=args.max_authors)
    print(f"built {n:,} edges (max_authors={args.max_authors}) -> {out}")
    return 0
