"""Select authors with works_count >= threshold into nodes.parquet."""
import duckdb

from pipeline.config import data_dir


def filter_authors(src_glob: str, out_path: str, min_works: int = 5) -> int:
    con = duckdb.connect()
    con.execute(
        f"""
        CREATE TABLE nodes AS
        SELECT
            id,
            display_name,
            works_count,
            cited_by_count,
            last_known_institutions[1].display_name AS institution,
            topics[1].field.display_name AS field
        FROM read_parquet('{src_glob}')
        WHERE works_count >= {int(min_works)}
        """
    )
    con.execute(f"COPY nodes TO '{out_path}' (FORMAT PARQUET)")
    return con.execute("SELECT count(*) FROM nodes").fetchone()[0]


def add_parser(parser) -> None:
    parser.add_argument("--min-works", type=int, default=5)
    parser.add_argument("--src", default=None, help="override authors glob")
    parser.add_argument("--out", default=None, help="override output path")


def run(args) -> int:
    src = args.src or str(data_dir() / "snapshot" / "authors" / "*" / "*.parquet")
    out = args.out or str(data_dir() / "nodes.parquet")
    kept = filter_authors(src, out, min_works=args.min_works)
    print(f"kept {kept:,} authors (min_works={args.min_works}) -> {out}")
    print("target window: 5-10M; re-run with a different --min-works to tune")
    return 0
