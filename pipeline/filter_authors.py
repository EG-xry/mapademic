"""Select authors with works_count >= threshold into nodes.parquet."""
from pathlib import Path

import duckdb

from pipeline.config import data_dir


def filter_authors(src_glob: str, out_path: str, min_works: int = 5) -> int:
    con = duckdb.connect()
    tmp_dir = Path(out_path).parent / ".duckdb_tmp"
    con.execute(f"SET temp_directory='{tmp_dir}'")
    con.execute("SET preserve_insertion_order=false")
    return con.execute(
        f"""
        COPY (
            SELECT
                id,
                display_name,
                works_count,
                cited_by_count,
                last_known_institutions[1].display_name AS institution,
                topics[1].field.display_name AS field
            FROM read_parquet('{src_glob}')
            WHERE works_count >= {int(min_works)}
        ) TO '{out_path}' (FORMAT PARQUET)
        """
    ).fetchone()[0]


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
