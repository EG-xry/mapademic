"""Stream work authorships from the S3 parquet snapshot; raw works never touch disk."""
from pathlib import Path

import duckdb

from pipeline.config import data_dir

DEFAULT_SRC = "s3://openalex/data/parquet/works"


def connect(src_root: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    if src_root.startswith("s3://"):
        con.execute("INSTALL httpfs; LOAD httpfs;")
        # long timeout + retries: VPN'd reads of ~200MB column chunks time out
        # on DuckDB's defaults
        con.execute(
            "SET s3_region='us-east-1';"
            "SET http_timeout=600000;"
            "SET http_retries=5;"
        )
    return con


def partitions(con, src_root: str) -> list[str]:
    rows = con.execute(
        "SELECT file FROM glob(?)", [f"{src_root}/*/*.parquet"]
    ).fetchall()
    return sorted({row[0].rsplit("/", 2)[1] for row in rows})


def extract_partition(con, src_root: str, partition: str, out_dir: Path) -> bool:
    """Extract one updated_date partition. Returns False if already done."""
    out = out_dir / f"{partition}.parquet"
    if out.exists():
        return False
    tmp = out.with_name(out.name + ".tmp")
    tmp.unlink(missing_ok=True)
    con.execute(
        f"""
        COPY (
            SELECT
                id AS work_id,
                authors_count AS n_authors,
                list_transform(authorships, a -> a.author.id) AS author_ids
            FROM read_parquet('{src_root}/{partition}/*.parquet')
        ) TO '{tmp}' (FORMAT PARQUET)
        """
    )
    tmp.rename(out)  # atomic: a crash mid-COPY never yields a half checkpoint
    return True


def add_parser(parser) -> None:
    parser.add_argument("--src", default=DEFAULT_SRC, help="works parquet root")
    parser.add_argument("--out", default=None, help="override output dir")


def run(args) -> int:
    out_dir = Path(args.out) if args.out else data_dir() / "works_authorships"
    out_dir.mkdir(parents=True, exist_ok=True)
    con = connect(args.src)
    parts = partitions(con, args.src)
    done = 0
    for i, part in enumerate(parts, 1):
        fresh = extract_partition(con, args.src, part, out_dir)
        done += fresh
        state = "extracted" if fresh else "skip (checkpoint)"
        print(f"[{i}/{len(parts)}] {part}: {state}", flush=True)
    print(f"{done} extracted, {len(parts) - done} skipped -> {out_dir}")
    return 0
