"""Sync the OpenAlex authors parquet snapshot (~53GB) from S3."""
import subprocess
from pathlib import Path

from pipeline.config import data_dir


def sync_command(dest: Path) -> list[str]:
    return [
        "aws", "s3", "sync",
        "s3://openalex/data/parquet/authors/",
        str(dest / "snapshot" / "authors"),
        "--no-sign-request",
        "--no-progress",
    ]


def add_parser(parser) -> None:
    parser.add_argument(
        "--dest", default=None,
        help="download root (default: DATA_DIR; put this on the external drive)",
    )


def run(args) -> int:
    dest = Path(args.dest) if args.dest else data_dir()
    cmd = sync_command(dest)
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return 0
