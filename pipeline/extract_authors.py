"""Extract the author fields the map uses (~4.6GB of the ~59GB authors snapshot)."""
from pathlib import Path

from pipeline import extract, snapshot
from pipeline.config import data_dir

DEFAULT_SRC = f"{snapshot.BUCKET_ROOT}/authors"

COLUMNS = (
    "id",
    "display_name",
    "works_count",
    "cited_by_count",
    "last_known_institutions.list.element.display_name",
    "topics.list.element.display_name",
    "topics.list.element.field.display_name",
)

# Same column names and nesting as the snapshot, pruned to the fields read
# downstream and to the first institution/topic, so `filter` and `regions`
# read this exactly as they read the full mirror.
SELECT_SQL = """
    SELECT
        id,
        display_name,
        works_count,
        cited_by_count,
        list_slice(last_known_institutions, 1, 1) AS last_known_institutions,
        list_slice(topics, 1, 1) AS topics
    FROM src
"""


def add_parser(parser) -> None:
    parser.add_argument("--src", default=DEFAULT_SRC,
                        help="authors parquet root (S3, or a local mirror of the same release)")
    parser.add_argument("--out", default=None,
                        help="override output dir (default DATA_DIR/authors_extract)")
    parser.add_argument("--manifest", default=None,
                        help="release manifest (default: DATA_DIR/snapshot/manifest.json "
                             "from `download`, else the bucket's current one)")
    parser.add_argument("--jobs", type=int, default=4, help="files fetched concurrently")
    parser.add_argument("--force", action="store_true",
                        help="restamp an output dir that holds another release")


def run(args) -> int:
    out_dir = Path(args.out) if args.out else data_dir() / "authors_extract"
    manifest = extract.load_manifest(args.manifest, data_dir())
    n = extract.extract_release("authors", COLUMNS, SELECT_SQL, args.src, out_dir,
                                manifest, jobs=args.jobs, force=args.force)
    print(f"{n} files extracted -> {out_dir} (release {manifest['date']})")
    return 0
