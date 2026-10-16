"""Extract work authorships + quality columns from the works snapshot (~20GB of ~707GB)."""
from pathlib import Path

from pipeline import extract, snapshot
from pipeline.config import data_dir
from pipeline.works import id_key_sql

DEFAULT_SRC = f"{snapshot.BUCKET_ROOT}/works"

# Leaf columns fetched from each source file (see pipeline/extract.py for why
# the nested author id is named down to its leaf).
COLUMNS = (
    "id",
    "authors_count",
    "authorships.list.element.author.id",
    "publication_year",
    "type",
    "is_xpac",
    "is_paratext",
    "is_retracted",
    "cited_by_count",
)

# Ids become numeric keys (works.id_key_sql); the quality columns feed
# works.quality_predicate, so the filter can change without re-extracting.
SELECT_SQL = f"""
    SELECT
        {id_key_sql('id')} AS work_id,
        authors_count AS n_authors,
        list_transform(authorships, a -> {id_key_sql('a.author.id')}) AS author_ids,
        CAST(publication_year AS SMALLINT) AS year,
        type,
        is_xpac,
        is_paratext,
        is_retracted,
        cited_by_count
    FROM src
"""


def extract_file(con, fs, src_path: str, out_path: Path,
                 expected_rows: int | None = None) -> bool:
    return extract.extract_file(con, fs, src_path, COLUMNS, SELECT_SQL, out_path,
                                expected_rows)


def add_parser(parser) -> None:
    parser.add_argument("--src", default=DEFAULT_SRC,
                        help="works parquet root (S3, or a local mirror of the same release)")
    parser.add_argument("--out", default=None, help="override output dir")
    parser.add_argument("--manifest", default=None,
                        help="release manifest (default: DATA_DIR/snapshot/manifest.json "
                             "from `download`, else the bucket's current one)")
    parser.add_argument("--jobs", type=int, default=4,
                        help="files fetched concurrently; S3 reads are latency-bound, "
                             "so a few in flight use the link far better than one")
    parser.add_argument("--force", action="store_true",
                        help="restamp an output dir that holds another release")


def run(args) -> int:
    out_dir = Path(args.out) if args.out else data_dir() / "works_authorships"
    manifest = extract.load_manifest(args.manifest, data_dir())
    n = extract.extract_release("works", COLUMNS, SELECT_SQL, args.src, out_dir,
                                manifest, jobs=args.jobs, force=args.force)
    print(f"{n} files extracted -> {out_dir} (release {manifest['date']})")
    return 0
