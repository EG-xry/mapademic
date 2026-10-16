"""Select authors with at least --min-works qualifying works into nodes.parquet."""
from pathlib import Path

import duckdb

from pipeline.config import apply_resource_limits, authors_glob, data_dir, spill_dir, works_glob
from pipeline.works import check_extract_schema, id_key_sql, qualifying_works_sql

# Author records that are not people, measured on the 2026-09-23 release:
# placeholder names (678 nodes; "et al." variants alone were coauthor hubs of
# up to ~2,000 edges bridging unrelated fields) and uncited bulk records with
# 1000+ works (426: archives, newspapers, government offices). Names compare
# after lowercasing and dropping whitespace, punctuation and symbols, so a
# name with no letters or digits at all (e.g. "-") normalizes to "".
PLACEHOLDER_NAMES = (
    "", "etal", "anonymous", "anon", "unav", "unknown", "unknownauthor",
    "sinautor", "autorsvaris", "variousauthors", "ほか", "等", "идр", "nn",
    "none", "null",
)
UNCITED_BULK_MIN_WORKS = 1000


def is_person_sql(works_col: str) -> str:
    """SQL boolean over an authors row (alias a): FALSE for the records above."""
    names = ", ".join(f"'{n}'" for n in PLACEHOLDER_NAMES)
    return (
        f"coalesce(regexp_replace(lower(trim(a.display_name)), '[\\s\\p{{P}}\\p{{S}}]', '', 'g')"
        f" NOT IN ({names}), FALSE)"
        f" AND NOT (coalesce(a.cited_by_count, 0) = 0 AND {works_col} >= {UNCITED_BULK_MIN_WORKS})"
    )


AUTHOR_COLUMNS = """
    a.id,
    a.display_name,
    a.works_count,
    a.cited_by_count,
    a.last_known_institutions[1].display_name AS institution,
    a.topics[1].field.display_name AS field
"""


def filter_authors(src_glob: str, out_path: str, min_works: int = 5,
                   works_glob: str | None = None, all_works: bool = False,
                   keep_placeholders: bool = False) -> int:
    """Write nodes.parquet; returns the number of authors kept.

    With `works_glob` (the extract), the threshold applies to `works_kept`:
    the author's works passing works.quality_predicate, each work counted once.
    The snapshot's own works_count includes xpac deposits (one 2026-07 node
    had 1.9M) and is kept only as display metadata. Without it, the snapshot
    works_count is thresholded directly and works_kept is NULL.
    Records that are not people (see PLACEHOLDER_NAMES) are dropped unless
    `keep_placeholders`.
    """
    con = duckdb.connect()
    apply_resource_limits(con)
    con.execute(f"SET temp_directory='{spill_dir(Path(out_path).parent)}'")
    con.execute("SET preserve_insertion_order=false")
    if works_glob is None:
        query = f"""
            SELECT {AUTHOR_COLUMNS}, CAST(NULL AS INTEGER) AS works_kept
            FROM read_parquet('{src_glob}') a
            WHERE a.works_count >= {int(min_works)}
              AND {"TRUE" if keep_placeholders else is_person_sql("a.works_count")}
        """
    else:
        check_extract_schema(con, works_glob)
        query = f"""
            WITH counts AS (
                SELECT author_key, CAST(count(*) AS INTEGER) AS works_kept
                FROM (
                    -- list_distinct: an author listed twice on a work counts once
                    SELECT unnest(list_distinct(author_ids)) AS author_key
                    FROM ({qualifying_works_sql(works_glob, all_works)})
                )
                WHERE author_key IS NOT NULL
                GROUP BY author_key
                HAVING count(*) >= {int(min_works)}
            )
            SELECT {AUTHOR_COLUMNS}, c.works_kept
            FROM read_parquet('{src_glob}') a
            JOIN counts c ON c.author_key = {id_key_sql('a.id')}
            WHERE {"TRUE" if keep_placeholders else is_person_sql("c.works_kept")}
        """
    return con.execute(
        f"COPY ({query}) TO '{out_path}' (FORMAT PARQUET)"
    ).fetchone()[0]


def add_parser(parser) -> None:
    parser.add_argument("--min-works", type=int, default=5)
    parser.add_argument("--src", default=None,
                        help="override authors glob (default: the `authors` extract, "
                             "else the snapshot mirror)")
    parser.add_argument("--out", default=None, help="override output path")
    parser.add_argument("--works", default=None,
                        help="override extracts glob (default DATA_DIR/works_authorships/*/*)")
    parser.add_argument("--count-from", choices=["works", "snapshot"], default="works",
                        help="threshold qualifying works counted from the extract "
                             "(default), or the snapshot's raw works_count (pre-2026-09 "
                             "behavior; includes xpac deposits)")
    parser.add_argument("--all-works", action="store_true",
                        help="count every extracted work, skipping the quality filter")
    parser.add_argument("--keep-placeholders", action="store_true",
                        help="keep author records that are not people (placeholder "
                             "names like 'et al.', uncited bulk records)")


def run(args) -> int:
    src = args.src or authors_glob()
    out = args.out or str(data_dir() / "nodes.parquet")
    works = None
    if args.count_from == "works":
        works = args.works or works_glob()
    kept = filter_authors(src, out, min_works=args.min_works, works_glob=works,
                          all_works=args.all_works,
                          keep_placeholders=args.keep_placeholders)
    basis = "snapshot works_count" if works is None else (
        "all extracted works" if args.all_works else "qualifying works")
    print(f"kept {kept:,} authors (min_works={args.min_works} over {basis}) -> {out}")
    print("target window: 5-10M; re-run with a different --min-works to tune")
    return 0
