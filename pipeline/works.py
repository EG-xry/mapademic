"""Which works count: the quality filter shared by node selection and edges.

OpenAlex's snapshot mixes scholarly output with bulk records that are not
evidence of collaboration. Two kinds distort the map badly:

- "xpac" records (is_xpac): the expansion pack added in the 2025-11 walden
  switch, mostly DataCite/repository deposits. The API hides them by default.
  In the 2026-07 build they gave one author 1.9M "works" (per-shot plasma
  datasets) and one coauthor pair an edge weight of 447,612 against a
  99.99th-percentile of 47.
- versioned deposit types (dataset, software, ...): every version is its own
  record, so a repeat upload re-counts the same collaboration.

Excluded here rather than at extract time so the rule can change without
re-downloading: the extract keeps every work plus the columns this filter
reads.
"""

# Types excluded from works counts and coauthor edges. Retracted *articles*
# stay in (the collaboration happened); the retraction notice type does not.
EXCLUDED_TYPES = (
    "dataset",
    "software",
    "other",
    "paratext",
    "libguides",
    "peer-review",
    "erratum",
    "retraction",
    "supplementary-materials",
    "reference-entry",
    "standard",
)

# Columns the extract stage writes (see extract_works.EXTRACT_SQL). Older
# extracts (string ids, no type columns) predate the filter and are refused.
EXTRACT_COLUMNS = (
    "work_id", "n_authors", "author_ids", "year", "type",
    "is_xpac", "is_paratext", "is_retracted", "cited_by_count",
)


def id_key_sql(col: str) -> str:
    """SQL: numeric part of an OpenAlex id (".../A5023888391" -> 5023888391).

    Integer keys make the billion-row explode/join/aggregate steps far cheaper
    than the 32-byte URL strings, and they are what the extract stores.
    """
    return f"CAST(regexp_extract({col}, '(\\d+)$', 1) AS BIGINT)"


def quality_predicate(all_works: bool = False) -> str:
    """SQL boolean over extract columns: TRUE for works that count."""
    if all_works:
        return "TRUE"
    types = ", ".join(f"'{t}'" for t in EXCLUDED_TYPES)
    return (
        "coalesce(NOT is_xpac, TRUE)"
        " AND coalesce(NOT is_paratext, TRUE)"
        f" AND coalesce(type NOT IN ({types}), TRUE)"
    )


def qualifying_works_sql(works_glob: str, all_works: bool = False,
                         min_authors: int | None = None,
                         max_authors: int | None = None,
                         dedupe: bool = True) -> str:
    """SQL query: one row per qualifying work (work_id, n_authors, author_ids).

    Deduped by work_id: a guard against the same work landing in two
    partition extracts. The extract is release-pinned so this should never
    fire, but a double-counted work silently inflates weights. The window
    materializes every qualifying work, so a caller that has checked the
    extract holds no duplicates may pass dedupe=False.
    """
    conds = [quality_predicate(all_works)]
    if min_authors is not None:
        conds.append(f"n_authors >= {int(min_authors)}")
    if max_authors is not None:
        conds.append(f"n_authors <= {int(max_authors)}")
    return f"""
        SELECT work_id, n_authors, author_ids
        FROM read_parquet('{works_glob}')
        WHERE {' AND '.join(conds)}
        {"QUALIFY row_number() OVER (PARTITION BY work_id) = 1" if dedupe else ""}
    """


def check_extract_schema(con, works_glob: str) -> None:
    """Refuse a pre-filter extract with a message that says what to do."""
    cols = {r[0] for r in con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{works_glob}')").fetchall()}
    missing = [c for c in EXTRACT_COLUMNS if c not in cols]
    if missing:
        raise SystemExit(
            f"works extract at {works_glob!r} lacks {missing}: it predates the "
            "work-quality filter. Re-run `python -m pipeline extract` into a "
            "fresh MAPADEMIC_DATA directory."
        )
