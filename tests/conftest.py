"""Fixture helpers: build tiny parquet files shaped like the OpenAlex snapshot."""
import re

import duckdb

# Snapshot works columns the extract reads beyond id/authors_count/authorships,
# with the values an ordinary journal article carries.
WORK_DEFAULTS = {
    "publication_year": 2020,
    "type": "article",
    "is_xpac": False,
    "is_paratext": False,
    "is_retracted": False,
    "cited_by_count": 0,
}

# The same defaults under the extract's output column names.
EXTRACT_DEFAULTS = {
    "year": 2020,
    "type": "article",
    "is_xpac": False,
    "is_paratext": False,
    "is_retracted": False,
    "cited_by_count": 0,
}


def lit(v) -> str:
    """Python value -> DuckDB SQL literal."""
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    return "'" + str(v).replace("'", "''") + "'"


def key(openalex_id: str) -> int:
    """Numeric key of an OpenAlex id, as the extract stores it."""
    return int(re.search(r"(\d+)$", openalex_id).group(1))


def _split(work):
    """(wid, count, aids[, extras]) -> (wid, count, aids, extras)."""
    wid, count, aids, *rest = work
    return wid, count, aids, (rest[0] if rest else {})


def authorships_sql(author_ids):
    if not author_ids:
        return "CAST([] AS STRUCT(author STRUCT(id VARCHAR))[])"
    inner = ", ".join(f"{{'author': {{'id': '{a}'}}}}" for a in author_ids)
    return f"[{inner}]"


def write_works_partition(part_dir, works):
    """works: list of (work_id, authors_count, [author_ids][, {column: value}]).

    Writes <part_dir>/part_0000.parquet shaped like the snapshot's works files
    (only the columns the extract stage touches). The optional dict overrides
    WORK_DEFAULTS, e.g. {"is_xpac": True} or {"type": "dataset"}.
    """
    part_dir.mkdir(parents=True, exist_ok=True)
    extra_cols = list(WORK_DEFAULTS)
    rows = []
    for work in works:
        wid, count, aids, extras = _split(work)
        vals = {**WORK_DEFAULTS, **extras}
        rows.append(
            f"('{wid}', {count}, {authorships_sql(aids)}, "
            + ", ".join(lit(vals[c]) for c in extra_cols) + ")"
        )
    duckdb.sql(
        f"COPY (SELECT id, authors_count, authorships,"
        f" publication_year::INTEGER AS publication_year, type::VARCHAR AS type,"
        f" is_xpac::BOOLEAN AS is_xpac, is_paratext::BOOLEAN AS is_paratext,"
        f" is_retracted::BOOLEAN AS is_retracted,"
        f" cited_by_count::INTEGER AS cited_by_count"
        f" FROM (VALUES {', '.join(rows)}) "
        f"t(id, authors_count, authorships, {', '.join(extra_cols)})) "
        f"TO '{part_dir / 'part_0000.parquet'}' (FORMAT PARQUET)"
    )


def write_extract_output(path, rows):
    """rows: list of (work_id, n_authors, [author_ids][, {column: value}]) in
    the EXTRACT OUTPUT schema. Ids may be given as OpenAlex strings ("W1",
    "https://openalex.org/A2"); they are stored as their numeric keys, as the
    extract stage does. The optional dict overrides EXTRACT_DEFAULTS."""
    extra_cols = list(EXTRACT_DEFAULTS)
    vals = []
    for row in rows:
        wid, n, aids, extras = _split(row)
        attrs = {**EXTRACT_DEFAULTS, **extras}
        ids = "[" + ", ".join(str(key(a)) for a in aids) + "]"
        vals.append(
            f"({key(wid)}, {n}, CAST({ids} AS BIGINT[]), "
            + ", ".join(lit(attrs[c]) for c in extra_cols) + ")"
        )
    duckdb.sql(
        f"COPY (SELECT work_id::BIGINT AS work_id, n_authors::INTEGER AS n_authors,"
        f" author_ids, year::SMALLINT AS year, type::VARCHAR AS type,"
        f" is_xpac::BOOLEAN AS is_xpac, is_paratext::BOOLEAN AS is_paratext,"
        f" is_retracted::BOOLEAN AS is_retracted,"
        f" cited_by_count::INTEGER AS cited_by_count"
        f" FROM (VALUES {', '.join(vals)})"
        f" t(work_id, n_authors, author_ids, {', '.join(extra_cols)})) "
        f"TO '{path}' (FORMAT PARQUET)"
    )
