"""Fixture helpers: build tiny parquet files shaped like the OpenAlex snapshot."""
import duckdb


def authorships_sql(author_ids):
    if not author_ids:
        return "CAST([] AS STRUCT(author STRUCT(id VARCHAR))[])"
    inner = ", ".join(f"{{'author': {{'id': '{a}'}}}}" for a in author_ids)
    return f"[{inner}]"


def write_works_partition(part_dir, works):
    """works: list of (work_id, authors_count, [author_ids]).

    Writes <part_dir>/part_0000.parquet shaped like the snapshot's works files
    (only the columns the extract stage touches).
    """
    part_dir.mkdir(parents=True, exist_ok=True)
    rows = ", ".join(
        f"('{wid}', {count}, {authorships_sql(aids)})"
        for wid, count, aids in works
    )
    duckdb.sql(
        f"COPY (SELECT * FROM (VALUES {rows}) t(id, authors_count, authorships)) "
        f"TO '{part_dir / 'part_0000.parquet'}' (FORMAT PARQUET)"
    )


def write_extract_output(path, rows):
    """rows: list of (work_id, n_authors, [author_ids]) in the EXTRACT OUTPUT schema."""
    vals = ", ".join(f"('{w}', {n}, {ids})" for w, n, ids in rows)
    duckdb.sql(
        f"COPY (SELECT * FROM (VALUES {vals}) t(work_id, n_authors, author_ids)) "
        f"TO '{path}' (FORMAT PARQUET)"
    )
