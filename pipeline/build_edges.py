"""Build weighted coauthorship edges from authorship extracts and selected nodes."""
import math
import shutil
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa

from pipeline.config import apply_resource_limits, data_dir, spill_dir, works_glob
from pipeline.works import check_extract_schema, id_key_sql, qualifying_works_sql

# Spec-locked: works with more than 50 authors are excluded from edge building
# (a 3,000-author CERN paper would otherwise contribute ~4.5M clique edges).
MAX_AUTHORS = 50

# Raw (pre-aggregation) pairs summed per bucket. Each bucket's pairs sit in
# memory as an Arrow table (~16 bytes/pair) while DuckDB aggregates them.
PAIRS_PER_BUCKET = 150_000_000
BATCH_ROWS = 2_000_000

# Why not one SQL statement: explode -> DISTINCT -> self-join on work_id ->
# GROUP BY pair spilled past 48GB, and was stopped, on a 36GB laptop for the
# 2026-09-23 release (145M coauthored works, 599M authorships, ~1.6B raw
# pairs). Here each work's author list is filtered to node indices by binary
# search, pairs are generated from per-work index rows in buckets of the
# smaller index, and each bucket is summed in memory; only aggregated edges
# touch disk.


def _node_keys(con, nodes_path: str) -> np.ndarray:
    """Sorted numeric keys of the selected nodes; position = node index."""
    keys = con.execute(
        f"SELECT k FROM (SELECT {id_key_sql('id')} AS k FROM read_parquet('{nodes_path}')) "
        "WHERE k IS NOT NULL ORDER BY k"
    ).fetchnumpy()["k"]
    return np.asarray(keys, dtype=np.int64)


def _kept_rows(con, works_sql: str, keys: np.ndarray) -> dict[int, tuple]:
    """Per work: its selected authors as sorted node indices, grouped by count.

    Returns {k: (matrix (m, k) int32, n_authors (m,) int16)} for k >= 2.
    `n_authors` stays the work's full author count: unselected coauthors still
    dilute the fractional credit, as they always have.
    """
    parts: dict[int, list] = {}
    reader = con.execute(
        f"SELECT n_authors, list_distinct(author_ids) AS ids FROM ({works_sql})"
    ).to_arrow_reader(BATCH_ROWS)
    for batch in reader:
        ids = batch.column("ids")
        lens = ids.value_lengths().fill_null(0).to_numpy(zero_copy_only=False)
        flat = ids.flatten().to_numpy(zero_copy_only=False)
        if len(keys) == 0 or len(flat) == 0:
            continue
        n_auth = batch.column("n_authors").to_numpy(zero_copy_only=False)
        pos = np.minimum(np.searchsorted(keys, flat), len(keys) - 1)
        member = keys[pos] == flat
        row = np.repeat(np.arange(len(lens)), lens)[member]
        idx = pos[member].astype(np.int32)          # grouped by row, in row order
        k = np.bincount(row, minlength=len(lens))
        starts = np.concatenate(([0], np.cumsum(k)[:-1]))
        for kk in np.unique(k[k >= 2]):
            rows = np.flatnonzero(k == kk)
            mat = idx[starts[rows][:, None] + np.arange(kk)]
            mat.sort(axis=1)
            parts.setdefault(int(kk), []).append((mat, n_auth[rows].astype(np.int16)))
    return {kk: (np.concatenate([m for m, _ in p]), np.concatenate([n for _, n in p]))
            for kk, p in parts.items()}


def _bucket_pairs(rows: dict, bucket: int, buckets: int) -> pa.Table | None:
    """Every pair whose smaller node index hashes to `bucket`, unaggregated."""
    s_parts, d_parts, w_parts = [], [], []
    for kk, (mat, n_auth) in rows.items():
        w = 1.0 / (n_auth.astype(np.float64) - 1)
        for p in range(kk - 1):
            sel = (mat[:, p] % buckets) == bucket
            if not sel.any():
                continue
            rest = mat[sel, p + 1:]                  # rows are sorted, so s < d
            s_parts.append(np.repeat(mat[sel, p], rest.shape[1]))
            d_parts.append(rest.ravel())
            w_parts.append(np.repeat(w[sel], rest.shape[1]))
    if not s_parts:
        return None
    return pa.table({"s": np.concatenate(s_parts), "d": np.concatenate(d_parts),
                     "w": np.concatenate(w_parts)})


def build_edges(
    works_glob: str, nodes_path: str, out_path: str, max_authors: int = MAX_AUTHORS,
    all_works: bool = False, buckets: int | None = None,
) -> int:
    con = duckdb.connect()
    apply_resource_limits(con)
    con.execute(f"SET temp_directory='{spill_dir(Path(out_path).parent)}'")
    con.execute("SET preserve_insertion_order=false")
    check_extract_schema(con, works_glob)
    # A release-pinned extract holds each work once (checked in seconds); the
    # dedupe window, which materializes every work, runs only if it doesn't.
    dupes = con.execute(
        f"SELECT count(*) - count(DISTINCT work_id) FROM read_parquet('{works_glob}')"
    ).fetchone()[0]
    works_sql = qualifying_works_sql(works_glob, all_works, min_authors=2,
                                     max_authors=max_authors, dedupe=dupes > 0)
    rows = _kept_rows(con, works_sql, _node_keys(con, nodes_path))
    total = sum(len(m) * kk * (kk - 1) // 2 for kk, (m, _) in rows.items())
    buckets = buckets or max(1, math.ceil(total / PAIRS_PER_BUCKET))

    parts_dir = Path(out_path).parent / (Path(out_path).name + ".parts")
    shutil.rmtree(parts_dir, ignore_errors=True)
    parts_dir.mkdir(parents=True)
    try:
        for b in range(buckets):
            pairs = _bucket_pairs(rows, b, buckets)
            if pairs is None:
                continue
            con.register("pairs", pairs)
            con.execute(
                f"COPY (SELECT s, d, sum(w) AS weight FROM pairs GROUP BY s, d) "
                f"TO '{parts_dir / f'part_{b:04d}.parquet'}' (FORMAT PARQUET)"
            )
            con.unregister("pairs")
            del pairs
        del rows
        # node index = rank of the numeric key, exactly as _node_keys sorted it
        con.execute(
            f"""CREATE TEMP TABLE node_idx AS
                SELECT id, CAST(row_number() OVER (ORDER BY k) - 1 AS INTEGER) AS idx
                FROM (SELECT id, {id_key_sql('id')} AS k FROM read_parquet('{nodes_path}'))
                WHERE k IS NOT NULL"""
        )
        src = (f"read_parquet('{parts_dir}/*.parquet')" if any(parts_dir.iterdir()) else
               "(SELECT 0::INTEGER AS s, 0::INTEGER AS d, 0.0::DOUBLE AS weight LIMIT 0)")
        return con.execute(
            f"""COPY (
                    SELECT ns.id AS src, nd.id AS dst, p.weight
                    FROM {src} p
                    JOIN node_idx ns ON ns.idx = p.s
                    JOIN node_idx nd ON nd.idx = p.d
                ) TO '{out_path}' (FORMAT PARQUET)"""
        ).fetchone()[0]
    finally:
        shutil.rmtree(parts_dir, ignore_errors=True)


def add_parser(parser) -> None:
    parser.add_argument("--max-authors", type=int, default=MAX_AUTHORS)
    parser.add_argument("--works", default=None, help="override extracts glob")
    parser.add_argument("--nodes", default=None, help="override nodes.parquet path")
    parser.add_argument("--out", default=None, help="override output path")
    parser.add_argument("--all-works", action="store_true",
                        help="use every extracted work, skipping the quality filter "
                             "(xpac deposits, datasets, ...; see pipeline/works.py)")
    parser.add_argument("--buckets", type=int, default=None,
                        help=f"pair-aggregation buckets (default: about {PAIRS_PER_BUCKET:,} "
                             "raw pairs each); raise it if memory is tight")


def run(args) -> int:
    works = args.works or works_glob()
    nodes = args.nodes or str(data_dir() / "nodes.parquet")
    out = args.out or str(data_dir() / "edges.parquet")
    n = build_edges(works, nodes, out, max_authors=args.max_authors,
                    all_works=args.all_works, buckets=args.buckets)
    print(f"built {n:,} edges (max_authors={args.max_authors}"
          f"{', all works' if args.all_works else ''}) -> {out}")
    return 0
