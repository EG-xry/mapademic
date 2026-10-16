"""Column-projected extraction of one snapshot entity, one output per source file.

DuckDB's parquet reader fetches a list-of-struct column whole: selecting
authorships[].author.id from one 421MB works file pulled 448MB over S3 in a
2026-09-26 measurement, i.e. the entire file, which made the full works
extract a ~33 hour download on a 6MB/s link. pyarrow reads only the named
leaf columns (28MB for the same file), so sources are read with pyarrow and
transformed by DuckDB, which queries the Arrow table in place.

Outputs mirror the source layout (<out>/<partition>/<part>.parquet), so each
manifest entry is its own checkpoint with its own expected row count.
"""
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from pipeline import snapshot
from pipeline.config import apply_resource_limits

RETRIES = 4          # per-file attempts on network errors before the run fails
BUFFER = 1 << 20     # streamed reads; pre-buffering coalesces neighbouring columns in


def filesystem(src_root: str) -> tuple[pafs.FileSystem, str]:
    """(pyarrow filesystem, base path) for an s3:// URL or a local directory."""
    if src_root.startswith("s3://"):
        fs = pafs.S3FileSystem(
            anonymous=True, region="us-east-1",
            connect_timeout=30, request_timeout=300,
            retry_strategy=pafs.AwsStandardS3RetryStrategy(max_attempts=10),
        )
        return fs, src_root[len("s3://"):].rstrip("/")
    return pafs.LocalFileSystem(), str(Path(src_root).resolve())


def extract_file(con, fs, src_path: str, columns, select_sql: str,
                 out_path: Path, expected_rows: int | None = None) -> bool:
    """Extract one source file. Returns False if its checkpoint exists.

    `select_sql` reads the projected source as table `src`. With
    `expected_rows` (from the release manifest) a short or long read raises
    instead of leaving a checkpoint behind.
    """
    out_path = Path(out_path)
    if out_path.exists():
        return False
    table = pq.ParquetFile(src_path, filesystem=fs, pre_buffer=False,
                           buffer_size=BUFFER).read(columns=list(columns))
    con.register("src", table)
    try:
        out = con.execute(select_sql).to_arrow_table()
    finally:
        con.unregister("src")
    if expected_rows is not None and out.num_rows != expected_rows:
        raise RuntimeError(
            f"{src_path}: read {out.num_rows} rows, expected {expected_rows} rows "
            "per the release manifest (source changed mid-run, or a partial mirror)"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    pq.write_table(out, tmp, compression="zstd")
    os.replace(tmp, out_path)  # atomic: a crash mid-write never yields a half checkpoint
    return True


def extract_release(entity: str, columns, select_sql: str, src_root: str,
                    out_dir: Path, manifest: dict, jobs: int = 4,
                    force: bool = False) -> int:
    """Extract every file of `entity` in the release; returns files written."""
    out_dir = Path(out_dir)
    release = manifest["date"]
    snapshot.ensure_release(out_dir, release, force=force)
    files = snapshot.entity_files(manifest, entity)
    fs, base = filesystem(src_root)
    jobs = max(1, jobs)
    local = threading.local()

    def connection():
        # one DuckDB instance per worker thread, splitting the cores between them
        if not hasattr(local, "con"):
            con = duckdb.connect()
            apply_resource_limits(con)
            if not os.environ.get("MAPADEMIC_THREADS"):
                con.execute(f"SET threads={max(2, (os.cpu_count() or 4) // jobs)}")
            local.con = con
        return local.con

    def one(f: snapshot.SnapshotFile) -> bool:
        src = f"{base}/{f.partition}/{f.name}"
        out = out_dir / f.partition / f.name
        for attempt in range(RETRIES):
            try:
                return extract_file(connection(), fs, src, columns, select_sql, out, f.rows)
            except OSError:                   # pyarrow's network/IO errors
                if attempt == RETRIES - 1:
                    raise
                time.sleep(5 * 2 ** attempt)
        raise AssertionError("unreachable")

    fresh = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(one, f): f for f in files}
        try:
            for i, fut in enumerate(as_completed(futures), 1):
                wrote = fut.result()
                fresh += wrote
                f = futures[fut]
                state = "extracted" if wrote else "skip (checkpoint)"
                print(f"[{i}/{len(files)}] {f.partition}/{f.name}: {state} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        except BaseException:
            pool.shutdown(wait=True, cancel_futures=True)
            raise
    expected = {(f.partition, f.name) for f in files}
    stray = sorted(p.relative_to(out_dir).as_posix() for p in out_dir.glob("*/*.parquet")
                   if (p.parent.name, p.name) not in expected)
    if stray:
        raise SystemExit(f"{len(stray)} extract file(s) not in release {release}: {stray[:5]}")
    return fresh


def load_manifest(manifest_arg: str | None, data: Path) -> dict:
    """The release to extract: --manifest, else the one `download` pinned into
    DATA_DIR, else the bucket's current release."""
    local = data / "snapshot" / "manifest.json"
    path = manifest_arg or (str(local) if local.exists()
                            else f"{snapshot.BUCKET_ROOT}/manifest.json")
    print(f"manifest: {path}", flush=True)
    return snapshot.load_manifest(snapshot.connect(path), path)
