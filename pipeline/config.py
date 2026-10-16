"""Shared paths and resource settings for pipeline stages."""
import os
import re
from pathlib import Path

_MEMORY_LIMIT_RE = re.compile(r"^\d+(\.\d+)?\s*[KMGT]i?B$", re.IGNORECASE)


def data_dir() -> Path:
    """Checkpoint root; override with MAPADEMIC_DATA (e.g. an external drive)."""
    return Path(os.environ.get("MAPADEMIC_DATA", "data")).expanduser()


def authors_glob() -> str:
    """The authors table: the slim `authors` extract when present, else the
    full snapshot mirror from `download`. Both share one schema."""
    slim = data_dir() / "authors_extract"
    if slim.exists():
        return str(slim / "*" / "*.parquet")
    return str(data_dir() / "snapshot" / "authors" / "*" / "*.parquet")


def works_glob() -> str:
    """The works extract: one file per snapshot file, grouped by partition."""
    return str(data_dir() / "works_authorships" / "*" / "*.parquet")


def spill_dir(default_parent: Path) -> Path:
    """DuckDB temp_directory: MAPADEMIC_TMP, else <default_parent>/.duckdb_tmp."""
    override = os.environ.get("MAPADEMIC_TMP")
    return Path(override).expanduser() if override else Path(default_parent) / ".duckdb_tmp"


def apply_resource_limits(con) -> None:
    """Cap DuckDB via MAPADEMIC_THREADS / MAPADEMIC_MEMORY_LIMIT (e.g. "24GB").

    DuckDB sizes itself to the whole machine; on shared hosts (cluster login
    nodes) the cgroup limit is far smaller and the process gets OOM-killed.
    """
    threads = os.environ.get("MAPADEMIC_THREADS")
    if threads:
        con.execute(f"SET threads={int(threads)}")
    memory_limit = os.environ.get("MAPADEMIC_MEMORY_LIMIT")
    if memory_limit:
        if not _MEMORY_LIMIT_RE.match(memory_limit.strip()):
            raise ValueError(f"invalid MAPADEMIC_MEMORY_LIMIT: {memory_limit!r}")
        con.execute(f"SET memory_limit='{memory_limit.strip()}'")
