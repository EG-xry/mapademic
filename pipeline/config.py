"""Shared paths and resource settings for pipeline stages."""
import os
import re
from pathlib import Path

_MEMORY_LIMIT_RE = re.compile(r"^\d+(\.\d+)?\s*[KMGT]i?B$", re.IGNORECASE)


def data_dir() -> Path:
    """Checkpoint root; override with MAPADEMIC_DATA (e.g. an external drive)."""
    return Path(os.environ.get("MAPADEMIC_DATA", "data")).expanduser()


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
