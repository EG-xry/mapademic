"""OpenAlex snapshot releases: manifest reading, release pinning, verification.

OpenAlex republishes the whole snapshot quarterly (second Wednesday of Jan,
Apr, Jul, Oct) and writes data/parquet/manifest.json last, so a manifest
names one complete release: every file with its byte size and row count.

Everything derived from one release lives in one MAPADEMIC_DATA directory,
stamped with a RELEASE file. Mixing releases inside one directory is how the
same work ends up in two partition extracts, so a stamp mismatch is refused
rather than merged.
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path

import duckdb

from pipeline.config import apply_resource_limits

BUCKET_ROOT = "s3://openalex/data/parquet"
STAMP = "RELEASE"


def connect(src_root: str = BUCKET_ROOT) -> duckdb.DuckDBPyConnection:
    """DuckDB connection able to read `src_root` (S3 needs httpfs)."""
    con = duckdb.connect()
    apply_resource_limits(con)
    if src_root.startswith("s3://"):
        con.execute("INSTALL httpfs; LOAD httpfs;")
        # long timeout + retries: VPN'd reads of ~200MB column chunks time out
        # on DuckDB's defaults
        con.execute(
            "SET s3_region='us-east-1';"
            "SET http_timeout=600000;"
            "SET http_retries=5;"
        )
    return con


@dataclass(frozen=True)
class SnapshotFile:
    entity: str
    partition: str     # "updated_date=2026-04-08"
    name: str          # "part_0000.parquet"
    url: str
    size: int
    rows: int

    @property
    def rel(self) -> str:
        return f"{self.entity}/{self.partition}/{self.name}"

    def local(self, snapshot_dir: Path) -> Path:
        return Path(snapshot_dir) / self.entity / self.partition / self.name


def load_manifest(con, path: str) -> dict:
    """Parse a manifest from a local path or an s3:// URL."""
    return json.loads(con.execute("SELECT content FROM read_text(?)", [path]).fetchone()[0])


def entity_files(manifest: dict, entity: str) -> list[SnapshotFile]:
    for e in manifest["entities"]:
        if e["entity"] == entity:
            out = []
            for f in e["files"]:
                *_, partition, name = f["url"].split("/")
                out.append(SnapshotFile(entity, partition, name, f["url"],
                                        int(f["meta"]["content_length"]),
                                        int(f["meta"]["record_count"])))
            return out
    raise KeyError(f"manifest has no {entity!r} entity")


def partition_rows(manifest: dict, entity: str) -> dict[str, int]:
    rows: dict[str, int] = {}
    for f in entity_files(manifest, entity):
        rows[f.partition] = rows.get(f.partition, 0) + f.rows
    return rows


def ensure_release(directory: Path, release: str, force: bool = False) -> None:
    """Stamp `directory` with `release`, or refuse if it holds another one."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = directory / STAMP
    if stamp.exists() and not force:
        held = stamp.read_text().strip()
        if held != release:
            raise SystemExit(
                f"{directory} holds OpenAlex release {held}, not {release}. "
                "Use a fresh MAPADEMIC_DATA directory per release (or --force "
                "to overwrite the stamp after clearing the old files)."
            )
    stamp.write_text(release + "\n")


def write_manifest(manifest: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest))
    os.replace(tmp, path)


def verify_entity(manifest: dict, entity: str, snapshot_dir: Path,
                  check_rows: bool = True, deep: bool = False) -> list[str]:
    """Compare a local mirror against the manifest; [] means complete.

    Sizes catch truncated or stale files; footer row counts (read from each
    parquet footer, no data pages) catch a file whose bytes were replaced.
    `deep` also decodes every row of every file, which is the only check that
    notices damaged data pages (parquet pages carry no checksum by default).
    That is a full read of the mirror: minutes per GB on a slow USB drive.
    """
    snapshot_dir = Path(snapshot_dir)
    expected = {f.rel: f for f in entity_files(manifest, entity)}
    problems = []
    present = {}
    for f in expected.values():
        p = f.local(snapshot_dir)
        if not p.exists():
            problems.append(f"missing: {f.rel}")
            continue
        size = p.stat().st_size
        if size != f.size:
            problems.append(f"size {size} != {f.size}: {f.rel}")
            continue
        present[str(p)] = f
    for p in sorted((snapshot_dir / entity).glob("*/*.parquet")):
        rel = p.relative_to(snapshot_dir).as_posix()
        if rel not in expected:
            problems.append(f"not in release: {rel}")
    if check_rows and present:
        paths = "[" + ", ".join(f"'{p}'" for p in present) + "]"
        for name, rows in duckdb.connect().execute(
                f"SELECT file_name, num_rows FROM parquet_file_metadata({paths})"
        ).fetchall():
            f = present[name]
            if rows != f.rows:
                problems.append(f"rows {rows} != {f.rows}: {f.rel}")
    if deep:
        con = duckdb.connect()
        for name, f in present.items():
            try:
                # hashing the whole row forces every column's pages to decode
                con.execute(f"SELECT bit_xor(hash(t)) FROM read_parquet('{name}') t").fetchone()
            except duckdb.Error as e:
                problems.append(f"unreadable: {f.rel} ({str(e).splitlines()[0][:120]})")
    return problems
