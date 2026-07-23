"""Binary author-id export for the full-point-cloud hover fallback (R30).

Standalone experiment script, mirrors export_points_bin.py exactly in source
table, WHERE NOT is_ring filter, and cited_by_count DESC, id ordering, so row
i here lines up with row i of positions.bin / attrs.bin. Writes a single flat
little-endian uint64 array under data/vector_experiment/ids.bin -- the
numeric part of each row's OpenAlex author id (the "A" + digits after
"https://openalex.org/") -- plus id-specific fields merged into meta.json.

Used by the viewer's cloud-on hover fallback: nearest-neighbor over the
in-memory point cloud resolves to a row index, and an HTTP Range request
against ids.bin at offset index*8 recovers that row's author id without
downloading the whole 67MB file.
"""
import json
import re
import time
from pathlib import Path

import duckdb
import numpy as np

from pipeline.config import apply_resource_limits, data_dir

# Matches both the real data's full "https://openalex.org/A<digits>" ids and
# bare "A<digits>" ids (used by tests), by taking the trailing A<digits> run.
_ID_RE = re.compile(r"A(\d+)$")


def _id_to_u64(raw_id: str) -> int:
    m = _ID_RE.search(raw_id)
    if not m:
        raise ValueError(f"id does not match expected OpenAlex author id shape: {raw_id!r}")
    return int(m.group(1))


def export_ids(web_path: str, out_dir: Path, sample: int | None = None) -> dict:
    """Core export: builds ids.bin under out_dir, merges id info into meta.json.

    Same deterministic ordering as export_points_bin.export_points:
    cited_by_count DESC, id. --sample takes the first N rows of that
    ordering (highest-cited first), matching positions.bin/attrs.bin exactly
    when the same --sample value is used for both.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    apply_resource_limits(con)

    limit_clause = f"LIMIT {int(sample)}" if sample else ""
    rows = con.execute(
        f"""SELECT id
            FROM read_parquet('{web_path}')
            WHERE NOT is_ring
            ORDER BY cited_by_count DESC, id
            {limit_clause}"""
    ).fetchnumpy()["id"]

    n = rows.shape[0]
    ids_u64 = np.empty(n, dtype="<u8")
    for i, raw_id in enumerate(rows):
        ids_u64[i] = _id_to_u64(raw_id)

    ids_path = out_dir / "ids.bin"
    ids_u64.tofile(ids_path)

    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["ids"] = {
        "count": int(n),
        "file": "ids.bin",
        "record_layout": "little-endian uint64 per row; numeric part of the "
                          "OpenAlex author id (row i aligns with positions.bin/"
                          "attrs.bin row i -- prepend 'A' to reconstruct the id)",
        "id_prefix": "A",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def add_parser(parser) -> None:
    parser.add_argument("--web", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--sample", type=int, default=None, help="export only the first N ids (for tests)")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    add_parser(parser)
    args = parser.parse_args()
    web = args.web or str(data_dir() / "coords_web.parquet")
    out = Path(args.out) if args.out else data_dir() / "vector_experiment"
    t0 = time.time()
    meta = export_ids(web, out, sample=args.sample)
    elapsed = time.time() - t0
    ids_size = (out / "ids.bin").stat().st_size
    print(f"{meta['ids']['count']:,} ids -> {out / 'ids.bin'} in {elapsed:.1f}s ({ids_size:,}B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
