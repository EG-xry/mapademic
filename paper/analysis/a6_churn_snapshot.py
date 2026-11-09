"""A6c: compare the main network with the cached 2026-10-14 author snapshot.

The cache's historical release is kept independently of today's S3 header.
Legacy caches lack a saved manifest, so their sidecar explicitly records the
project-memory evidence and that limitation. A new acquisition requires the
expected release and an unchanged manifest. The comparison uses bounded
DuckDB joins rather than 20-million-row pandas objects and Python dicts.
"""
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from a6_institutions import RELEASE, release_state
from common import CACHE, ND, PG, con, log, save_result

BUCKET = "openalex/data/parquet/authors"


def file_digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_provenance(cached, sidecar):
    fingerprint = {"bytes": cached.stat().st_size, "sha256": file_digest(cached),
                   "rows": pq.ParquetFile(cached).metadata.num_rows}
    if sidecar.exists():
        provenance = json.loads(sidecar.read_text())
        if provenance["release"] != RELEASE:
            raise RuntimeError("comparison cache belongs to a different release")
        if provenance.get("cache", {}).get("sha256") != fingerprint["sha256"]:
            raise RuntimeError("comparison cache changed since its provenance was recorded")
        return provenance
    provenance = {"release": RELEASE, "cache": fingerprint,
        "evidence": "legacy snapshot_20261014.parquet filename and saved project run record",
        "acquired_mtime_utc": datetime.fromtimestamp(cached.stat().st_mtime, timezone.utc).isoformat(),
        "limitation": "Original acquisition saved no S3 manifest. The historical release cannot be independently "
                      "verified from these columns; today's release header is not evidence for cached data."}
    sidecar.write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def read_part(s3, path, mapped, attempts=5):
    for attempt in range(attempts):
        try:
            with s3.open_input_file(path) as f:
                table = pq.ParquetFile(f, pre_buffer=False).read(columns=["id", "works_count", "cited_by_count"])
            break
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(5 * (attempt + 1))
    ids = pc.cast(pc.utf8_slice_codeunits(table.column("id"), 22), "int64").to_numpy()
    keep = np.isin(ids, mapped)
    return pd.DataFrame({"id": ids[keep], "works_count": table.column("works_count").to_numpy()[keep],
                         "cited_by_count": table.column("cited_by_count").to_numpy()[keep]})


def acquire(cached, sidecar):
    s3 = pafs.S3FileSystem(anonymous=True, region="us-east-1")
    state = release_state(s3)  # refuses future snapshots before any cache can be mislabeled
    mapped = np.sort(np.load(PG / "ids.npy").astype(np.int64))
    files = sorted(f.path for f in s3.get_file_info(pafs.FileSelector(BUCKET, recursive=True))
                   if f.path.endswith(".parquet"))
    parts = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, part in enumerate(pool.map(lambda path: read_part(s3, path, mapped), files)):
            parts.append(part)
            if i % 50 == 0:
                log(f"{i + 1}/{len(files)} snapshot files; {sum(len(p) for p in parts):,} mapped ids")
    if release_state(s3) != state:
        raise RuntimeError("OpenAlex release or manifest changed during the snapshot scan")
    new = pd.concat(parts, ignore_index=True).drop_duplicates("id", keep="last")
    temporary = cached.with_suffix(".tmp.parquet")
    new.to_parquet(temporary, index=False)
    temporary.replace(cached)
    provenance = {**state, "acquired_at_utc": datetime.now(timezone.utc).isoformat(), "source_files": len(files),
                  "cache": {"bytes": cached.stat().st_size, "sha256": file_digest(cached), "rows": len(new)}}
    sidecar.write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def compare_snapshot(c, nodes, snapshot, attrs, identity):
    c.execute(f"""CREATE TEMP VIEW snapshot_comparison AS
        WITH joined AS (
          SELECT n.works_count, n.cited_by_count, s.works_count AS works_new, s.cited_by_count AS cited_new,
                 coalesce(k.key_n >= 200, FALSE) AS common
          FROM read_parquet('{nodes}') n
          LEFT JOIN read_parquet('{snapshot}') s
            ON s.id = CAST(regexp_extract(n.id, '(\\d+)$', 1) AS BIGINT)
          LEFT JOIN read_parquet('{attrs}') a ON a.id = n.id
          LEFT JOIN read_parquet('{identity}') k ON k.node_idx = a.node_idx)
        SELECT *, works_new IS NOT NULL AS alive,
               (works_new - works_count) / greatest(works_count, 1) AS rw,
               (cited_new - cited_by_count) / greatest(cited_by_count, 1) AS rc FROM joined""")
    rows = c.execute("""SELECT CASE WHEN grouping(common) = 1 THEN 'all'
                        WHEN common THEN 'common' ELSE 'other' END AS label,
        count(*) AS mapped, count(*) FILTER (WHERE NOT alive) AS gone,
        avg((NOT alive)::INT) AS gone_share,
        avg(coalesce(rw < -0.1, FALSE)::INT) FILTER (WHERE alive) AS down10,
        avg(coalesce(rw < -0.5, FALSE)::INT) FILTER (WHERE alive) AS down50,
        avg(coalesce(rw > 0.1, FALSE)::INT) FILTER (WHERE alive) AS up10,
        avg(coalesce(rw > 0.5, FALSE)::INT) FILTER (WHERE alive) AS up50,
        avg(coalesce(rw = 0, FALSE)::INT) FILTER (WHERE alive) AS unchanged,
        avg(coalesce(rc < -0.1, FALSE)::INT) FILTER (WHERE alive) AS cited_down10,
        avg(coalesce(rc > 0.5, FALSE)::INT) FILTER (WHERE alive) AS cited_up50
        FROM snapshot_comparison GROUP BY GROUPING SETS ((), (common))""").fetchdf().set_index("label")
    x = rows.loc["all"]
    expected = c.execute(f"SELECT count(*) FROM read_parquet('{nodes}')").fetchone()[0]
    if int(x.mapped) != expected:
        raise RuntimeError("snapshot joins duplicated researchers")
    out = {"release": RELEASE, "mapped": int(x.mapped), "gone": int(x.gone), "gone_share": float(x.gone_share),
        "works": {key: float(x[col]) for key, col in (("down_10pct", "down10"), ("down_50pct", "down50"),
            ("up_10pct", "up10"), ("up_50pct", "up50"), ("unchanged", "unchanged"))},
        "citations": {"down_10pct": float(x.cited_down10), "up_50pct": float(x.cited_up50)}, "by_name": {}}
    for label in ("common", "other"):
        out["by_name"][label] = {"gone_share": float(rows.loc[label, "gone_share"]),
                                 "works_down_10pct": float(rows.loc[label, "down10"])} if label in rows.index else {
                                     "gone_share": None, "works_down_10pct": None}
    c.execute("DROP VIEW snapshot_comparison")
    return out


def main():
    cached = CACHE / "snapshot_20261014.parquet"
    sidecar = CACHE / "snapshot_20261014.provenance.json"
    provenance = cache_provenance(cached, sidecar) if cached.exists() else acquire(cached, sidecar)
    log(f"{provenance['cache']['rows']:,} rows from cached {RELEASE} snapshot")
    c = con()
    c.execute("SET memory_limit='6GB'")
    c.execute("SET threads=4")
    out = compare_snapshot(c, ND / "nodes.parquet", cached, CACHE / "node_attrs.parquet", PG / "identity_signals.parquet")
    c.close()
    out["provenance"] = provenance
    save_result("a6_churn_census", out)
    log(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
