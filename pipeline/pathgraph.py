"""Coauthor graph as compressed sparse rows, for pathserver/server.py.

Writes <data>/pathgraph/: offsets.npy (int64, n+1), neighbors.npy (int32,
both directions of every coauthor link, each author's coauthors ordered by
link weight, strongest first, so the first k are the coauthor rays),
ids.npy (uint64 numeric part of each OpenAlex id, by node index) and
names.bin + names_offsets.npy (UTF-8 display names). About 3.2GB for the
2026-09-23 release; the server memory-maps it.
"""
import json
import os
import re
import time
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq

from pipeline.config import apply_resource_limits, data_dir, spill_dir


def build_pathgraph(nodes_path: str, edges_path: str, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    con = duckdb.connect()
    apply_resource_limits(con)
    con.execute(f"SET temp_directory='{spill_dir(out_dir)}'")
    n = con.execute(f"SELECT max(node_idx) + 1 FROM read_parquet('{nodes_path}')").fetchone()[0]
    adj = out_dir / "adjacency.tmp.parquet"
    con.execute(f"""
        COPY (SELECT u, v FROM (
                SELECT src AS u, dst AS v, weight AS w FROM read_parquet('{edges_path}')
                UNION ALL
                SELECT dst, src, weight FROM read_parquet('{edges_path}'))
              ORDER BY u, w DESC, v)
        TO '{adj}' (FORMAT PARQUET, ROW_GROUP_SIZE 4000000)""")
    m = pq.ParquetFile(adj).metadata.num_rows
    counts = np.zeros(n, np.int64)
    neighbors = np.lib.format.open_memmap(out_dir / "neighbors.npy", mode="w+", dtype=np.int32, shape=(m,))
    pos = 0
    for batch in pq.ParquetFile(adj).iter_batches(batch_size=8_000_000, columns=["u", "v"]):
        u = batch.column(0).to_numpy(); v = batch.column(1).to_numpy()
        counts += np.bincount(u, minlength=n)
        neighbors[pos:pos + len(v)] = v
        pos += len(v)
    neighbors.flush()
    del neighbors
    adj.unlink()
    offsets = np.zeros(n + 1, np.int64)
    np.cumsum(counts, out=offsets[1:])
    np.save(out_dir / "offsets.npy", offsets)
    t = con.execute(f"""SELECT node_idx, id, display_name FROM read_parquet('{nodes_path}')
                        ORDER BY node_idx""").fetchnumpy()
    assert len(t["node_idx"]) == n and int(t["node_idx"][-1]) == n - 1, "node_idx must be dense"
    np.save(out_dir / "ids.npy", np.array([int(re.search(r"(\d+)$", s).group(1)) for s in t["id"]], np.uint64))
    blobs = [(s or "").encode("utf-8") for s in t["display_name"]]
    name_offsets = np.zeros(n + 1, np.int64)
    np.cumsum([len(b) for b in blobs], out=name_offsets[1:])
    (out_dir / "names.bin").write_bytes(b"".join(blobs))
    np.save(out_dir / "names_offsets.npy", name_offsets)
    meta = {"nodes": int(n), "directed_links": int(m), "seconds": round(time.time() - t0, 1)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def add_parser(parser) -> None:
    parser.add_argument("--out", default=None, help="default <data>/pathgraph")


def run(args) -> int:
    d = data_dir()
    meta = build_pathgraph(str(d / "graph" / "nodes_int32.parquet"),
                           str(d / "graph" / "edges_int32.parquet"),
                           Path(args.out) if args.out else d / "pathgraph")
    print(f"{meta['nodes']:,} authors, {meta['directed_links']:,} directed links -> pathgraph/ "
          f"in {meta['seconds']}s")
    return 0
