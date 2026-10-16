import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.pathgraph import build_pathgraph
from scripts.export_paths_bin import ID_BASE, export_paths, read_neighbors, varint_encode


def test_varint_matches_leb128():
    assert varint_encode(np.array([0, 1, 127, 128, 300, 2 ** 31])) == bytes(
        [0, 1, 127, 0x80, 1, 0xAC, 2, 0x80, 0x80, 0x80, 0x80, 8])


def test_round_trip_every_list_across_small_chunks(tmp_path):
    rng = np.random.default_rng(0)
    n = 400
    edges = {(int(a), int(b)) for a, b in rng.integers(0, n, size=(1500, 2)) if a != b}
    edges = sorted({(min(a, b), max(a, b)) for a, b in edges})
    ids = [f"https://openalex.org/A{ID_BASE + 1000 + 7 * i}" for i in range(n)]
    pq.write_table(pa.table({"node_idx": pa.array(range(n), pa.int32()), "id": ids,
                             "display_name": [f"N{i}" for i in range(n)]}), tmp_path / "nodes.parquet")
    pq.write_table(pa.table({"src": pa.array([a for a, _ in edges], pa.int32()),
                             "dst": pa.array([b for _, b in edges], pa.int32()),
                             "weight": pa.array(rng.integers(1, 5, len(edges)), pa.float32())}),
                   tmp_path / "edges.parquet")
    pq.write_table(pa.table({"id": ids, "xw": rng.random(n), "yw": rng.random(n)}), tmp_path / "web.parquet")
    build_pathgraph(str(tmp_path / "nodes.parquet"), str(tmp_path / "edges.parquet"), tmp_path / "pg")
    suspect = edges[::7]                                   # every 7th link flagged
    meta = export_paths(tmp_path / "pg", str(tmp_path / "web.parquet"), tmp_path / "paths",
                        chunk_bytes=600, batch=500,
                        suspect=(np.array([a for a, _ in suspect]), np.array([b for _, b in suspect])))
    assert len(meta["chunks"]) > 3 and meta["directed_links"] == 2 * len(edges)
    assert meta["suspect_links"] == 2 * len(suspect)
    new_of_old = np.load(tmp_path / "pg" / "new_of_old.npy")
    flagged = {frozenset((int(new_of_old[a]), int(new_of_old[b]))) for a, b in suspect}
    want = [set() for _ in range(n)]
    for a, b in edges:
        want[new_of_old[a]].add(int(new_of_old[b])); want[new_of_old[b]].add(int(new_of_old[a]))
    for node in range(n):
        got = read_neighbors(tmp_path / "paths", node)
        assert [v for v, _ in got] == sorted(want[node]), node
        assert all(f == (frozenset((node, v)) in flagged) for v, f in got), node
    files = meta["files"]
    assert all(name.startswith(stem + "-") for stem, name in files.items())
    new_ids = np.fromfile(tmp_path / "paths" / files["ids"], dtype="<u4").astype(np.int64) + ID_BASE
    assert new_ids[new_of_old[5]] == ID_BASE + 1000 + 35
    lengths = np.diff(np.fromfile(tmp_path / "paths" / files["offsets"], dtype="<u4").astype(np.int64))
    assert (tmp_path / "paths" / files["lengths"]).read_bytes() == varint_encode(lengths.astype(np.uint64))
    assert json.loads((tmp_path / "paths" / "manifest.json").read_text())["chunks"][0]["byte_start"] == 0
