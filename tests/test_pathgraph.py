import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.pathgraph import build_pathgraph


def test_csr_lists_both_directions_strongest_first(tmp_path):
    pq.write_table(pa.table({"node_idx": pa.array([0, 1, 2, 3], pa.int32()),
                             "id": [f"https://openalex.org/A{i}" for i in (10, 11, 12, 13)],
                             "display_name": ["Ann", "Bo", "Céline", None]}), tmp_path / "nodes.parquet")
    pq.write_table(pa.table({"src": pa.array([0, 0, 1], pa.int32()), "dst": pa.array([1, 2, 2], pa.int32()),
                             "weight": pa.array([1.0, 3.0, 2.0], pa.float32())}), tmp_path / "edges.parquet")
    meta = build_pathgraph(str(tmp_path / "nodes.parquet"), str(tmp_path / "edges.parquet"), tmp_path / "pg")
    assert meta == {"nodes": 4, "directed_links": 6, "seconds": meta["seconds"]}
    off = np.load(tmp_path / "pg" / "offsets.npy"); nb = np.load(tmp_path / "pg" / "neighbors.npy")
    adj = [nb[off[i]:off[i + 1]].tolist() for i in range(4)]
    assert adj == [[2, 1], [2, 0], [0, 1], []]          # node 0: weight-3 link to 2 before weight-1 to 1
    assert np.load(tmp_path / "pg" / "ids.npy").tolist() == [10, 11, 12, 13]
    noff = np.load(tmp_path / "pg" / "names_offsets.npy"); blob = (tmp_path / "pg" / "names.bin").read_bytes()
    assert [blob[noff[i]:noff[i + 1]].decode() for i in range(4)] == ["Ann", "Bo", "Céline", ""]
