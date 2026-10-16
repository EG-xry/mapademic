import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.identity import suspect_links


def test_rare_excursion_of_a_common_name_is_suspect(tmp_path):
    # "W Sun" (node 0) has a common name: 3 records share it (common_name=3 here).
    # Node 0 works with 9 people in region 1 and once with node 10 in region 2.
    names = ["Wei Sun", "Wen Sun", "Weiwei Sun"] + [f"P{i} Lee" for i in range(1, 10)] + ["Geoff Rare"]
    comms = [1, 1, 1] + [1] * 9 + [2]
    ids = [f"A{i}" for i in range(13)]
    pq.write_table(pa.table({"node_idx": pa.array(range(13), pa.int32()), "id": ids, "display_name": names}),
                   tmp_path / "nodes.parquet")
    pq.write_table(pa.table({"id": ids, "community": pa.array(comms, pa.int32())}), tmp_path / "web.parquet")
    src = [0] * 9 + [0, 3]
    dst = list(range(3, 12)) + [12, 4]
    pq.write_table(pa.table({"src": pa.array(src, pa.int32()), "dst": pa.array(dst, pa.int32()),
                             "weight": pa.array([1.0] * len(src), pa.float32())}), tmp_path / "edges.parquet")
    s, d = suspect_links(duckdb.connect(), str(tmp_path / "nodes.parquet"), str(tmp_path / "edges.parquet"),
                         str(tmp_path / "web.parquet"), common_name=3, rare_region=0.15, legend_min=1)
    assert sorted(zip(s.tolist(), d.tolist())) == [(0, 12)]      # the 1-in-10 excursion, and only it
