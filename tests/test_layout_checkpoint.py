"""Unit tests for the checkpoint helpers in slurm/run_layout.py (loaded by path; slurm/ is not a package)."""
import importlib.util
from pathlib import Path

import pytest

RUN_LAYOUT = Path(__file__).resolve().parent.parent / "slurm" / "run_layout.py"

spec = importlib.util.spec_from_file_location("run_layout", RUN_LAYOUT)
run_layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_layout)


def test_empty_dir_returns_zero_none(tmp_path):
    assert run_layout.latest_checkpoint(tmp_path) == (0, None)


def test_picks_highest_iteration_checkpoint(tmp_path):
    (tmp_path / "pos_25.parquet").touch()
    (tmp_path / "pos_50.parquet").touch()
    done, ckpt = run_layout.latest_checkpoint(tmp_path)
    assert done == 50
    assert ckpt == tmp_path / "pos_50.parquet"


def test_ignores_junk_names(tmp_path):
    (tmp_path / "pos_abc.parquet").touch()
    (tmp_path / "other.parquet").touch()
    (tmp_path / "pos_10.parquet.tmp").touch()
    assert run_layout.latest_checkpoint(tmp_path) == (0, None)
    (tmp_path / "pos_10.parquet").touch()
    done, ckpt = run_layout.latest_checkpoint(tmp_path)
    assert done == 10
    assert ckpt == tmp_path / "pos_10.parquet"


def make_graph(d, nodes=b"n" * 10, edges=b"e" * 20):
    d.mkdir(parents=True, exist_ok=True)
    (d / "nodes_int32.parquet").write_bytes(nodes)
    (d / "edges_int32.parquet").write_bytes(edges)
    return run_layout.graph_fingerprint(d)


def test_first_run_stamps_the_checkpoint_dir(tmp_path):
    fp = make_graph(tmp_path / "graph")
    ckpt = tmp_path / "layout_ckpt"
    ckpt.mkdir()
    run_layout.check_checkpoints(ckpt, fp)
    run_layout.check_checkpoints(ckpt, fp)          # same graph resumes fine
    assert (ckpt / "graph.json").exists()


def test_checkpoints_from_another_graph_are_refused(tmp_path):
    # a new release's graph uploaded into an old DATA dir: its pos_1000 must
    # not be taken as "layout done" for a differently numbered graph
    ckpt = tmp_path / "layout_ckpt"
    ckpt.mkdir()
    run_layout.check_checkpoints(ckpt, make_graph(tmp_path / "g1"))
    (ckpt / "pos_1000.parquet").touch()
    other = make_graph(tmp_path / "g2", edges=b"e" * 21)
    with pytest.raises(SystemExit, match="different graph"):
        run_layout.check_checkpoints(ckpt, other)


def test_unstamped_legacy_checkpoints_need_explicit_adoption(tmp_path):
    fp = make_graph(tmp_path / "graph")
    ckpt = tmp_path / "layout_ckpt"
    ckpt.mkdir()
    (ckpt / "pos_100.parquet").touch()
    with pytest.raises(SystemExit, match="--adopt-checkpoints"):
        run_layout.check_checkpoints(ckpt, fp)
    run_layout.check_checkpoints(ckpt, fp, adopt=True)
    run_layout.check_checkpoints(ckpt, fp)


def test_assemble_coords_joins_positions_and_communities_by_node_index():
    import pyarrow as pa
    nodes = pa.table({"node_idx": [0, 1, 2], "id": ["A", "B", "C"], "display_name": ["a", "b", "c"],
                      "works_count": [20, 30, 40], "cited_by_count": [1, 2, 3],
                      "institution": ["i", None, "k"], "field": ["f", "g", None],
                      "works_kept": [15, 16, 17]})
    pos = pa.table({"vertex": [1, 0], "x": [10.0, 0.0], "y": [11.0, 1.0]})   # vertex 2: edgeless
    parts = pa.table({"vertex": [0, 1], "partition": [7, 8]})
    out = run_layout.assemble_coords(nodes, pos, parts)
    assert out.column_names == ["id", "display_name", "x", "y", "community", "works_count",
                                "cited_by_count", "institution", "field"]
    rows = {r["id"]: (r["x"], r["y"], r["community"]) for r in out.to_pylist()}
    assert rows == {"A": (0.0, 1.0, 7), "B": (10.0, 11.0, 8), "C": (None, None, None)}


def test_write_host_writes_atomically_through_arrow(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    class FakeGpuFrame:          # what cudf offers: a bulk device->host copy
        def to_arrow(self):
            return pa.table({"vertex": [0, 1], "x": [0.5, 1.5], "y": [2.5, 3.5]})

    path = tmp_path / "pos_100.parquet"
    run_layout.write_host(FakeGpuFrame(), path)
    assert pq.read_table(path).to_pydict()["x"] == [0.5, 1.5]
    assert not list(tmp_path.glob("*.tmp"))
