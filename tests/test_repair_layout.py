import math

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipeline.repair_layout import repair_webcoords


def write(path, rows):
    pq.write_table(pa.Table.from_pylist(rows), path)
    return str(path)


def inputs(tmp_path):
    old = write(tmp_path / "old.parquet", [
        {"id": "A1", "xw": .2, "yw": .3, "community": 1, "sub": 2, "is_ring": False},
        {"id": "A2", "xw": .8, "yw": .7, "community": 3, "sub": 4, "is_ring": False},
        {"id": "A3", "xw": .5, "yw": .5, "community": 5, "sub": 6, "is_ring": False},
    ])
    rows = [{"id": aid, "display_name": name, "works_count": 1, "works_kept": 1,
             "cited_by_count": 2, "institution": "Home", "field": "Science", "source_ids": sources}
            for aid, name, sources in [("A1", "One", ["A1"]), ("A2", "Two", ["A2"]),
                                       ("M10", "Split One", ["A3"]), ("M11", "Split Two", ["A3"]),
                                       ("M12", "Solo", ["A3"]), ("M13", "No old source", [])]]
    nodes = write(tmp_path / "nodes.parquet", rows)
    edges = write(tmp_path / "edges.parquet", [
        {"src": "A1", "dst": "M10", "weight": 3.0},
        {"src": "A2", "dst": "M10", "weight": 1.0},
        {"src": "M10", "dst": "M11", "weight": 1.0},
    ])
    return old, nodes, edges


def test_preserves_untouched_positions_and_places_split_nodes_from_real_neighbors(tmp_path):
    out = tmp_path / "web.parquet"
    stats = repair_webcoords(*inputs(tmp_path), str(out))
    rows = {r["id"]: r for r in pq.read_table(out).to_pylist()}
    assert set(rows) == {"A1", "A2", "M10", "M11", "M12", "M13"}
    assert (rows["A1"]["xw"], rows["A1"]["yw"], rows["A1"]["community"], rows["A1"]["sub"]) == (.2, .3, 1, 2)
    assert (rows["A2"]["xw"], rows["A2"]["yw"]) == (.8, .7)
    assert rows["M10"]["xw"] == pytest.approx(.35, abs=.0002)
    assert rows["M10"]["yw"] == pytest.approx(.4, abs=.0002)
    assert rows["M11"]["xw"] == pytest.approx(rows["M10"]["xw"], abs=.0002)
    assert rows["M10"]["community"] == 1
    assert rows["M12"]["xw"] == pytest.approx(.5, abs=.0002)
    assert rows["M12"]["is_ring"] and rows["M12"]["is_dust"]
    assert rows["M12"]["works_count"] == 1
    assert stats == {"carried": 2, "from_neighbours": 2, "from_sources": 1, "scattered": 1, "nodes": 6}
    assert all(math.isfinite(r[k]) and 0 <= r[k] <= 1 for r in rows.values() for k in ["xw", "yw"])


def test_reruns_are_deterministic_and_metadata_is_partition_specific(tmp_path):
    paths = inputs(tmp_path)
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    repair_webcoords(*paths, str(a)); repair_webcoords(*paths, str(b))
    assert pq.read_table(a).to_pylist() == pq.read_table(b).to_pylist()
    assert {r["display_name"] for r in pq.read_table(a).to_pylist()} == {"One", "Two", "Split One", "Split Two", "Solo", "No old source"}


@pytest.mark.parametrize("problem", ["duplicate", "dangling", "nan", "bounds"])
def test_bad_graph_inputs_fail_without_publishing_partial_output(tmp_path, problem):
    old, nodes, edges = inputs(tmp_path)
    if problem == "duplicate":
        rows = pq.read_table(nodes).to_pylist(); rows.append(rows[0]); write(nodes, rows)
    elif problem == "dangling":
        write(edges, [{"src": "A1", "dst": "M999", "weight": 1.0}])
    else:
        rows = pq.read_table(old).to_pylist(); rows[0]["xw"] = float("nan") if problem == "nan" else 1.5; write(old, rows)
    out = tmp_path / "out.parquet"
    with pytest.raises(ValueError): repair_webcoords(old, nodes, edges, str(out))
    assert not out.exists()
