import json

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipeline.identity import suspect_links
from pipeline.identifiers import decode_author_id
from scripts.export_paths_bin import read_neighbors
from scripts.repair_map import repair_map
from tests.test_resolve_identities import make_inputs, duplicate_inputs, homogeneous_inputs


def setup_release(tmp_path):
    works, nodes = make_inputs(tmp_path)
    nrows = pq.read_table(nodes).to_pylist()
    rows = [{**n, "xw": .2 + i * .15, "yw": .3 + i * .1,
             "community": 1, "sub": -1, "is_ring": False, "is_dust": False}
            for i, n in enumerate(nrows)]
    web = tmp_path / "previous_web.parquet"
    pq.write_table(pa.Table.from_pylist(rows), web)
    return works, nodes, web


def test_complete_release_preserves_map_and_removes_false_namesake_bridge(tmp_path, monkeypatch):
    works, nodes, web = setup_release(tmp_path)
    monkeypatch.setenv("MAPADEMIC_DATA", "leave-this-setting-alone")
    out = tmp_path / "repaired"
    report = repair_map(str(works), str(nodes), str(web), out, zooms="0-2")
    import os
    assert os.environ["MAPADEMIC_DATA"] == "leave-this-setting-alone"
    assert report["identity_resolution"]["split_profiles"] == 1
    rows = {r["id"]: r for r in pq.read_table(out / "coords_web.parquet").to_pylist()}
    original = pq.read_table(web).to_pylist()
    source = "https://openalex.org/A5023477610"
    assert source not in rows
    for row in original:
        if row["id"] != source:
            assert (rows[row["id"]]["xw"], rows[row["id"]]["yw"]) == (row["xw"], row["yw"])
    assert len([aid for aid in rows if aid.startswith("M")]) == 2
    manifest = json.loads((out / "paths/manifest.json").read_text())
    assert manifest["version"] == 3
    ids = [decode_author_id(x) for x in np.load(out / "pathgraph/ids.npy")]
    assert set(ids) == {aid.rsplit("/", 1)[-1] for aid in rows}
    path_ids = [decode_author_id(x) for x in np.fromfile(out / "paths" / manifest["files"]["ids"], dtype="<u8")]
    adj = {aid: [path_ids[n] for n, _ in read_neighbors(out / "paths", i)] for i, aid in enumerate(path_ids)}
    queue, seen = ["A5000000001"], {"A5000000001"}
    while queue:
        for v in adj[queue.pop()]:
            if v not in seen: seen.add(v); queue.append(v)
    assert "A5000000002" not in seen
    assert any(x.startswith("M") for x in seen)
    alias = json.loads((out / "profiles.json").read_text())["aliases"]["A5023477610"]
    assert len(alias) == 2
    coords = json.loads((out / "index/ids" / f"{int(alias[0][1:]) % 1000}.json").read_text())
    assert alias[0] in coords and len(coords[alias[0]]) == 3
    assert json.loads((out / "identities/manifest.json").read_text())["graph_repaired"]
    assert json.loads((out / "vector_experiment/meta.json").read_text())["count"] == len(rows)
    assert (out / "tiles/0/0/0.png").exists()
    assert (out / "index/legend.json").exists()
    assert (out / "vector_experiment/edges_meta.json").exists()


def test_output_is_atomic_and_does_not_overwrite_existing_release(tmp_path):
    works, nodes, web = setup_release(tmp_path)
    out = tmp_path / "existing"; out.mkdir(); (out / "keep").write_text("original")
    with pytest.raises(ValueError, match="exist"):
        repair_map(str(works), str(nodes), str(web), out, zooms="0")
    assert (out / "keep").read_text() == "original"
    bad = tmp_path / "bad.parquet"
    pq.write_table(pq.read_table(works).drop(["authorship_evidence"]), bad)
    out = tmp_path / "failed"
    with pytest.raises(ValueError): repair_map(str(bad), str(nodes), str(web), out, zooms="0")
    assert not out.exists()
    assert not list(tmp_path.glob("failed.build-*"))


def test_complete_release_accepts_apostrophes_in_input_output_and_spill_paths(tmp_path, monkeypatch):
    folder = tmp_path / "researcher's map"
    folder.mkdir()
    works, nodes, web = setup_release(folder)
    monkeypatch.setenv("MAPADEMIC_TMP", str(folder / "DuckDB's spill"))
    output = folder / "repaired release's data"
    report = repair_map(str(works), str(nodes), str(web), output, zooms="0")
    assert report["identity_resolution"]["split_profiles"] == 1
    assert (output / "tiles/0/0/0.png").exists()
    assert (output / "index/legend.json").exists()
    assert (output / "identities/manifest.json").exists()
    assert (output / "vector_experiment/edges_meta.json").exists()
    assert not list(folder.glob("*.build-*"))


def test_duplicate_profiles_become_one_searchable_identity_without_losing_collaboration(tmp_path):
    works, nodes = duplicate_inputs(tmp_path)
    previous = tmp_path / "previous.parquet"
    rows = [{**n, "xw": .2 + i * .15, "yw": .3, "community": 1, "sub": -1,
             "is_ring": False, "is_dust": False} for i, n in enumerate(pq.read_table(nodes).to_pylist())]
    pq.write_table(pa.Table.from_pylist(rows), previous)
    output = tmp_path / "merged"
    report = repair_map(str(works), str(nodes), str(previous), output, zooms="0")
    assert report["identity_resolution"]["merged_profiles"] == 1
    profiles = json.loads((output / "profiles.json").read_text())
    identity = profiles["aliases"]["A5088470719"][0]
    assert profiles["aliases"]["A5088470720"] == [identity]
    assert len(profiles["profiles"]) == 1
    assert len(pq.read_table(output / "coords_web.parquet")) == 2
    edges = pq.read_table(output / "edges.parquet").to_pylist()
    assert len(edges) == 1 and edges[0]["weight"] == 4
    shard = json.loads((output / "index/search-v2/pa.json").read_text())
    assert [entry[2] for entry in shard] == [identity]


def test_repaired_paths_store_namesake_flags_from_staged_graph(tmp_path, monkeypatch):
    works, nodes, web = setup_release(tmp_path)
    coords = pq.read_table(web).to_pylist()
    # The biology split will inherit its main coauthor's region. Set a very
    # small common-name threshold so these fixture names trigger the heuristic.
    coords[2]["community"] = 2
    pq.write_table(pa.Table.from_pylist(coords), web)
    monkeypatch.setattr("scripts.repair_map.suspect_links",
                        lambda *args: suspect_links(*args, common_name=1, rare_region=.75, legend_min=1))
    out = tmp_path / "flagged"
    report = repair_map(str(works), str(nodes), str(web), out, zooms="0")
    paths = report["paths"]
    flags = sum(flag for i in range(paths["authors"]) for _, flag in read_neighbors(out / "paths", i))
    assert flags == paths["suspect_links"] > 0
    assert report["wall_seconds"] > 0 and report["peak_rss_bytes"] > 0


def test_retained_id_keeps_its_map_position_and_only_clean_publication_evidence(tmp_path):
    works, nodes, web = setup_release(tmp_path)
    _, _, rows = homogeneous_inputs(tmp_path)
    rows[0]["authorship_evidence"].append(dict(rows[0]["authorship_evidence"][0], raw_name="David A. Chalmers"))
    pq.write_table(pa.Table.from_pylist(rows), works)
    out = tmp_path / "retained"
    repair_map(str(works), str(nodes), str(web), out, zooms="0")
    aid = "A5023477610"
    old, new = [{r["id"]: (r["xw"], r["yw"]) for r in pq.read_table(p).to_pylist()}
                for p in (web, out / "coords_web.parquet")]
    assert new["https://openalex.org/" + aid] == old["https://openalex.org/" + aid]
    aliases = json.loads((out / "profiles.json").read_text())["aliases"]
    assert aid not in aliases
    retained = json.loads((out / f"identities/profiles/{aid}.json").read_text())
    assert retained["work_ids"] == ["W2", "W3", "W4"]
    assert retained["papers"]["A5000000001"]["work_id"] == "W2"
    flags = json.loads((out / "identities/flags.json").read_text())
    assert len(flags["entries"]) == 1 and flags["entries"][0]["kind"] == "quarantined"
    assert flags["entries"][0]["id"].startswith("M")
