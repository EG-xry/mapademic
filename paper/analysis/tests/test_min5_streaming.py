"""Small reproducible parity/resume tests for the memory-bounded min5 continuation.

Run from the repository root:
NUMBA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .tmp/venv-paper/bin/python -m unittest discover -s paper/analysis/tests -v
No production analysis outputs are read or modified; all fixtures use temporary directories.
"""
import hashlib
import json
import os
import sys
import tempfile
import unittest
from collections import deque
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ANALYSIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS))
import a2c_team_cap as team
import b3_hubs as hubs
import export_deposit as deposit
import identities
import kernels


def csr(adjacency):
    offsets = np.zeros(len(adjacency) + 1, np.int64)
    offsets[1:] = np.cumsum([len(row) for row in adjacency])
    return offsets, np.array([v for row in adjacency for v in row], np.int32)


class TeamCapTests(unittest.TestCase):
    def test_build_resume_projection_and_histogram_cache(self):
        with tempfile.TemporaryDirectory(prefix="mapademic-a2c-test-") as directory:
            base = Path(directory)
            cache, results, data = base / "cache", base / "results", base / "data"
            pg = data / "pg"
            for path in (cache, results, pg):
                path.mkdir(parents=True)
            ids = np.array([30, 10, 20, 40, 50, 60, 70], np.int64)
            np.save(pg / "ids.npy", ids)
            rows = [(1, 2, [30, 10], False, "article"), (2, 4, [30, 20, 20, 999], False, "article"),
                    (3, 75, [20, 40], False, "article"), (4, 500, [40, 50], False, "article"),
                    (5, 501, [50, 60], False, "article"), (6, 2, [10, 70], True, "article"),
                    (7, 2, [10, 70], False, "dataset"), (8, 2, [10, 999], False, "article")]
            for part, subset in enumerate((rows[:2], rows[2:])):
                path = data / "works_authorships" / f"part{part}" / "works.parquet"
                path.parent.mkdir(parents=True)
                pq.write_table(pa.table({"work_id": [r[0] for r in subset], "n_authors": [r[1] for r in subset],
                                         "author_ids": pa.array([r[2] for r in subset], type=pa.list_(pa.int64())),
                                         "is_xpac": [r[3] for r in subset], "type": [r[4] for r in subset],
                                         "is_paratext": [False] * len(subset)}), path)
            with mock.patch.multiple(team, CACHE=cache, RESULTS=results, D=data, PG=pg, BUILD_BATCH=1,
                                     load_ids=lambda: np.load(pg / "ids.npy")):
                actual_batch = team._batch_members
                calls = 0

                def interrupted(*args):
                    nonlocal calls
                    calls += 1
                    if calls == 4:
                        raise RuntimeError("simulated partition interruption")
                    return actual_batch(*args)

                with mock.patch.object(team, "_batch_members", interrupted):
                    with self.assertRaisesRegex(RuntimeError, "simulated"):
                        team.build()
                state = json.loads((cache / "bipartite_build/progress.json").read_text())
                self.assertEqual((state["files_done"], state["works"]), (1, 2))
                graph = team.build()
                self.assertIsInstance(graph["w_auth"], np.memmap)
                np.testing.assert_array_equal(graph["w_size"], [2, 4, 75, 500, 501])
                np.testing.assert_array_equal(graph["w_off"], [0, 2, 4, 6, 8, 10])
                np.testing.assert_array_equal(graph["a_off"], [0, 2, 3, 5, 7, 9, 10, 10])
                for key, array in team.build().items():
                    np.testing.assert_array_equal(graph[key], array)
                sources = np.array([0, 1, 2], np.int32)

                def projected_hist(cap):
                    adjacency = [set() for _ in ids]
                    for work, size in enumerate(graph["w_size"]):
                        if size > cap:
                            continue
                        authors = graph["w_auth"][graph["w_off"][work]:graph["w_off"][work + 1]]
                        for author in authors:
                            adjacency[author].update(set(authors) - {author})
                            self.assertIn(len(ids) + work, graph["a_nb"][graph["a_off"][author]:graph["a_off"][author + 1]])
                    hist = np.zeros((len(sources), team.MAXD + 1), np.int64)
                    for row, source in enumerate(sources):
                        distances, queue = {int(source): 0}, deque([int(source)])
                        while queue:
                            u = queue.popleft()
                            for v in adjacency[u]:
                                if v not in distances:
                                    distances[v] = distances[u] + 1
                                    queue.append(v)
                        for distance in distances.values():
                            hist[row, distance] += 1
                    return hist

                for cap in team.CAPS:
                    actual = team.run(graph["a_off"], graph["a_nb"], graph["w_off"], graph["w_auth"], graph["w_size"], sources, cap, team.MAXD)
                    np.testing.assert_array_equal(actual, projected_hist(cap))
                identity = json.loads((cache / "bipartite_manifest.json").read_text())["fingerprint"]
                partial = team.checkpointed_run(graph, sources[:2], 50, identity, sources)
                full = team.checkpointed_run(graph, sources, 50, identity, sources)
                np.testing.assert_array_equal(partial, projected_hist(50)[:2])
                np.testing.assert_array_equal(full, projected_hist(50))
                with mock.patch.object(team, "run", side_effect=AssertionError("cached BFS repeated")):
                    np.testing.assert_array_equal(team.checkpointed_run(graph, sources[:2], 50, identity, sources), partial)
                np.testing.assert_array_equal(team.checkpointed_run(graph, sources, 50, "changed-graph", sources), full)
                (results / "a2_distances.json").write_text(json.dumps({"K": 3, "all_links": team.summarize(full)}))
                self.assertTrue(team.validate_cap50(full, sources, sources)["full_histogram_exact"])
                damaged = full.copy()
                damaged[0, 1] -= 1
                damaged[0, 2] += 1
                with self.assertRaisesRegex(AssertionError, "histogram differs"):
                    team.validate_cap50(damaged, sources, sources)

    def test_bipartite_graph_survives_moving_the_project_folder(self):
        def layout(base):
            return base / "cache", base / "results", base / "data", base / "data" / "pg"

        with tempfile.TemporaryDirectory(prefix="mapademic-a2c-move-") as directory:
            before, after = Path(directory) / "before", Path(directory) / "after"
            cache, results, data, pg = layout(before)
            for path in (cache, results, pg):
                path.mkdir(parents=True)
            np.save(pg / "ids.npy", np.array([30, 10, 20], np.int64))
            part = data / "works_authorships" / "part0" / "works.parquet"
            part.parent.mkdir(parents=True)
            pq.write_table(pa.table({"work_id": [1, 2], "n_authors": [2, 2],
                                     "author_ids": pa.array([[30, 10], [10, 20]], type=pa.list_(pa.int64())),
                                     "is_xpac": [False, False], "type": ["article", "article"],
                                     "is_paratext": [False, False]}), part)
            with mock.patch.multiple(team, CACHE=cache, RESULTS=results, D=data, PG=pg,
                                     load_ids=lambda: np.load(before / "data/pg/ids.npy")):
                built = {key: np.array(value) for key, value in team.build().items()}
            os.rename(before, after)
            cache, results, data, pg = layout(after)
            with mock.patch.object(identities, "ROOT", after), \
                    mock.patch.multiple(team, CACHE=cache, RESULTS=results, D=data, PG=pg,
                                        load_ids=lambda: np.load(after / "data/pg/ids.npy"),
                                        _batch_members=mock.Mock(side_effect=AssertionError("graph rebuilt"))):
                reused = team.build()
            for key, array in built.items():
                np.testing.assert_array_equal(reused[key], array)


class HubTests(unittest.TestCase):
    def test_streaming_matches_full_table(self):
        rng = np.random.default_rng(271828)
        n = 2500
        src = rng.integers(0, n, 15003, dtype=np.int32)
        dst = rng.integers(0, n, len(src), dtype=np.int32)
        weight = rng.random(len(src)).astype(np.float32)
        degree = np.bincount(src, minlength=n) + np.bincount(dst, minlength=n)
        flagged = rng.random(n) < .03
        clubs = (10, 100, 1000)
        order = np.argsort(-np.where(flagged, -1, degree), kind="stable")
        ranks = np.full(n, max(clubs), np.int32)
        ranks[order[:max(clubs)]] = np.arange(max(clubs), dtype=np.int32)
        with tempfile.TemporaryDirectory(prefix="mapademic-b3-test-") as directory:
            path = Path(directory) / "edges.parquet"
            pq.write_table(pa.table({"src": src, "dst": dst, "weight": weight}), path, row_group_size=301)
            strength, inside, links, assortativity = hubs.edge_statistics(path, degree, ranks, clubs, batch_size=137)
            original = np.bincount(src, weight.astype(float), minlength=n) + np.bincount(dst, weight.astype(float), minlength=n)
            np.testing.assert_array_equal(strength, original)
            self.assertEqual(links, len(src))
            for i, size in enumerate(clubs):
                club = np.zeros(n, bool)
                club[order[:size]] = True
                self.assertEqual(inside[i], (club[src] & club[dst]).sum())
            for i, transform in enumerate((lambda v: v, np.log)):
                x, y = transform(degree[src].astype(float)), transform(degree[dst].astype(float))
                total, squares, cross = x.sum() + y.sum(), (x*x).sum() + (y*y).sum(), 2*(x*y).sum()
                mean = total / (2*len(src))
                reference = (cross/(2*len(src)) - mean**2) / (squares/(2*len(src)) - mean**2)
                np.testing.assert_allclose(assortativity[i], reference, rtol=1e-11, atol=1e-13)
            old = pd.DataFrame({"degree": degree, "strength": original, "betweenness": rng.random(n),
                                "closeness": -rng.random(n), "citations": rng.integers(0, 50, n)})
            new = old.copy()
            new["strength"] = strength
            pd.testing.assert_frame_equal(old[~flagged].sample(500, random_state=1), new[~flagged].sample(500, random_state=1))
            for column in old:
                self.assertTrue(old[~flagged].nlargest(100, column).index.equals(new[~flagged].nlargest(100, column).index))
            attrs = pd.DataFrame({"node_idx": np.arange(n), "name": [f"Name {i}" for i in range(n)],
                                  "country": ["US", "CN", "JP", "GB"] * (n//4), "field": ["Physics"] * n})
            metadata = Path(directory) / "attrs.parquet"
            attrs.to_parquet(metadata, index=False, row_group_size=53)
            selected = {1, 27, 50, 101, 217, 2499}
            pd.testing.assert_frame_equal(hubs.selected_metadata(metadata, selected, batch_size=29).sort_index(),
                                          attrs.set_index("node_idx").loc[sorted(selected)])


class RepairKernelTests(unittest.TestCase):
    def test_fused_distances_resume(self):
        offsets, neighbors = csr([[1], [0,2], [1,3], [2,4], [3], [6,7], [5,7], [5,6], []])
        sources = np.array([0,2,5,8,0,7,4], np.int64)
        for width in (2, 64):
            expected_h = kernels.histograms(offsets, neighbors, sources, width)
            for chunks in (1,2,9):
                expected_s, expected_c = kernels.distance_sums(offsets, neighbors, sources, chunks)
                actual = kernels.histograms_distance_sums(offsets, neighbors, sources, width, chunks)
                for a, b in zip(actual, (expected_h, expected_s, expected_c)):
                    np.testing.assert_array_equal(a, b)
        isolated = kernels.histograms_distance_sums(offsets, neighbors, np.array([8]), 64, 2)
        self.assertEqual(isolated[0].sum(), 0)
        self.assertEqual(isolated[2][8], 1)
        with tempfile.TemporaryDirectory(prefix="mapademic-a8-test-") as directory:
            base = Path(directory)
            checkpoint = base / "distance.npz"
            real_kernel, calls = kernels.histograms_distance_sums, 0

            def interrupted(*args):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("simulated interruption")
                return real_kernel(*args)

            with mock.patch.object(kernels, "histograms_distance_sums", interrupted):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    kernels.checkpointed_distance_summaries(offsets, neighbors, sources, 64, 2, checkpoint, {"graph":"fixture"}, batch_size=3)
            with np.load(checkpoint) as saved:
                self.assertEqual(len(saved["histograms"]), 3)
            expected = real_kernel(offsets, neighbors, sources, 64, 2)
            resumed = kernels.checkpointed_distance_summaries(offsets, neighbors, sources, 64, 2, checkpoint, {"graph":"fixture"}, batch_size=2)
            for a, b in zip(expected, resumed):
                np.testing.assert_array_equal(a, b)
            with mock.patch.object(kernels, "histograms_distance_sums", side_effect=AssertionError("cached BFS repeated")):
                kernels.checkpointed_distance_summaries(offsets, neighbors, sources, 64, 2, checkpoint, {"graph":"fixture"})
            changed = kernels.checkpointed_distance_summaries(offsets, neighbors, sources[::-1].copy(), 64, 2, checkpoint, {"graph":"changed"}, batch_size=4)
            for a, b in zip(real_kernel(offsets, neighbors, sources[::-1].copy(), 64, 2), changed):
                np.testing.assert_array_equal(a, b)


class DepositTests(unittest.TestCase):
    def test_streamed_export_matches_original_and_is_atomic(self):
        with tempfile.TemporaryDirectory(prefix="mapademic-export-test-") as directory:
            base = Path(directory)
            n = 9
            ids = np.arange(100, 100+n, dtype=np.int64)
            rep = np.array([-1,-1,-1,-1,-1,0,0,1,1], np.int64)
            flags = {"merged":np.array([0,1,2]), "split":np.array([1,2,3]), "notPerson":np.array([2,4])}
            codes = deposit.flag_codes(n, flags, rep)
            np.testing.assert_array_equal(codes, [1,2,3,2,3,0,0,0,0])
            offsets, neighbors = csr([[j for j in range(n) if j != i] for i in range(n)])
            repair = {"merged_pairs": {**{f"twins_{s}":{"threshold":3} for s, _ in deposit.SCRIPT_LABELS}, "splits":2}}
            for script, _ in deposit.SCRIPT_LABELS:
                pairs = {"zh":[(5,6,3,8),(0,8,4,8)], "ja":[(6,5,2,8)], "ko":[(7,8,3,8)], "ru":[]}[script]
                pd.DataFrame(pairs, columns=["focal_idx","twin_idx","shared","coauthors"]).to_csv(base / f"a6_twins_{script}.csv", index=False)
            pd.DataFrame({"a":[5,0,6,7], "b":[6,8,7,8], "shared":[3,4,2,3], "group_size":[2,3,2,6]}).to_parquet(base / "split_pairs.parquet", index=False)
            twins, splits, selected = deposit.selected_pairs(repair, offsets, cache=base)
            self.assertEqual((len(twins),len(splits)), (3,2))
            attrs = pa.table({"node_idx":pa.array(np.arange(n), type=pa.int32()), "name":[f"Name {i}" for i in range(n)],
                              "works_kept":pa.array(np.arange(5,5+n),type=pa.int32()), "works":pa.array(np.arange(6,6+n),type=pa.int32()),
                              "cited":pa.array(np.arange(n),type=pa.int32()), "degree":pa.array(np.diff(offsets),type=pa.int32()),
                              "community":pa.array([0]*n,type=pa.int32()), "country":pa.array([None,None,"US","CN",None,"JP","US","CN","JP"],type=pa.string()),
                              "institution":pa.array([None]*n,type=pa.string()), "field":pa.array(["Physics"]*n,type=pa.string()),
                              "on_map":[False,False,True,True,False,True,True,True,False]})
            attrs_path, target = base / "attrs.parquet", base / "researchers.parquet"
            pq.write_table(attrs, attrs_path, row_group_size=3)
            mapped = np.array([False,False,True,True,False,True,True,True,False])
            supplementary = {"is_fallback_topic":np.array([True,False,False,False,True,False,False,False,False]),
                             "on_map":mapped,"xw":np.arange(n,dtype=np.float64)/10,"yw":np.arange(n,dtype=np.float64)/20,
                             "xw_valid":mapped,"yw_valid":mapped}
            names = deposit.export_researchers(attrs_path,target,ids,rep,codes,selected,network="min5",batch_size=2,
                                               supplementary=supplementary)
            a = attrs.to_pandas()
            expected = pd.DataFrame({"node":np.arange(n,dtype=np.int32), "openalex_id":"A"+pd.Series(ids).astype(str),
                                     "name":a.name, "works":a.works_kept, "works_openalex":a.works, "citations":a.cited,
                                     "coauthors":a.degree,"community":a.community,"on_map":a.on_map.astype(bool),
                                     "country":a.country,"institution":a.institution,"field":a.field,
                                     "flag":np.asarray(deposit.FLAG_LABELS,dtype=object)[codes],"repaired_node":rep.astype(np.int32),
                                     "is_fallback_topic":supplementary["is_fallback_topic"],
                                     "xw":np.where(mapped,supplementary["xw"],np.nan),
                                     "yw":np.where(mapped,supplementary["yw"],np.nan)})
            expected_table = pa.Table.from_pandas(expected,preserve_index=False)
            actual = pq.read_table(target)
            self.assertTrue(actual.schema.equals(expected_table.schema,check_metadata=False))
            self.assertTrue(actual.equals(expected_table,check_metadata=False))
            pd.testing.assert_series_equal(names.sort_index(),pd.Series(a.name.to_numpy()[selected],index=pd.Index(selected,name="node"),name="name"),check_dtype=False)
            twin_frame, split_frame = deposit.pair_frames(twins,splits,ids,rep,names)
            self.assertEqual(twin_frame.joined.tolist(),[True,False,True])
            self.assertEqual(split_frame.joined.tolist(),[True,False])
            self.assertEqual(twin_frame.native_id.tolist(),["A105","A100","A107"])
            self.assertEqual(twin_frame.latin_name.tolist(),["Name 6","Name 8","Name 8"])
            mask = np.zeros(len(neighbors),bool)
            full_src = np.repeat(np.arange(n,dtype=np.int32),np.diff(offsets))
            for left,right in ((0,8),(5,6)):
                mask |= ((full_src==left)&(neighbors==right))|((full_src==right)&(neighbors==left))
            flagged_path = base / "flagged_links.parquet"
            self.assertEqual(deposit.export_flagged_links(offsets,neighbors,mask,flagged_path,batch_size=7),(2,4))
            keep = mask & (neighbors>full_src)
            self.assertTrue(pq.read_table(flagged_path).equals(pa.table({"src":full_src[keep],"dst":neighbors[keep]})))
            previous = hashlib.sha256(flagged_path.read_bytes()).hexdigest()
            asymmetric = mask.copy(); asymmetric[np.flatnonzero(asymmetric)[0]]=False
            with self.assertRaisesRegex(AssertionError,"symmetric"):
                deposit.export_flagged_links(offsets,neighbors,asymmetric,flagged_path,batch_size=7)
            self.assertEqual(previous,hashlib.sha256(flagged_path.read_bytes()).hexdigest())
            previous = hashlib.sha256(target.read_bytes()).hexdigest()
            broken = attrs.set_column(0,"node_idx",pa.array([0,1,2,3,4,5,6,8,7],type=pa.int32()))
            pq.write_table(broken,attrs_path)
            with self.assertRaisesRegex(AssertionError,"node order"):
                deposit.export_researchers(attrs_path,target,ids,rep,codes,selected,network="min5",batch_size=2,
                                           supplementary=supplementary)
            self.assertEqual(previous,hashlib.sha256(target.read_bytes()).hexdigest())
            source, links = base / "source.bin", base / "links.parquet"
            source.write_bytes(b"fresh-link-source"); links.write_bytes(b"stale-link-source")
            self.assertEqual(deposit.stage_links(source,links),"hard link")
            self.assertTrue(os.path.samefile(source,links))
            self.assertEqual(deposit.stage_links(source,links),"existing hard link")
            copied = base / "copied.bin"
            with mock.patch.object(deposit.os,"link",side_effect=OSError("cross-device")):
                self.assertEqual(deposit.stage_links(source,copied),"copy")
            self.assertEqual(copied.read_bytes(),source.read_bytes())
            for name in deposit.PACKAGE_FILES:
                path = base / name
                if not path.exists(): path.write_text(name)
            extra = base / ".unrelated.tmp"; extra.write_text("keep me")
            deposit.write_checksums(base)
            sums = (base / "SHA256SUMS").read_text().splitlines()
            self.assertEqual(len(sums),len(deposit.PACKAGE_FILES))
            for row in sums:
                digest,name = row.split("  ",1)
                self.assertEqual(digest,hashlib.sha256((base/name).read_bytes()).hexdigest())
            self.assertEqual(extra.read_text(),"keep me")


if __name__ == "__main__":
    unittest.main()
