"""Require source-identified A2c parity before skipping A8's delivered BFS."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ANALYSIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS))
import identities
import kernels
from repair_metrics import guarded_summary


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def identity(path):
    stat = path.stat()
    return [str(path), stat.st_size, stat.st_mtime_ns]


def fixture(root):
    results, cache, graph = root / "results", root / "cache", root / "graph"
    for path in (results, cache, graph):
        path.mkdir()
    ids = np.arange(100, 104, dtype=np.int64)
    np.save(graph / "ids.npy", ids)
    np.save(graph / "offsets.npy", np.array([0, 1, 3, 5, 6], np.int64))
    np.save(graph / "neighbors.npy", np.array([1, 0, 2, 1, 3, 2], np.int32))
    sources = np.array([0, 2], np.int32)
    np.save(cache / "a2_sources.npy", sources)
    target_mask = np.ones(len(ids), np.bool_)
    np.save(cache / "lcc_all.npy", target_mask)
    target_digest = hashlib.sha256(target_mask).hexdigest()
    works = root / "works.parquet"
    works.write_bytes(b"fixture input identity")
    inputs = {"version": 2, "ids": identity(graph / "ids.npy"),
              "ids_sha256": hashlib.sha256(ids).hexdigest(), "inputs": [identity(works)], "filter": "fixture"}
    manifest = {"identity": inputs, "fingerprint": digest(inputs), "researchers": len(ids)}
    (cache / "bipartite_manifest.json").write_text(json.dumps(manifest))
    hist = np.zeros((2, 65), np.int64)
    hist[0, :4], hist[1, :4] = [1, 1, 1, 1], [1, 2, 1, 0]
    without_self = hist.copy()
    without_self[:, 0] = 0
    expected = kernels.summary(without_self)
    original = {**expected, "p90_effective_diameter": expected["p90"]}
    del original["p90"]
    (results / "a2_distances.json").write_text(json.dumps({"K": len(sources), "all_links": original}))
    cap = {"sources": len(sources), "researchers": len(ids), "network_fingerprint": manifest["fingerprint"],
           "distance_kernel_version": 2,
           "baseline_targets": {"records": len(ids), "sha256": target_digest,
                                "source_self_excluded_from_summaries": True},
           "cap50_validation": {"full_histogram_exact": True, "reachable_count_exact": True,
                                "fixed_targets_exact": True,
                                "sources_checked": len(sources), "a2_sources": len(sources)}, "cap_50": expected}
    (results / "a2c_team_cap.json").write_text(json.dumps(cap))
    signature = digest({"graph": manifest["fingerprint"], "sources": sources.tolist(),
                        "cap": 50, "maxd": 64, "kernel_version": 2, "target_mask_sha256": target_digest})
    np.savez(cache / "a2c_hist_cap50.npz", sources=sources, hist=hist, hist_fixed=hist,
             signature=np.array(signature), target_mask_sha256=np.array(target_digest), kernel_version=np.array(2))
    return results, cache, graph, sources, hist, expected, signature


class A2BaselineReuse(unittest.TestCase):
    def test_valid_histogram_reuse_matches_direct_summary_without_self_distances(self):
        with tempfile.TemporaryDirectory() as directory:
            results, cache, graph, sources, hist, expected, _ = fixture(Path(directory))
            self.assertEqual(kernels.reuse_a2_summary(results, cache, graph, sources), expected)
            with np.load(cache / "a2c_hist_cap50.npz") as saved:
                np.testing.assert_array_equal(saved["hist"], hist)  # reuse never edits the checkpoint

    def test_reuse_survives_moving_the_project_folder(self):
        # The fixture's manifest holds absolute paths in the folder it was built in, like the 3 November records.
        with tempfile.TemporaryDirectory() as directory:
            before, after = Path(directory) / "before", Path(directory) / "after"
            before.mkdir()
            _, _, _, sources, _, expected, _ = fixture(before)
            os.rename(before, after)
            with mock.patch.object(identities, "ROOT", after):
                summary = kernels.reuse_a2_summary(after / "results", after / "cache", after / "graph", sources)
            self.assertEqual(summary, expected)

    def test_source_replacement_preserving_timestamp_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            results, cache, graph, sources, _, _, _ = fixture(Path(directory))
            path = cache / "a2_sources.npy"
            stat = path.stat()
            changed = np.array([0, 1], np.int32)  # same histogram totals on this symmetric path
            np.save(path, changed)
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            self.assertIsNone(kernels.reuse_a2_summary(results, cache, graph, changed))

    def test_changed_checkpoint_or_manifest_is_rejected(self):
        for damage in ("signature", "histogram", "partial", "manifest", "incomplete_result"):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as directory:
                results, cache, graph, sources, hist, _, signature = fixture(Path(directory))
                with np.load(cache / "a2c_hist_cap50.npz") as saved:
                    fields = {name: saved[name] for name in saved.files}
                if damage == "signature":
                    signature = "wrong-signature"
                elif damage == "histogram":
                    hist[0, 1] -= 1
                    hist[0, 2] += 1
                elif damage == "partial":
                    hist = hist[:1]
                elif damage == "manifest":
                    path = cache / "bipartite_manifest.json"
                    manifest = json.loads(path.read_text())
                    manifest["identity"]["filter"] = "changed-without-rebuilding"
                    path.write_text(json.dumps(manifest))
                elif damage == "incomplete_result":
                    path = results / "a2c_team_cap.json"
                    cap = json.loads(path.read_text())
                    cap["cap50_validation"]["full_histogram_exact"] = False
                    path.write_text(json.dumps(cap))
                fields.update(sources=sources, hist=hist, signature=np.array(signature))
                np.savez(cache / "a2c_hist_cap50.npz", **fields)
                self.assertIsNone(kernels.reuse_a2_summary(results, cache, graph, sources))

    def test_fixed_target_mask_counts_and_schema_are_verified(self):
        for damage in ("mask", "fixed_counts", "version", "target_digest", "result_target_count"):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as directory:
                results, cache, graph, sources, _, _, _ = fixture(Path(directory))
                path = cache / "a2c_hist_cap50.npz"
                with np.load(path) as saved:
                    fields = {name: saved[name] for name in saved.files}
                if damage == "mask":
                    np.save(cache / "lcc_all.npy", np.array([True, True, True, False]))
                elif damage == "fixed_counts":
                    fields["hist_fixed"][0, 1] -= 1
                    fields["hist_fixed"][0, 2] += 1
                elif damage == "version":
                    fields["kernel_version"] = np.array(1)
                elif damage == "target_digest":
                    fields["target_mask_sha256"] = np.array("wrong-target-mask")
                else:
                    result_path = results / "a2c_team_cap.json"
                    cap = json.loads(result_path.read_text())
                    cap["baseline_targets"]["records"] -= 1
                    result_path.write_text(json.dumps(cap))
                np.savez(path, **fields)
                self.assertIsNone(kernels.reuse_a2_summary(results, cache, graph, sources))

    def test_changed_node_content_and_newer_graph_are_rejected(self):
        for change in ("node_content", "graph_time", "work_input"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                results, cache, graph, sources, _, _, _ = fixture(Path(directory))
                if change == "node_content":
                    path = graph / "ids.npy"
                    stat = path.stat()
                    np.save(path, np.arange(100, 104, dtype=np.int64)[::-1])
                    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                else:
                    path = graph / "neighbors.npy" if change == "graph_time" else Path(directory) / "works.parquet"
                    later = (results / "a2_distances.json").stat().st_mtime_ns + 1_000_000_000
                    os.utime(path, ns=(later, later))
                self.assertIsNone(kernels.reuse_a2_summary(results, cache, graph, sources))

    def test_missing_a2c_uses_safe_wrapper_bfs_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            results, cache, graph, sources, hist, expected, _ = fixture(Path(directory))
            (results / "a2c_team_cap.json").unlink()
            self.assertIsNone(kernels.reuse_a2_summary(results, cache, graph, sources))
            tree = ast.parse((ANALYSIS / "a8_repair.py").read_text())
            helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "delivered_baseline")
            hist[:, 0] = 0
            fallback = mock.Mock(return_value=hist)
            environment = {"RESULTS": results, "CACHE": cache, "PG": graph, "reuse_a2_summary": kernels.reuse_a2_summary,
                           "guarded_summary": guarded_summary, "histograms": fallback, "log": lambda *args: None}
            exec(compile(ast.Module(body=[helper], type_ignores=[]), "<baseline-wrapper>", "exec"), environment)
            self.assertEqual(environment["delivered_baseline"](None, None, sources), expected)
            fallback.assert_called_once()


if __name__ == "__main__":
    unittest.main()
