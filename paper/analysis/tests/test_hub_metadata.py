"""Regression test for selective A4b metadata loading; no real network is read."""
import contextlib
import io
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common


class HubMetadataParity(unittest.TestCase):
    def test_rankings_and_names_match_full_metadata_join(self):
        rng = np.random.default_rng(7)
        n = 3000
        names = [f"Person {i}" if i % 4 else f"研究者{i}" for i in range(n)]
        attrs = pd.DataFrame({"node_idx": np.arange(n), "name": names,
                              "country": [None if i % 7 == 0 else "US" for i in range(n)],
                              "field": ["Medicine" if i % 2 else "Social Sciences" for i in range(n)],
                              "cited": rng.integers(0, 100, n), "degree": rng.integers(1, 100, n),
                              "community": rng.integers(0, 4, n)})
        hubs = pd.DataFrame({"node_idx": np.arange(n), "btw": rng.random(n),
                             "btw_a": rng.random(n), "btw_b": rng.random(n), "participation": rng.random(n)})
        attrs.loc[hubs.nlargest(4, "btw").node_idx, "cited"] = 0
        flagged = np.zeros(n, bool)
        flagged[::97] = True
        expected = hubs.merge(attrs, on="node_idx")
        expected = expected[(expected.degree > 0) & ~flagged].copy()
        for col, rank in (("btw", "btw_pct"), ("cited", "cited_pct"), ("degree", "degree_pct")):
            expected[rank] = expected[col].rank(pct=True)
        expected["script"] = common.name_script(expected.name)
        gates = expected[(expected.btw_pct >= .999) & (expected.cited_pct <= .5)].sort_values("btw", ascending=False)
        top = expected.sort_values("btw", ascending=False).head(25)
        offsets = np.arange(n + 1, dtype=np.int64) * 2
        neighbors = np.column_stack(((np.arange(n) - 1) % n, (np.arange(n) + 1) % n)).ravel().astype(np.int32)
        captured = {}
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            attrs.to_parquet(cache / "node_attrs.parquet", index=False)
            hubs.to_parquet(cache / "hubs.parquet", index=False)
            with patch.object(common, "CACHE", cache), patch.object(common, "load_csr", return_value=(offsets, neighbors)), \
                    patch.object(common, "flagged_mask", return_value=flagged), \
                    patch.object(common, "SCRIPT_COMMUNITIES", {"Japanese script": 0, "Chinese characters": 1, "Korea": 2}), \
                    patch.object(common, "save_result", side_effect=lambda name, data: captured.update(data)), \
                    contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(Path(__file__).resolve().parents[1] / "a4b_hubs_analysis.py"))
        self.assertEqual(captured["researchers"], len(expected))
        self.assertEqual([row["name"] for row in captured["top25"]], top.name.tolist())
        self.assertEqual([row["name"] for row in captured["gatekeepers"]["all"]], gates.name.tolist())
        self.assertEqual(captured["gatekeepers"]["scripts"], gates.script.value_counts().to_dict())
        cited = expected.sort_values("cited", ascending=False).head(1000)
        self.assertEqual(captured["top1000_cited"]["median_betweenness_pct"], cited.btw_pct.median())
        self.assertAlmostEqual(captured["spearman"]["betweenness_degree"],
                               expected[["btw", "degree"]].corr("spearman").iloc[0, 1])
        comm = attrs.community.to_numpy()
        for row, node in zip(captured["gatekeepers"]["all"], gates.node_idx):
            self.assertEqual(row["inside"], np.mean(comm[neighbors[offsets[node]:offsets[node + 1]]] == comm[node]))


if __name__ == "__main__":
    unittest.main()
