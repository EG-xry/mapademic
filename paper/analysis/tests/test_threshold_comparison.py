"""JSON-only report tests: missing evidence stays pending and groups align by name."""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import threshold_comparison as comparison


class ThresholdComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mapademic-threshold-test-")
        self.paper = Path(self.temporary.name) / "paper"
        for folder in ("results", "results_min15"):
            (self.paper / folder).mkdir(parents=True)
        self.addCleanup(self.temporary.cleanup)

    def write(self, net, stem, value):
        folder = "results" if net == "min5" else "results_min15"
        (self.paper / folder / f"{stem}.json").write_text(json.dumps(value))

    @staticmethod
    def metrics(report):
        return {row["id"]: row for row in report["metrics"]}

    def test_missing_results_and_baselines_never_fall_back(self):
        self.write("min5", "a1_basics", {"nodes": 100})
        self.write("min15", "a1_basics", {"nodes": 20, "links": 40})
        self.write("min5", "a2_distances", {"K": 1000, "all_links": {"mean": 4.65}})
        self.write("min5", "a7_remedy_peak25_strict", {"sources": 99, "all": {"mean": 4.8}})
        report = comparison.build_report(self.paper)
        rows = self.metrics(report)
        self.assertEqual(rows["nodes"]["min5"]["value"], 100)
        self.assertEqual(rows["nodes"]["min15"]["value"], 20)
        self.assertEqual(rows["links"]["min5"]["status"], "pending")
        self.assertEqual(rows["links"]["min15"]["value"], 40)
        self.assertEqual(rows["removal_baseline_mean"]["min5"]["status"], "pending")
        self.assertEqual(rows["a7_baseline_sources"]["min5"]["value"], 99)
        self.assertEqual(rows["a7_baseline_mean"]["min5"]["value"], 4.8)
        self.write("min5", "a2c_team_cap_100", {"sources": 100, "cap_50": {"mean": 4.64}})
        updated = comparison.build_report(self.paper)
        value = self.metrics(updated)["removal_baseline_mean"]["min5"]
        self.assertEqual(value["value"], 4.64)
        self.assertEqual(value["provenance"], [{"file":"paper/results/a2c_team_cap_100.json", "key":"/cap_50/mean"}])
        self.assertEqual(updated["team_size_sensitivity"]["min5"]["status"], "pending")
        self.assertIn("sources affected by its comparison sets", comparison.markdown(updated))
        self.assertIn("No cross-threshold differences are paired estimates", " ".join(updated["interpretation"]))

    def test_name_alignment_rank_invariants_and_removal_row_provenance(self):
        arts, immunology = comparison.ARTS, comparison.IMMUNOLOGY
        for net, names, matrix, within, countries, rows in (
            ("min5", [arts, immunology], [[5.4,4.8],[4.7,3.8]], {arts:5.4,immunology:3.8}, {"CN":3.4,"KR":3.5,"JP":3.6},
             [{"how":"random","share":.01,"mean":4.6},{"how":"top","share":.01,"mean":5.0}]),
            ("min15", [immunology, arts], [[3.5,4.4],[4.5,5.1]], {arts:5.1,immunology:3.5}, {"JP":3.2,"CN":3.0,"KR":3.1},
             [{"how":"top","share":.01,"mean":4.4},{"how":"random","share":.01,"mean":4.2}]),
        ):
            self.write(net,"a9_structure",{"field":{"names":names,"mean_distance":matrix,"within":within},
                                           "country":{"names":list(countries),"within":countries},"hub_removal":rows})
        report = comparison.build_report(self.paper)
        rows = self.metrics(report)
        self.assertEqual(rows["arts_to_immunology"]["min5"]["value"],4.8)
        self.assertEqual(rows["arts_to_immunology"]["min15"]["value"],4.5)
        self.assertEqual(rows["arts_to_immunology"]["min5"]["provenance"][0]["key"],"/field/mean_distance/0/1")
        self.assertEqual(rows["arts_to_immunology"]["min15"]["provenance"][0]["key"],"/field/mean_distance/1/0")
        self.assertEqual(rows["hub_removal_top_0.01"]["min5"]["provenance"][0]["key"],"/hub_removal/1/mean")
        self.assertEqual(rows["hub_removal_top_0.01"]["min15"]["provenance"][0]["key"],"/hub_removal/0/mean")
        country = report["rankings"]["country"]["comparison"]
        self.assertTrue(country["same_nearest_three_in_order"])
        self.assertEqual(country["nearest_three"]["min5"],["CN","KR","JP"])
        self.assertTrue(country["all_common_distances_larger_at_min5"])
        self.assertAlmostEqual(country["spearman_on_common_groups"]["value"],1.)
        self.assertTrue(all(item["value"] for item in report["arts_immunology_ordering"].values()))
        rendered = comparison.markdown(report)
        self.assertIn("5+: `a9_structure.json#/field/mean_distance/0/1`; 15+: `a9_structure.json#/field/mean_distance/1/0`",rendered)
        for item in report["source_manifest"].values():
            if item["status"] == "available":
                self.assertEqual(len(item["sha256"]),64)
        json.dumps(report,allow_nan=False)

    def test_average_ties_and_invalid_json_fail_explicitly(self):
        self.assertEqual(comparison.ranks([3,1,1,2]),[4.,1.5,1.5,3.])
        self.assertAlmostEqual(comparison.correlation([1,2,3],[3,2,1]),-1.)
        path=self.paper/"results/a1_basics.json"
        path.write_text("{bad-json")
        with self.assertRaises(json.JSONDecodeError):
            comparison.build_report(self.paper)
        path.write_text("null")
        with self.assertRaisesRegex(ValueError,"JSON result object"):
            comparison.build_report(self.paper)

    def complete_min5(self):
        """Tiny complete reporting schema, deliberately leaving all min15 files absent."""
        files = {}
        for _, _, stem, path, unit in comparison.METRICS:
            obj = files.setdefault(stem, {})
            for key in path[:-1]:
                obj = obj.setdefault(key, {})
            obj[path[-1]] = True if unit == "boolean" else 1
        files["a2_distances"]["K"] = 1000
        arts, immunology = comparison.ARTS, comparison.IMMUNOLOGY
        field = {"names": [arts, immunology], "within": {arts: 5.4, immunology: 3.8},
                 "mean_distance": [[5.4, 4.8], [4.7, 3.8]]}
        country = {"names": ["CN", "JP"], "within": {"CN": 3.4, "JP": 3.6}}
        files["a9_structure"] = {"field": field, "country": country,
                                  "hub_removal": [{"how": how, "share": share, "mean": 5.}
                                                  for share in (.0001, .001, .01, .05) for how in ("top", "random")]}
        files["a9b_repaired"] = {"field": field, "country": country}
        files["a2c_team_cap"] = {"sources": 1000, "baseline_targets": {"records": 10},
                                  "cap50_validation": {"sources_checked": 1000, "full_histogram_exact": True,
                                                       "fixed_targets_exact": True}}
        for cap in ("cap_50", "cap_100", "cap_500", "any"):
            files["a2c_team_cap"][cap] = {"mean": 4., "source_mean_se": .01, "p90": 6, "reached_share": .9,
                                         "works_left_out": 0, "pairs_left_out": 0,
                                         "fixed_baseline_targets": {"mean": 4., "source_mean_se": .01,
                                                                    "paired_cap_minus_50": {"mean": 0., "se": 0.}}}
        for stem, obj in files.items():
            self.write("min5", stem, obj)
        return files

    def test_final_separates_archival_absence_without_changing_progress_mode(self):
        self.complete_min5()
        progress = comparison.build_report(self.paper)
        self.assertEqual(self.metrics(progress)["nodes"]["min15"]["status"], "pending")
        final = comparison.build_report(self.paper, final=True)
        self.assertTrue(final["availability"]["required_min5_complete"])
        self.assertEqual(final["pending_metric_values"], 0)
        self.assertEqual(self.metrics(final)["nodes"]["min15"]["status"], "unavailable")
        self.assertEqual(final["team_size_sensitivity"]["min15"]["status"], "unavailable")
        self.assertEqual(final["rankings"]["field"]["comparison"]["status"], "unavailable")
        self.assertTrue(all(row["status"] == "unavailable" for path, row in final["source_manifest"].items()
                            if "/results_min15/" in path))
        rendered = comparison.markdown(final)
        self.assertIn("Unavailable (archive)", rendered)
        self.assertIn("threshold_comparison.py --final", rendered)
        self.assertNotIn("Pending", rendered)
        self.assertNotIn("pending until that output exists", " ".join(final["interpretation"]))
        json.dumps(final, allow_nan=False)

    def test_final_requires_every_min5_cap_field_and_full_source_validation(self):
        files = self.complete_min5()
        del files["a2c_team_cap"]["cap_500"]["fixed_baseline_targets"]["paired_cap_minus_50"]["se"]
        self.write("min5", "a2c_team_cap", files["a2c_team_cap"])
        with self.assertRaisesRegex(ValueError, "five-work results"):
            comparison.build_report(self.paper, final=True)
        files = self.complete_min5()
        files["a2c_team_cap"]["sources"] = 100
        self.write("min5", "a2c_team_cap", files["a2c_team_cap"])
        with self.assertRaisesRegex(ValueError, "1,000-source"):
            comparison.build_report(self.paper, final=True)

    def test_failed_final_cli_preserves_existing_report(self):
        for path in (self.paper / "results/threshold_comparison.json", self.paper / "THRESHOLD_COMPARISON.md"):
            path.write_text("previous report")
        with patch.object(sys, "argv", ["threshold_comparison.py", "--paper-dir", str(self.paper), "--final"]):
            with self.assertRaisesRegex(ValueError, "five-work results"):
                comparison.main()
        self.assertEqual((self.paper / "results/threshold_comparison.json").read_text(), "previous report")
        self.assertEqual((self.paper / "THRESHOLD_COMPARISON.md").read_text(), "previous report")


if __name__ == "__main__":
    unittest.main()
