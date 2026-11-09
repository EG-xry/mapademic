"""Exact reconstruction on a graph with removals, contractions and parallel edges."""
import copy
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validate_min5 import (check_fixed_targets, check_histogram_summary, check_repaired_center_results,
                          contains, finite_json, pack, reconstruct_edge_count)


def test_repaired_center_and_group_denominators_are_explicit_and_consistent():
    selection = {"selection_version": 2, "sources": 97, "surviving_sources_total": 100,
                 "excluded_sources_outside_component": 3, "component_size": 900, "candidates": 200}
    repaired = {"nodes": {"after": 1000}, "center_selection": selection,
                "all_sources": {"sources_after": 100}, "diameter_lower_bound": {"giant_component": 900}}
    center = {**selection, "estimate_cut": 4., "top20_largest_estimate": 3.,
              "most_central": [{"rank": i + 1, "mean_distance": 2. + i / 1000, "estimate": 3.} for i in range(200)]}
    group = {"names": ["A", "B"], "sources_per_group": [10, 100],
             "reachable_pairs": [[90, 500], [1000, 4900]], "mean_distance": [[2., 3.], [3.1, 2.2]]}
    groups = {"component": {"size": 900, "nodes": 1000, "verified_largest": True},
              "field": group, "country": group}
    assert check_repaired_center_results(repaired, center, groups)["sources"] == 97
    for target, key, value in ((repaired["center_selection"], "sources", 100),
                                (groups["component"], "size", 500),
                                (center["most_central"][3], "mean_distance", float("nan")),
                                (group, "sources_per_group", [0, 100]),
                                (group, "reachable_pairs", [[90, 0], [1000, 4900]])):
        old = target[key]
        target[key] = value
        try:
            check_repaired_center_results(repaired, center, groups)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"invalid repaired selection or group field accepted: {key}")
        finally:
            target[key] = old


def test_reconstruction_against_python_set():
    # Merge 0/1, remove 4. (0,1) becomes a loop; (0,2)/(1,2) become parallel.
    src = np.array([0, 0, 3, 1, 1, 2, 2, 3], np.int32)
    dst = np.array([1, 2, 0, 2, 4, 3, 5, 4], np.int32)
    rep = np.array([0, 0, 1, 2, -1, 3], np.int64)
    flagged = pack(np.array([0], np.int32), np.array([3], np.int32))
    table = pa.table({"src": src, "dst": dst})
    expected = {tuple(sorted((int(rep[a]), int(rep[b])))) for a, b in zip(src, dst)
                if tuple(sorted((a, b))) != (0, 3) and rep[a] >= 0 and rep[b] >= 0 and rep[a] != rep[b]}
    for size in (1, 3, 100):
        with tempfile.TemporaryDirectory() as temporary:
            result = reconstruct_edge_count(table.to_batches(max_chunksize=size), rep, flagged, Path(temporary), buckets=3)
        assert result["repaired_links"] == len(expected) == 3
        assert result["flagged_links"] == 1
        assert result["contraction_loops"] == 1
        assert result["parallel_edges_merged"] == 1
        assert result["unflagged_links_touching_removed"] == 2


def test_membership_bounds_and_nonfinite_json():
    assert contains(np.array([3, 7], np.uint64), np.array([0, 3, 4, 7, 9], np.uint64)).tolist() == [False, True, False, True, False]
    assert not contains(np.array([], np.uint64), np.array([3], np.uint64)).any()
    finite_json({"null_is_allowed": None, "number": 2.0})
    try:
        finite_json({"nested": [float("nan")]})
    except ValueError as error:
        assert "root.nested[0]" in str(error)
    else:
        raise AssertionError("nonfinite JSON passed validation")


def test_only_documented_metadata_nan_is_allowed():
    paths = []
    finite_json({"nodes": [{"xw": float("nan"), "yw": float("nan")}]},
                "paper/results/a3b_far_edge.json", paths)
    assert len(paths) == 2
    finite_json({"prolific_uncited": {"top": [["A researcher", 50, float("nan"), float("nan")]]}},
                "paper/results/b1_uncited.json", paths)
    assert len(paths) == 4
    for value, path in ((float("inf"), "paper/results/a3b_far_edge.json.nodes[0].xw"),
                        (float("nan"), "paper/results/a2_distances.json.all_links.mean")):
        try:
            finite_json(value, path, [])
        except ValueError:
            pass
        else:
            raise AssertionError("nonfinite statistic or infinite coordinate was accepted")


def fixed_target_fixture():
    baseline = np.array([[1, 1, 1, 1, 0], [1, 2, 1, 0, 0]], np.int64)
    fixed = np.array([[1, 2, 1, 0, 0], [1, 3, 0, 0, 0]], np.int64)
    reported = {"pairs": 6, "mean": 7 / 6, "median": 1, "p90": 2, "pmf": {1: 5 / 6, 2: 1 / 6},
                "source_mean_se": 1 / 6, "target_records": 4, "target_pairs_per_source": 3,
                "paired_cap_minus_50": {"mean": -.5, "se": 1 / 6, "sources": 2,
                                        "minimum_source_change": -2 / 3, "maximum_source_change": -1 / 3},
                "validation": {"constant_target_count": True, "source_self_count_exact": True,
                               "every_source_cdf_nondecreasing_vs_previous_cap": True,
                               "every_source_cdf_nondecreasing_vs_cap50": True}}
    return baseline, fixed, reported


def test_fixed_targets_recompute_paired_effect_and_standard_error():
    baseline, fixed, reported = fixed_target_fixture()
    np.testing.assert_allclose(check_histogram_summary(fixed, reported), [4 / 3, 1])
    check_fixed_targets(fixed, baseline, baseline, reported, 4)
    for part, key in ((None, "pairs"), (None, "source_mean_se"), ("paired_cap_minus_50", "se"),
                      ("paired_cap_minus_50", "mean"), ("paired_cap_minus_50", "maximum_source_change")):
        bad = copy.deepcopy(reported)
        target = bad if part is None else bad[part]
        target[key] += .1
        try:
            check_fixed_targets(fixed, baseline, baseline, bad, 4)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"corrupt fixed-target {part}/{key} accepted")


def test_fixed_targets_reject_source_cdf_worsening_despite_unchanged_pooled_mean():
    baseline, _, reported = fixed_target_fixture()
    # Source 0 worsens by 1/3 while source 1 improves by 1/3: a pooled mean
    # comparison alone would miss the impossible worsening under added edges.
    fixed = np.array([[1, 1, 0, 2, 0], [1, 3, 0, 0, 0]], np.int64)
    reported.update(mean=5 / 3, p90=3, pmf={1: 2 / 3, 3: 1 / 3}, source_mean_se=2 / 3)
    reported["paired_cap_minus_50"].update(mean=0, se=1 / 3, minimum_source_change=-1 / 3, maximum_source_change=1 / 3)
    try:
        check_fixed_targets(fixed, baseline, baseline, reported, 4)
    except AssertionError as error:
        assert "CDF worsened" in str(error)
    else:
        raise AssertionError("per-source CDF worsening was hidden by pooled results")
