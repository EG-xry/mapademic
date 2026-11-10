"""Build a provenance-bearing 5-work versus 15-work comparison from result JSON only.

Run from any directory: python paper/analysis/threshold_comparison.py
Missing files/keys remain pending in their own threshold; no cross-threshold or alternate-file fallback.
Use --final after the five-work run: require all displayed five-work evidence and
label absent historical measurements unavailable, rather than pending work.
No graphs, arrays, Parquet files or network services are read. Output is descriptive sensitivity evidence,
not a paired experiment: the eligible populations and sampled researchers differ across thresholds.
"""
import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PAPER = Path(__file__).resolve().parents[1]
NETWORKS = ("min5", "min15")
ARTS = "Arts and Humanities"
IMMUNOLOGY = "Immunology and Microbiology"

# id, description, result file stem, JSON path, unit
METRICS = (
    ("nodes", "Eligible author records", "a1_basics", ("nodes",), "records"),
    ("links", "Undirected coauthor links", "a1_basics", ("links",), "links"),
    ("giant_component_share", "Largest-component share", "a1_basics", ("components_all", "largest_share"), "fraction"),
    ("median_degree", "Median coauthors", "a1_basics", ("degree", "median"), "coauthors"),
    ("isolated", "Isolated author records", "a1_basics", ("degree", "isolated"), "records"),
    ("flagged_link_share", "Flagged-link share", "a1_basics", ("flagged_share",), "fraction"),
    ("distance_sources", "A2 random sources", "a2_distances", ("K",), "sources"),
    ("mean_distance", "A2 mean distance", "a2_distances", ("all_links", "mean"), "steps"),
    ("mean_distance_se", "A2 source-level standard error", "a2_distances", ("per_source_mean", "all_se"), "steps"),
    ("median_distance", "Median distance", "a2_distances", ("all_links", "median"), "steps"),
    ("p90_distance", "90th-percentile distance", "a2_distances", ("all_links", "p90_effective_diameter"), "steps"),
    ("diameter_lower", "Diameter lower bound", "a3_farthest", ("all_links", "diameter_lower"), "steps"),
    ("diameter_upper", "Diameter upper bound", "a3_farthest", ("all_links", "diameter_upper"), "steps"),
    ("diameter_exact", "Diameter established exactly", "a3_farthest", ("all_links", "exact"), "boolean"),
    ("mean_local_clustering", "Mean local clustering (degree >= 2)", "a1b_clustering", ("all", "mean_local"), "coefficient"),
    ("transitivity", "Global transitivity", "a1b_clustering", ("all", "transitivity"), "coefficient"),
    ("random_clustering", "Degree-matched random expectation", "a1b_clustering", ("all", "random_expectation"), "coefficient"),
    ("field_additive_r2", "Additive field-distance model R-squared", "a9c_field_order", ("additive_r2",), "coefficient"),
    ("field_degree_correlation", "Within-field distance versus median coauthors", "a9c_field_order", ("spearman_within_vs_median_coauthors",), "Spearman rho"),
    ("communities_total", "Communities (A5 count)", "a5_clusters", ("communities", "total"), "communities"),
    ("communities_1000plus", "Communities with >= 1,000 members (A5)", "a5_clusters", ("communities", "with_1000_plus"), "communities"),
    ("national_communities", "National communities with >= 1,000 members", "a5_clusters", ("communities", "national_1000_plus"), "communities"),
    ("national_community_members", "Members in national communities", "a5_clusters", ("communities", "members_in_national"), "records"),
    ("within_community_link_share", "Link weight within communities", "a5_clusters", ("within_community_link_share",), "fraction"),
    ("leiden_1000plus", "Communities with >= 1,000 members (Leiden diagnostics)", "leiden", ("communities_1000plus",), "communities"),
    ("modularity", "Weighted modularity, resolution 1", "a10_checks", ("modularity",), "coefficient"),
    ("uncited_share", "Uncited author-record share", "b1_uncited", ("all", "uncited"), "fraction"),
    ("removal_baseline_sources", "A9 initial-baseline sources (A2c cap 50)", "a2c_team_cap_100", ("sources",), "sources"),
    ("removal_baseline_mean", "A9 initial 100-source baseline (A2c cap 50)", "a2c_team_cap_100", ("cap_50", "mean"), "steps"),
    ("a7_baseline_sources", "A7 strict retained baseline sources", "a7_remedy_peak25_strict", ("sources",), "sources"),
    ("a7_baseline_mean", "A7 strict retained-source unmodified baseline", "a7_remedy_peak25_strict", ("all", "mean"), "steps"),
    ("repair_nodes", "Repaired nodes", "a8_repair", ("nodes", "after"), "nodes"),
    ("repair_links", "Repaired links", "a8_repair", ("links_after",), "links"),
    ("repair_removed", "Removed author records", "a8_repair", ("removed_total",), "records"),
    ("repair_merged_away", "Records absorbed by contraction", "a8_repair", ("nodes", "merged_away"), "records"),
    ("repair_sources", "Distinct retained repaired sources", "a8_repair", ("all_sources", "sources_after"), "sources"),
    ("repair_mean", "Mean distance after repair", "a8_repair", ("all_sources", "after", "mean"), "steps"),
    ("repair_p90", "90th-percentile distance after repair", "a8_repair", ("all_sources", "after", "p90"), "steps"),
)


def pointer(path):
    return "/" + "/".join(str(key).replace("~", "~0").replace("/", "~1") for key in path) if path else ""


def pending(unit, provenance, reason):
    return {"status": "pending", "value": None, "unit": unit, "provenance": provenance, "reason": reason}


def derived(value, unit, inputs, calculation):
    provenance = [source for item in inputs for source in item["provenance"]]
    if any(item["status"] != "available" for item in inputs):
        return pending(unit, provenance, "One or more required inputs are pending")
    return {"status": "available", "value": value, "unit": unit, "provenance": provenance, "calculation": calculation}


class Results:
    def __init__(self, paper):
        self.paper = Path(paper)
        self.folders = {"min5": self.paper / "results", "min15": self.paper / "results_min15"}
        self.cache = {}
        self.manifest = {}

    def file(self, net, stem):
        return self.folders[net] / f"{stem}.json"

    def provenance(self, net, stem, path):
        return [{"file": str(self.file(net, stem).relative_to(self.paper.parent)), "key": pointer(path)}]

    def load(self, net, stem):
        key = (net, stem)
        if key not in self.cache:
            path = self.file(net, stem)
            relative = str(path.relative_to(self.paper.parent))
            if not path.exists():
                self.cache[key] = None
                self.manifest[relative] = {"status": "pending", "reason": "File does not exist"}
            else:
                raw = path.read_bytes()
                self.cache[key] = json.loads(raw)  # malformed JSON is an error, never silently treated as absent
                if not isinstance(self.cache[key], dict):
                    raise ValueError(f"Expected a JSON result object: {path}")
                self.manifest[relative] = {"status": "available", "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        return self.cache[key]

    def get(self, net, stem, path, unit):
        provenance = self.provenance(net, stem, path)
        value = self.load(net, stem)
        if value is None:
            return pending(unit, provenance, "Source file has not been produced")
        try:
            for key in path:
                value = value[key]
        except (KeyError, IndexError, TypeError):
            return pending(unit, provenance, "Source JSON does not contain this key")
        if value is None or isinstance(value, float) and not math.isfinite(value):
            return pending(unit, provenance, "Source value is null or non-finite")
        return {"status": "available", "value": value, "unit": unit, "provenance": provenance}

    def matrix_cell(self, net, stem, source, target):
        data = self.load(net, stem)
        if data is None or "field" not in data:
            return pending("steps", self.provenance(net, stem, ("field", "mean_distance")), "Field-distance output is pending")
        names = data["field"]["names"]
        if source not in names or target not in names:
            return pending("steps", self.provenance(net, stem, ("field", "names")), "Requested field is not present")
        observation = self.get(net, stem, ("field", "mean_distance", names.index(source), names.index(target)), "steps")
        observation["source_group"], observation["target_group"] = source, target
        observation["note"] = "Directional estimate: sampled source group to all reachable target-group members; matrix is stored to four decimals."
        return observation

    def removal(self, net, share, how, key, unit):
        stem = "a9_structure"
        data = self.load(net, stem)
        rows = data.get("hub_removal", []) if data else []
        matches = [i for i, row in enumerate(rows) if row["how"] == how and math.isclose(row["share"], share, abs_tol=1e-12)]
        if not matches:
            return pending(unit, self.provenance(net, stem, ("hub_removal",)), f"Removal result is pending: {how}, share={share}")
        if len(matches) > 1:
            raise ValueError(f"Duplicate removal rows in {net}: {how}, {share}")
        return self.get(net, stem, ("hub_removal", matches[0], key), unit)


def ranks(values):
    """Average ranks, so the descriptive correlation handles ties explicitly."""
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average = (start + 1 + end) / 2
        for index, _ in ordered[start:end]:
            result[index] = average
        start = end
    return result


def correlation(x, y):
    xm, ym = sum(x)/len(x), sum(y)/len(y)
    numerator = sum((a-xm)*(b-ym) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a-xm)**2 for a in x) * sum((b-ym)**2 for b in y))
    return numerator / denominator if denominator else None


def group_rankings(results, kind, stem="a9_structure"):
    by_network, observations = {}, {}
    for net in NETWORKS:
        root = results.get(net, stem, (kind, "within"), "steps")
        observations[net] = root
        if root["status"] == "pending":
            by_network[net] = {**root, "rows": []}
            continue
        values = root["value"]
        ordered = sorted(values, key=lambda group: (values[group], group))
        ranked = ranks([values[group] for group in ordered])
        by_network[net] = {"status": "available", "rows": [{"group": group, "rank": rank,
                            "distance": results.get(net, stem, (kind, "within", group), "steps")}
                            for group, rank in zip(ordered, ranked)]}
    complete = all(item["status"] == "available" for item in by_network.values())
    comparison = {"status": "pending", "reason": "Both thresholds must have this result"}
    if complete:
        a, b = (observations[net]["value"] for net in NETWORKS)
        common = sorted(set(a) & set(b))
        rho = correlation(ranks([a[key] for key in common]), ranks([b[key] for key in common])) if len(common) >= 2 else None
        first = {net: [row["group"] for row in by_network[net]["rows"][:3]] for net in NETWORKS}
        comparison = {"status": "available", "common_groups": common,
                      "only_min5": sorted(set(a)-set(b)), "only_min15": sorted(set(b)-set(a)),
                      "spearman_on_common_groups": derived(rho, "Spearman rho", list(observations.values()), "Pearson correlation of average ranks on the common groups; descriptive, no hypothesis test."),
                      "nearest_three": first, "same_nearest_three_in_order": first["min5"] == first["min15"],
                      "same_farthest_group": (by_network["min5"]["rows"][-1]["group"] == by_network["min15"]["rows"][-1]["group"]) if a and b else None,
                      "all_common_distances_larger_at_min5": all(a[key] > b[key] for key in common) if common else None,
                      "provenance": [source for item in observations.values() for source in item["provenance"]]}
    return {"measure": "Mean within-group shortest-path distance, ascending rank", "unit": "steps",
            "population_note": "Countries are the top 16 by researcher count in each network; field records with the fallback topic are excluded.",
            "networks": by_network, "comparison": comparison}


def build_report(paper=PAPER, final=False):
    results = Results(paper)
    metrics = []
    for key, label, stem, path, unit in METRICS:
        metrics.append({"id": key, "label": label, "unit": unit,
                        **{net: results.get(net, stem, path, unit) for net in NETWORKS}})
    for key, label, source, target in (("arts_within", "Arts and humanities within-field distance", ARTS, ARTS),
                                       ("arts_to_immunology", "Arts and humanities sources to immunology", ARTS, IMMUNOLOGY),
                                       ("immunology_to_arts", "Immunology sources to arts and humanities", IMMUNOLOGY, ARTS)):
        metrics.append({"id": key, "label": label, "unit": "steps",
                        **{net: results.matrix_cell(net, "a9_structure", source, target) for net in NETWORKS}})
    for share in (.0001, .001, .01, .05):
        for how in ("top", "random"):
            metrics.append({"id": f"hub_removal_{how}_{share:g}", "label": f"Mean after removing {share:.2%} ({how})", "unit": "steps",
                            **{net: results.removal(net, share, how, "mean", "steps") for net in NETWORKS}})
    metric_map = {item["id"]: item for item in metrics}
    invariants = {}
    for net in NETWORKS:
        within, cross = metric_map["arts_within"][net], metric_map["arts_to_immunology"][net]
        available = within["status"] == cross["status"] == "available"
        invariants[net] = derived(within["value"] > cross["value"] if available else None, "boolean", [within, cross],
                                  "Arts-within distance exceeds distance from arts sources to immunology targets.")
    team_size = {}
    for net in NETWORKS:
        stem = "a2c_team_cap"
        data = results.load(net, stem)
        team_size[net] = {"status": "available" if data is not None else "pending",
                          "sources": results.get(net, stem, ("sources",), "sources"),
                          "baseline_targets": results.get(net, stem, ("baseline_targets",), "target population"),
                          "cap50_validation": results.get(net, stem, ("cap50_validation",), "validation record"), "rows": []}
        for cap, key in ((50, "cap_50"), (100, "cap_100"), (500, "cap_500"), ("any", "any")):
            row = {"maximum_authors_per_work": cap}
            for column, unit in (("mean", "steps"), ("source_mean_se", "steps"), ("p90", "steps"),
                                 ("reached_share", "fraction"), ("works_left_out", "works"), ("pairs_left_out", "pair contributions")):
                row[column] = results.get(net, stem, (key, column), unit)
            for column, path in (("fixed_mean", ("mean",)), ("fixed_source_mean_se", ("source_mean_se",)),
                                 ("fixed_paired_cap_minus50_mean", ("paired_cap_minus_50", "mean")),
                                 ("fixed_paired_cap_minus50_se", ("paired_cap_minus_50", "se"))):
                row[column] = results.get(net, stem, (key, "fixed_baseline_targets", *path), "steps")
            team_size[net]["rows"].append(row)
    rankings = {kind: group_rankings(results, kind) for kind in ("field", "country")}
    repaired_rankings = {kind: group_rankings(results, kind, "a9b_repaired") for kind in ("field", "country")}
    report = {"schema_version": 1, "generated_at": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds"),
              "thresholds": {"min5": {"minimum_quality_filtered_works": 5, "folder": "paper/results"},
                             "min15": {"minimum_quality_filtered_works": 15, "folder": "paper/results_min15"}},
              "interpretation": ["Descriptive sensitivity comparison; eligible populations and random source draws differ between thresholds.",
                                 "No cross-threshold differences are paired estimates; no cross-threshold p-values or confidence intervals are inferred.",
                                 "A2 uses 1,000 sources. A7/A9 removal analyses use the first 100, dropping removed sources; their baseline must not be replaced by A2's overall mean.",
                                 "A9 does not serialize its unmodified baseline. The initial-100 baseline row uses A2c_team_cap_100's cap-50 mean; it remains pending until that output exists.",
                                 "A7 strict removes any source affected by its comparison sets before running its baseline. Its retained-source count and mean are separate rows and are not substituted for A9's initial-100 baseline.",
                                 "Reachable-pair means can change with connectivity as well as path length. Repair also removes and contracts records.",
                                 "Each network's A2 standard error describes variation among its source means, not independent uncertainty for every source-target pair.",
                                 "Missing JSON files or keys are pending; neither threshold borrows the other's values. Archived outputs may remain unavailable indefinitely."],
              "metrics": metrics, "arts_immunology_ordering": invariants, "rankings": rankings,
              "repaired_rankings": repaired_rankings, "team_size_sensitivity": team_size,
              "source_manifest": results.manifest}
    report["pending_metric_values"] = sum(item[net]["status"] == "pending" for item in metrics for net in NETWORKS)
    report["final"] = False
    report["unavailable_archival_metric_values"] = 0
    if final:
        finalize_report(report)
    return report


def missing_observations(value, path):
    """Inventory unresolved selected report cells, including nested experiment rows."""
    missing = []
    if isinstance(value, dict):
        if value.get("status") == "pending":
            missing.append({"path": path, "reason": value.get("reason", "Required output is missing")})
        if value.get("status") == "available" and "value" in value:
            def invalid(item):
                if isinstance(item, float):
                    return not math.isfinite(item)
                if isinstance(item, dict):
                    return any(invalid(v) for v in item.values())
                if isinstance(item, list):
                    return any(invalid(v) for v in item)
                return False
            if invalid(value["value"]):
                missing.append({"path": path, "reason": "Required value contains a non-finite number"})
        for key, item in value.items():
            missing.extend(missing_observations(item, f"{path}/{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            missing.extend(missing_observations(item, f"{path}/{index}"))
    return missing


def network_sections(report, net):
    for row in report["metrics"]:
        yield f"/metrics/{row['id']}/{net}", row[net]
    yield f"/arts_immunology_ordering/{net}", report["arts_immunology_ordering"][net]
    for kind in ("rankings", "repaired_rankings"):
        for group, value in report[kind].items():
            yield f"/{kind}/{group}/networks/{net}", value["networks"][net]
    yield f"/team_size_sensitivity/{net}", report["team_size_sensitivity"][net]


def archival_unavailable(value):
    if isinstance(value, dict):
        if value.get("status") == "pending":
            value["status"] = "unavailable"
            value["reason"] = "The archived 15-work outputs do not supply this measurement. " + value.get("reason", "")
        for item in value.values():
            archival_unavailable(item)
    elif isinstance(value, list):
        for item in value:
            archival_unavailable(item)


def finalize_report(report):
    """Strict current-run completeness; missing archives are never backfilled.

    This is a report-completeness gate. Production validity remains the job of
    validate_min5.py, which checks source identities and exact graph statistics.
    """
    missing = [item for path, value in network_sections(report, "min5")
               for item in missing_observations(value, path)]
    for kind in ("rankings", "repaired_rankings"):
        for group, value in report[kind].items():
            if not value["networks"]["min5"]["rows"]:
                missing.append({"path": f"/{kind}/{group}/networks/min5/rows", "reason": "No ranked groups"})
    if missing:
        raise ValueError(f"Final comparison requires complete five-work results; {len(missing)} unresolved observations: "
                         + ", ".join(item["path"] for item in missing[:12]))
    experiment = report["team_size_sensitivity"]["min5"]
    proof = experiment["cap50_validation"]["value"]
    if (experiment["sources"]["value"] != 1000 or not isinstance(proof, dict)
            or proof.get("full_histogram_exact") is not True or proof.get("fixed_targets_exact") is not True
            or proof.get("sources_checked") != 1000):
        raise ValueError("Final comparison requires the full 1,000-source team-cap result with exact cap-50 validation")
    absent = [item for path, value in network_sections(report, "min15")
              for item in missing_observations(value, path)]
    for _, value in network_sections(report, "min15"):
        archival_unavailable(value)
    for kind in ("rankings", "repaired_rankings"):
        for value in report[kind].values():
            if value["comparison"]["status"] == "pending":
                archival_unavailable(value["comparison"])
    for path, value in report["source_manifest"].items():
        if "/results_min15/" in path:
            archival_unavailable(value)
    report["final"] = True
    report["availability"] = {"required_min5_complete": True, "missing_required_min5": [],
                              "unavailable_archival_observations": absent}
    report["pending_metric_values"] = 0
    report["unavailable_archival_metric_values"] = sum(row["min15"]["status"] == "unavailable" for row in report["metrics"])
    report["interpretation"][3] = ("A9 does not serialize its unmodified baseline. The initial-100 baseline row uses only "
                                    "A2c_team_cap_100's cap-50 mean; an absent archived baseline is unavailable and is never inferred from A2 or A7.")
    report["interpretation"][-1] = ("All required five-work report evidence is present. Unavailable (archive) denotes an absent "
                                     "15-work file or key, not a pending five-work analysis; no value is borrowed or imputed.")


def display(item):
    if item["status"] != "available":
        return "Unavailable (archive)" if item["status"] == "unavailable" else "Pending"
    value, unit = item["value"], item["unit"]
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if value is None:
        return "Undefined"
    if unit == "fraction":
        return f"{value:.3%}"
    if unit in ("records", "nodes", "links", "sources", "communities", "works", "pair contributions"):
        return f"{value:,}"
    if unit == "steps":
        return f"{value:.5f}" if "standard error" in item.get("label", "") else f"{value:.4f}"
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def source_link(observation):
    source = observation["provenance"][0]
    relative = source["file"].removeprefix("paper/")
    return f"[{Path(relative).name}]({relative}) `{source['key']}`"


def markdown(report):
    final = report.get("final", False)
    command = "python paper/analysis/threshold_comparison.py" + (" --final" if final else "")
    availability = ("All required five-work measurements are present. Unavailable (archive) identifies a missing historical "
                    "15-work result, not work still pending for the five-work run; no old-core value is substituted." if final else
                    "Pending means the indicated result file or key is unavailable; no old-core value is substituted.")
    lines = ["# Five-work versus fifteen-work sensitivity comparison", "",
             f"Generated {report['generated_at']} from saved JSON outputs only. Regenerate with `{command}`.", "",
             "This is a descriptive comparison of different eligible populations and source samples. It does not estimate a paired threshold effect. " + availability, "",
             "## Core measurements", "", "| Measurement | 5+ works | 15+ works | Unit | Source key |", "|---|---:|---:|---|---|"]
    for row in report["metrics"]:
        sources = [row[net]["provenance"][0] for net in NETWORKS]
        keys = [f"{Path(source['file']).name}#{source['key']}" for source in sources]
        source_text = f"`{keys[0]}`" if keys[0] == keys[1] else f"5+: `{keys[0]}`; 15+: `{keys[1]}`"
        lines.append(f"| {row['label']} | {display(row['min5'])} | {display(row['min15'])} | {row['unit']} | {source_text} |")
    lines += ["", "Source files for the two columns are respectively in [results](results/) and [results_min15](results_min15/). Every value, including name-aligned matrix cells, has its exact file and JSON key in [threshold_comparison.json](results/threshold_comparison.json). Source bytes are SHA-256 fingerprinted there.", "",
              "A9 itself saves no zero-removal row. Its initial-100 baseline comes only from the cap-50 result in `a2c_team_cap_100.json`. A7 strict excludes sources affected by its comparison sets, so its retained-source count and baseline are separate rows. A2's 1,000-source mean is also separate. Removal means exclude disconnected pairs and removed sources. National-community counts use the A5 classification, not the Leiden diagnostic count.", "",
              "## Observed ordering checks", ""]
    for net, label in (("min5", "5+"), ("min15", "15+")):
        lines.append(f"- Arts-within distance exceeds arts-to-immunology distance at {label}: {display(report['arts_immunology_ordering'][net])}.")
    for kind, ranked in report["rankings"].items():
        comparison = ranked["comparison"]
        if comparison["status"] == "available":
            lines.append(f"- {kind.capitalize()} ranking correlation on {len(comparison['common_groups'])} common groups: {display(comparison['spearman_on_common_groups'])}. Same nearest three in order: {'yes' if comparison['same_nearest_three_in_order'] else 'no'}; all common within-group distances larger at 5+: {'yes' if comparison['all_common_distances_larger_at_min5'] else 'no'}.")
            if kind == "country":
                lines.append(f"- Nearest countries: 5+ = {', '.join(comparison['nearest_three']['min5'])}; 15+ = {', '.join(comparison['nearest_three']['min15'])}.")
        else:
            lines.append(f"- {kind.capitalize()} ordering comparison: {display(comparison)}.")
    for title, key in (("Delivered-network rankings", "rankings"), ("Repaired-network rankings", "repaired_rankings")):
        lines += ["", f"## {title}", "", "Ranks are ascending mean within-group distance; ties receive average ranks. Countries are limited to the top 16 by researcher count in each network. Fields exclude fallback-topic records."]
        for kind, ranked in report[key].items():
            lines += ["", f"### {kind.capitalize()}", ""]
            by_network = {net: {row['group']: row for row in ranked['networks'][net]['rows']} for net in NETWORKS}
            groups = sorted(set(by_network['min5']) | set(by_network['min15']),
                            key=lambda name: (by_network['min5'].get(name, {}).get('rank', float('inf')), name))
            if not groups:
                lines.append("; ".join(f"{net}: " + ("No selected groups" if ranked['networks'][net]['status'] == 'available'
                                                    else display(ranked['networks'][net])) for net in NETWORKS) + ".")
                continue
            lines += ["| Group | 5+ rank | 5+ steps | 15+ rank | 15+ steps |", "|---|---:|---:|---:|---:|"]
            for group in groups:
                values = []
                for net in NETWORKS:
                    row = by_network[net].get(group)
                    absent = ("Not in selected groups" if ranked["networks"][net]["status"] == "available"
                              else display(ranked["networks"][net]))
                    values.extend([f"{row['rank']:g}", display(row['distance'])] if row else [absent, absent])
                lines.append(f"| {group} | " + " | ".join(values) + " |")
            stem = "a9_structure" if key == "rankings" else "a9b_repaired"
            lines += ["", f"Sources: [5+]({{folder}}/{stem}.json) and [15+](results_min15/{stem}.json), `/{kind}/within/<group>`; missing groups are not substituted.".replace("{folder}", "results")]
    lines += ["", "## Separate team-size sensitivity", "", "A2c holds the eligible researcher set fixed and varies the maximum authors per work. The all-reachable summaries can change because newly connected targets enter the average. A second summary fixes targets to the original cap-50 largest component and reports source-paired cap-minus-50 changes (self distances excluded). These within-network paired estimates are distinct from the unpaired work-count-threshold comparison. The full result file is used; a smaller-source result never fills a missing full result."]
    for net, label in (("min5", "5+"), ("min15", "15+")):
        experiment = report['team_size_sensitivity'][net]
        lines += ["", f"### {label} researcher eligibility", ""]
        if experiment['status'] != 'available':
            folder = 'results' if net == 'min5' else 'results_min15'
            detail = "is absent from the archive" if experiment['status'] == 'unavailable' else "has not been produced"
            lines.append(f"{display(experiment)}: `{folder}/a2c_team_cap.json` {detail}.")
            continue
        proof = ("The recorded cap-50 validation is preserved in the JSON report." if experiment['cap50_validation']['status'] == 'available'
                 else f"Cap-50 validation: {display(experiment['cap50_validation'])}.")
        lines += [f"Sources: {display(experiment['sources'])}. {proof}", "",
                  "| Maximum authors/work | Mean steps | Source-mean SE | P90 steps | Reached share | Works omitted |", "|---|---:|---:|---:|---:|---:|"]
        for row in experiment['rows']:
            lines.append(f"| {row['maximum_authors_per_work']} | " + " | ".join(display(row[key]) for key in ('mean','source_mean_se','p90','reached_share','works_left_out')) + " |")
        lines += ["", "Fixed original-LCC targets:", "",
                  "| Maximum authors/work | Mean steps | Source-mean SE | Paired cap-minus-50 | Paired SE |", "|---|---:|---:|---:|---:|"]
        for row in experiment['rows']:
            lines.append(f"| {row['maximum_authors_per_work']} | " + " | ".join(display(row[key]) for key in ('fixed_mean','fixed_source_mean_se','fixed_paired_cap_minus50_mean','fixed_paired_cap_minus50_se')) + " |")
    lines += ["", "## Interpretation and availability" if final else "## Interpretation and pending work", ""]
    lines += [f"- {note}" for note in report['interpretation']]
    if final:
        lines += ["", f"All five-work core cells are complete; {report['unavailable_archival_metric_values']} historical core cells are unavailable. The exact archival gaps, including separate team-size evidence, are listed in the JSON availability inventory.", ""]
    else:
        lines += ["", f"{report['pending_metric_values']} cells in the core table are pending. Re-running this builder fills only outputs actually produced.", ""]
    return "\n".join(lines)


def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, default=PAPER)
    parser.add_argument("--final", action="store_true", help="require complete five-work evidence; mark absent archived metrics unavailable")
    args = parser.parse_args()
    report = build_report(args.paper_dir, final=args.final)
    atomic_write(args.paper_dir / "results/threshold_comparison.json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    atomic_write(args.paper_dir / "THRESHOLD_COMPARISON.md", markdown(report))
    print(f"Wrote threshold comparison: {len(report['metrics'])} core metrics, {report['pending_metric_values']} pending cells, "
          f"{report['unavailable_archival_metric_values']} unavailable archive cells; JSON files only.")


if __name__ == "__main__":
    main()
