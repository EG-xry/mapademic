"""The numbers the paper quotes, from results/*.json, next to the same numbers on the map's 15-work network
(results_min15/). For revising the text: python quoted_numbers.py > ../results/quoted_numbers.txt
"""
import json

from common import PAPER

NETS = {"min5": PAPER / "results", "min15": PAPER / "results_min15"}


def load(net, name):
    p = NETS[net] / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def get(obj, path):
    for k in path.split("."):
        if obj is None:
            return None
        obj = obj[int(k)] if isinstance(obj, list) else obj.get(k)
    return obj


ROWS = [
    ("a1_basics", "nodes"), ("a1_basics", "links"), ("a1_basics", "flagged_share"), ("a1_basics", "degree.median"),
    ("a1_basics", "degree.max"), ("a1_basics", "degree.isolated"), ("a1_basics", "components_all.largest_share"),
    ("a2_distances", "all_links.mean"), ("a2_distances", "all_links.median"), ("a2_distances", "all_links.p90_effective_diameter"),
    ("a2_distances", "all_links.pairs"), ("a2_distances", "per_source_mean.all_se"), ("a2_distances", "all_links.pmf"),
    ("a2_distances", "source_eccentricity.all_mean"), ("a2_distances", "source_eccentricity.all_max"),
    ("a2_distances", "flagged_removed.mean"),
    ("a1b_clustering", None), ("a3_farthest", None), ("a3b_far_edge", "width_by_step"),
    ("a9c_field_order", None), ("a5_clusters", "communities"), ("a5_clusters", "within_community_link_share"),
    ("a10_checks", "modularity"), ("a10_checks", "touching"), ("a10_checks", "social_sciences"), ("a10_checks", "country_known"),
    ("b3_hubs", "overlap_top1000"), ("b3_hubs", "spearman"), ("b3_hubs", "rich_club"), ("b3_hubs", "assortativity_degree"),
    ("b3_hubs", "assortativity_log_degree"), ("a4b_hubs", "top1000_cited"), ("a4b_hubs", "spearman"),
    ("b2_ties", None), ("b2b_long_links", None), ("b1_uncited", "all"), ("b1_uncited", "position"),
    ("b1_uncited", "uncited_twins"), ("a8_repair", "removed"), ("a8_repair", "merged_pairs"), ("a8_repair", "nodes"),
    ("a8_repair", "links_after"), ("a8_repair", "all_sources.after.mean"), ("a8_repair", "diameter_lower_bound"),
    ("a7b_paired_peak25_strict", None), ("a2c_team_cap", None), ("leiden", None),
]

if __name__ == "__main__":
    for name, path in ROWS:
        print(f"## {name}{'.' + path if path else ''}")
        for net in NETS:
            v = load(net, name)
            v = get(v, path) if path else v
            s = json.dumps(v, default=str)
            print(f"  {net}: {s[:1500]}")
