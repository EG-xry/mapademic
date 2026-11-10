"""A8: repair the network with what we can detect, and measure again.

Removed: flagged links (region share) whose common-name endpoint also has a common full name (common.repair_link_mask); merged hubs by the strict rule (A6d: 25+ institutions in a year, a name
key shared by 200+ records and a full name by 10+ retained profiles); common-name hubs with fewer than half
their works remaining in the 2026-10-14 snapshot or cached API response; and records flagged as not
people (A6e). Merged into one node: cross-script twins (A6g, at the smallest shared-coauthor threshold
where decoys pass at most 5% of the time) and likely splits (A6e: same name, 3+ shared coauthors making up 20%+ of the smaller one's in name groups of at most
five profiles, 50%+ in groups of six to twenty; common.split_keep). Then distances from the A2 sources, distances from
members of the script-separated communities, the diameter (double sweeps) and the most central researchers,
against the network as delivered.

Repaired-source histograms and center estimates share each BFS and checkpoint every 64 sources.
The delivered baseline reuses A2c's source histograms only after verifying source order, the input
manifest/checkpoint identity, and exact integer parity with A2; otherwise it is recomputed.
Completed community and center results are also cached.
Center screening uses only surviving sources in a verified largest component; all-source distance summaries
retain sources in other components and their reachable targets. Occupied terminal distance bins fail clearly.
"""
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numba import get_num_threads
from center_cache import (reusable_center, screening_candidates, signature, source_with_largest_reach,
                          verified_largest_component)

from common import CACHE, ND, Names, PG, RESULTS, SCRIPT_COMMUNITIES, con, full_name_counts, load_csr, log, save_result, repair_link_mask, split_keep
from repair_inputs import hub_metadata
from repair_metrics import guarded_summary, verify_cached_summary
from kernels import (_bfs, checkpointed_distance_summaries, contract, double_sweeps, exact_mean,
                     highest_degree_members, histograms, reuse_a2_summary, union_pairs)

rng = np.random.default_rng(20261024)


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=1))
    temporary.replace(path)


def delivered_baseline(offsets, neighbors, sources):
    """Reuse verified per-source cap-50 counts; legacy or invalid caches get fresh BFS."""
    baseline = reuse_a2_summary(RESULTS, CACHE, PG, sources, maxd=64, progress=log)
    if baseline is not None:
        return baseline
    return guarded_summary(histograms(offsets, neighbors, sources, 64), "delivered A8 baseline")


# --- what to remove and what to merge ---------------------------------------------------------------
offsets, neighbors = load_csr(mmap=False)
n = len(offsets) - 1
deg = np.diff(offsets)
skip = np.asarray(repair_link_mask(offsets, neighbors))
comm = pq.read_table(CACHE / "node_attrs.parquet", columns=["community"]).column("community").to_numpy()
aff = pd.read_csv(CACHE / "a6_affiliations.csv")
aff = aff[(aff.group == "top") & aff.alive]
top = np.argsort(-deg, kind="stable")[:50_000]
selected = np.union1d(top, aff.node_idx.to_numpy())
ids = np.load(PG / "ids.npy", mmap_mode="r")
targets = pd.DataFrame({"id": ids[selected].astype(np.int64), "node_idx": selected})
if not (CACHE / "full_name_counts.parquet").exists():
    full_name_counts()
c = con()
c.execute("SET memory_limit='4GB'")
try:
    metadata = hub_metadata(c, targets, ND / "nodes.parquet", CACHE / "snapshot_20261014.parquet",
                            PG / "identity_signals.parquet", CACHE / "full_name_counts.parquet")
finally:
    c.close()
aff = aff[(aff.peak_insts >= 25) & (metadata.key_n.reindex(aff.node_idx).to_numpy() >= 200)
          & (metadata.full_n.reindex(aff.node_idx).to_numpy() >= 10)]
hub = metadata.reindex(top)
ratio = hub.new_works.to_numpy() / hub.old_works.to_numpy()
live = pd.read_csv(CACHE / "a6_live_works.csv").set_index("node_idx").works_ratio      # October responses reused in November continuation
split_any = (ratio < 0.5) | (live.reindex(top).to_numpy() < 0.5)                      # either source
split = top[split_any & (hub.key_n.to_numpy() >= 200)]
del metadata, hub, targets, selected, ids, live, ratio, split_any, top
removed = np.zeros(n, bool)
parts = {"merged_strict": aff.node_idx.to_numpy(), "split_since_snapshot": split,
         "not_a_person": np.load(CACHE / "nonperson_idx.npy")}
for k, v in parts.items():
    removed[v] = True
out = {"removed": {k: int(len(v)) for k, v in parts.items()}, "removed_total": int(removed.sum()),
       "removed_degree": int(deg[removed].sum())}
pairs, out["merged_pairs"] = [], {}
for script in ("zh", "ja", "ru", "ko"):
    res = json.load(open(RESULTS / f"a6_twins_{script}.json"))
    ok = [int(t) for t, r in res["pass_rate"].items() if r["decoys"] <= 0.05]
    t = min(ok)
    tw = pd.read_csv(CACHE / f"a6_twins_{script}.csv")
    tw = tw[tw.shared >= t]
    pairs.append(tw[["focal_idx", "twin_idx"]].to_numpy())
    out["merged_pairs"][f"twins_{script}"] = {"threshold": t, "decoy_rate": res["pass_rate"][str(t)]["decoys"], "pairs": int(len(tw))}
sp = pq.read_table(CACHE / "split_pairs.parquet").to_pandas()
small = np.minimum(deg[sp.a.to_numpy()], deg[sp.b.to_numpy()])
sp = sp[split_keep(sp.group_size, sp.shared, small)]
pairs.append(sp[["a", "b"]].to_numpy())
out["merged_pairs"]["splits"] = int(len(sp))
p = np.concatenate(pairs).astype(np.int64)
p = p[~(removed[p[:, 0]] | removed[p[:, 1]])]
root = union_pairs(n, p[:, 0], p[:, 1])
keep = ~removed
uniq, inv = np.unique(root[keep], return_inverse=True)
rep = np.full(n, -1, np.int64)
rep[keep] = inv
m = len(uniq)
size = np.bincount(inv)
out["nodes"] = {"before": int(n), "after": int(m), "merged_away": int(keep.sum() - m),
                "group_sizes": {str(k): int(v) for k, v in zip(*np.unique(np.minimum(size, 6), return_counts=True))}}
log(out)
del pairs, p, sp, small, root, uniq, inv, tw, aff, parts
off2, nb2 = contract(offsets, neighbors, skip, rep, m)
out["links_after"] = int(len(nb2) // 2)
log(f"repaired network: {m:,} nodes, {len(nb2) // 2:,} links")
np.save(CACHE / "repair_rep.npy", rep)

# --- distances ----------------------------------------------------------------------------------------
sources = np.load(CACHE / "a2_sources.npy")
s_after = np.unique(rep[sources][rep[sources] >= 0])
repair_signature = signature(rep, sources)
h_after, sums, cnt = checkpointed_distance_summaries(
    off2, nb2, s_after, 64, get_num_threads(), CACHE / "repair_distance_summaries.npz", repair_signature, progress=log)
out["all_sources"] = {"before": delivered_baseline(offsets, neighbors, sources),
                      "after": guarded_summary(h_after, "repaired all-source distances"), "sources_after": int(len(s_after))}
log("all sources", {k: (v["mean"] if isinstance(v, dict) else v) for k, v in out["all_sources"].items()})
out["community_sources"] = {}
for label, c in SCRIPT_COMMUNITIES.items():
    members = np.flatnonzero((comm == c) & keep & (deg > 0))
    src = rng.choice(members, size=min(100, len(members)), replace=False)
    after = rep[src]
    community_signature = {"repair": repair_signature, "community": int(c),
                           "sources_sha256": hashlib.sha256(src).hexdigest()}
    checkpoint = CACHE / f"repair_community_{c}.json"
    previous = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
    if previous.get("signature") == community_signature:
        result = previous["result"]
        for variant in ("before", "after"):
            verify_cached_summary(result[variant], f"{label} {variant}")
        log(f"{label}: reused completed community searches")
    else:
        result = {"before": guarded_summary(histograms(offsets, neighbors, src, 64), f"{label} delivered"),
                  "after": guarded_summary(histograms(off2, nb2, np.unique(after), 64), f"{label} repaired"),
                  "share_of_sources_merged": float(np.mean(size[after] > 1))}
        atomic_json(checkpoint, {"signature": community_signature, "result": result})
    out["community_sources"][label] = result
    log(label, {k: (v["mean"] if isinstance(v, dict) else v) for k, v in out["community_sources"][label].items()})

# --- the far edge and the center ----------------------------------------------------------------------
reach = np.empty(m, np.int16)
_bfs(off2, nb2, source_with_largest_reach(s_after, h_after), reach, np.empty(m, np.int32))
lcc = np.flatnonzero(verified_largest_component(reach))
d, a, b = double_sweeps(off2, nb2, rng.choice(lcc, 8, replace=False))
names = Names()
first = highest_degree_members(rep, np.argsort(-deg), m)
out["diameter_lower_bound"] = {"after": d, "pair": [names[first[a]], names[first[b]]], "giant_component": int(len(lcc))}
log("diameter lower bound", out["diameter_lower_bound"])
cache_path = CACHE / "repair_center.json"
previous = json.loads(cache_path.read_text()) if cache_path.exists() else {}
if reusable_center(previous, repair_signature, len(s_after), m):
    center = previous["result"]
    log("center: reused completed 200-candidate calculation")
else:
    est, best, selection = screening_candidates(s_after, reach, sums, cnt)
    exact = exact_mean(off2, nb2, best)
    order = np.argsort(exact)
    center_rows = [{"rank": k + 1, "name": names[int(first[best[i]])],
                    "mean_distance": float(exact[i]), "estimate": float(est[best[i]]),
                    "degree": int(off2[best[i] + 1] - off2[best[i]]), "members": int(size[best[i]])}
                   for k, i in enumerate(order)]
    center = {**selection, "candidates": len(best), "estimate_cut": float(est[best[-1]]),
              "top20_largest_estimate": max(row["estimate"] for row in center_rows[:20]), "most_central": center_rows}
    # Preserve all 200 for A8c instead of repeating identical searches to recover rows 31-200.
    atomic_json(cache_path, {"signature": repair_signature, "result": center})
out["most_central"] = [{key: row[key] for key in ("name", "mean_distance", "degree", "members")}
                       for row in center["most_central"][:30]]
out["center_selection"] = {key: center[key] for key in ("selection_version", "sources", "surviving_sources_total",
                                                      "excluded_sources_outside_component", "component_size", "candidates")}
assert center["component_size"] == len(lcc), "cached center component differs from the verified largest component"
log("center source selection", out["center_selection"])
log("most central", [(r["name"], round(r["mean_distance"], 3)) for r in out["most_central"][:15]])
save_result("a8_repair", out)
