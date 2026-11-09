"""A8c: the center of the repaired network, kept in full.

A8's last step again, on A8's repair (cache/repair_rep.npy): surviving unique A2 sources in the verified largest
component pick the 200 best candidates. Their exact mean distance to other members of that component is then
computed. A8 keeps the top 30; this keeps all 200 and reports the selection counts and estimate cut.
"""
import json
import sys

import numpy as np
from numba import get_num_threads
from center_cache import reusable_center, screening_candidates, signature, verified_largest_component

from common import CACHE, Names, load_csr, log, save_result, repair_link_mask
from kernels import _bfs, contract, distance_sums, exact_mean, highest_degree_members

rep = np.load(CACHE / "repair_rep.npy")
sources = np.load(CACHE / "a2_sources.npy")
cached = CACHE / "repair_center.json"
if cached.exists():
    previous = json.loads(cached.read_text())
    surviving = np.unique(rep[sources][rep[sources] >= 0])
    if reusable_center(previous, signature(rep, sources), len(surviving), int(rep.max()) + 1):
        result = previous["result"]
        save_result("a8c_center", result)
        log("reused A8's identical 200-candidate calculation; repair, sources, graph and names verified")
        sys.exit(0)

offsets, neighbors = load_csr(mmap=False)
deg = np.diff(offsets)
skip = np.asarray(repair_link_mask(offsets, neighbors))
rep = np.load(CACHE / "repair_rep.npy")
m = int(rep.max()) + 1
size = np.bincount(rep[rep >= 0], minlength=m)
off2, nb2 = contract(offsets, neighbors, skip, rep, m)
del neighbors, skip
log(f"repaired network: {m:,} nodes, {len(nb2) // 2:,} links")

sources = np.load(CACHE / "a2_sources.npy")
s_after = np.unique(rep[sources][rep[sources] >= 0])
reach = np.empty(m, np.int16)
_bfs(off2, nb2, int(np.argmax(np.diff(off2))), reach, np.empty(m, np.int32))
component = verified_largest_component(reach)
included_sources = s_after[component[s_after]]
if len(included_sources) == 0:
    raise ValueError("No surviving baseline sources lie in the repaired largest component")
sums, cnt = distance_sums(off2, nb2, included_sources, get_num_threads())
est, best, selection = screening_candidates(s_after, reach, sums, cnt)
exact = exact_mean(off2, nb2, best)
order = np.argsort(exact)

names = Names()
first = highest_degree_members(rep, np.argsort(-deg), m)    # as in A8: preserve the default degree tie order
rows = [{"rank": k + 1, "name": names[int(first[best[i]])], "mean_distance": float(exact[i]), "estimate": float(est[best[i]]),
         "degree": int(off2[best[i] + 1] - off2[best[i]]), "members": int(size[best[i]])} for k, i in enumerate(order)]
out = {**selection, "candidates": len(best), "estimate_cut": float(est[best[-1]]),
       "top20_largest_estimate": max(row["estimate"] for row in rows[:20]), "most_central": rows}
save_result("a8c_center", out)
log({k: v for k, v in out.items() if k != "most_central"})
for row in rows[:80]:
    log(row["rank"], row["name"], round(row["mean_distance"], 4), round(row["estimate"], 4), row["degree"], row["members"])
