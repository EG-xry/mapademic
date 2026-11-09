"""A3c: how solid are the links at the far edge (A3b)?

For every link among the researchers on some shortest path between the two farthest researchers: the qualifying
works (at most 50 authors) the two researchers share, whether the link is flagged as a probable namesake join
(common.suspect_mask) or removed by the repair (repair_link_mask), and each researcher's number of coauthors. Summaries
for the path drawn in Fig. 3b and for the whole corridor.
"""
import json

import numpy as np

from common import CACHE, RESULTS, load_csr, log, repair_link_mask, save_result, suspect_mask

MAX_TEAM = 50

d = json.loads((RESULTS / "a3b_far_edge.json").read_text())
nodes = {r["node"]: r for r in d["nodes"]}
offsets, neighbors = load_csr(mmap=True)                   # read-only maps: only the corridor's rows are touched
n = len(offsets) - 1
a_off = np.load(CACHE / "bipartite_a_off.npy", mmap_mode="r")
a_nb = np.load(CACHE / "bipartite_a_nb.npy", mmap_mode="r")
w_size = np.load(CACHE / "bipartite_w_size.npy", mmap_mode="r")
flagged, removed = suspect_mask(offsets, neighbors), repair_link_mask(offsets, neighbors)


def entry(u, v):
    """Position of v in u's row of the CSR, where the link masks are indexed."""
    hit = np.flatnonzero(np.asarray(neighbors[offsets[u]:offsets[u + 1]]) == v)
    assert len(hit) == 1, (u, v)
    return int(offsets[u] + hit[0])


def shared(u, v):
    """Works numbered after the n researchers in the author-work graph, as in B2c."""
    both = np.intersect1d(np.asarray(a_nb[a_off[u]:a_off[u + 1]]), np.asarray(a_nb[a_off[v]:a_off[v + 1]]))
    return int((np.asarray(w_size[both - n]) <= MAX_TEAM).sum())


# the path drawn in Fig. 3b: from one end, at each step the linked record with the most coauthors (figures.far_edge)
following = {}
for u, v in d["links"]:
    following.setdefault(u, []).append(v)
path = [d["pair"][0]]
while path[-1] != d["pair"][1]:
    path.append(max((v for v in following[path[-1]] if v not in d["placeholder"]), key=lambda v: nodes[v]["coauthors"]))
on_path = set(zip(path, path[1:]))
rows = []
for u, v in d["links"]:
    e = entry(u, v)
    rows.append({"u": u, "v": v, "step": nodes[u]["step"], "shared_works": shared(u, v), "flagged": bool(flagged[e]),
                 "repair_removed": bool(removed[e]), "drawn_path": (u, v) in on_path})
assert sum(r["drawn_path"] for r in rows) == len(path) - 1 == d["distance"]


def summary(part):
    works = np.array([r["shared_works"] for r in part])
    assert works.min() >= 1, "a link without a shared qualifying work"
    return {"links": len(part), "shared_works_median": float(np.median(works)), "shared_works_max": int(works.max()),
            "one_shared_work": int((works == 1).sum()), "flagged": sum(r["flagged"] for r in part),
            "repair_removed": sum(r["repair_removed"] for r in part)}


coauthors = np.array([nodes[u]["coauthors"] for u in path])
out = {"drawn_path": summary([r for r in rows if r["drawn_path"]]), "corridor": summary(rows),
       "drawn_path_researchers": {"count": len(path), "with_at_most_two_coauthors": int((coauthors <= 2).sum()),
                                  "coauthors_median": float(np.median(coauthors))},
       "links": rows}
save_result("a3c_far_edge_links", out)
log({k: v for k, v in out.items() if k != "links"})
