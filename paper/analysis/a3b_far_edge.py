"""A3b: the corridor between the two researchers farthest apart (A3), for the far-edge figure.

Breadth-first searches from both ends of the 28-step pair: a researcher is on some shortest path between them when
its two distances add up to 28. Records the corridor (researchers by step, the links between consecutive steps, map
coordinates) and checks that the pair stays 28 steps apart without the placeholder record on one of the paths
("مؤطر", Arabic for "supervisor", credited on theses).
"""
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numba import njit

from common import CACHE, D, RESULTS, load_csr, log, save_result

PLACEHOLDER = "مؤطر"


@njit(cache=True)
def bfs(offsets, neighbors, src, dist, queue, banned):
    for i in range(dist.shape[0]):
        dist[i] = -1
    dist[src] = 0
    queue[0] = src
    head, tail = 0, 1
    while head < tail:
        u = queue[head]
        head += 1
        for j in range(offsets[u], offsets[u + 1]):
            v = neighbors[j]
            if v != banned and dist[v] < 0:
                dist[v] = dist[u] + 1
                queue[tail] = v
                tail += 1


far = json.loads((RESULTS / "a3_farthest.json").read_text())["all_links"]
a, b = far["pair_idx"]
offsets, neighbors = load_csr(mmap=False)
n = offsets.shape[0] - 1
q = np.empty(n, np.int32)
da, db = np.empty(n, np.int16), np.empty(n, np.int16)
bfs(offsets, neighbors, a, da, q, -1)
bfs(offsets, neighbors, b, db, q, -1)
dist = int(da[b])
on = np.flatnonzero((da >= 0) & (db >= 0) & (da + db == dist))
width = np.bincount(da[on], minlength=dist + 1)
log(f"pair {a}-{b}: {dist} steps; corridor of {len(on)} researchers, width by step {width.tolist()}")

attrs = pq.read_table(CACHE / "node_attrs.parquet", columns=["node_idx", "id", "name", "works_kept", "cited", "degree",
                                                             "community", "country", "institution"]).to_pandas()
xy = pq.read_table(D / "coords_web.parquet", columns=["id", "xw", "yw"]).to_pandas().set_index("id")
c = attrs.set_index("node_idx").loc[on].join(xy, on="id")
c["step"] = da[on]
placeholder = [int(i) for i in c.index[c.name == PLACEHOLDER]]
banned = placeholder[0] if placeholder else -1
ea, eb = np.empty(n, np.int16), np.empty(n, np.int16)
bfs(offsets, neighbors, a, ea, q, banned)
bfs(offsets, neighbors, b, eb, q, banned)
log(f"without the placeholder {placeholder}: {int(ea[b])} steps; eccentricities {int(ea.max())} and {int(eb.max())}")
inside = set(on.tolist())
links = [(int(u), int(v)) for u in on for v in neighbors[offsets[u]:offsets[u + 1]]
         if int(v) in inside and da[v] == da[u] + 1]
save_result("a3b_far_edge", {
    "pair": [int(a), int(b)], "distance": dist, "width_by_step": width.tolist(), "placeholder": placeholder,
    "distance_without_placeholder": int(ea[b]), "eccentricity_without_placeholder": [int(ea.max()), int(eb.max())],
    "nodes": [{"node": int(i), "step": int(r.step), "name": r["name"], "works": int(r.works_kept), "cited": int(r.cited),
               "coauthors": int(r.degree), "community": int(r.community),
               "country": None if pd.isna(r.country) else r.country,
               "institution": None if pd.isna(r.institution) else r.institution, "xw": float(r.xw), "yw": float(r.yw)}
              for i, r in c.sort_values(["step", "degree"], ascending=[True, False]).iterrows()],
    "links": links})
