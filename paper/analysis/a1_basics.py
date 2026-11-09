"""A1: network basics -- size, degrees, components, flagged links, isolates."""
import numpy as np

from numba import njit

from common import (CACHE, NO_SKIP, components, load_csr, log, save_result, suspect_mask)


@njit(cache=True)
def per_node_count(offsets, mask):
    n = offsets.shape[0] - 1
    out = np.zeros(n, np.int32)
    for u in range(n):
        c = 0
        for j in range(offsets[u], offsets[u + 1]):
            if mask[j]:
                c += 1
        out[u] = c
    return out

offsets, neighbors = load_csr(mmap=False)
n, m2 = len(offsets) - 1, len(neighbors)
deg = np.diff(offsets)
log(f"{n:,} nodes, {m2 // 2:,} links")

mask = np.asarray(suspect_mask(offsets, neighbors))
flagged = int(mask.sum()) // 2

out = {"nodes": n, "links": m2 // 2, "flagged_links": flagged, "flagged_share": flagged / (m2 // 2)}
q = np.percentile(deg, [50, 90, 99, 99.9])
out["degree"] = {"mean": float(deg.mean()), "median": float(q[0]), "p90": float(q[1]), "p99": float(q[2]),
                 "p999": float(q[3]), "max": int(deg.max()), "isolated": int((deg == 0).sum())}
# authors whose links are mostly flagged: the candidate merged profiles
flag_per_node = per_node_count(offsets, mask)
np.save(CACHE / "flagged_per_node.npy", flag_per_node)
share = flag_per_node / np.maximum(deg, 1)
out["nodes_with_flagged_links"] = int((flag_per_node > 0).sum())
out["nodes_majority_flagged"] = int((share > 0.5).sum())

for tag, skip in (("all", NO_SKIP), ("clean", mask)):
    log(f"components ({tag})")
    lab = components(offsets, neighbors, skip)
    labs, sizes = np.unique(lab, return_counts=True)
    big = labs[np.argmax(sizes)]
    np.save(CACHE / f"lcc_{tag}.npy", (lab == big))
    order = np.sort(sizes)[::-1]
    out[f"components_{tag}"] = {"count": int(len(sizes)), "largest": int(order[0]), "largest_share": float(order[0] / n),
                                 "second": int(order[1]) if len(order) > 1 else 0,
                                 "singletons": int((sizes == 1).sum()),
                                 "size_hist_top": [int(s) for s in order[:10]]}
    log(out[f"components_{tag}"])
save_result("a1_basics", out)
log(out)
