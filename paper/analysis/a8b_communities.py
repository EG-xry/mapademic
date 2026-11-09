"""A8b: do the script communities rejoin when twins are merged?

For each community A8 follows: the share of its members merged into a node led by another community's member
(the highest-degree member names a merged node), and the share of its members' links that stay inside it, on
the network as delivered, with removals only (flagged links, merged and split hubs, non-persons) and repaired
(removals plus twins and splits merged, A8's cache/repair_rep.npy). Mean distance from the same 100 members as
A8 on the removals-only network, which A8 does not run.
"""
import numpy as np
import pyarrow.parquet as pq
from numba import njit, prange

from common import CACHE, SCRIPT_COMMUNITIES, load_csr, log, save_result, repair_link_mask
from kernels import contract, highest_degree_members, histograms
from repair_metrics import guarded_summary


@njit(parallel=True, cache=True)
def inside_share(offsets, neighbors, label, c):
    """Share of the links of nodes labelled c that lead to nodes labelled c."""
    n = offsets.shape[0] - 1
    ins = np.zeros(n, np.int64)
    tot = np.zeros(n, np.int64)
    for u in prange(n):
        if label[u] == c:
            for j in range(offsets[u], offsets[u + 1]):
                tot[u] += 1
                if label[neighbors[j]] == c:
                    ins[u] += 1
    return ins.sum() / max(tot.sum(), 1)


def main():
    offsets, neighbors = load_csr(mmap=False)
    deg = np.diff(offsets)
    skip = np.asarray(repair_link_mask(offsets, neighbors))
    comm = pq.read_table(CACHE / "node_attrs.parquet", columns=["community"]).column(0).to_numpy()
    rep = np.load(CACHE / "repair_rep.npy")
    keep = rep >= 0
    m = int(rep.max()) + 1
    rep_rm = np.where(keep, np.cumsum(keep) - 1, -1).astype(np.int64)
    lab2 = comm[highest_degree_members(rep, np.argsort(-deg, kind="stable"), m)]
    rng = np.random.default_rng(20261024)  # Draw every community in the same order as A8.
    out, samples = {}, {}
    for label, c in SCRIPT_COMMUNITIES.items():
        members = np.flatnonzero((comm == c) & keep & (deg > 0))
        samples[label] = rng.choice(members, size=min(100, len(members)), replace=False)
        out[label] = {"members": int(len(members)),
                      "absorbed_by_other_community": float(np.mean(lab2[rep[members]] != c)),
                      "inside_share": {"delivered": float(inside_share(offsets, neighbors, comm, c))}}
    del deg

    # Finish removals-only statistics before allocating the repaired adjacency.
    # Only the original and one transformed CSR coexist at any point.
    off_rm, nb_rm = contract(offsets, neighbors, skip, rep_rm, int(keep.sum()))
    log(f"removals only: {keep.sum():,} nodes, {len(nb_rm) // 2:,} links")
    lab_rm = comm[keep]
    for label, c in SCRIPT_COMMUNITIES.items():
        out[label]["inside_share"]["removals_only"] = float(inside_share(off_rm, nb_rm, lab_rm, c))
        out[label]["mean_distance_removals_only"] = guarded_summary(
            histograms(off_rm, nb_rm, rep_rm[samples[label]], 64), f"{label} removals only")["mean"]
    del off_rm, nb_rm, lab_rm, rep_rm, samples, comm, keep

    off2, nb2 = contract(offsets, neighbors, skip, rep, m)
    del offsets, neighbors, skip, rep
    log(f"repaired: {m:,} nodes, {len(nb2) // 2:,} links")
    for label, c in SCRIPT_COMMUNITIES.items():
        out[label]["inside_share"]["repaired"] = float(inside_share(off2, nb2, lab2, c))
        log(label, out[label])
    save_result("a8b_communities", out)


if __name__ == "__main__":
    main()
