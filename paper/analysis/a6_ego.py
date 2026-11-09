"""A6a: merged profiles from ego-network fragmentation.

A real researcher's coauthors also write with each other; a profile that merges
several people with one name has coauthors that fall into separate groups, one
per hidden person, with no links between them. For every author with at least
DMIN coauthors we take the ego network without the ego, find its connected
components, and record the shares of the largest and second-largest group and
the number of sizeable groups. Name commonness (identity_signals.key_n: author
records sharing "first initial + surname") is the outside check: fragmentation
that tracks name commonness is what merges look like.
"""
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numba import njit, prange

from common import CACHE, PG, con, load_csr, log, save_result

DMIN = int(sys.argv[1]) if len(sys.argv) > 1 else 300


@njit(cache=True)
def _find(p, x):
    r = x
    while p[r] != r:
        r = p[r]
    while p[x] != r:
        nx = p[x]
        p[x] = r
        x = nx
    return r


@njit(parallel=True, cache=True)
def fragmentation(offsets, neighbors, egos, nchunks):
    n = offsets.shape[0] - 1
    m = egos.shape[0]
    out = np.zeros((m, 5), np.int64)    # degree, components, largest, second, groups >= max(5, 1% of degree)
    per = (m + nchunks - 1) // nchunks
    for c in prange(nchunks):
        stamp = np.full(n, -1, np.int64)
        loc = np.zeros(n, np.int32)
        for k in range(c * per, min(m, (c + 1) * per)):
            e = egos[k]
            lo, hi = offsets[e], offsets[e + 1]
            d = hi - lo
            for t in range(d):
                x = neighbors[lo + t]
                stamp[x] = k
                loc[x] = t
            p = np.arange(d)
            for t in range(d):
                x = neighbors[lo + t]
                for j in range(offsets[x], offsets[x + 1]):
                    y = neighbors[j]
                    if stamp[y] == k:
                        a = _find(p, t)
                        b = _find(p, loc[y])
                        if a != b:
                            p[max(a, b)] = min(a, b)
            size = np.zeros(d, np.int64)
            for t in range(d):
                size[_find(p, t)] += 1
            comps = 0
            first = 0
            second = 0
            big = 0
            thr = max(5, d // 100)
            for t in range(d):
                s = size[t]
                if s > 0:
                    comps += 1
                    if s >= thr:
                        big += 1
                    if s > first:
                        second = first
                        first = s
                    elif s > second:
                        second = s
            out[k, 0] = d
            out[k, 1] = comps
            out[k, 2] = first
            out[k, 3] = second
            out[k, 4] = big
    return out


offsets, neighbors = load_csr(mmap=False)
deg = np.diff(offsets)
egos = np.flatnonzero(deg >= DMIN).astype(np.int64)
log(f"{len(egos):,} authors with >= {DMIN} coauthors")
res = fragmentation(offsets, neighbors, egos, 56)
log("done")
key_n = pq.read_table(PG / "identity_signals.parquet", columns=["node_idx", "key_n"]).to_pandas().set_index("node_idx")["key_n"]
tab = pa.table({"node_idx": egos.astype(np.int32), "degree": res[:, 0], "components": res[:, 1], "largest": res[:, 2],
                "second": res[:, 3], "big_groups": res[:, 4], "key_n": key_n.reindex(egos).to_numpy()})
pq.write_table(tab, CACHE / f"ego_fragmentation_{DMIN}.parquet")
df = tab.to_pandas()
df["second_share"] = df["second"] / df["degree"]
df["largest_share"] = df["largest"] / df["degree"]
df["common"] = df["key_n"] >= 200
summary = {"DMIN": DMIN, "egos": int(len(df))}
for lab, sel in (("fragmented (second group >= 10% of coauthors)", df.second_share >= 0.10),
                 ("cohesive (largest group >= 90%)", df.largest_share >= 0.90)):
    summary[lab] = {"count": int(sel.sum()), "share_common_name": float(df.loc[sel, "common"].mean())}
summary["baseline_share_common_name"] = float(df["common"].mean())
bins = [0, 0.02, 0.05, 0.10, 0.20, 0.35, 1.01]
df["bin"] = np.digitize(df["second_share"], bins)
summary["common_name_share_by_second_group_share"] = {
    f"{bins[b - 1]:.2f}-{bins[b]:.2f}": {"n": int((df.bin == b).sum()), "common": float(df.loc[df.bin == b, "common"].mean())}
    for b in range(1, len(bins)) if (df.bin == b).any()}
save_result(f"a6_ego_{DMIN}", summary)
log(summary)
c = con()
c.register("f", tab)
print(c.execute(f"""SELECT a.name, a.country, f.degree, round(f.largest / f.degree, 3) AS largest, round(f.second / f.degree, 3) AS second,
                           f.big_groups, f.key_n FROM f JOIN read_parquet('{CACHE / 'node_attrs.parquet'}') a USING (node_idx)
                    ORDER BY f.degree DESC LIMIT 25""").fetchdf().to_string())
