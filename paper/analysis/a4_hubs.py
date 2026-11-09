"""A4: hubs and brokers -- sampled shortest-path betweenness, closeness and participation.

Betweenness is estimated with Brandes' dependency accumulation from 2,000 random sources in two independent halves
of 1,000, on the network as delivered. Each half gives an unbiased estimate (up to the factor n/1,000), but a
researcher next to a sampled source collects that source's whole reach at once, so one source can lift anyone to
the top. We keep the smaller of the two estimates: a researcher ranks high only if both halves agree. Closeness is
the mean distance to the 1,000 A2 sources. Participation coefficient P = 1 - sum_c (k_c / k)^2 over the
communities of an author's coauthors (Guimera & Amaral): near 0 for authors whose coauthors sit in one community,
higher for those linking communities.
Writes cache/hubs.parquet and cache/closeness_1000.npy (all nodes) and results/a4_hubs.json.
"""
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numba import njit, prange

from common import CACHE, con, load_csr, log, save_result, suspect_mask
from kernels import distance_sums

K = int(sys.argv[1]) if len(sys.argv) > 1 else 2000     # two halves
CHUNKS = 14


@njit(parallel=True, cache=True)
def betweenness(offsets, neighbors, skip, use_skip, sources, nchunks):
    n = offsets.shape[0] - 1
    k = sources.shape[0]
    acc = np.zeros((nchunks, n), np.float64)
    per = (k + nchunks - 1) // nchunks
    for c in prange(nchunks):
        dist = np.empty(n, np.int16)
        sigma = np.empty(n, np.float64)
        delta = np.empty(n, np.float64)
        order = np.empty(n, np.int32)
        for t in range(c * per, min(k, (c + 1) * per)):
            s = sources[t]
            for i in range(n):
                dist[i] = -1
                sigma[i] = 0.0
                delta[i] = 0.0
            dist[s] = 0
            sigma[s] = 1.0
            order[0] = s
            head, tail = 0, 1
            while head < tail:
                u = order[head]
                head += 1
                du = dist[u] + 1
                for j in range(offsets[u], offsets[u + 1]):
                    if use_skip and skip[j]:
                        continue
                    v = neighbors[j]
                    if dist[v] < 0:
                        dist[v] = du
                        order[tail] = v
                        tail += 1
                    if dist[v] == du:
                        sigma[v] += sigma[u]
            for idx in range(tail - 1, 0, -1):
                w = order[idx]
                coeff = (1.0 + delta[w]) / sigma[w]
                dw = dist[w] - 1
                for j in range(offsets[w], offsets[w + 1]):
                    if use_skip and skip[j]:
                        continue
                    v = neighbors[j]
                    if dist[v] == dw:
                        delta[v] += sigma[v] * coeff
                acc[c, w] += delta[w]
    return acc.sum(axis=0)


@njit(parallel=True, cache=True)
def participation(offsets, neighbors, comm):
    n = offsets.shape[0] - 1
    out = np.zeros(n, np.float64)
    for u in prange(n):
        lo, hi = offsets[u], offsets[u + 1]
        d = hi - lo
        if d == 0:
            continue
        cs = np.empty(d, np.int32)
        for t in range(d):
            cs[t] = comm[neighbors[lo + t]]
        cs.sort()
        s = 0.0
        run = 1
        for t in range(1, d + 1):
            if t < d and cs[t] == cs[t - 1]:
                run += 1
            else:
                s += (run / d) ** 2
                run = 1
        out[u] = 1.0 - s
    return out


offsets, neighbors = load_csr(mmap=False)
n = len(offsets) - 1
skip = np.asarray(suspect_mask(offsets, neighbors))
lcc = np.load(CACHE / "lcc_all.npy") & np.load(CACHE / "lcc_clean.npy")
rng = np.random.default_rng(42)
sources = rng.choice(np.flatnonzero(lcc), K, replace=False).astype(np.int32)
half = K // 2
cols = {}
for tag, src in (("a", sources[:half]), ("b", sources[half:])):
    log(f"betweenness, half {tag}: {len(src)} sources")
    cols[f"btw_{tag}"] = betweenness(offsets, neighbors, skip, False, src, CHUNKS) * (n / len(src))
cols["btw"] = np.minimum(cols["btw_a"], cols["btw_b"])
log("closeness from the A2 sources")
a2 = np.load(CACHE / "a2_sources.npy")
sums, cnt = distance_sums(offsets, neighbors, a2, CHUNKS)
np.save(CACHE / "closeness_1000.npy", np.where(cnt == len(a2), sums / len(a2), np.inf))
c = con()
comm = c.execute(f"SELECT community FROM read_parquet('{CACHE / 'node_attrs.parquet'}') ORDER BY node_idx").fetchnumpy()["community"].astype(np.int32)
log("participation")
cols["participation"] = participation(offsets, neighbors, comm)
tab = pa.table({"node_idx": np.arange(n, dtype=np.int32), **cols})
pq.write_table(tab, CACHE / "hubs.parquet")
c.register("h", tab)
q = f"""SELECT a.node_idx, a.name, a.country, a.field, a.cited, a.degree, a.flagged, h.btw, h.btw_a, h.btw_b, h.participation
        FROM h JOIN read_parquet('{CACHE / 'node_attrs.parquet'}') a USING (node_idx)"""
top = c.execute(q + " ORDER BY btw DESC LIMIT 1000").fetchdf()
ranks = c.execute("""SELECT corr(ra, rb) FROM (SELECT rank() OVER (ORDER BY btw_a) ra, rank() OVER (ORDER BY btw_b) rb
                     FROM h WHERE btw_a > 0 OR btw_b > 0)""").fetchone()[0]
top_a = set(c.execute("SELECT node_idx FROM h ORDER BY btw_a DESC LIMIT 1000").fetchdf().node_idx)
top_b = set(c.execute("SELECT node_idx FROM h ORDER BY btw_b DESC LIMIT 1000").fetchdf().node_idx)
out = {"K": K, "halves_spearman": float(ranks), "halves_top1000_overlap": len(top_a & top_b),
       "top": top.head(40).to_dict("records")}
top.to_csv(CACHE / "top_brokers.csv", index=False)
save_result("a4_hubs", out)
log({k: v for k, v in out.items() if k != "top"})
print(top.head(25).to_string())
