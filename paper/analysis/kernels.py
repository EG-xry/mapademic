"""Shared graph kernels for the repair experiments (A8, A8b): union-find, CSR contraction, BFS distance
histograms, distance sums, exact mean distances and double sweeps. Plain-CSR (no skip mask): contract first."""
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
from numba import njit, prange

from identities import file_identity, locate, same_identity


@njit(cache=True)
def find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@njit(cache=True)
def union_pairs(n, a, b):
    parent = np.arange(n)
    for k in range(a.shape[0]):
        ra, rb = find(parent, a[k]), find(parent, b[k])
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    for x in range(n):
        parent[x] = find(parent, x)
    return parent


@njit(cache=True)
def contract(offsets, neighbors, skip, rep, m):
    """CSR of the network with nodes relabelled by rep (-1 = removed), skipped links dropped, duplicates merged."""
    n = offsets.shape[0] - 1
    start = np.zeros(m + 1, np.int64)
    for u in range(n):
        if rep[u] >= 0:
            start[rep[u] + 1] += 1
    start = np.cumsum(start)
    pos = start[:-1].copy()
    order = np.empty(start[-1], np.int64)
    for u in range(n):
        if rep[u] >= 0:
            order[pos[rep[u]]] = u
            pos[rep[u]] += 1
    mark = np.full(m, -1, np.int64)
    off = np.zeros(m + 1, np.int64)
    for r in range(m):
        k = 0
        for t in range(start[r], start[r + 1]):
            u = order[t]
            for j in range(offsets[u], offsets[u + 1]):
                v = rep[neighbors[j]]
                if not skip[j] and v >= 0 and v != r and mark[v] != r:
                    mark[v] = r
                    k += 1
        off[r + 1] = off[r] + k
    nb = np.empty(off[-1], np.int32)
    mark[:] = -1
    for r in range(m):
        k = off[r]
        for t in range(start[r], start[r + 1]):
            u = order[t]
            for j in range(offsets[u], offsets[u + 1]):
                v = rep[neighbors[j]]
                if not skip[j] and v >= 0 and v != r and mark[v] != r:
                    mark[v] = r
                    nb[k] = v
                    k += 1
    return off, nb


@njit(cache=True)
def _bfs(offsets, neighbors, src, dist, queue):
    n = offsets.shape[0] - 1
    for i in range(n):
        dist[i] = -1
    dist[src] = 0
    queue[0] = src
    head, tail = 0, 1
    while head < tail:
        u = queue[head]
        head += 1
        for j in range(offsets[u], offsets[u + 1]):
            v = neighbors[j]
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                queue[tail] = v
                tail += 1
    return tail


@njit(parallel=True, cache=True)
def histograms(offsets, neighbors, sources, maxd):
    n = offsets.shape[0] - 1
    h = np.zeros((sources.shape[0], maxd + 1), np.int64)
    for s in prange(sources.shape[0]):
        d = np.empty(n, np.int16)
        q = np.empty(n, np.int32)
        _bfs(offsets, neighbors, sources[s], d, q)
        for i in range(n):
            if d[i] > 0:
                h[s, min(d[i], maxd)] += 1
    return h


@njit(parallel=True, cache=True)
def distance_sums(offsets, neighbors, sources, nchunks):
    n = offsets.shape[0] - 1
    acc = np.zeros((nchunks, n), np.int32)
    cnt = np.zeros((nchunks, n), np.int32)
    per = (sources.shape[0] + nchunks - 1) // nchunks
    for c in prange(nchunks):
        d = np.empty(n, np.int16)
        q = np.empty(n, np.int32)
        for t in range(c * per, min(sources.shape[0], (c + 1) * per)):
            _bfs(offsets, neighbors, sources[t], d, q)
            for i in range(n):
                if d[i] >= 0:
                    acc[c, i] += d[i]
                    cnt[c, i] += 1
    return acc.sum(axis=0), cnt.sum(axis=0)


@njit(parallel=True, cache=True)
def histograms_distance_sums(offsets, neighbors, sources, maxd, nchunks):
    """One BFS per source for histograms and center estimates; self counts only in sums/counts.

    Integer accumulation is identical to calling histograms and distance_sums separately.
    The source order determines histogram rows; each thread owns its node accumulators.
    """
    n = offsets.shape[0] - 1
    h = np.zeros((sources.shape[0], maxd + 1), np.int64)
    acc = np.zeros((nchunks, n), np.int32)
    cnt = np.zeros((nchunks, n), np.int32)
    per = (sources.shape[0] + nchunks - 1) // nchunks
    for c in prange(nchunks):
        d = np.empty(n, np.int16)
        q = np.empty(n, np.int32)
        for t in range(c * per, min(sources.shape[0], (c + 1) * per)):
            _bfs(offsets, neighbors, sources[t], d, q)
            for i in range(n):
                if d[i] >= 0:
                    acc[c, i] += d[i]
                    cnt[c, i] += 1
                    if d[i] > 0:
                        h[t, min(d[i], maxd)] += 1
    return h, acc.sum(axis=0), cnt.sum(axis=0)


def checkpointed_distance_summaries(offsets, neighbors, sources, maxd, nchunks, path, identity,
                                   batch_size=64, progress=None):
    """Atomically checkpoint fused results after each source batch, with graph/source verification.

    Only one cumulative checkpoint is kept (two int64 node arrays plus source histograms).
    Graph identity is supplied by the caller and must include both repair and skipped links.
    """
    if batch_size < 1 or nchunks < 1:
        raise ValueError("batch_size and nchunks must be positive")
    path = Path(path)
    stamp = json.dumps({"version": 1, "identity": identity, "maxd": maxd,
                        "sources": hashlib.sha256(np.ascontiguousarray(sources)).hexdigest()}, sort_keys=True)
    n = len(offsets) - 1
    h = np.zeros((len(sources), maxd + 1), np.int64)
    sums, counts = np.zeros(n, np.int64), np.zeros(n, np.int64)
    done = 0
    if path.exists():
        with np.load(path, allow_pickle=False) as previous:
            if str(previous["signature"]) == stamp:
                saved_hist = previous["histograms"]
                sums, counts = previous["sums"], previous["counts"]
                done = len(saved_hist)
                if done > len(sources) or sums.shape != (n,) or counts.shape != (n,):
                    raise ValueError(f"Invalid distance checkpoint shapes: {path}")
                if saved_hist.shape[1] != maxd + 1:
                    raise ValueError(f"Invalid histogram width: {path}")
                h[:done] = saved_hist
    if progress:
        progress(f"fused repaired distances: {done}/{len(sources)} sources cached")
    for start in range(done, len(sources), batch_size):
        end = min(start + batch_size, len(sources))
        hh, ss, cc = histograms_distance_sums(offsets, neighbors, sources[start:end], maxd, nchunks)
        h[start:end] = hh
        sums += ss
        counts += cc
        del hh, ss, cc
        temporary = path.with_name(path.stem + ".tmp.npz")
        np.savez(temporary, signature=np.array(stamp), histograms=h[:end], sums=sums, counts=counts)
        temporary.replace(path)
        if progress:
            progress(f"fused repaired distances: {end}/{len(sources)} sources complete")
    return h, sums, counts


@njit(parallel=True, cache=True)
def exact_mean(offsets, neighbors, nodes):
    n = offsets.shape[0] - 1
    out = np.zeros(nodes.shape[0], np.float64)
    for k in prange(nodes.shape[0]):
        d = np.empty(n, np.int16)
        q = np.empty(n, np.int32)
        reached = _bfs(offsets, neighbors, nodes[k], d, q)
        tot = 0
        for i in range(n):
            if d[i] > 0:
                tot += d[i]
        out[k] = tot / max(reached - 1, 1)
    return out


@njit(cache=True)
def highest_degree_members(rep, by_degree, count):
    """First original member in the supplied degree order, without a Python dict."""
    first = np.full(count, -1, np.int64)
    for u in by_degree:
        r = rep[u]
        if r >= 0 and first[r] < 0:
            first[r] = u
    return first


def summary(h):
    t = h.sum(axis=0).astype(float)
    cdf = np.cumsum(t) / t.sum()
    return {"pairs": int(t.sum()), "mean": float((np.arange(len(t)) * t).sum() / t.sum()), "median": int(np.searchsorted(cdf, 0.5)),
            "p90": int(np.searchsorted(cdf, 0.9)), "share_le_3": float(t[:4].sum() / t.sum()), "share_ge_6": float(t[6:].sum() / t.sum()),
            "pmf": {str(k): float(t[k] / t.sum()) for k in range(1, len(t)) if t[k] > 0}}


def reuse_a2_summary(results, cache, pg, sources, maxd=64, progress=None):
    """Return a verified A2c cap-50 baseline, or None to require fresh BFS.

    Reuse requires the completed A2c full-source result, its checkpoint
    signature, exact source ordering, a self-consistent/current input manifest,
    and integer aggregate parity with A2. The returned summary is calculated
    from saved per-source counts, with self distances removed. No-A2c/min15 or
    incomplete/inconsistent caches safely fall back; timestamps alone cannot
    authorize reuse. CSR chronology remains an additional stale-file guard.
    """
    results, cache, pg = Path(results), Path(cache), Path(pg)
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    def json_digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    try:
        a2_path = results / "a2_distances.json"
        a2 = json.loads(a2_path.read_text())
        a2c = json.loads((results / "a2c_team_cap.json").read_text())
        manifest = json.loads((cache / "bipartite_manifest.json").read_text())
        sources = np.asarray(sources)
        require(sources.ndim == 1 and len(sources) > 0, "invalid requested source array")
        require(np.array_equal(np.load(cache / "a2_sources.npy"), sources), "requested and saved A2 sources differ")
        require(a2["K"] == a2c["sources"] == len(sources), "full-source result counts differ")
        validation = a2c["cap50_validation"]
        require(validation["full_histogram_exact"] is True and validation["reachable_count_exact"] is True
                and validation["fixed_targets_exact"] is True,
                "A2c has no completed exact cap-50 validation")
        require(validation["sources_checked"] == validation["a2_sources"] == len(sources),
                "A2c validated a different number of sources")
        fingerprint = manifest["fingerprint"]
        require(fingerprint == a2c["network_fingerprint"] == json_digest(manifest["identity"]),
                "A2c result and input-manifest fingerprints differ")
        require(same_identity(manifest["identity"]["ids"], file_identity(pg / "ids.npy")), "node-order file identity changed")
        for expected in manifest["identity"]["inputs"]:
            require(same_identity(expected, file_identity(locate(expected[0]))), "bipartite work input changed")
        # Same normalization as A2c.build; hashing slices avoids another full ID copy.
        ids = np.load(pg / "ids.npy", mmap_mode="r")
        digest = hashlib.sha256()
        for begin in range(0, len(ids), 1_000_000):
            digest.update(np.ascontiguousarray(ids[begin:begin + 1_000_000], dtype=np.int64))
        require(digest.hexdigest() == manifest["identity"]["ids_sha256"], "node-order contents changed")
        require(len(ids) == a2c["researchers"] == manifest["researchers"], "researcher populations differ")
        require(np.all((sources >= 0) & (sources < len(ids))), "source outside researcher population")
        target_mask = np.load(cache / "lcc_all.npy", mmap_mode="r")
        require(target_mask.dtype == np.bool_ and target_mask.shape == (len(ids),),
                "original largest-component mask is malformed")
        require(np.all(target_mask[sources]), "source outside original largest component")
        target_digest = hashlib.sha256(target_mask).hexdigest()
        target_records = int(target_mask.sum())
        targets = a2c["baseline_targets"]
        require(a2c["distance_kernel_version"] == 2 and targets["sha256"] == target_digest
                and targets["records"] == target_records and targets["source_self_excluded_from_summaries"] is True,
                "A2c target population or checkpoint version differs")
        written = a2_path.stat().st_mtime_ns
        require(all((pg / name).stat().st_mtime_ns <= written
                    for name in ("offsets.npy", "neighbors.npy", "ids.npy")), "CSR files postdate A2")
        # A2c checkpoint schema version 2; reject future schemas until reviewed.
        expected_signature = json_digest({"graph": fingerprint, "sources": sources.tolist(),
                                          "cap": 50, "maxd": maxd, "kernel_version": 2,
                                          "target_mask_sha256": target_digest})
        with np.load(cache / "a2c_hist_cap50.npz", allow_pickle=False) as saved:
            require(str(saved["signature"]) == expected_signature, "cap-50 checkpoint signature differs")
            require(np.array_equal(saved["sources"], sources), "cap-50 checkpoint source order differs")
            require(int(saved["kernel_version"]) == 2 and str(saved["target_mask_sha256"]) == target_digest,
                    "cap-50 checkpoint target population or version differs")
            hist = saved["hist"].copy()
            require(np.array_equal(saved["hist_fixed"], hist), "cap-50 fixed-target and all-reachable counts differ")
        require(hist.shape == (len(sources), maxd + 1) and np.issubdtype(hist.dtype, np.integer),
                "cap-50 checkpoint histogram is incomplete or malformed")
        require(np.all(hist >= 0) and np.all(hist[:, 0] == 1) and not np.any(hist[:, -1]),
                "invalid or truncated cap-50 distance counts")
        original = a2["all_links"]
        pairs = original["pairs"]
        require(pairs > 0 and pairs % len(sources) == 0, "A2 pair count is inconsistent with its giant-component sources")
        require(target_records == pairs // len(sources) + 1
                and np.all(hist.sum(axis=1) == target_records), "per-source reachable populations differ")
        hist[:, 0] = 0
        expected = np.zeros(maxd + 1, np.int64)
        for distance, probability in original["pmf"].items():
            distance = int(distance)
            require(0 < distance < maxd and np.isfinite(probability) and probability >= 0,
                    "invalid A2 distance distribution")
            expected[distance] = round(probability * pairs)
        require(np.array_equal(hist.sum(axis=0), expected), "cap-50 aggregate histogram differs from A2")
        baseline = summary(hist)
        require(baseline["pairs"] == pairs == a2c["cap_50"]["pairs"], "baseline pair totals differ")
        require(abs(baseline["mean"] - original["mean"]) < 1e-12 and
                abs(baseline["mean"] - a2c["cap_50"]["mean"]) < 1e-12 and
                baseline["median"] == original["median"] and
                baseline["p90"] == original["p90_effective_diameter"], "baseline summary statistics differ")
    except (OSError, EOFError, zipfile.BadZipFile, KeyError, TypeError, ValueError, IndexError, OverflowError) as error:
        if progress:
            progress(f"delivered baseline: verified A2c reuse unavailable ({error}); fresh BFS required")
        return None
    if progress:
        progress(f"delivered baseline: reused {len(sources)} exact-source cap-50 histograms; checkpoint signature, "
                 "input fingerprint, node IDs and A2 aggregate counts verified")
    return baseline


def double_sweeps(offsets, neighbors, starts):
    n = len(offsets) - 1
    d, q = np.empty(n, np.int16), np.empty(n, np.int32)
    best = (0, -1, -1)
    for s in starts:
        tail = _bfs(offsets, neighbors, int(s), d, q)
        a = int(q[tail - 1])
        tail = _bfs(offsets, neighbors, a, d, q)
        b = int(q[tail - 1])
        if d[b] > best[0]:
            best = (int(d[b]), a, b)
    return best


@njit(parallel=True, cache=True)
def group_distances(offsets, neighbors, sources, group, ngroups):
    """Per source: sum and count of distances to the members of each group (group -1 skipped)."""
    n = offsets.shape[0] - 1
    k = sources.shape[0]
    s = np.zeros((k, ngroups), np.float64)
    c = np.zeros((k, ngroups), np.int64)
    for t in prange(k):
        d = np.empty(n, np.int16)
        q = np.empty(n, np.int32)
        _bfs(offsets, neighbors, sources[t], d, q)
        for v in range(n):
            g = group[v]
            if d[v] > 0 and g >= 0:
                s[t, g] += d[v]
                c[t, g] += 1
    return s, c
