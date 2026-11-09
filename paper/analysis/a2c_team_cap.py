"""A2c: do works with more than 50 authors change the distances?

The network leaves out works with more than 50 authors. Their effect on distances needs no pairs:
in the bipartite graph of researchers and works (a researcher
joined to each of their works), two researchers are d coauthor steps apart exactly when they are 2d steps apart,
so a cap on team size becomes a mask on work nodes. From A2's first SOURCES sources, the distance to every
researcher with works of up to 50 authors (which must reproduce A2), up to 100, up to 500, and of any size.
Works count when they pass the network's quality filter and have at least two of its researchers. The bipartite
graph is cached in cache/bipartite_*.npy.

Builds stream 50,000 work rows at a time into disk-backed arrays, committing each input file in
cache/bipartite_build/progress.json. An interrupted file is replayed from its last committed boundary.
The final manifest fingerprints the source files, node ordering and quality filter. Distance histograms
checkpoint each small source batch; a 1,000-source run also writes its first-100-source summary without
repeating searches. The cap-50 histogram must match A2 exactly before the other caps are accepted.

Each traversal also counts distances restricted to the original cap-50 largest component
(cache/lcc_all.npy). These fixed-target histograms separate path shortening from changes in which
researchers become reachable. Their source-paired cap-minus-50 mean and SE compare identical sources
and target sets; the original all-reachable summaries remain unchanged and are reported separately.
"""
import hashlib
import gc
import json
import os
import sys

import duckdb
import numpy as np
from numba import config, get_num_threads, njit, prange, set_num_threads

from common import CACHE, D, PG, RESULTS, ROOT, load_ids, log, save_result
from identities import file_identity, same_identity

sys.path.insert(0, str(ROOT))
from pipeline.works import qualifying_works_sql  # noqa: E402

CAPS = (50, 100, 500, 1_000_000)
MAXD = 64
BUILD_VERSION = 2
BUILD_BATCH = 50_000
DISTANCE_VERSION = 2
STAT_BATCH = 1_000_000
GiB = 1024 ** 3


def work_cap_statistics(work_offsets, work_sizes, caps=CAPS, batch_size=STAT_BATCH):
    """Exact work/authorship/pair counts in bounded chunks, released before any BFS buffers exist."""
    if batch_size < 1:
        raise ValueError("work statistics batch_size must be positive")
    result = {cap: {"works_left_out": 0, "authorships_left_out": 0, "pairs_left_out": 0,
                    "pairs_kept": 0, "max_researchers_on_a_work": 0} for cap in caps}
    for start in range(0, len(work_sizes), batch_size):
        stop = min(start + batch_size, len(work_sizes))
        k = np.diff(work_offsets[start:stop + 1])
        pairs = k * (k - 1) // 2
        total_pairs = int(pairs.sum())
        for cap in caps:
            big = work_sizes[start:stop] > cap
            row = result[cap]
            excluded = int(pairs[big].sum())
            row["works_left_out"] += int(big.sum())
            row["authorships_left_out"] += int(k[big].sum())
            row["pairs_left_out"] += excluded
            row["pairs_kept"] += total_pairs - excluded
            if big.any():
                row["max_researchers_on_a_work"] = max(row["max_researchers_on_a_work"], int(k[big].max()))
    return result


def bfs_thread_plan(n, works, authorships, sources, configured_threads, guard_bytes=22 * GiB,
                    headroom_bytes=4 * GiB):
    """Conservative array bound for eight workers, otherwise four, beneath the footprint guard.

    Shared CSR: 8N + 12W + 8M + 16 bytes, where M is authorships. Each worker owns
    int16 researcher distances, bool visited works and two int32 researcher frontiers:
    10N + W bytes. The original-LCC mask, histogram buffers and 4 GiB of allocator/runtime
    headroom are additional. This is a planned upper array allowance, not measured RSS.
    """
    if min(n, works, authorships, sources, configured_threads) < 1:
        raise ValueError("Positive graph dimensions, sources and configured threads are required")
    shared = 8 * n + 12 * works + 8 * authorships + 16
    per_worker = 10 * n + works
    target = n
    # All/fixed output and checkpoint copies, plus retained baseline/previous histograms.
    histogram_buffers = 8 * sources * (MAXD + 1) * np.dtype(np.int64).itemsize
    base = shared + target + histogram_buffers
    statistics_temporaries = 40 * min(STAT_BATCH, works)
    statistics_peak = shared + target + statistics_temporaries + headroom_bytes
    estimates = {threads: max(base + threads * per_worker + headroom_bytes, statistics_peak) for threads in (8, 4)}
    candidates = [threads for threads in (8, 4) if threads <= configured_threads]
    if not candidates:
        candidates = [configured_threads]  # respect deliberately smaller configured pools (e.g. fixtures)
        estimates[configured_threads] = max(base + configured_threads * per_worker + headroom_bytes, statistics_peak)
    selected = next((threads for threads in candidates if estimates[threads] <= guard_bytes), None)
    if selected is None:
        raise MemoryError(f"A2c shared graph and {candidates[-1]} worker buffers plus headroom need "
                          f"{estimates[candidates[-1]] / GiB:.2f} GiB, above the {guard_bytes / GiB:.2f} GiB guard")
    return {"threads": selected, "configured_max_threads": int(configured_threads),
            "shared_csr_bytes": shared, "private_bytes_per_worker": per_worker,
            "fixed_target_mask_bytes": target, "histogram_buffer_allowance_bytes": histogram_buffers,
            "runtime_headroom_bytes": headroom_bytes, "guard_bytes": guard_bytes,
            "estimated_total_bytes": estimates[selected],
            "candidate_estimated_total_bytes": {str(key): value for key, value in estimates.items()},
            "work_statistics_chunk_works": STAT_BATCH,
            "work_statistics_temporary_allowance_bytes": statistics_temporaries,
            "work_statistics_estimated_peak_bytes": statistics_peak,
            "estimate_kind": "Conservative array allowance plus runtime headroom; not measured physical footprint.",
            "reason": "Eight workers fit the configured pool and memory allowance." if selected == 8 else
                      "Eight workers exceed the configured pool or memory allowance; using the smaller safe pool."}


@njit(cache=True)
def _author_side_into(n, w_off, w_auth, deg, out):
    """Fill disk-backed author rows; only the insertion positions need a RAM copy."""
    deg[:] = 0
    for t in range(w_auth.shape[0]):
        deg[w_auth[t] + 1] += 1
    for i in range(n):
        deg[i + 1] += deg[i]
    pos = deg[:-1].copy()
    for w in range(w_off.shape[0] - 1):
        for t in range(w_off[w], w_off[w + 1]):
            a = w_auth[t]
            out[pos[a]] = n + w
            pos[a] += 1


def _atomic_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1))
    tmp.replace(path)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _batch_members(batch, keys, order):
    """Match deduplicated authors to network indices, retaining works with two or more."""
    col = batch.column("ids")
    lens = col.value_lengths().fill_null(0).to_numpy(zero_copy_only=False)
    flat = col.flatten().to_numpy(zero_copy_only=False)
    pos = np.minimum(np.searchsorted(keys, flat), len(keys) - 1)
    hit = keys[pos] == flat
    row = np.repeat(np.arange(len(lens)), lens)[hit]
    counts = np.bincount(row, minlength=len(lens))
    keep = counts >= 2
    sizes = batch.column("n_authors").to_numpy(zero_copy_only=False)[keep].astype(np.int32)
    members = order[pos[hit][keep[row]]].astype(np.int32)
    return sizes, counts[keep].astype(np.int64), members


def _raw_to_npy(raw, path, dtype, count):
    """Convert an append-only raw array without a second in-memory copy."""
    out = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(count,))
    with raw.open("rb") as source:
        for start in range(0, count, 1_000_000):
            chunk = np.fromfile(source, dtype=dtype, count=min(1_000_000, count - start))
            if not len(chunk):
                raise ValueError(f"Truncated bipartite build array: {raw}")
            out[start:start + len(chunk)] = chunk
    out.flush()
    return out


def build():
    paths = {k: CACHE / f"bipartite_{k}.npy" for k in ("a_off", "a_nb", "w_off", "w_auth", "w_size")}
    files = sorted((D / "works_authorships").glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError("No works_authorships parquet files")
    # Content of ids is hashed as well as file metadata: node order defines every CSR entry.
    ids = load_ids().astype(np.int64, copy=False)
    identity = {"version": BUILD_VERSION, "ids": file_identity(PG / "ids.npy"),
                "ids_sha256": hashlib.sha256(ids).hexdigest(),
                "inputs": [file_identity(p) for p in files],
                "filter": qualifying_works_sql("INPUT", min_authors=2, dedupe=False)}
    fingerprint = _digest(identity)
    manifest = CACHE / "bipartite_manifest.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        # The saved record may hold paths from the folder the graph was built in; its fingerprint stays the
        # network's identity in results and checkpoints.
        if (saved["fingerprint"] == _digest(saved["identity"]) and same_identity(saved["identity"], identity)
                and all(p.exists() for p in paths.values())):
            graph = {k: np.load(p, mmap_mode="r") for k, p in paths.items()}
            assert len(graph["a_off"]) == len(ids) + 1
            assert graph["a_off"][-1] == graph["w_off"][-1] == len(graph["w_auth"]) == len(graph["a_nb"])
            assert len(graph["w_off"]) == len(graph["w_size"]) + 1
            return graph
    stage = CACHE / "bipartite_build"
    stage.mkdir(exist_ok=True)
    state_path = stage / "progress.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state.get("fingerprint") != fingerprint:
        state = {"fingerprint": fingerprint, "files_done": 0, "works": 0, "authorships": 0}
        _atomic_json(state_path, state)
    order = np.argsort(ids)
    keys = ids[order]
    n = len(ids)
    del ids
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    con.execute("SET threads=2")
    con.execute("SET enable_progress_bar=false")
    raw_paths = {k: stage / f"{k}.raw" for k in ("size", "count", "auth")}
    handles = {k: p.open("r+b" if p.exists() else "w+b") for k, p in raw_paths.items()}
    try:
        for k, count, width in (("size", state["works"], 4), ("count", state["works"], 8),
                                ("auth", state["authorships"], 4)):
            handle = handles[k]
            if handle.seek(0, 2) < count * width:
                raise ValueError(f"Truncated build checkpoint: {raw_paths[k]}")
            handle.truncate(count * width)      # discard an interrupted file's uncommitted tail
            handle.seek(0, 2)
        log(f"bipartite build: resume at file {state['files_done']}/{len(files)}, "
            f"{state['works']:,} works, {state['authorships']:,} authorships")
        for index in range(state["files_done"], len(files)):
            sql = qualifying_works_sql(str(files[index]), min_authors=2, dedupe=False)
            # Relation streaming avoids execute()'s materialized DB-API result and extra Arrow copy.
            reader = con.sql(f"SELECT n_authors, list_distinct(author_ids) AS ids FROM ({sql})").to_arrow_reader(BUILD_BATCH)
            for batch in reader:
                sizes, counts, members = _batch_members(batch, keys, order)
                sizes.tofile(handles["size"])
                counts.tofile(handles["count"])
                members.tofile(handles["auth"])
                state["works"] += len(sizes)
                state["authorships"] += len(members)
            for handle in handles.values():
                handle.flush()
                os.fsync(handle.fileno())
            state["files_done"] = index + 1
            _atomic_json(state_path, state)
            if (index + 1) % 10 == 0 or index + 1 == len(files):
                log(f"bipartite build: {index + 1}/{len(files)} files, "
                    f"{state['works']:,} works, {state['authorships']:,} authorships")
    finally:
        for handle in handles.values():
            handle.close()
        con.close()
    del keys, order
    W, memberships = state["works"], state["authorships"]
    if n + W > np.iinfo(np.int32).max:
        raise OverflowError("Bipartite node indices exceed int32")
    log("bipartite build: writing memory-mapped CSR")
    w_size = _raw_to_npy(raw_paths["size"], paths["w_size"], np.int32, W)
    w_auth = _raw_to_npy(raw_paths["auth"], paths["w_auth"], np.int32, memberships)
    w_off = np.lib.format.open_memmap(paths["w_off"], mode="w+", dtype=np.int64, shape=(W + 1,))
    w_off[0] = 0
    with raw_paths["count"].open("rb") as source:
        for start in range(0, W, 1_000_000):
            chunk = np.fromfile(source, dtype=np.int64, count=min(1_000_000, W - start))
            np.cumsum(chunk, out=w_off[start + 1:start + len(chunk) + 1])
            w_off[start + 1:start + len(chunk) + 1] += w_off[start]
    a_off = np.lib.format.open_memmap(paths["a_off"], mode="w+", dtype=np.int64, shape=(n + 1,))
    a_nb = np.lib.format.open_memmap(paths["a_nb"], mode="w+", dtype=np.int32, shape=(memberships,))
    _author_side_into(n, w_off, w_auth, a_off, a_nb)
    out = {"a_off": a_off, "a_nb": a_nb, "w_off": w_off, "w_auth": w_auth, "w_size": w_size}
    for array in out.values():
        array.flush()
    assert a_off[-1] == w_off[-1] == memberships
    _atomic_json(manifest, {"fingerprint": fingerprint, "identity": identity,
                            "researchers": n, "works": W, "authorships": memberships})
    log(f"bipartite build complete: {n:,} researchers, {W:,} works, {memberships:,} authorships")
    return {k: np.load(p, mmap_mode="r") for k, p in paths.items()}


@njit(parallel=True, cache=True)
def run_with_targets(a_off, a_nb, w_off, w_auth, w_size, sources, cap, maxd, target_mask):
    """All-reachable and fixed-target histograms from the same bipartite BFS traversal."""
    n = a_off.shape[0] - 1
    W = w_size.shape[0]
    k = sources.shape[0]
    hist = np.zeros((k, maxd + 1), np.int64)
    fixed = np.zeros((k, maxd + 1), np.int64)
    for s in prange(k):
        da = np.full(n, -1, np.int16)
        seen = np.zeros(W, np.bool_)
        frontier = np.empty(n, np.int32)
        nxt = np.empty(n, np.int32)
        frontier[0] = sources[s]
        da[sources[s]] = 0
        nf, d = 1, 0
        while nf > 0:
            nn = 0
            for f in range(nf):
                u = frontier[f]
                for j in range(a_off[u], a_off[u + 1]):
                    w = a_nb[j] - n
                    if seen[w] or w_size[w] > cap:
                        continue
                    seen[w] = True
                    for t in range(w_off[w], w_off[w + 1]):
                        v = w_auth[t]
                        if da[v] < 0:
                            da[v] = d + 1
                            nxt[nn] = v
                            nn += 1
            frontier, nxt = nxt, frontier
            nf = nn
            d += 1
        for i in range(n):
            if da[i] >= 0:
                hist[s, min(da[i], maxd)] += 1
                if target_mask[i]:
                    fixed[s, min(da[i], maxd)] += 1
    return hist, fixed


def run(a_off, a_nb, w_off, w_auth, w_size, sources, cap, maxd):
    """Backward-compatible all-reachable histogram interface for small reference tests."""
    mask = np.zeros(len(a_off) - 1, np.bool_)
    return run_with_targets(a_off, a_nb, w_off, w_auth, w_size, sources, cap, maxd, mask)[0]


def summarize(h):
    per_source_pairs = h[:, 1:].sum(axis=1)
    per_source_means = (h[:, 1:] * np.arange(1, h.shape[1])).sum(axis=1) / per_source_pairs
    h = h.sum(axis=0).astype(np.float64)
    h[0] = 0
    d = np.arange(len(h))
    tot = h.sum()
    cdf = np.cumsum(h) / tot
    return {"pairs": int(tot), "mean": float((d * h).sum() / tot), "median": int(np.searchsorted(cdf, 0.5)),
            "p90": int(np.searchsorted(cdf, 0.9)), "pmf": {int(i): float(h[i] / tot) for i in np.flatnonzero(h)},
            "source_mean_se": float(per_source_means.std(ddof=1) / np.sqrt(len(per_source_means)))
                              if len(per_source_means) > 1 else None}


def checkpointed_run(graph, sources, cap, network_fingerprint, all_sources, target_mask=None):
    """Resume only matching graph/source/target-mask checkpoints; return both histograms when masked."""
    return_fixed = target_mask is not None
    if target_mask is None:
        target_mask = np.zeros(len(graph["a_off"]) - 1, np.bool_)
    target_mask = np.asarray(target_mask, dtype=np.bool_)
    assert target_mask.shape == (len(graph["a_off"]) - 1,)
    target_digest = hashlib.sha256(np.ascontiguousarray(target_mask)).hexdigest()
    checkpoint = CACHE / f"a2c_hist_cap{cap}.npz"
    signature = _digest({"graph": network_fingerprint, "sources": all_sources.tolist(),
                         "cap": cap, "maxd": MAXD, "kernel_version": DISTANCE_VERSION,
                         "target_mask_sha256": target_digest})
    hist = np.zeros((len(sources), MAXD + 1), np.int64)
    fixed = np.zeros_like(hist)
    done = 0
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            if str(saved["signature"]) == signature:
                done = min(len(saved["hist"]), len(sources))
                assert np.array_equal(saved["sources"][:done], sources[:done])
                hist[:done] = saved["hist"][:done]
                assert str(saved["target_mask_sha256"]) == target_digest
                assert saved["hist_fixed"].shape == saved["hist"].shape
                fixed[:done] = saved["hist_fixed"][:done]
    batch_size = int(os.environ.get("A2C_SOURCE_BATCH", max(4, get_num_threads())))
    if batch_size < 1:
        raise ValueError("A2C_SOURCE_BATCH must be positive")
    log(f"cap {cap}: {done}/{len(sources)} sources cached; batch={batch_size}, threads={get_num_threads()}")
    for start in range(done, len(sources), batch_size):
        end = min(start + batch_size, len(sources))
        hist[start:end], fixed[start:end] = run_with_targets(
            graph["a_off"], graph["a_nb"], graph["w_off"], graph["w_auth"],
            graph["w_size"], sources[start:end], cap, MAXD, target_mask)
        tmp = checkpoint.with_name(checkpoint.stem + ".tmp.npz")
        np.savez(tmp, signature=np.array(signature), sources=sources[:end], hist=hist[:end], hist_fixed=fixed[:end],
                 target_mask_sha256=np.array(target_digest), kernel_version=np.array(DISTANCE_VERSION))
        tmp.replace(checkpoint)
        log(f"cap {cap}: {end}/{len(sources)} sources complete")
    return (hist, fixed) if return_fixed else hist


def fixed_target_summary(hist, baseline, previous, target_records):
    """Validate fixed denominators/CDF monotonicity and summarize source-paired cap-minus-50 changes."""
    assert hist.shape == baseline.shape == previous.shape
    assert target_records > 1
    for values in (hist, baseline, previous):
        assert np.all(values.sum(axis=1) == target_records), "fixed-target population changed"
        assert np.all(values[:, 0] == 1), "each source must belong to the fixed target population"
        assert not values[:, -1].any(), "fixed distances reached the histogram limit"
    cumulative = np.cumsum(hist[:, 1:], axis=1)
    assert np.all(cumulative >= np.cumsum(previous[:, 1:], axis=1)), "fixed-target CDF worsened when the cap increased"
    assert np.all(cumulative >= np.cumsum(baseline[:, 1:], axis=1)), "fixed-target CDF worsened relative to cap 50"
    distance = np.arange(1, hist.shape[1], dtype=np.int64)
    differences = ((hist[:, 1:] - baseline[:, 1:]) * distance).sum(axis=1) / (target_records - 1)
    assert np.all(differences <= 0), "a source's fixed-target mean increased when works were added"
    return {**summarize(hist), "target_records": int(target_records),
            "target_pairs_per_source": int(target_records - 1),
            "paired_cap_minus_50": {"mean": float(differences.mean()),
                                     "se": float(differences.std(ddof=1) / np.sqrt(len(differences))) if len(differences) > 1 else None,
                                     "sources": len(differences), "minimum_source_change": float(differences.min()),
                                     "maximum_source_change": float(differences.max()),
                                     "definition": "Per-source mean difference on the same original-LCC targets, excluding the source itself; cap minus 50."},
            "validation": {"constant_target_count": True, "source_self_count_exact": True,
                           "every_source_cdf_nondecreasing_vs_previous_cap": True,
                           "every_source_cdf_nondecreasing_vs_cap50": True}}


def validate_cap50(hist, sources, all_sources):
    """A50's projected graph must reproduce the original CSR's exact distance histogram."""
    reference = json.loads((RESULTS / "a2_distances.json").read_text())
    pairs = reference["all_links"]["pairs"]
    K = reference["K"]
    assert np.array_equal(sources, all_sources[:len(sources)])
    expected_reached = pairs // K + 1
    assert pairs % K == 0
    assert np.all(hist.sum(axis=1) == expected_reached), "cap50 reachable population differs from A2"
    result = {"a2_sources": K, "sources_checked": len(sources), "reachable_count_exact": True,
              "full_histogram_exact": False}
    if len(sources) == K:
        counts = hist.sum(axis=0)
        counts[0] = 0
        expected = np.zeros_like(counts)
        for distance, probability in reference["all_links"]["pmf"].items():
            expected[int(distance)] = round(probability * pairs)
        if not np.array_equal(counts, expected):
            raise AssertionError("cap50 distance histogram differs from A2: "
                                 + str({int(i): int(counts[i] - expected[i]) for i in np.flatnonzero(counts != expected)}))
        assert abs(summarize(hist)["mean"] - reference["all_links"]["mean"]) < 1e-12
        result["full_histogram_exact"] = True
    log("cap50 validation:", result)
    return result


if __name__ == "__main__":
    SOURCES = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    g = build()
    n, W = len(g["a_off"]) - 1, len(g["w_size"])
    if W and int(g["w_size"].max()) > CAPS[-1]:
        raise ValueError("The uncapped sentinel is smaller than an observed team")
    log(f"bipartite graph: {n:,} researchers, {W:,} works with 2+ of them, {len(g['w_auth']):,} authorships")
    all_sources = np.load(CACHE / "a2_sources.npy")
    if not 1 <= SOURCES <= len(all_sources):
        raise ValueError(f"Requested {SOURCES} sources; A2 has {len(all_sources)}")
    sources = all_sources[:SOURCES]
    target_mask = np.load(CACHE / "lcc_all.npy", mmap_mode="r")
    assert target_mask.dtype == np.bool_ and target_mask.shape == (n,)
    assert np.all(target_mask[sources]), "A2 sources must belong to the original largest component"
    target_records = int(target_mask.sum())
    target_digest = hashlib.sha256(target_mask).hexdigest()
    manifest = json.loads((CACHE / "bipartite_manifest.json").read_text())
    out = {"sources": int(len(sources)), "researchers": n, "works": W, "authorships": int(len(g["w_auth"]))}
    out["network_fingerprint"] = manifest["fingerprint"]
    out["distance_kernel_version"] = DISTANCE_VERSION
    out["baseline_targets"] = {"definition": "Original cap-50 largest component (cache/lcc_all.npy).",
                               "records": target_records, "sha256": target_digest,
                               "source_self_excluded_from_summaries": True}
    log("precomputing team-cap statistics in bounded work chunks")
    cap_statistics = work_cap_statistics(g["w_off"], g["w_size"])
    gc.collect()  # no global work-count, pair-count or cap mask arrays survive into BFS
    execution = bfs_thread_plan(n, W, len(g["w_auth"]), len(sources), int(config.NUMBA_NUM_THREADS))
    assert execution["shared_csr_bytes"] == sum(array.nbytes for array in g.values()), "CSR dtypes differ from the memory model"
    set_num_threads(execution["threads"])
    execution["actual_numba_threads"] = get_num_threads()
    execution["source_batch"] = int(os.environ.get("A2C_SOURCE_BATCH", max(4, get_num_threads())))
    if execution["source_batch"] < 1:
        raise ValueError("A2C_SOURCE_BATCH must be positive")
    execution["maximum_active_workers_per_batch"] = min(get_num_threads(), execution["source_batch"], len(sources))
    execution["thread_count_scope"] = "Current invocation's Numba mask; cached rows may come from an earlier mask (4/8 parity tested)."
    out["execution"] = execution
    log(f"A2c BFS: {get_num_threads()} Numba threads, source batch {execution['source_batch']}; "
        f"estimated array/runtime allowance {execution['estimated_total_bytes'] / GiB:.2f} GiB "
        f"including {execution['runtime_headroom_bytes'] / GiB:.1f} GiB headroom, "
        f"guard {execution['guard_bytes'] / GiB:.1f} GiB")
    log("A2c memory/thread plan", execution)
    first100 = {**out, "sources": 100, "derived_from_first_sources": SOURCES} if SOURCES > 100 else None
    baseline_fixed = previous_fixed = None
    for cap in CAPS:
        h, fixed = checkpointed_run(g, sources, cap, manifest["fingerprint"], all_sources, target_mask)
        if h[:, MAXD].any():
            raise ValueError("Distances reached the histogram limit; increase MAXD before interpretation")
        if cap == 50:
            assert np.array_equal(fixed, h), "cap50 fixed-target histogram differs from the full baseline"
            baseline_fixed = previous_fixed = fixed
            out["cap50_validation"] = validate_cap50(h, sources, all_sources)
            out["cap50_validation"]["fixed_targets_exact"] = True
            if first100 is not None:
                first100["cap50_validation"] = validate_cap50(h[:100], sources[:100], all_sources)
                first100["cap50_validation"]["fixed_targets_exact"] = True
        res = summarize(h)
        stats = cap_statistics[cap]
        res.update({**stats, "reached_share": float(h.sum(axis=1).mean() / n)})
        res["fixed_baseline_targets"] = fixed_target_summary(fixed, baseline_fixed, previous_fixed, target_records)
        key = "any" if cap >= 1_000_000 else f"cap_{cap}"
        out[key] = res
        if first100 is not None:
            first100[key] = {**summarize(h[:100]), **stats,
                             "reached_share": float(h[:100].sum(axis=1).mean() / n),
                             "fixed_baseline_targets": fixed_target_summary(fixed[:100], baseline_fixed[:100], previous_fixed[:100], target_records)}
        log(cap, {k: v for k, v in res.items() if k not in ("pmf", "fixed_baseline_targets")},
            "fixed-target mean", res["fixed_baseline_targets"]["mean"],
            "paired cap-minus-50", res["fixed_baseline_targets"]["paired_cap_minus_50"])
        previous_fixed = fixed
    save_result(f"a2c_team_cap{'' if SOURCES == 1000 else f'_{SOURCES}'}", out)
    if first100 is not None:
        first100["cap50_validation"]["derived_from_validated_full_run"] = out["cap50_validation"]["full_histogram_exact"]
        save_result("a2c_team_cap_100", first100)
