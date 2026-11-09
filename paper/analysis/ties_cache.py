"""Bounded-memory, exact edge features and scratch reductions shared by B2 and B2b.

The cache stores float32 weights/lengths and one packed label byte per link. Metadata is aligned
by numeric OpenAlex ID in batches; no full-table string join or per-edge source/destination copy.
Exact quantiles/medians use one reusable disk-backed float32 scratch array, preserving edge order
before reductions and NumPy's float32 sum/mean and default quantile interpolation semantics.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common import CACHE, D, G, PG, log
from identities import file_identity, same_identity

BATCH = 500_000
IN_COMM, BOTH_CC, IN_CC, BOTH_FD, IN_FD, ON_MAP = 1, 2, 4, 8, 16, 32
FEATURE_VERSION = 1


def identity(attrs, coords, edges, ids):
    return {"version": FEATURE_VERSION, "feature_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "inputs": [file_identity(path) for path in (attrs, coords, edges, ids)]}


def category_codes(column, known):
    encoded = pc.dictionary_encode(column)
    dictionary = encoded.dictionary.to_pylist()
    code = np.array([known.setdefault(value, len(known)) for value in dictionary], np.int16)
    assert len(known) < np.iinfo(np.int16).max
    indices = pc.fill_null(encoded.indices, -1).to_numpy()
    result = np.full(len(column), -1, np.int16)
    present = indices >= 0
    result[present] = code[indices[present]]
    return result


def metadata_arrays(attrs, coords, ids, batch_size=BATCH):
    n = len(ids)
    comm = np.empty(n, np.int32)
    cc, fd = np.empty(n, np.int16), np.empty(n, np.int16)
    country_names, field_names = {}, {}
    start = 0
    for batch in pq.ParquetFile(attrs).iter_batches(batch_size=batch_size, columns=["node_idx", "community", "country", "field"]):
        stop = start + len(batch)
        assert np.array_equal(batch.column("node_idx").to_numpy(), np.arange(start, stop)), "node metadata order changed"
        community = batch.column("community")
        if community.null_count and comm.dtype != np.float64:
            expanded = np.empty(n, np.float64)
            expanded[:start] = comm[:start]
            comm = expanded
        comm[start:stop] = community.to_numpy(zero_copy_only=False)
        cc[start:stop] = category_codes(batch.column("country"), country_names)
        fd[start:stop] = category_codes(batch.column("field"), field_names)
        start = stop
    assert start == n
    x, y = np.full(n, np.nan, np.float32), np.full(n, np.nan, np.float32)
    assigned = np.zeros(n, bool)
    order = np.argsort(ids)
    keys = ids[order]
    for batch in pq.ParquetFile(coords).iter_batches(batch_size=batch_size, columns=["id", "xw", "yw"]):
        numeric = pc.extract_regex(batch.column("id"), r"(?P<num>\d+)$").field("num").cast(pa.int64()).to_numpy()
        positions = np.minimum(np.searchsorted(keys, numeric), n - 1)
        present = keys[positions] == numeric
        nodes = order[positions[present]]
        assert len(np.unique(nodes)) == len(nodes) and not assigned[nodes].any(), "duplicate map coordinate IDs"
        assigned[nodes] = True
        x[nodes] = batch.column("xw").to_numpy().astype(np.float32)[present]
        y[nodes] = batch.column("yw").to_numpy().astype(np.float32)[present]
    return {"community": comm, "country": cc, "field": fd, "x": x, "y": y}


def edge_features(src, dst, metadata):
    comm, cc, fd = (metadata[key] for key in ("community", "country", "field"))
    in_comm = comm[src] == comm[dst]  # missing communities remain NaN and compare false, as in pandas
    both_cc = (cc[src] >= 0) & (cc[dst] >= 0)
    both_fd = (fd[src] >= 0) & (fd[dst] >= 0)
    flags = in_comm.astype(np.uint8)
    flags |= both_cc.astype(np.uint8) * BOTH_CC
    flags |= (both_cc & (cc[src] == cc[dst])).astype(np.uint8) * IN_CC
    flags |= both_fd.astype(np.uint8) * BOTH_FD
    flags |= (both_fd & (fd[src] == fd[dst])).astype(np.uint8) * IN_FD
    x, y = metadata["x"], metadata["y"]
    length = np.hypot(x[src] - x[dst], y[src] - y[dst])
    flags |= np.isfinite(length).astype(np.uint8) * ON_MAP
    return length, flags


def load_features(attrs=CACHE / "node_attrs.parquet", coords=D / "coords_web.parquet",
                  edges=G / "edges_int32.parquet", ids_path=PG / "ids.npy", folder=CACHE / "ties", batch_size=BATCH):
    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    manifest_path = folder / "manifest.json"
    stamp = identity(attrs, coords, edges, ids_path)
    fingerprint = hashlib.sha256(json.dumps(stamp, sort_keys=True).encode()).hexdigest()
    log(f"tie feature identity: {fingerprint}; helper code {stamp['feature_code_sha256']}")
    names = ("weights", "length", "flags", "community", "country", "field", "x", "y")
    paths = {key: folder / f"{key}.npy" for key in names}
    saved = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if not same_identity(saved.get("identity"), stamp) or not all(path.exists() for path in paths.values()):
        manifest_path.unlink(missing_ok=True)
        ids = np.load(ids_path, mmap_mode="r")
        metadata = metadata_arrays(attrs, coords, ids, batch_size)
        for key, value in metadata.items():
            np.save(paths[key], value)
        reader = pq.ParquetFile(edges)
        count = reader.metadata.num_rows
        weight = np.lib.format.open_memmap(paths["weights"], mode="w+", dtype=np.float32, shape=(count,))
        length = np.lib.format.open_memmap(paths["length"], mode="w+", dtype=np.float32, shape=(count,))
        flags = np.lib.format.open_memmap(paths["flags"], mode="w+", dtype=np.uint8, shape=(count,))
        start = 0
        for batch in reader.iter_batches(batch_size=batch_size, columns=["src", "dst", "weight"]):
            stop = start + len(batch)
            src, dst, values = (batch.column(key).to_numpy() for key in ("src", "dst", "weight"))
            assert values.dtype == np.float32, "review sum/quantile semantics for a non-float32 weight file"
            weight[start:stop] = values
            length[start:stop], flags[start:stop] = edge_features(src, dst, metadata)
            if stop % 50_000_000 < len(batch) or stop == count:
                log(f"tie features: {stop:,}/{count:,} links")
            start = stop
        assert start == count
        for array in (weight, length, flags):
            array.flush()
            array._mmap.close()
        del metadata, ids
        saved = {"identity": stamp, "fingerprint": fingerprint, "links": count, "nodes": int(len(np.load(ids_path, mmap_mode="r")))}
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(saved, indent=1))
        temporary.replace(manifest_path)
    arrays = {key: np.load(path, mmap_mode="r") for key, path in paths.items()}
    assert all(len(arrays[key]) == saved["links"] for key in ("weights", "length", "flags"))
    assert all(len(arrays[key]) == saved["nodes"] for key in names[3:])
    return arrays, saved


def close_features(arrays):
    for array in arrays.values():
        if isinstance(array, np.memmap):
            array._mmap.close()


@contextmanager
def selected_values(values, selector, scratch, batch_size=BATCH, finite=False):
    """One exact, original-order masked vector on disk, never an in-memory edge-table copy."""
    scratch = Path(scratch)
    buffer = np.lib.format.open_memmap(scratch, mode="w+", dtype=values.dtype, shape=(len(values),))
    count = 0
    try:
        for start in range(0, len(values), batch_size):
            stop = min(start + batch_size, len(values))
            chunk = values[start:stop]
            keep = np.ones(len(chunk), bool) if selector is None else selector(start, stop)
            if finite:
                keep = keep & np.isfinite(chunk)
            chosen = chunk[keep]
            buffer[count:count+len(chosen)] = chosen
            count += len(chosen)
        buffer.flush()
        yield buffer[:count]
    finally:
        buffer._mmap.close()


def describe(values, total_count, total_weight):
    """Original B2 reductions, including NumPy float32 arithmetic and median semantics."""
    count = len(values)
    if not count:
        return {"links": 0, "share_links": 0., "share_weight": 0., "median_weight": float("nan"),
                "mean_weight": float("nan"), "share_weight_ge_1": float("nan")}
    weight_sum = values.sum()
    mean_weight = values.mean()
    at_least_one = sum(np.count_nonzero(values[start:start+BATCH] >= 1) for start in range(0,count,BATCH))
    median = np.median(values, overwrite_input=True)  # compute only after original-order sums/means
    return {"links": count, "share_links": count / total_count, "share_weight": float(weight_sum / total_weight),
            "median_weight": float(median), "mean_weight": float(mean_weight), "share_weight_ge_1": at_least_one/count}


def read_edge_rows(path, positions, batch_size=BATCH):
    """Source/destination arrays in the caller's row order, including seeded unsorted sample order."""
    positions = np.asarray(positions, dtype=np.int64)
    order = np.argsort(positions, kind="stable")
    sorted_positions = positions[order]
    src, dst = np.empty(len(positions), np.int32), np.empty(len(positions), np.int32)
    start = found = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=["src", "dst"]):
        stop = start + len(batch)
        low, high = np.searchsorted(sorted_positions, [start, stop])
        if high > low:
            at = sorted_positions[low:high] - start
            src[order[low:high]] = batch.column("src").to_numpy()[at]
            dst[order[low:high]] = batch.column("dst").to_numpy()[at]
            found += high-low
        start = stop
    assert found == len(positions)
    return src, dst
