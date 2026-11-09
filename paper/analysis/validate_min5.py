"""Validate the completed five-work run without materializing the full graph.

Default: checks declared outputs, finite JSON numbers, population/CSR metadata,
A2c cap-50 parity and source checkpoints, representative/flag agreement, export
rows, SHA256 checksums, eligible-work uniqueness, and an exact reconstruction
of repaired edges from the deposit. Reconstruction partitions uint64 edge keys
on disk and deduplicates one partition at a time (about 5 GB scratch; no full
edge table in RAM). DuckDB is capped at 6 GB and four threads.

--quick omits checksums and scans of graph/deposit/works. --skip-reconstruction
omits only the disk partition pass. --require-manuscript also rejects numeric
TODO markers, stale/missing figure formats, and builds without an exact successful
receipt; it binds code/results/deliverables and large-input identities before and
after validation. These checks do not
certify human identity labels, statistical design, figure appearance, or
publication; those limitations are always written as warnings.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common import CACHE, D, G, ND, NET, PAPER, PG, RESULTS, ROOT, log
from identities import file_identity, same_identity
from kernels import reuse_a2_summary, union_pairs
from run_min5 import EARLIER, STEPS
from delivery_checks import validate_build_receipt, validate_figures, validation_fingerprint, verify_fingerprint

sys.path.insert(0, str(ROOT))
from pipeline.works import id_key_sql, qualifying_works_sql  # noqa: E402
from pipeline.filter_authors import is_person_sql  # noqa: E402

BATCH = 250_000
EXPECTED_N = 19_848_920
EXPECTED_E = 525_109_095
TZ = ZoneInfo("America/Los_Angeles")


def finite_json(value, path="root", metadata_missing=None):
    if isinstance(value, float) and not math.isfinite(value):
        # Old outputs use NaN for absent display metadata. These precise fields
        # are not statistics and must not trigger expensive scientific reruns.
        permitted = (
            r"paper/results/a3b_far_edge\.json\.nodes\[\d+\]\.(xw|yw)",
            r"paper/results/a4_hubs\.json\.top\[\d+\]\.(country|field|institution)",
            r"paper/results/a4b_hubs\.json\.(top25|gatekeepers\.examples)\[\d+\]\.(country|field|institution)",
            r"paper/results/a5b_communities\.json\.communities\[\d+\]\.top_country",
            r"paper/results/b1_uncited\.json\.prolific_uncited\.top\[\d+\]\[[23]\]",
        )
        if metadata_missing is not None and math.isnan(value) and any(re.fullmatch(pattern, path) for pattern in permitted):
            metadata_missing.append(path)
            return
        raise ValueError(f"nonfinite number at {path}: {value}")
    if isinstance(value, dict):
        for key, item in value.items():
            finite_json(item, f"{path}.{key}", metadata_missing)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_json(item, f"{path}[{index}]", metadata_missing)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def contains(sorted_keys, queries):
    if not len(sorted_keys):
        return np.zeros(len(queries), bool)
    positions = np.searchsorted(sorted_keys, queries)
    return (positions < len(sorted_keys)) & (sorted_keys[np.minimum(positions, len(sorted_keys) - 1)] == queries)


def pack(src, dst):
    return (src.astype(np.uint64) << np.uint64(32)) | dst.astype(np.uint64)


def keys_from_parquet(path, repair_only=False):
    """Packed (src, dst) keys of the deposited flagged links; with repair_only, those the repaired network removes."""
    table = pq.ParquetFile(path)
    out = np.empty(table.metadata.num_rows, np.uint64)
    position = 0
    columns = ["src", "dst", "repair"] if repair_only else ["src", "dst"]
    for batch in table.iter_batches(batch_size=BATCH, columns=columns):
        src, dst = batch.column("src").to_numpy(), batch.column("dst").to_numpy()
        assert np.all(src < dst), "flagged links must be canonical src < dst"
        if repair_only:
            chosen = batch.column("repair").to_numpy(zero_copy_only=False).astype(bool)
            src, dst = src[chosen], dst[chosen]
        out[position:position + len(src)] = pack(src, dst)
        position += len(src)
    out = out[:position]
    out.sort()
    assert not np.any(out[1:] == out[:-1]), "duplicate flagged links"
    return out


def reconstruct_edge_count(batches, representative, flagged, scratch, buckets=128):
    """Exact distinct edge count after node contraction, loop and flagged removal."""
    scratch.mkdir(parents=True, exist_ok=True)
    paths = [scratch / f"edges-{i:03d}.bin" for i in range(buckets)]
    handles = [path.open("wb") for path in paths]
    count, flagged_hits, removed_incident, loops, candidates = 0, 0, 0, 0, 0
    try:
        for batch in batches:
            src, dst = batch.column("src").to_numpy(), batch.column("dst").to_numpy()
            assert np.all(src >= 0) and np.all(dst >= 0) and np.all(src != dst)
            assert np.all(src < len(representative)) and np.all(dst < len(representative))
            if "weight" in batch.schema.names:
                weight = batch.column("weight").to_numpy()
                assert np.all(np.isfinite(weight) & (weight > 0)), "nonpositive or nonfinite edge weights"
            # Graph preparation orders string IDs while edge building orders
            # numeric IDs; source rows need not have src < dst.
            marked = contains(flagged, pack(np.minimum(src, dst), np.maximum(src, dst)))
            flagged_hits += int(marked.sum())
            a, b = representative[src], representative[dst]
            kept = ~marked & (a >= 0) & (b >= 0)
            removed_incident += int((~marked & ((a < 0) | (b < 0))).sum())
            loops += int((kept & (a == b)).sum())
            kept &= a != b
            a, b = a[kept], b[kept]
            keys = pack(np.minimum(a, b), np.maximum(a, b))
            bucket = ((keys ^ (keys >> np.uint64(32))) % np.uint64(buckets)).astype(np.int16)
            order = np.argsort(bucket, kind="stable")
            keys = keys[order]
            lengths = np.bincount(bucket, minlength=buckets)
            at = 0
            for length, handle in zip(lengths, handles):
                if length:
                    keys[at:at + length].tofile(handle)
                at += length
            count += len(src)
            candidates += len(keys)
        for handle in handles:
            handle.close()
        unique, largest = 0, 0
        for path in paths:
            values = np.fromfile(path, dtype=np.uint64)
            largest = max(largest, values.nbytes)
            values.sort()
            unique += int(len(values) > 0) + int(np.count_nonzero(values[1:] != values[:-1]))
            path.unlink()
    finally:
        for handle in handles:
            handle.close()
    assert flagged_hits == len(flagged), "deposit flagged links are not all present in the source graph"
    return {"source_links": count, "flagged_links": flagged_hits, "unflagged_links_touching_removed": removed_incident,
            "contraction_loops": loops, "candidate_edges": candidates, "repaired_links": unique,
            "parallel_edges_merged": candidates - unique, "largest_sort_partition_bytes": largest}


class Report:
    def __init__(self):
        self.checks = []
        self.delivery_fingerprint = None
        self.warnings = [
            "Human identity labels and the hand-judged merge sample remain external judgments; structural checks do not validate them.",
            "The dataset and analysis-code release remain unpublished unless separately confirmed; this validator performs no upload.",
            "Historical snapshot and institution caches predate manifest capture; their sidecars state the saved evidence and its limitation.",
            "Map figures show the 15-work layout recoloured by five-work communities, not positions for all five-work profiles.",
            "Automated validation cannot certify manuscript interpretation or visual legibility."]
    def run(self, name, function):
        log("validation", name)
        try:
            detail = function()
            self.checks.append({"name": name, "status": "pass", "detail": detail})
            return detail
        except Exception as error:
            self.checks.append({"name": name, "status": "fail", "error": f"{type(error).__name__}: {error}"})
            log("validation FAILED", name, error)
            return None
    def save(self, args):
        failed = [c for c in self.checks if c["status"] == "fail"]
        data = {"network": NET, "time": datetime.now(TZ).isoformat(timespec="seconds"),
                "status": "failed" if failed else "passed", "quick": args.quick,
                "reconstruction_requested": not (args.quick or args.skip_reconstruction),
                "manuscript_required": args.require_manuscript, "checks": self.checks, "warnings": self.warnings,
                "delivery_fingerprint": self.delivery_fingerprint}
        target = RESULTS / "validation_min5.json"
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n")
        tmp.replace(target)
        lines = ["# Five-work validation", "", f"Status: **{data['status']}**. Checked {data['time']}.", "",
                 "| Check | Result |", "|---|---|"]
        lines += [f"| {c['name']} | {c['status']} |" for c in self.checks]
        lines += ["", "Failures:", ""] + [f"- {c['name']}: {c['error']}" for c in failed]
        if not failed:
            lines += ["None."]
        lines += ["", "Limits and pending human/publication work:", ""] + [f"- {w}" for w in self.warnings]
        (PAPER / "VALIDATION_MIN5.md").write_text("\n".join(lines) + "\n")
        log("validation", data["status"], f"{len(failed)} failed of {len(self.checks)} checks")
        return bool(failed)


def dependencies():
    declared = {p for outputs in EARLIER.values() for p in outputs}
    declared |= {p for step in STEPS if step["command"][0] != "validate_min5.py" for p in step["outputs"]}
    declared |= {"paper/results/leiden.json", "paper/figures/fig_hub_definitions.pdf",
                 "paper/cache/a6_institutions.provenance.json", "paper/cache/snapshot_20261014.provenance.json"}
    missing, metadata_missing = [], []
    for relative in sorted(declared):
        path = ROOT / relative
        if not path.is_file() or not path.stat().st_size:
            missing.append(relative)
        elif path.suffix == ".json":
            finite_json(json.loads(path.read_text()), relative, metadata_missing)
    assert not missing, "missing outputs: " + ", ".join(missing)
    return {"required_outputs": len(declared), "finite_core_statistics": True,
            "allowed_missing_metadata_nan": metadata_missing}


def population():
    assert NET == "min5", "validation must use NET=min5"
    basic = json.loads((RESULTS / "a1_basics.json").read_text())
    ids = np.load(PG / "ids.npy", mmap_mode="r")
    offsets = np.load(PG / "offsets.npy", mmap_mode="r")
    neighbors = np.load(PG / "neighbors.npy", mmap_mode="r")
    assert len(ids) == basic["nodes"] == EXPECTED_N
    assert len(offsets) == EXPECTED_N + 1 and offsets[0] == 0
    assert offsets[-1] == len(neighbors) == 2 * basic["links"] == 2 * EXPECTED_E
    for begin in range(0, len(offsets) - 1, BATCH):
        x = offsets[begin:min(begin + BATCH + 1, len(offsets))]
        assert np.all(x[1:] >= x[:-1]), "CSR offsets are not monotone"
    for path in (ND / "nodes.parquet", CACHE / "node_attrs.parquet", ND / "deposit/researchers.parquet"):
        assert pq.ParquetFile(path).metadata.num_rows == EXPECTED_N, f"wrong population in {path}"
    for path in (G / "edges_int32.parquet", ND / "deposit/links.parquet"):
        assert pq.ParquetFile(path).metadata.num_rows == EXPECTED_E, f"wrong link count in {path}"
    c = duckdb.connect()
    c.execute("SET memory_limit='1GB'")
    c.execute("SET threads=2")
    minimum = c.execute(f"SELECT min(works_kept) FROM read_parquet('{ND / 'nodes.parquet'}')").fetchone()[0]
    c.close()
    assert minimum >= 5
    return {"researchers": EXPECTED_N, "links": EXPECTED_E, "minimum_qualifying_works": minimum}


def check_histogram_summary(hist, reported):
    """Independently reproduce JSON summary statistics from integer source counts."""
    assert hist.ndim == 2 and len(hist) > 1 and np.issubdtype(hist.dtype, np.integer)
    assert np.all(hist >= 0) and np.all(hist[:, 0] == 1)
    assert not np.any(hist[:, -1]), "histogram overflow at the maximum distance"
    counts = hist[:, 1:]
    distance = np.arange(1, hist.shape[1])
    source_pairs = counts.sum(axis=1)
    assert np.all(source_pairs > 0)
    aggregate = counts.sum(axis=0)
    pairs = int(aggregate.sum())
    assert pairs == reported["pairs"]
    mean = float((aggregate * distance).sum() / pairs)
    assert abs(mean - reported["mean"]) < 1e-12
    cumulative = np.cumsum(aggregate) / pairs
    assert int(np.searchsorted(cumulative, .5)) + 1 == reported["median"]
    assert int(np.searchsorted(cumulative, .9)) + 1 == reported["p90"]
    expected_pmf = {int(d): float(count / pairs) for d, count in zip(distance, aggregate) if count}
    actual_pmf = {int(d): probability for d, probability in reported["pmf"].items()}
    assert expected_pmf.keys() == actual_pmf.keys()
    assert all(abs(probability - actual_pmf[d]) < 1e-12 for d, probability in expected_pmf.items())
    source_mean = (counts * distance).sum(axis=1) / source_pairs
    se = float(source_mean.std(ddof=1) / np.sqrt(len(hist)))
    assert abs(se - reported["source_mean_se"]) < 1e-12
    return source_mean


def check_fixed_targets(hist, baseline, previous, reported, target_records):
    """Check fixed denominators, per-source distance CDFs and paired mean/SE."""
    assert hist.shape == baseline.shape == previous.shape
    for values in (hist, baseline, previous):
        assert np.all(values.sum(axis=1) == target_records), "fixed-target population changed"
        assert np.all(values[:, 0] == 1), "fixed-target source self-count differs"
    check_histogram_summary(hist, reported)
    assert reported["pairs"] == len(hist) * (target_records - 1)
    assert reported["target_records"] == target_records
    assert reported["target_pairs_per_source"] == target_records - 1
    cumulative = np.cumsum(hist[:, 1:], axis=1)
    assert np.all(cumulative >= np.cumsum(previous[:, 1:], axis=1)), "fixed-target CDF worsened at higher cap"
    assert np.all(cumulative >= np.cumsum(baseline[:, 1:], axis=1)), "fixed-target CDF worsened versus cap50"
    distance = np.arange(1, hist.shape[1])
    changes = ((hist[:, 1:] - baseline[:, 1:]) * distance).sum(axis=1) / (target_records - 1)
    assert np.all(changes <= 0), "fixed-target mean increased"
    paired = reported["paired_cap_minus_50"]
    assert paired["sources"] == len(hist)
    for key, expected in (("mean", changes.mean()), ("se", changes.std(ddof=1) / np.sqrt(len(hist))),
                          ("minimum_source_change", changes.min()), ("maximum_source_change", changes.max())):
        assert abs(paired[key] - float(expected)) < 1e-12, f"paired cap-minus-50 {key} differs"
    for key in ("constant_target_count", "source_self_count_exact",
                "every_source_cdf_nondecreasing_vs_previous_cap", "every_source_cdf_nondecreasing_vs_cap50"):
        assert reported["validation"][key] is True


def cap_parity():
    a2 = json.loads((RESULTS / "a2_distances.json").read_text())
    cap = json.loads((RESULTS / "a2c_team_cap.json").read_text())
    first = json.loads((RESULTS / "a2c_team_cap_100.json").read_text())
    manifest = json.loads((CACHE / "bipartite_manifest.json").read_text())
    sources = np.load(CACHE / "a2_sources.npy")
    assert cap["sources"] == a2["K"] == len(sources) == 1000
    assert cap["researchers"] == EXPECTED_N
    assert cap["network_fingerprint"] == manifest["fingerprint"]
    # Includes input-manifest self-digest/current metadata, streamed node-ID
    # content digest, exact source ordering and current fixed-target mask.
    assert reuse_a2_summary(RESULTS, CACHE, PG, sources, progress=log) is not None
    target_mask = np.load(CACHE / "lcc_all.npy", mmap_mode="r")
    assert target_mask.dtype == np.bool_ and target_mask.shape == (EXPECTED_N,)
    assert np.all(target_mask[sources])
    target_digest = hashlib.sha256(target_mask).hexdigest()
    target_records = int(target_mask.sum())
    for value, count in ((cap, len(sources)), (first, 100)):
        assert value["sources"] == count and value["distance_kernel_version"] == 2
        assert value["network_fingerprint"] == manifest["fingerprint"]
        assert value["baseline_targets"]["sha256"] == target_digest
        assert value["baseline_targets"]["records"] == target_records
        assert value["baseline_targets"]["source_self_excluded_from_summaries"] is True
        assert value["cap50_validation"]["fixed_targets_exact"] is True
        assert value["cap50_validation"]["reachable_count_exact"] is True
        assert value["cap50_validation"]["sources_checked"] == count
    assert cap["cap50_validation"]["full_histogram_exact"] is True
    assert first["derived_from_first_sources"] == len(sources)
    assert first["cap50_validation"]["derived_from_validated_full_run"] is True
    independent_baseline = RESULTS / "a7_remedy_peak25_strict.json"
    baseline_crosscheck = False
    if independent_baseline.exists():
        original_csr = json.loads(independent_baseline.read_text())
        if original_csr["sources"] == 100:
            assert original_csr["all"]["pairs"] == first["cap_50"]["pairs"]
            assert abs(original_csr["all"]["mean"] - first["cap_50"]["mean"]) < 1e-12
            baseline_crosscheck = True
    assert cap["cap_50"]["pairs"] == a2["all_links"]["pairs"]
    assert abs(cap["cap_50"]["mean"] - a2["all_links"]["mean"]) < 1e-12
    previous = baseline_fixed = previous_fixed = None
    for limit, key in ((50, "cap_50"), (100, "cap_100"), (500, "cap_500"), (1_000_000, "any")):
        with np.load(CACHE / f"a2c_hist_cap{limit}.npz", allow_pickle=False) as saved:
            assert np.array_equal(saved["sources"], sources)
            signature = {"graph": manifest["fingerprint"], "sources": sources.tolist(), "cap": limit,
                         "maxd": 64, "kernel_version": 2, "target_mask_sha256": target_digest}
            assert str(saved["signature"]) == hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
            assert int(saved["kernel_version"]) == 2 and str(saved["target_mask_sha256"]) == target_digest
            hist = saved["hist"]
            fixed = saved["hist_fixed"]
        assert hist.shape == fixed.shape == (1000, 65)
        check_histogram_summary(hist, cap[key])
        check_histogram_summary(hist[:100], first[key])
        assert np.all(fixed <= hist), "fixed-target counts exceed all-reachable counts"
        sums = hist.sum(axis=0)
        if limit == 50:
            assert np.array_equal(fixed, hist)
            baseline_fixed = previous_fixed = fixed
            expected = np.zeros(65, np.int64)
            for distance, probability in a2["all_links"]["pmf"].items():
                expected[int(distance)] = round(probability * a2["all_links"]["pairs"])
            sums[0] = 0
            assert np.array_equal(sums, expected)
        check_fixed_targets(fixed, baseline_fixed, previous_fixed, cap[key]["fixed_baseline_targets"], target_records)
        check_fixed_targets(fixed[:100], baseline_fixed[:100], previous_fixed[:100],
                            first[key]["fixed_baseline_targets"], target_records)
        previous_fixed = fixed
        cumulative = np.cumsum(hist, axis=1)
        if previous is not None:
            assert np.all(cumulative >= previous), "adding large-team works increased a source-target distance"
        previous = cumulative
    assert int(np.load(CACHE / "bipartite_w_size.npy", mmap_mode="r").max()) <= 1_000_000
    return {"full_cap50_histogram_exact": True, "sources": 1000, "first100_exact": True,
            "distance_monotonicity_per_source": True, "histogram_overflow": False,
            "fixed_target_records": target_records, "fixed_target_sha256": target_digest,
            "fixed_target_cdf_monotonicity_per_source": True, "paired_fixed_target_means_and_se_exact": True,
            "checkpoint_signatures_verified": True, "distance_kernel_version": 2,
            "independent_a7_first100_baseline_checked": baseline_crosscheck}


def representatives():
    rep = np.load(CACHE / "repair_rep.npy", mmap_mode="r")
    repair = json.loads((RESULTS / "a8_repair.json").read_text())
    assert len(rep) == repair["nodes"]["before"] == EXPECTED_N
    flags = np.zeros(EXPECTED_N, bool)
    with np.load(CACHE / "flags.npz") as values:
        for kind in values.files:
            indices = values[kind]
            assert np.all(indices >= 0) and np.all(indices < EXPECTED_N)
            assert len(np.unique(indices)) == len(indices)
            assert not np.any(flags[indices]), "flag categories overlap despite precedence rule"
            flags[indices] = True
    assert np.array_equal(rep < 0, flags), "removals differ between flags and repair representatives"
    assert int(flags.sum()) == repair["removed_total"]
    assert int(rep.max()) + 1 == repair["nodes"]["after"]
    counts = np.bincount(rep[~flags])
    assert np.all(counts > 0), "repaired indices are not contiguous"
    assert EXPECTED_N - int(flags.sum()) - len(counts) == repair["nodes"]["merged_away"]
    pairs = []
    for file, a, b in (("twin_pairs.csv", "native_node", "latin_node"), ("split_pairs.csv", "node_a", "node_b")):
        for chunk in pd.read_csv(ND / "deposit" / file, chunksize=BATCH):
            joined = chunk.joined.to_numpy(dtype=bool)
            x, y = chunk[a].to_numpy(dtype=np.int64), chunk[b].to_numpy(dtype=np.int64)
            assert np.all((x >= 0) & (x < EXPECTED_N)) and np.all((y >= 0) & (y < EXPECTED_N))
            assert np.array_equal(joined, ~(flags[x] | flags[y]))
            pairs.append(np.column_stack([x[joined], y[joined]]))
    pairs = np.concatenate(pairs) if pairs else np.empty((0, 2), np.int64)
    root = union_pairs(EXPECTED_N, pairs[:, 0], pairs[:, 1])
    unique, expected = np.unique(root[~flags], return_inverse=True)
    assert len(unique) == len(counts) and np.array_equal(rep[~flags], expected), "deposit joins do not reconstruct representatives"
    return {"removed": int(flags.sum()), "repaired_researchers": len(counts),
            "exported_pair_rows_used": len(pairs), "representatives_reconstructed_exactly": True}


def expected_researcher_metadata(connection, attrs, authors_glob, coordinates, batch_size=BATCH):
    """Independent, bounded SQL source join; no exporter helper/cache is reused."""
    return connection.sql(f"""WITH fallback AS (
        SELECT DISTINCT id FROM read_parquet('{authors_glob}')
        WHERE topics[1].display_name = 'Legal and Regulatory Analysis'
    ), layout AS (
        SELECT {id_key_sql('id')} AS author_key, xw, yw FROM read_parquet('{coordinates}')
    ) SELECT a.node_idx, a.on_map AS attributes_on_map,
             (w.author_key IS NOT NULL) AS source_on_map,
             (f.id IS NOT NULL) AS is_fallback_topic, w.xw, w.yw
      FROM read_parquet('{attrs}') a
      LEFT JOIN layout w ON {id_key_sql('a.id')} = w.author_key
      LEFT JOIN fallback f ON a.id = f.id
      ORDER BY a.node_idx""").to_arrow_reader(batch_size)


def aligned_batches(left, right):
    """Pair bounded slices even when independent readers use different boundaries."""
    left, right = iter(left), iter(right)
    a, b = next(left, None), next(right, None)
    ia = ib = 0
    while a is not None and b is not None:
        assert len(a) > 0 and len(b) > 0, "unexpected empty validation batch"
        count = min(len(a) - ia, len(b) - ib)
        yield a.slice(ia, count), b.slice(ib, count)
        ia, ib = ia + count, ib + count
        if ia == len(a):
            a, ia = next(left, None), 0
        if ib == len(b):
            b, ib = next(right, None), 0
    assert a is None and b is None, "deposit/source metadata row counts differ (possibly duplicate layout IDs)"


def check_supplemental_batch(batch, expected):
    """Check boolean semantics and distinguish coordinate nulls from IEEE NaNs."""
    assert np.array_equal(batch.column("node").to_numpy(), expected.column("node_idx").to_numpy()), (
        "source metadata is not one row per deposited node (possibly duplicate layout IDs)")
    for column in ("on_map", "is_fallback_topic"):
        assert batch.column(column).type == pa.bool_() and batch.column(column).null_count == 0
    mapped = batch.column("on_map").to_numpy(zero_copy_only=False)
    assert np.array_equal(mapped, expected.column("attributes_on_map").to_numpy(zero_copy_only=False)), "on_map differs from node attributes"
    assert np.array_equal(mapped, expected.column("source_on_map").to_numpy(zero_copy_only=False)), "on_map differs from coordinate-ID membership"
    fallback = batch.column("is_fallback_topic").to_numpy(zero_copy_only=False)
    assert np.array_equal(fallback, expected.column("is_fallback_topic").to_numpy(zero_copy_only=False)), "fallback-topic flag differs from the first-topic source rule"
    stats = {"fallback_topic_records": int(fallback.sum()), "coordinate_id_matches": int(mapped.sum())}
    for coordinate in ("xw", "yw"):
        actual, source = batch.column(coordinate), expected.column(coordinate)
        assert actual.type == source.type == pa.float64(), "coordinates must retain source float64 precision"
        null = actual.is_null().to_numpy(zero_copy_only=False)
        assert np.array_equal(null, source.is_null().to_numpy(zero_copy_only=False)), f"{coordinate} null mask differs from source"
        assert np.all(null[~mapped]), f"off-map {coordinate} values must be null"
        actual_values = actual.to_numpy(zero_copy_only=False)
        source_values = source.to_numpy(zero_copy_only=False)
        assert np.array_equal(actual_values[~null], source_values[~null], equal_nan=True), f"{coordinate} values differ from source"
        stats[f"{coordinate}_nulls"] = int(null.sum())
        stats[f"{coordinate}_nan_values"] = int(np.isnan(actual_values[~null]).sum())
        stats[f"{coordinate}_infinite_values"] = int(np.isinf(actual_values[~null]).sum())
    return stats


def deposit_rows():
    rep = np.load(CACHE / "repair_rep.npy", mmap_mode="r")
    ids = np.load(PG / "ids.npy", mmap_mode="r")
    leiden = np.load(CACHE / "leiden.npy", mmap_mode="r")
    offsets = np.load(PG / "offsets.npy", mmap_mode="r")
    flag = np.zeros(EXPECTED_N, np.int8)
    labels = {"": 0, "merged": 1, "split": 2, "not_a_person": 3}
    with np.load(CACHE / "flags.npz") as values:
        for kind, code in (("merged", 1), ("split", 2), ("notPerson", 3)):
            flag[values[kind]] = code
    at, on_map, supplemental = 0, 0, {}
    coordinates = D / "coords_web.parquet"
    layout = pq.ParquetFile(coordinates)
    assert layout.schema_arrow.field("xw").type == layout.schema_arrow.field("yw").type == pa.float64()
    db = duckdb.connect()
    db.execute("SET memory_limit='6GB'")
    db.execute("SET threads=4")
    scratch = CACHE / "duck_deposit_validation"
    scratch.mkdir(exist_ok=True)
    db.execute(f"SET temp_directory='{scratch}'")
    db.execute("SET max_temp_directory_size='20GB'")
    expected = None
    try:
        expected = expected_researcher_metadata(db, CACHE / "node_attrs.parquet",
            str(D / "authors_extract/*/*.parquet"), coordinates)
        batches = pq.ParquetFile(ND / "deposit/researchers.parquet").iter_batches(batch_size=BATCH,
            columns=["node", "openalex_id", "works", "coauthors", "community", "flag", "repaired_node", "on_map",
                     "is_fallback_topic", "xw", "yw"])
        for batch, source in aligned_batches(batches, expected):
            for key, value in check_supplemental_batch(batch, source).items():
                supplemental[key] = supplemental.get(key, 0) + value
            end = at + len(batch)
            assert np.array_equal(batch.column("node").to_numpy(), np.arange(at, end))
            numeric = pc.cast(pc.utf8_slice_codeunits(batch.column("openalex_id"), 1), "int64").to_numpy()
            assert np.array_equal(numeric, ids[at:end])
            assert np.all(batch.column("works").to_numpy() >= 5)
            assert np.array_equal(batch.column("coauthors").to_numpy(), np.diff(offsets[at:end + 1]))
            assert np.array_equal(batch.column("community").to_numpy(), leiden[at:end])
            assert np.array_equal(batch.column("repaired_node").to_numpy(), rep[at:end])
            assert np.array_equal(np.array([labels[s] for s in batch.column("flag").to_pylist()]), flag[at:end])
            on_map += int(batch.column("on_map").to_numpy(zero_copy_only=False).sum())
            at = end
    finally:
        if expected is not None:
            expected.close()
        db.close()
    assert at == EXPECTED_N
    core = json.loads((PAPER / "results_min15/a1_basics.json").read_text())["nodes"]
    assert on_map == core, "on_map count differs from archived core"
    assert on_map == layout.metadata.num_rows == supplemental["coordinate_id_matches"], "layout source population differs from mapped records"
    return {"exported_researchers_checked": at, "on_map": on_map, "flags_and_representatives_exact": True,
            "fallback_and_coordinates_match_sources": True, "supplemental_metadata": supplemental}


def csr_and_flags():
    offsets = np.load(PG / "offsets.npy", mmap_mode="r")
    neighbors = np.load(PG / "neighbors.npy", mmap_mode="r")
    mask = np.load(CACHE / "suspect_mask.npy", mmap_mode="r")
    repair = np.load(CACHE / "repair_link_mask.npy", mmap_mode="r")
    flagged = keys_from_parquet(ND / "deposit/flagged_links.parquet")
    repaired = keys_from_parquet(ND / "deposit/flagged_links.parquet", repair_only=True)
    assert len(mask) == len(neighbors) == len(repair)
    counts, matched, repair_counts = 0, 0, 0
    for begin in range(0, len(neighbors), 1_000_000):
        end = min(begin + 1_000_000, len(neighbors))
        dst = neighbors[begin:end]
        assert np.all((dst >= 0) & (dst < EXPECTED_N))
        src = np.searchsorted(offsets, np.arange(begin, end), side="right") - 1
        assert not np.any(src == dst), "CSR contains self loops"
        selected = mask[begin:end]
        chosen = np.asarray(repair[begin:end])
        assert not np.any(chosen & ~np.asarray(selected)), "repair links must be suspect links"
        counts += int(selected.sum())
        repair_counts += int(chosen.sum())
        if selected.any():
            keys = pack(np.minimum(src[selected], dst[selected]), np.maximum(src[selected], dst[selected]))
            assert np.all(contains(flagged, keys)), "CSR flags disagree with deposited flagged links"
            matched += len(keys)
        if chosen.any():
            keys = pack(np.minimum(src[chosen], dst[chosen]), np.maximum(src[chosen], dst[chosen]))
            assert np.all(contains(repaired, keys)), "repair mask disagrees with the deposited repair column"
    expected = json.loads((RESULTS / "a1_basics.json").read_text())["flagged_links"]
    assert counts == matched == 2 * len(flagged) == 2 * expected
    assert repair_counts == 2 * len(repaired)
    return {"directed_neighbors_checked": len(neighbors), "flagged_undirected_links": len(flagged),
            "repair_undirected_links": len(repaired)}


def checksums():
    directory = ND / "deposit"
    entries = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        path = directory / relative
        assert path.resolve().parent == directory.resolve(), "checksum entry leaves deposit"
        assert path.is_file()
        assert digest(path) == expected, f"checksum mismatch: {relative}"
        entries[relative] = path.stat().st_size
    actual = {p.name for p in directory.iterdir() if p.is_file() and p.name != "SHA256SUMS"}
    assert set(entries) == actual, "checksum manifest does not cover every deposited file"
    deposited_links, source_links = directory / "links.parquet", G / "edges_int32.parquet"
    assert deposited_links.samefile(source_links) or digest(deposited_links) == digest(source_links), "deposit links differ from analyzed graph"
    return {"files": len(entries), "bytes": sum(entries.values()), "links_match_analyzed_graph": True}


def qualifying_work_counts(connection, works_glob):
    # Solo-authored works affect eligibility and the Section 2 census, even
    # though they do not enter the projected coauthor graph.
    query = qualifying_works_sql(str(works_glob), dedupe=False)
    rows, unique, maximum, multiple = connection.execute(f"""SELECT count(*), count(DISTINCT work_id),
        max(n_authors), count(*) FILTER (WHERE n_authors >= 2) FROM ({query})""").fetchone()
    assert rows == unique, f"{rows - unique:,} duplicate or null qualifying work IDs invalidate author/edge-incidence counts"
    assert maximum <= 1_000_000, "uncapped sensitivity sentinel excludes an eligible work"
    return {"qualifying_work_rows": rows, "unique_work_ids": unique,
            "qualifying_multi_author_rows": multiple, "maximum_team_size": maximum,
            "includes_solo_authored_works": True}


def census_provenance():
    def encoded_digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    census = json.loads((RESULTS / "a6_quality.json").read_text())["qualifying_author_census"]
    counts = json.loads((CACHE / "qualifying_author_counts.provenance.json").read_text())
    population = json.loads((CACHE / "qualifying_author_census.provenance.json").read_text())
    assert counts["fingerprint"] == census["counts_fingerprint"] == encoded_digest(counts["identity"])
    assert same_identity(counts["identity"]["works_inputs"],
                         [file_identity(p) for p in sorted((D / "works_authorships").glob("*/*.parquet"))])
    assert counts["identity"]["query"] == qualifying_works_sql("INPUT", dedupe=False)
    assert population["fingerprint"] == census["population_fingerprint"] == encoded_digest(population["identity"])
    assert population["identity"]["counts_fingerprint"] == counts["fingerprint"]
    assert same_identity(population["identity"]["authors_inputs"],
                         [file_identity(p) for p in sorted((D / "authors_extract").glob("*/*.parquet"))])
    assert same_identity(population["identity"]["nodes"], file_identity(ND / "nodes.parquet"))
    assert population["identity"]["person_predicate"] == is_person_sql("c.works_kept")
    assert census["count_cache_sha256"] == counts["cache"]["sha256"] == digest(CACHE / "qualifying_author_counts.parquet")
    threshold = census["threshold"]
    assert threshold["min_works"] == population["identity"]["min_works"] == 5
    assert threshold["retained_author_ids"] == EXPECTED_N
    assert threshold["eligible_author_ids"] == sum(threshold[k] for k in (
        "retained_author_ids", "excluded_nonperson_records", "missing_author_records"))
    assert counts["cache"]["author_rows"] == census["author_ids_with_qualifying_work"]
    assert sum(row["author_ids"] for row in census["work_count_bands"].values()) == census["author_ids_with_qualifying_work"]
    assert sum(row["authorships"] for row in census["work_count_bands"].values()) == census["authorships_total"]
    assert abs(threshold["retained_author_share"] - EXPECTED_N / census["author_ids_with_qualifying_work"]) < 1e-12
    assert abs(threshold["retained_authorship_share"] - threshold["retained_authorships"] / census["authorships_total"]) < 1e-12
    for key in ("node_works_kept_exact", "retained_id_set_exact", "eligible_ids_partition_exact", "eligible_author_records_unique"):
        assert census["validation"][key] is True
    return {"author_ids": census["author_ids_with_qualifying_work"], "retained_author_ids": EXPECTED_N,
            "counts_cache_checksum_verified": True, "current_input_provenance_verified": True}


def work_uniqueness(scratch):
    scratch.mkdir(parents=True, exist_ok=True)
    c = duckdb.connect()
    c.execute("SET memory_limit='6GB'")
    c.execute("SET threads=4")
    c.execute(f"SET temp_directory='{scratch}'")
    c.execute("SET max_temp_directory_size='30GB'")
    try:
        result = qualifying_work_counts(c, D / "works_authorships/*/*.parquet")
    finally:
        c.close()
    result["author_census"] = census_provenance()
    return result


def reconstruct(scratch):
    free = shutil.disk_usage(CACHE).free
    assert free >= 10_000_000_000, "need at least 10 GB scratch space for exact edge reconstruction"
    rep = np.load(CACHE / "repair_rep.npy", mmap_mode="r")
    flagged = keys_from_parquet(ND / "deposit/flagged_links.parquet", repair_only=True)
    batches = pq.ParquetFile(ND / "deposit/links.parquet").iter_batches(batch_size=BATCH, columns=["src", "dst", "weight"])
    result = reconstruct_edge_count(batches, rep, flagged, scratch)
    expected = json.loads((RESULTS / "a8_repair.json").read_text())["links_after"]
    assert result["source_links"] == EXPECTED_E
    assert result["repaired_links"] == expected, f"reconstructed {result['repaired_links']} links, A8 reports {expected}"
    return result


def manuscript(required, report):
    text = (PAPER / "draft2.md").read_text()
    markers = [line.strip() for line in text.splitlines() if re.search(r"\{\{\s*TODO\b", line)]
    details = {}
    if required:
        assert not markers, f"{len(markers)} manuscript lines still contain numeric TODO placeholders"
        details["figures"] = validate_figures(ROOT)
        details["build"] = validate_build_receipt(ROOT)
    elif markers:
        report.warnings.append(f"{len(markers)} manuscript lines still contain TODO markers; final manuscript validation not requested.")
    if "[TODO, authors:" in text:
        report.warnings.append("Author/publication TODO remains in the manuscript; it is not a missing numerical result.")
    return {"numeric_todo_lines": len(markers), "build_required": required, **details}


def repaired_center_selection():
    """JSON-only corroboration of A8/A8c's component proof and candidate selection."""
    repaired = json.loads((RESULTS / "a8_repair.json").read_text())
    center = json.loads((RESULTS / "a8c_center.json").read_text())
    groups = json.loads((RESULTS / "a9b_repaired.json").read_text())
    return check_repaired_center_results(repaired, center, groups)


def check_repaired_center_results(repaired, center, groups):
    nodes = repaired["nodes"]["after"]
    selection = repaired["center_selection"]
    assert set(selection) == {"selection_version", "sources", "surviving_sources_total",
                              "excluded_sources_outside_component", "component_size", "candidates"}
    assert selection == {key: center[key] for key in selection}
    assert selection["selection_version"] == 2
    assert 0 < selection["sources"] <= selection["surviving_sources_total"]
    assert selection["surviving_sources_total"] == repaired["all_sources"]["sources_after"]
    assert selection["excluded_sources_outside_component"] == selection["surviving_sources_total"] - selection["sources"]
    assert nodes / 2 < selection["component_size"] <= nodes
    assert selection["component_size"] == repaired["diameter_lower_bound"]["giant_component"]
    assert center["candidates"] == len(center["most_central"]) == 200
    assert math.isfinite(center["estimate_cut"]) and math.isfinite(center["top20_largest_estimate"])
    for rank, row in enumerate(center["most_central"], 1):
        assert row["rank"] == rank
        assert math.isfinite(row["mean_distance"]) and row["mean_distance"] >= 0
        assert math.isfinite(row["estimate"]) and row["estimate"] >= 0
    assert all(a["mean_distance"] <= b["mean_distance"] for a, b in zip(center["most_central"], center["most_central"][1:]))
    component = groups["component"]
    assert component["verified_largest"] is True and component["nodes"] == nodes
    assert component["size"] == selection["component_size"]
    for kind in ("field", "country"):
        group = groups[kind]
        n_groups = len(group["names"])
        sources = np.asarray(group["sources_per_group"])
        pairs = np.asarray(group["reachable_pairs"])
        means = np.asarray(group["mean_distance"])
        assert sources.shape == (n_groups,) and np.all((sources >= 1) & (sources <= 100))
        assert np.issubdtype(sources.dtype, np.integer)
        assert pairs.shape == means.shape == (n_groups, n_groups)
        assert np.issubdtype(pairs.dtype, np.integer) and np.all(pairs > 0)
        assert np.all(np.isfinite(means) & (means >= 0))
    return {**selection, "candidate_rows_finite": True, "a9b_component_agrees": True,
            "field_country_source_counts_and_pair_denominators_valid": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-reconstruction", action="store_true")
    parser.add_argument("--require-manuscript", action="store_true")
    args = parser.parse_args()
    report = Report()
    if args.require_manuscript:
        def capture_delivery():
            report.delivery_fingerprint = validation_fingerprint(ROOT)
            value = report.delivery_fingerprint
            return {"sha256": value["sha256"], "content_hashed_files": len(value["content_hashed"]),
                    "metadata_only_files": len(value["metadata_only"])}
        report.run("delivery fingerprint capture", capture_delivery)
    dependency_result = report.run("dependency completeness and finite JSON", dependencies)
    if dependency_result and dependency_result["allowed_missing_metadata_nan"]:
        paths = dependency_result["allowed_missing_metadata_nan"]
        report.warnings.append(f"{len(paths)} NaN display metadata values were allowed only at documented paths (off-map coordinates or unknown country/field); exact paths are in the JSON check details. Core numerical statistics remain strictly finite.")
    report.run("five-work population and CSR dimensions", population)
    report.run("cap-50 parity and sensitivity checkpoints", cap_parity)
    report.run("flags and exported joins reconstruct representatives", representatives)
    report.run("repaired center selection", repaired_center_selection)
    if not args.quick:
        report.run("deposited researcher rows", deposit_rows)
        report.run("CSR neighbors and deposited flagged links", csr_and_flags)
        report.run("deposit SHA256 checksums", checksums)
        with tempfile.TemporaryDirectory(prefix="validate-min5-", dir=CACHE) as temporary:
            scratch = Path(temporary)
            report.run("unique eligible work identifiers", lambda: work_uniqueness(scratch / "duck"))
            if not args.skip_reconstruction:
                report.run("exact repaired edge reconstruction", lambda: reconstruct(scratch / "edges"))
    else:
        report.warnings.append("Quick mode omits full graph/deposit/works scans and all checksums; it is not final run validation.")
    if args.skip_reconstruction:
        report.warnings.append("Exact repaired edge reconstruction was explicitly skipped.")
    report.run("manuscript placeholders and build", lambda: manuscript(args.require_manuscript, report))
    if args.require_manuscript:
        report.run("delivery inputs unchanged during validation", lambda: verify_fingerprint(ROOT, report.delivery_fingerprint))
    sys.exit(1 if report.save(args) else 0)


if __name__ == "__main__":
    main()
