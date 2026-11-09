"""Largest-component candidate screening and verified reuse of A8's full center calculation."""
import hashlib

import numpy as np

from common import CACHE, PG

CENTER_SELECTION_VERSION = 2


def source_with_largest_reach(sources, histograms):
    """Pick a sampled component seed using exact reached counts (histograms exclude self)."""
    if len(sources) == 0 or len(histograms) != len(sources):
        raise ValueError("The repaired center needs nonempty source-aligned distance histograms")
    return int(sources[np.argmax(histograms.sum(axis=1) + 1)])


def verified_largest_component(reach):
    """A component containing more than half of all nodes is uniquely the largest."""
    component = reach >= 0
    size = int(component.sum())
    if 2 * size <= len(reach):
        raise ValueError(f"Cannot establish the repaired largest component: reached {size:,} of {len(reach):,} nodes, "
                         "not more than half. Run a full component census before reporting a giant-component result.")
    return component


def screening_candidates(sources, reach, sums, counts, candidates=200):
    """Screen only in the proven largest component, excluding other sampled components.

    A8's fused sums/counts can contain sources from several components. Outside-component
    sources contribute nothing to nodes inside this component, so dividing those sums by the
    included source count exactly matches a fresh traversal from only the included sources.
    """
    component = verified_largest_component(reach)
    if sums.shape != reach.shape or counts.shape != reach.shape:
        raise ValueError("Center sums/counts must align with repaired node indices")
    included = int(component[sources].sum())
    if included == 0:
        raise ValueError("No surviving baseline sources lie in the repaired largest component")
    if np.any(counts[component] != included):
        raise ValueError("Largest-component target counts differ from the included screening-source count")
    estimates = np.where(component & (counts == included), sums / included, np.inf)
    eligible = np.isfinite(estimates)
    if int(eligible.sum()) < candidates:
        raise ValueError(f"Only {int(eligible.sum())} finite largest-component candidates; {candidates} required")
    best = np.argsort(estimates)[:candidates].astype(np.int64)
    metadata = {"selection_version": CENTER_SELECTION_VERSION, "sources": included,
                "surviving_sources_total": int(len(sources)),
                "excluded_sources_outside_component": int(len(sources)) - included,
                "component_size": int(component.sum())}
    return estimates, best, metadata


def reusable_center(previous, expected_signature, surviving_sources_total, nodes, candidates=200):
    """Reject pre-selection-version centers without invalidating reusable fused BFS summaries."""
    try:
        result = previous["result"]
        return (previous["signature"] == expected_signature
                and result["selection_version"] == CENTER_SELECTION_VERSION
                and result["surviving_sources_total"] == surviving_sources_total
                and 0 < result["sources"] <= surviving_sources_total
                and result["excluded_sources_outside_component"] == surviving_sources_total - result["sources"]
                and nodes / 2 < result["component_size"] <= nodes
                and result["candidates"] == len(result["most_central"]) == candidates
                and result["component_size"] >= candidates
                and np.isfinite(result["estimate_cut"]) and np.isfinite(result["top20_largest_estimate"])
                and all(np.isfinite(row["mean_distance"]) and np.isfinite(row["estimate"])
                        for row in result["most_central"]))
    except (KeyError, TypeError, ValueError):
        return False


def signature(rep, sources):
    # Repair mapping contains both exclusions and contractions. Graph and name
    # identities also matter because the cached result includes researcher names.
    return {"version": 1, "repair_sha256": hashlib.sha256(rep).hexdigest(),
            "sources_sha256": hashlib.sha256(sources).hexdigest(),
            "skip_mask": [(CACHE / "repair_link_mask.npy").stat().st_size,
                          (CACHE / "repair_link_mask.npy").stat().st_mtime_ns],
            "graph": {name: [p.stat().st_size, p.stat().st_mtime_ns]
                      for name in ("offsets.npy", "neighbors.npy", "ids.npy", "names.bin", "names_offsets.npy")
                      for p in [PG / name]}}
