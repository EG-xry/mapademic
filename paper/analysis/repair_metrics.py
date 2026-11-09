"""Fail clearly when repair distance summaries would hide clipped or undefined observations."""
import numpy as np

from kernels import summary


def guarded_summary(histograms, context):
    """Summarize reached non-self pairs only when their distance bins remain exact."""
    if histograms.ndim != 2 or histograms.shape[1] < 2:
        raise ValueError(f"{context}: invalid distance histogram shape")
    if histograms[:, -1].any():
        raise ValueError(f"{context}: distances reached the {histograms.shape[1] - 1}-bin limit; "
                         "increase the histogram limit before reporting exact means")
    if histograms.sum() <= 0:
        raise ValueError(f"{context}: no reachable non-self source-target pairs")
    return summary(histograms)


def verify_cached_summary(result, context, maxd=64):
    """Old completed community caches must also have an empty terminal distance bin."""
    if result["pmf"].get(str(maxd), result["pmf"].get(maxd, 0)):
        raise ValueError(f"{context}: cached distances reached the {maxd}-bin limit; recompute with a larger limit")
    if result["pairs"] <= 0:
        raise ValueError(f"{context}: cached summary has no reachable non-self source-target pairs")


def group_summary(distance_sums, pair_counts, source_groups, names):
    """Report pooled group means and denominators; an unobserved cell is never distance zero."""
    groups = len(names)
    if (groups == 0 or distance_sums.shape != pair_counts.shape
            or distance_sums.shape != (len(source_groups), groups)):
        raise ValueError("Invalid repaired group-distance matrix dimensions")
    source_counts = np.bincount(source_groups, minlength=groups)
    if len(source_counts) != groups or np.any(source_counts == 0):
        missing = [names[i] for i in range(groups) if source_counts[i] == 0]
        raise ValueError(f"No repaired-component sources for groups: {missing}")
    pairs = np.array([pair_counts[source_groups == i].sum(axis=0) for i in range(groups)])
    if np.any(pairs <= 0):
        i, j = np.argwhere(pairs <= 0)[0]
        raise ValueError(f"No reachable non-self target pairs for repaired group cell {names[i]!r} -> {names[j]!r}")
    sums = np.array([distance_sums[source_groups == i].sum(axis=0) for i in range(groups)])
    mean = sums / pairs
    return {"names": names, "mean_distance": mean.round(4).tolist(),
            "within": {name: float(mean[i, i]) for i, name in enumerate(names)},
            "sources_per_group": source_counts.tolist(), "reachable_pairs": pairs.tolist()}
