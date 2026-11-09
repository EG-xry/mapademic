"""A7b: paired uncertainty for A7. The variants share their sources, so differences between them are judged by
the spread of per-source differences, not by the (much larger) spread of per-source means. Uses the merged hubs,
matched merged hubs and controls A7 saved (cache/a7_sets_<tag>.npz) and reports, for each comparison, the mean
per-source difference in mean distance and its standard error.

    python a7b_paired.py peak25|peak50|peak25_strict
"""
import sys

import numpy as np

from common import CACHE, load_csr, log, save_result, repair_link_mask
from kernels import contract, histograms

TAG = sys.argv[1] if len(sys.argv) > 1 else "peak25"
offsets, neighbors = load_csr(mmap=False)
n = len(offsets) - 1
skip = np.asarray(repair_link_mask(offsets, neighbors))
sets = np.load(CACHE / f"a7_sets_{TAG}.npz")
merged, matched, ctrl = sets["merged"], sets["matched"], sets["control"]
sources = np.load(CACHE / "a2_sources.npy")[:100]          # the original 100 (A2)


def per_source(removed, use_skip):
    """Per-source mean distance on the network with `removed` nodes (and flagged links) dropped."""
    if not use_skip:
        assert not removed.any(), "the delivered baseline must not remove researchers"
        h = histograms(offsets, neighbors, sources, 64).astype(float)
        return (h * np.arange(h.shape[1])).sum(axis=1) / h.sum(axis=1)
    keep = ~removed
    rep = np.where(keep, np.cumsum(keep) - 1, -1).astype(np.int64)
    off, nb = contract(offsets, neighbors, skip, rep, int(keep.sum()))
    h = histograms(off, nb, rep[sources], 64).astype(float)
    return (h * np.arange(h.shape[1])).sum(axis=1) / h.sum(axis=1)


none = np.zeros(n, bool)
rm = {k: none.copy() for k in ("merges", "matched", "control")}
rm["merges"][merged] = True
rm["matched"][matched] = True
rm["control"][ctrl] = True
sources = sources[~(rm["merges"][sources] | rm["control"][sources])]          # as A7 does
means = {"all": per_source(none, False), "flagged": per_source(none, True)}
for k, r in rm.items():
    means[k] = per_source(r, True)
    log(k, float(means[k].mean()))


def diff(a, b):
    d = means[a] - means[b]
    return {"mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d)))}


out = {"tag": TAG, "mean": {k: float(v.mean()) for k, v in means.items()},
       "se_of_mean": {k: float(v.std(ddof=1) / np.sqrt(len(v))) for k, v in means.items()},
       "flagged_minus_all": diff("flagged", "all"), "merges_minus_flagged": diff("merges", "flagged"),
       "matched_minus_flagged": diff("matched", "flagged"), "control_minus_flagged": diff("control", "flagged"),
       "control_minus_matched": diff("control", "matched")}
save_result(f"a7b_paired_{TAG}", out)
log(out)
