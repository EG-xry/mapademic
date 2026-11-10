"""A2b: sampling error of the mean distance: per-source mean distances from the 100 A2 sources (delivered network)."""
import numpy as np

from common import CACHE, load_csr, save_result
from kernels import histograms

offsets, neighbors = load_csr(mmap=False)
h = histograms(offsets, neighbors, np.load(CACHE / "a2_sources.npy"), 64).astype(float)
per = (h * np.arange(h.shape[1])).sum(axis=1) / h.sum(axis=1)
out = {"sources": len(per), "mean_of_source_means": float(per.mean()), "sd_of_source_means": float(per.std(ddof=1)),
       "standard_error": float(per.std(ddof=1) / np.sqrt(len(per))), "min": float(per.min()), "max": float(per.max())}
save_result("a2b_se", out)
print(out)
