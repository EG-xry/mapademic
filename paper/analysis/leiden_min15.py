"""Our Leiden implementation (leiden.py) on the map's 15-work network.

The paper's 15-work communities come from the map's GPU run; the five-work ones from leiden.py. Running leiden.py on
the 15-work network separates the effect of the population threshold from that of the implementation. Same settings
as the five-work run: resolution 1, beta 0.01, ten iterations, seed 20261029.

Writes cache_min15/leiden_cpu.npy (a new file; the archived 15-work results stay untouched) and
paper/results/leiden_min15_cpu.json.
"""
import json
import os

os.environ["NET"] = "min15"

import numpy as np  # noqa: E402

from common import CACHE, G, PAPER, load_ids, log  # noqa: E402
from leiden import leiden, weighted_csr  # noqa: E402

assert CACHE.name == "cache_min15", CACHE
n = len(load_ids())
offsets, nbr, w = weighted_csr(G / "edges_int32.parquet", n)
log(f"weighted CSR: {n:,} nodes, {len(nbr) // 2:,} links")
# Keep the 5.9 GB input graph as read-only memory maps: the same values, but clean pages the system can drop
# under pressure instead of swapping (a first attempt was stopped by the swap-growth guard). Removed at the end.
scratch = {name: CACHE / f"leiden_cpu_{name}.npy" for name in ("offsets", "nbr", "w")}
for name, arr in (("offsets", offsets), ("nbr", nbr), ("w", w)):
    np.save(scratch[name], arr)
del offsets, nbr, w
offsets, nbr, w = (np.load(scratch[name], mmap_mode="r") for name in ("offsets", "nbr", "w"))
memb, q = leiden(offsets, nbr, w, gamma=1.0, log=log,
                 checkpoint=lambda m, q, p: np.save(CACHE / "leiden_cpu_checkpoint.npy", m.astype(np.int32)))
np.save(CACHE / "leiden_cpu.npy", memb.astype(np.int32))
sizes = np.bincount(memb)
out = {"network": "min15", "resolution": 1.0, "beta": 0.01, "passes": 10, "seed": 20261029, "modularity": float(q),
       "nodes": int(n), "communities": int(len(sizes)), "communities_1000plus": int((sizes >= 1000).sum()),
       "share_in_1000plus": float(sizes[sizes >= 1000].sum() / n), "largest": sizes[:40].tolist()}
(PAPER / "results" / "leiden_min15_cpu.json").write_text(json.dumps(out, indent=1))
del offsets, nbr, w
for path in scratch.values():
    path.unlink()
log(f"modularity {q:.4f}; {len(sizes):,} communities, {(sizes >= 1000).sum()} with 1,000+ members")
