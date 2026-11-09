"""A9: how far apart are fields and countries, and does the small world hang on its hubs?

1. Field x field: from 100 random giant-component members of each OpenAlex field (an author's first topic's
   field; authors on the fallback topic left out), the mean distance to the members of every field. Same
   for the 16 countries with the most researchers.
2. Hub removal: mean distance and giant-component share after removing the top 0.01%, 0.1%, 1% and 5% of
   researchers by degree, against removing as many at random (from the first 100 A2 sources, minus removed ones).
Both on the network as delivered.

Standard errors treat each source as one cluster of pairs (the linearized standard error of a ratio of sums), and
the per-source sums are kept in cache/a9_per_source.npz for paired comparisons. hub_removal_detail adds, for each
removal, the surviving sources, their mean on the unmodified network (from A2c's saved per-source counts), the share
of surviving giant-component members they still reach, and four more random draws per share; hub_removal keeps the
published rows (top and the first random draw).
"""
import gc

import numpy as np
import pyarrow.parquet as pq

from common import CACHE, D, con, load_csr, log, save_result
from kernels import _bfs, contract, group_distances, histograms, summary

PER_GROUP = 100
EXTRA_DRAWS = 4
rng = np.random.default_rng(20261025)
extra_rng = np.random.default_rng(20261108)          # the extra random removals, so rng's published draws stay put


def ratio_se(s, c):
    """Standard error of s.sum() / c.sum() with each row (source) as one cluster."""
    m, total = len(s), c.sum()
    if m < 2 or total == 0:
        return 0.0
    r = s.sum() / total
    return float(np.sqrt(m / (m - 1) * ((s - r * c) ** 2).sum()) / total)


def matrix(offsets, neighbors, labels, lcc, key, top=None, saved=None):
    """Mean distance between groups of `labels` (a pandas Series by node), rows = source groups, with standard
    errors and the per-source sums and counts behind them. The sources are always drawn (so the random stream stays
    as published); their sums come from `saved`, an earlier run's a9_per_source.npz, when it holds exactly these
    sources, and from fresh breadth-first searches otherwise."""
    names = labels.value_counts().index.tolist()
    names = [x for x in names if x][:top] if top else [x for x in names if x]
    code = {x: i for i, x in enumerate(names)}
    group = labels.map(code).fillna(-1).astype(np.int64).to_numpy()
    sources, src_group = [], []
    for i in range(len(names)):
        members = np.flatnonzero((group == i) & lcc)
        pick = rng.choice(members, size=min(PER_GROUP, len(members)), replace=False)
        sources.extend(pick.tolist())
        src_group.extend([i] * len(pick))
    sources, src_group = np.array(sources, np.int64), np.array(src_group)
    if (saved is not None and np.array_equal(saved.get(f"{key}_sources"), sources)
            and np.array_equal(saved.get(f"{key}_src_group"), src_group) and list(saved[f"{key}_names"]) == names):
        s, c = saved[f"{key}_s"], saved[f"{key}_c"]
        log(f"{key}: the drawn sources match a9_per_source.npz; reusing their sums")
    else:
        s, c = group_distances(offsets, neighbors, sources, group, len(names))
    mean = np.array([[s[src_group == i, j].sum() / max(c[src_group == i, j].sum(), 1) for j in range(len(names))]
                     for i in range(len(names))])
    se = np.array([[ratio_se(s[src_group == i, j], c[src_group == i, j]) for j in range(len(names))]
                   for i in range(len(names))])
    return names, mean, se, {"sources": sources, "src_group": src_group, "s": s, "c": c}


offsets, neighbors = load_csr(mmap=True)                  # read-only maps: clean pages, dropped rather than swapped
n = len(offsets) - 1
deg = np.diff(offsets)
attrs = pq.read_table(CACHE / "node_attrs.parquet", columns=["node_idx", "id", "field", "country"]).to_pandas()
fallback = con().execute(f"""SELECT id FROM read_parquet('{D}/authors_extract/*/*.parquet')
                             WHERE topics[1].display_name = 'Legal and Regulatory Analysis'""").fetchdf().id
attrs.loc[attrs.id.isin(set(fallback)), "field"] = None       # the fallback topic says nothing about the field
log(f"{attrs.field.isna().sum():,} authors without a usable field")
lcc = np.load(CACHE / "lcc_all.npy")
out = {}
per_source = {}
saved_path = CACHE / "a9_per_source.npz"
saved = None
if saved_path.exists():
    with np.load(saved_path) as z:
        saved = {k: z[k] for k in z.files}
for key, top in (("field", None), ("country", 16)):
    names, mean, se, raw = matrix(offsets, neighbors, attrs[key].fillna(""), lcc, key, top, saved)
    out[key] = {"names": names, "mean_distance": mean.round(4).tolist(),
                "within": {nm: float(mean[i, i]) for i, nm in enumerate(names)},
                "se": se.round(4).tolist(), "within_se": {nm: float(se[i, i]) for i, nm in enumerate(names)}}
    per_source.update({f"{key}_{k}": v for k, v in raw.items()})
    log(key, {nm: round(float(mean[i, i]), 2) for i, nm in enumerate(names)})
    log(f"{key} within-group standard errors", {nm: round(float(se[i, i]), 3) for i, nm in enumerate(names)})
np.savez_compressed(saved_path, **per_source,
                    field_names=np.array(out["field"]["names"]), country_names=np.array(out["country"]["names"]))
del attrs, fallback, saved, per_source
gc.collect()
sources = np.load(CACHE / "a2_sources.npy")[:100]
saved = np.load(CACHE / "a2c_hist_cap50.npz")
assert (saved["sources"][:100] == sources).all(), "A2c's saved counts are for other sources"
before = saved["hist"][:100].astype(np.float64)
before[:, 0] = 0                                    # distance 0 is the source itself
steps = np.arange(before.shape[1])
NO_SKIP = np.zeros(len(neighbors), np.bool_)
out["hub_removal"], out["hub_removal_detail"] = [], []
order = np.argsort(-deg, kind="stable")             # ranked once, on the network as delivered
for share in (0.0001, 0.001, 0.01, 0.05):
    k = int(round(share * n))
    draws = [("top", 0, order[:k]), ("random", 0, rng.choice(n, k, replace=False))]
    draws += [("random", r, extra_rng.choice(n, k, replace=False)) for r in range(1, EXTRA_DRAWS + 1)]
    for how, realization, removed_nodes in draws:
        keep = np.ones(n, bool)
        keep[removed_nodes] = False
        rep = np.where(keep, np.cumsum(keep) - 1, -1).astype(np.int64)
        off2, nb2 = contract(offsets, neighbors, NO_SKIP, rep, int(keep.sum()))
        alive = keep[sources]
        src = rep[sources[alive]]
        h = histograms(off2, nb2, src, 64)
        res = summary(h)
        reach = np.empty(len(off2) - 1, np.int16)
        reached = _bfs(off2, nb2, int(src[0]), reach, np.empty(len(off2) - 1, np.int32))
        row = {"share": share, "removed": k, "how": how, "links_removed_share": float(deg[removed_nodes].sum() / deg.sum()),
               "mean": res["mean"], "median": res["median"], "p90": res["p90"], "giant_share": float(reached / n)}
        if realization == 0:
            out["hub_removal"].append(row)
        b = before[alive]
        kept_giant = int((lcc & keep).sum())
        detail = {**row, "realization": realization, "sources": int(alive.sum()),
                  "baseline_mean": float((b * steps).sum() / b.sum()),
                  "targets_reached_share": float(h[:, 1:].sum() / (alive.sum() * (kept_giant - 1))),
                  "giant_members_kept": kept_giant}
        out["hub_removal_detail"].append(detail)
        log(detail)
        del off2, nb2, h, reach, keep, rep                  # one contracted copy at a time
        gc.collect()
save_result("a9_structure", out)
