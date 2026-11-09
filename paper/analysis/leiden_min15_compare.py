"""Threshold or implementation? Three partitions of the same 8,396,458 researchers with at least 15 works:

  gpu      the map's communities (the 15-work partition used in the paper's comparison)
  cpu15    leiden.py run on the 15-work network (leiden_min15.py)
  cpu5     leiden.py's five-work communities, restricted to these researchers

For each: modularity where available, communities of 1,000+ members, national communities (a country holding at
least half of the members with a known country, among communities where at least a quarter have one), and Fig. 5's
cell, the share of a country's researchers in a field who sit in a community their own country leads (as a5_clusters,
dust communities left out; dust as in attrs.py: under 1,000 members, or three quarters without an institution and a
median of at most 10 citations). Agreement between partitions: normalized mutual information and adjusted Rand index.

Writes paper/results/leiden_min15_compare.json and paper/cache/leiden_min15_field_country.csv.
"""
import json
import os

os.environ["NET"] = "min15"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from common import CACHE, D, PAPER, RESULTS, log  # noqa: E402

assert CACHE.name == "cache_min15", CACHE
FIELDS = ["Medicine", "Physics and Astronomy", "Mathematics", "Earth and Planetary Sciences", "Engineering",
          "Computer Science", "Chemistry", "Social Sciences"]
MIN_CELL = 2000

attrs = pq.read_table(CACHE / "node_attrs.parquet",
                      columns=["node_idx", "id", "field", "country", "institution", "cited", "community", "is_dust"]).to_pandas()
n = len(attrs)
assert np.array_equal(attrs.node_idx.to_numpy(), np.arange(n))
ids15 = np.load(D / "pathgraph" / "ids.npy").astype(np.int64)
ids5 = np.load(D / "min5" / "pathgraph" / "ids.npy").astype(np.int64)
order5 = np.argsort(ids5)
pos = np.searchsorted(ids5, ids15, sorter=order5)
where5 = order5[np.minimum(pos, len(ids5) - 1)]
assert np.array_equal(ids5[where5], ids15), "a 15-work researcher is missing from the five-work network"
parts = {"gpu": attrs.community.to_numpy().astype(np.int64),
         "cpu15": np.load(CACHE / "leiden_cpu.npy").astype(np.int64),
         "cpu5": np.load(PAPER / "cache" / "leiden.npy")[where5].astype(np.int64)}
log({k: int(len(np.unique(v))) for k, v in parts.items()})


def dense(x):
    return np.unique(x, return_inverse=True)[1]


def agreement(a, b):
    a, b = dense(a), dense(b)
    key = a * (b.max() + 1) + b
    _, nab = np.unique(key, return_counts=True)
    na, nb = np.bincount(a), np.bincount(b)
    N = float(len(a))
    h = lambda c: -np.sum((c / N) * np.log(c / N))
    pa = na[a] / N
    pb = nb[b] / N
    # mutual information from the contingency cells
    uk, inv = np.unique(key, return_inverse=True)
    first = np.zeros(len(uk), np.int64)
    first[inv] = np.arange(len(a))
    mi = np.sum((nab / N) * np.log((nab / N) / (pa[first] * pb[first])))
    nmi = 2 * mi / (h(na) + h(nb))
    c2 = lambda c: (c.astype(np.float64) * (c - 1) / 2).sum()
    sab, sa, sb, tot = c2(nab), c2(na), c2(nb), N * (N - 1) / 2
    expected = sa * sb / tot
    ari = (sab - expected) / (0.5 * (sa + sb) - expected)
    return {"nmi": float(nmi), "ari": float(ari)}


def describe(name, comm):
    df = attrs[["field", "country", "institution", "cited"]].copy()
    df["community"] = comm
    size = df.community.value_counts()
    known = df.country.notna().groupby(df.community).mean()
    cc = df.dropna(subset=["country"]).groupby(["community", "country"], observed=True).size().rename("k").reset_index()
    known_n = cc.groupby("community").k.sum()
    top = cc.sort_values(["community", "k"], ascending=[True, False]).drop_duplicates("community").set_index("community")
    tab = pd.DataFrame({"size": size, "known": known})
    tab["top_country"] = top.country
    tab["top_share"] = top.k / known_n
    if name == "gpu":
        dust = attrs.is_dust.fillna(True).to_numpy()
    else:
        noinst = df.institution.isna().groupby(df.community).mean()
        med = df.cited.groupby(df.community).median()
        dust_comm = (size < 1000) | ((noinst.reindex(size.index) >= 0.75) & (med.reindex(size.index) <= 10))
        dust = df.community.map(dust_comm).to_numpy().astype(bool)
    big = size[size >= 1000].index
    nat = tab.loc[big][(tab.loc[big].top_share >= 0.5) & (tab.loc[big].known >= 0.25)]
    # Fig. 5's cells (a5_clusters): any community whose known-country majority is the researcher's own country
    lead = tab[tab.top_share >= 0.5].top_country
    df["national_top"] = df.community.map(lead)
    tab = tab.loc[big]
    use = df[df.country.notna() & df.field.notna() & ~dust]
    use = use.assign(own=(use.national_top == use.country).astype(float))
    cells = use.groupby(["field", "country"]).agg(n=("own", "size"), own_national_share=("own", "mean")).reset_index()
    cells = cells[cells.n >= MIN_CELL]
    cells["partition"] = name
    summary = {"communities": int(len(size)), "communities_1000plus": int(len(big)),
               "share_in_1000plus": float(size.loc[big].sum() / n), "national": int(len(nat)),
               "members_in_national": int(nat["size"].sum()), "national_countries": sorted(nat.top_country.tolist()),
               "largest": [[int(c), int(s), (None if pd.isna(tab.top_country.get(c)) else str(tab.top_country.get(c))),
                            float(round(tab.top_share.get(c, np.nan), 3))] for c, s in size.head(12).items()]}
    return summary, cells


out, cells = {}, []
for name, comm in parts.items():
    s, c = describe(name, comm)
    out[name] = s
    cells.append(c)
    log(name, {k: v for k, v in s.items() if k != "largest"})
cells = pd.concat(cells, ignore_index=True)
cells.to_csv(PAPER / "cache" / "leiden_min15_field_country.csv", index=False)
countries = attrs.country.value_counts().head(14).index.tolist()
table = cells[cells.field.isin(FIELDS) & cells.country.isin(countries)].pivot_table(
    index=["field", "country"], columns="partition", values="own_national_share")
out["field_country"] = {f"{f}|{c}": {p: (None if pd.isna(v) else round(float(v), 4)) for p, v in row.items()}
                        for (f, c), row in table.iterrows()}
out["countries"] = countries
out["agreement"] = {"gpu_vs_cpu15": agreement(parts["gpu"], parts["cpu15"]),
                    "cpu15_vs_cpu5": agreement(parts["cpu15"], parts["cpu5"]),
                    "gpu_vs_cpu5": agreement(parts["gpu"], parts["cpu5"])}
out["modularity"] = {"gpu": json.loads((RESULTS / "a10_checks.json").read_text()).get("modularity"),
                     "cpu15": json.loads((PAPER / "results" / "leiden_min15_cpu.json").read_text())["modularity"]}
# reproduce the archived Fig. 5 cells from the GPU partition as a check of the definition
old = pd.read_csv(CACHE / "a5_field_country.csv").set_index(["field", "country"]).own_national_share
mine = cells[cells.partition == "gpu"].set_index(["field", "country"]).own_national_share
common_idx = old.index.intersection(mine.index)
out["gpu_reproduces_archived_fig5_max_abs_diff"] = float((old.loc[common_idx] - mine.loc[common_idx]).abs().max())
(PAPER / "results" / "leiden_min15_compare.json").write_text(json.dumps(out, indent=1))
log("agreement", out["agreement"], "modularity", out["modularity"], "fig5 check", out["gpu_reproduces_archived_fig5_max_abs_diff"])
print(table.loc[["Physics and Astronomy", "Medicine"]].round(2).to_string())
