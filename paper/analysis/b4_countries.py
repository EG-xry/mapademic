"""B4: collaboration across borders, aggregated in bounded edge batches.

Only links with a known country at both ends enter country comparisons. A country's
all-field denominator counts each incident link once, even when both endpoints are
in that country. Field denominators count each matching endpoint, as in the original
analysis. Affinity matrices retain the original top-25 plus 'other countries' bucket.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numba import njit

from common import CACHE, G, log, save_result

TOP = 25
BATCH_SIZE = 1_000_000
REPORT_FIELDS = (
    "Medicine", "Physics and Astronomy", "Computer Science", "Social Sciences", "Mathematics", "Engineering",
    "Biochemistry, Genetics and Molecular Biology", "Earth and Planetary Sciences",
)


@njit(cache=True)
def _accumulate(s, d, w, cc, country, field, pair_weight, incident, domestic, field_weight, field_domestic):
    """Accumulate one edge batch; cc includes the other-country bucket, country does not."""
    known_links = 0
    known_weight = 0.0
    domestic_weight = 0.0
    for j in range(len(s)):
        u, v, weight = s[j], d[j], w[j]
        if country[u] < 0 or country[v] < 0:
            continue
        a, b = cc[u], cc[v]
        same = country[u] == country[v]
        known_links += 1
        known_weight += weight
        pair_weight[a, b] += weight
        incident[a] += weight
        if a != b:
            incident[b] += weight
        if same:
            domestic_weight += weight
            domestic[a] += weight
            if a != b:
                domestic[b] += weight
        fu, fv = field[u], field[v]
        if fu >= 0:
            field_weight[fu, a] += weight
            if same:
                field_domestic[fu, a] += weight
        if fv >= 0:
            field_weight[fv, b] += weight
            if same:
                field_domestic[fv, b] += weight
    return known_links, known_weight, domestic_weight


def aggregate_edges(path, cc, country, field, country_count, field_count, batch_size=BATCH_SIZE):
    """Read only one src/dst/weight batch at a time; node codes must be in node-index order."""
    k = country_count + 1
    pair_weight = np.zeros((k, k), np.float64)
    incident = np.zeros(k, np.float64)
    domestic = np.zeros(k, np.float64)
    field_weight = np.zeros((field_count, k), np.float64)
    field_domestic = np.zeros_like(field_weight)
    links = known_links = 0
    known_weight = domestic_weight = 0.0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=["src", "dst", "weight"]):
        s = batch.column("src").to_numpy()
        d = batch.column("dst").to_numpy()
        w = batch.column("weight").to_numpy().astype(np.float64, copy=False)
        n, total, same = _accumulate(s, d, w, cc, country, field, pair_weight, incident, domestic,
                                     field_weight, field_domestic)
        links += len(s)
        known_links += n
        known_weight += total
        domestic_weight += same
    return {"links": links, "known_links": known_links, "known_weight": known_weight,
            "domestic_weight": domestic_weight, "pair_weight": pair_weight,
            "incident": incident, "domestic": domestic,
            "field_weight": field_weight, "field_domestic": field_domestic}


def summarize(agg, names, field_names, report_fields=REPORT_FIELDS):
    top = len(names)
    W = agg["pair_weight"] + agg["pair_weight"].T       # diagonal counts each link twice
    strength = W.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        affinity = W / (np.outer(strength, strength) / strength.sum())
        abroad = {c: float(1 - agg["domestic"][i] / agg["incident"][i]) for i, c in enumerate(names)}
    out = {"countries": names, "share_links_known": float(agg["known_links"] / agg["links"]),
           "share_weight_abroad_overall": float(1 - agg["domestic_weight"] / agg["known_weight"]),
           "share_abroad": abroad}
    pairs = [(names[i], names[j], float(affinity[i, j]), float(W[i, j]))
             for i in range(top) for j in range(i + 1, top)]
    out["affinity_top"] = sorted(pairs, key=lambda p: -p[2])[:25]
    out["affinity_bottom"] = sorted(pairs, key=lambda p: p[2])[:15]
    out["affinity_self"] = {c: float(affinity[i, i]) for i, c in enumerate(names)}
    out["affinity_matrix"] = affinity[:top, :top].round(3).tolist()
    # Preserve the original convention: zero every bucket's diagonal, including
    # other/other. The latter may contain links between two different small countries.
    Wi = W.copy()
    np.fill_diagonal(Wi, 0)
    si = Wi.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        intl = Wi / (np.outer(si, si) / si.sum())
    out["intl_weight_matrix"] = Wi[:top, :top].round(1).tolist()
    out["intl_affinity_matrix"] = intl[:top, :top].round(3).tolist()
    ipairs = [(names[i], names[j], float(intl[i, j]), float(Wi[i, j]))
              for i in range(top) for j in range(i + 1, top)]
    out["intl_affinity_top"] = sorted(ipairs, key=lambda p: -p[2])[:25]
    out["intl_affinity_bottom"] = sorted(ipairs, key=lambda p: p[2])[:15]
    by_field = {}
    for f in report_fields:
        j = field_names.index(f)
        by_field[f] = {
            c: float(1 - agg["field_domestic"][j, i] / agg["field_weight"][j, i])
            if agg["field_weight"][j, i] > 0 else None
            for i, c in enumerate(names[:15])}
    out["share_abroad_by_field"] = by_field
    return out


def main():
    attrs = pq.read_table(CACHE / "node_attrs.parquet", columns=["node_idx", "country", "field"]).to_pandas()
    assert np.array_equal(attrs.node_idx.to_numpy(), np.arange(len(attrs))), "attributes are not in node order"
    names = attrs.country.value_counts().head(TOP).index.tolist()
    code = pd.Series(np.arange(len(names)), index=names)
    cc = attrs.country.map(code).fillna(len(names)).to_numpy(np.int32)
    country = attrs.country.astype("category").cat.codes.to_numpy()
    fields = attrs.field.astype("category")
    fd, fd_names = fields.cat.codes.to_numpy(), list(fields.cat.categories)
    del attrs, fields
    agg = aggregate_edges(G / "edges_int32.parquet", cc, country, fd, len(names), len(fd_names))
    out = summarize(agg, names, fd_names)
    save_result("b4_countries", out)
    log(f"{out['share_links_known']:.1%} of links have a known country at both ends; "
        f"{out['share_weight_abroad_overall']:.1%} of their weight crosses a border")
    log("abroad", {c: round(v, 3) for c, v in out["share_abroad"].items()})
    for key in ("affinity_top", "affinity_bottom", "intl_affinity_top", "intl_affinity_bottom"):
        log(key, [(a, b, round(x, 2)) for a, b, x, _ in out[key]])
    for f, row in out["share_abroad_by_field"].items():
        log(f[:20], {c: round(v, 2) for c, v in row.items() if v is not None})


if __name__ == "__main__":
    main()
