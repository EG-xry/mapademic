"""A5: communities, national membership and domestic collaboration by field.

Community/cell counts use the attribute table. Weighted edge summaries stream the
Parquet file once instead of materializing two joined rows for every link. Domestic
mixes count each field-bearing endpoint of links whose two countries are known.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numba import njit

from common import CACHE, G, con, log, save_result

MIN_CELL = 2000
BATCH_SIZE = 1_000_000


@njit(cache=True)
def _accumulate(s, d, w, community, country, field, country_weight, domestic, endpoints, domestic_ends):
    total_weight = 0.0
    inside_weight = 0.0
    inside_links = 0
    for j in range(len(s)):
        u, v, weight = s[j], d[j], w[j]
        total_weight += weight
        if community[u] >= 0 and community[u] == community[v]:
            inside_weight += weight
            inside_links += 1
        cu, cv = country[u], country[v]
        if cu < 0 or cv < 0:
            continue
        fu, fv = field[u], field[v]
        if fu >= 0:
            country_weight[fu, cu] += weight
            endpoints[fu] += 1
            if cu == cv:
                domestic[fu] += weight
                domestic_ends[fu] += 1
        if fv >= 0:
            country_weight[fv, cv] += weight
            endpoints[fv] += 1
            if cu == cv:
                domestic[fv] += weight
                domestic_ends[fv] += 1
    return total_weight, inside_weight, inside_links


def aggregate_edges(path, community, country, field, country_count, field_count, batch_size=BATCH_SIZE):
    """Aggregate bounded batches; all category codes follow original node-index order.

    A missing community has code -1. An actual stored community value of -1 must
    be encoded as an ordinary category, since SQL considers -1 equal to -1.
    """
    country_weight = np.zeros((field_count, country_count), np.float64)
    domestic = np.zeros(field_count, np.float64)
    endpoints = np.zeros(field_count, np.int64)
    domestic_ends = np.zeros(field_count, np.int64)
    total_weight = inside_weight = 0.0
    inside_links = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=["src", "dst", "weight"]):
        s = batch.column("src").to_numpy()
        d = batch.column("dst").to_numpy()
        w = batch.column("weight").to_numpy().astype(np.float64, copy=False)
        total, inside, count = _accumulate(s, d, w, community, country, field, country_weight,
                                          domestic, endpoints, domestic_ends)
        total_weight += total
        inside_weight += inside
        inside_links += count
    return {"total_weight": total_weight, "inside_weight": inside_weight, "inside_links": inside_links,
            "country_weight": country_weight, "domestic": domestic,
            "endpoints": endpoints, "domestic_ends": domestic_ends}


def summarize(agg, field_names):
    """Keep the original SQL's weighting and NULL behavior for absent domestic links."""
    use = agg["endpoints"] > 0
    mix = agg["country_weight"][use]
    total = mix.sum(axis=1)
    domestic = np.where(agg["domestic_ends"][use] > 0, agg["domestic"][use], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        observed = domestic / total
        expected = ((mix / total[:, None]) ** 2).sum(axis=1)
        homophily = (observed - expected) / (1 - expected)
        within = float(agg["inside_weight"] / agg["total_weight"]) if agg["inside_links"] else float("nan")
    dom = pd.DataFrame({"field": np.asarray(field_names)[use], "observed": observed, "expected": expected,
                        "homophily": homophily, "weight": total})
    return within, dom.sort_values("homophily", ascending=False, na_position="last", kind="stable").reset_index(drop=True)


def main():
    c = con()
    A = f"read_parquet('{CACHE / 'node_attrs.parquet'}')"
    log("communities")
    c.execute(f"""CREATE TABLE comm AS
        WITH m AS (SELECT community, country, count(*) n FROM {A} GROUP BY 1, 2),
             t AS (SELECT community, sum(n) AS size, sum(n) FILTER (WHERE country IS NOT NULL) AS known FROM m GROUP BY 1),
             top AS (SELECT community, arg_max(country, n) AS top_country, max(n) AS top_n FROM m WHERE country IS NOT NULL GROUP BY 1)
        SELECT t.community, t.size, t.known, top.top_country, top.top_n / t.known AS top_share,
               top.top_n / t.known >= 0.5 AS national
        FROM t LEFT JOIN top USING (community)""")
    big = c.execute("""SELECT count(*), sum(size), count(*) FILTER (WHERE national), sum(size) FILTER (WHERE national)
                       FROM comm WHERE size >= 1000""").fetchone()
    communities = {"total": c.execute("SELECT count(*) FROM comm").fetchone()[0],
                   "with_1000_plus": big[0], "members_in_1000_plus": big[1],
                   "national_1000_plus": big[2], "members_in_national": big[3],
                   "largest": c.execute("""SELECT community, size, top_country, round(top_share, 3), national FROM comm
                                           ORDER BY size DESC LIMIT 30""").fetchall()}
    log(communities["with_1000_plus"], "communities with 1000+;", communities["national_1000_plus"], "national")

    log("field x country")
    cells = c.execute(f"""
        SELECT a.field, a.country, count(*) AS n,
               avg(CASE WHEN cm.national AND cm.top_country = a.country THEN 1 ELSE 0 END) AS own_national_share
        FROM {A} a JOIN comm cm USING (community)
        WHERE a.country IS NOT NULL AND a.field IS NOT NULL AND NOT coalesce(a.is_dust, false)
        GROUP BY 1, 2 HAVING count(*) >= {MIN_CELL} ORDER BY n DESC""").fetchdf()
    cells.to_csv(CACHE / "a5_field_country.csv", index=False)
    c.close()                         # release DuckDB tables before loading compact node codes

    log("domestic collaboration by field (streaming links)")
    attrs = pq.read_table(CACHE / "node_attrs.parquet", columns=["node_idx", "community", "country", "field"]).to_pandas()
    assert np.array_equal(attrs.node_idx.to_numpy(), np.arange(len(attrs))), "attributes are not in node order"
    community = pd.factorize(attrs.community, sort=False)[0].astype(np.int32)
    country = attrs.country.astype("category")
    cc, country_count = country.cat.codes.to_numpy(), len(country.cat.categories)
    field = attrs.field.astype("category")
    fd, field_names = field.cat.codes.to_numpy(), list(field.cat.categories)
    del attrs, country, field
    agg = aggregate_edges(G / "edges_int32.parquet", community, cc, fd, country_count, len(field_names))
    within, dom = summarize(agg, field_names)
    dom.to_csv(CACHE / "a5_domestic_by_field.csv", index=False)
    out = {"communities": communities, "within_community_link_share": within,
           "domestic_by_field": dom.to_dict("records"), "field_country_cells": int(len(cells))}
    save_result("a5_clusters", out)
    log("within-community link weight share:", round(within, 3))
    print(dom.to_string())
    piv = cells.pivot(index="field", columns="country", values="own_national_share")
    top_countries = cells.groupby("country")["n"].sum().sort_values(ascending=False).head(14).index
    print(piv[top_countries].round(2).to_string())


if __name__ == "__main__":
    main()
