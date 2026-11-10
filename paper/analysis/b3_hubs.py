"""B3: what is a hub? Five definitions compared, and whether hubs work with each other.

Definitions: degree (distinct coauthors), strength (summed link weight, i.e. coauthored works shared out by team
size), betweenness (A4: 2,000 sources in two halves, the smaller estimate), closeness (mean distance to A2's 1,000
sources) and citations.
Profiles flagged as merged, split or not a person (flags.py) are left out of the rankings. For
each pair of definitions: the overlap of their top 1,000 and the Spearman correlation. Then the rich club: the
links among the k best-connected researchers against what a random network with the same degrees would give,
and degree assortativity (do hubs link to hubs?).

The edge table is streamed in 500,000-row batches. Strength retains separate source/destination accumulators
and each one's original row order, matching the former full-table bincount. Only citations and betweenness are
loaded for all researchers; names, countries and fields are read for the selected top profiles after ranking.
"""
import gc

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numba import njit

from common import CACHE, G, flagged_mask, load_csr, log, save_result

TOP = 1000
CLUBS = (100, 1000, 10_000, 100_000)
EDGE_BATCH = 500_000


@njit(cache=True)
def _add_strength(src, dst, weight, left, right):
    # Separate accumulators preserve the two np.bincounts' addition order and final sum.
    for i in range(len(src)):
        left[src[i]] += weight[i]
        right[dst[i]] += weight[i]


def edge_statistics(path, deg, club_rank, clubs=CLUBS, batch_size=EDGE_BATCH):
    """Strength, rich-club counts and degree/log-degree Pearson moments in bounded memory."""
    n = len(deg)
    left, right = np.zeros(n, np.float64), np.zeros(n, np.float64)
    links_inside = np.zeros(len(clubs), np.int64)
    moments = np.zeros((2, 3), np.float64)       # degree/log-degree: sum, square sum, cross sum
    links = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=["src", "dst", "weight"]):
        src, dst = batch.column("src").to_numpy(), batch.column("dst").to_numpy()
        weight = batch.column("weight").to_numpy().astype(np.float64, copy=False)
        _add_strength(src, dst, weight, left, right)
        rank = np.maximum(club_rank[src], club_rank[dst])
        for i, k in enumerate(clubs):
            links_inside[i] += np.count_nonzero(rank < k)
        x, y = deg[src].astype(np.float64), deg[dst].astype(np.float64)
        for i in range(2):
            moments[i, 0] += x.sum() + y.sum()
            moments[i, 1] += np.dot(x, x) + np.dot(y, y)
            moments[i, 2] += 2 * np.dot(x, y)
            if i == 0:
                np.log(x, out=x)
                np.log(y, out=y)
        links += len(src)
        if links % 50_000_000 < len(src):
            log(f"hub edge statistics: {links:,} links")
    left += right
    mean = moments[:, 0] / (2 * links)
    assortativity = (moments[:, 2] / (2 * links) - mean ** 2) / (moments[:, 1] / (2 * links) - mean ** 2)
    return left, links_inside, links, assortativity


def selected_metadata(path, selected, batch_size=250_000):
    """Read string metadata only for selected node indices, preserving index-based lookup semantics."""
    selected = np.asarray(sorted(selected), dtype=np.int64)
    found = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=["node_idx", "name", "country", "field"]):
        mask = np.isin(batch.column("node_idx").to_numpy(), selected)
        if mask.any():
            found.append(batch.filter(mask).to_pandas())
    attrs = pd.concat(found, ignore_index=True).set_index("node_idx")
    assert attrs.index.is_unique and len(attrs) == len(selected)
    return attrs


def main():
    offsets, _ = load_csr()
    deg = np.diff(offsets).astype(np.int64)
    n = len(deg)
    flagged = flagged_mask(n)
    order = np.argsort(-np.where(flagged, -1, deg), kind="stable")
    club_rank = np.full(n, max(CLUBS), np.int32)
    club_rank[order[:max(CLUBS)]] = np.arange(min(n, max(CLUBS)), dtype=np.int32)
    strength, inside, links, assortativity = edge_statistics(G / "edges_int32.parquet", deg, club_rank)
    del club_rank
    out = {"top": TOP, "rich_club": []}
    m2 = 2 * links
    for i, k in enumerate(CLUBS):
        club = order[:k]
        kc = deg[club].astype(np.float64)
        expected = (kc.sum() ** 2 - (kc ** 2).sum()) / (2 * m2)
        out["rich_club"].append({"club": k, "min_degree": int(kc.min()), "links_inside": int(inside[i]),
                                 "density": float(inside[i] / (k * (k - 1) / 2)), "random_expectation": float(expected),
                                 "ratio": float(inside[i] / expected)})
        log(out["rich_club"][-1])
    del order, kc, club
    citations = pq.read_table(CACHE / "node_attrs.parquet", columns=["cited"]).column(0).to_numpy()
    hubs = pq.read_table(CACHE / "hubs.parquet", columns=["node_idx", "btw"]).to_pandas().set_index("node_idx")
    between = hubs.btw.reindex(np.arange(n)).fillna(0).to_numpy()
    del hubs
    close = np.load(CACHE / "closeness_1000.npy")
    df = pd.DataFrame({"degree": deg, "strength": strength, "betweenness": between,
                       "closeness": -close, "citations": citations})
    ok = ~flagged & np.isfinite(close)
    log(f"{flagged.sum():,} flagged profiles left out of the rankings")
    eligible = df[ok]
    del df, deg, strength, citations, between, close, flagged, ok, offsets
    gc.collect()
    measures = list(eligible.columns)
    top = {m: set(eligible.nlargest(TOP, m).index) for m in measures}
    top10 = {m: eligible.nlargest(10, m).index for m in measures}
    sub = eligible.sample(500_000, random_state=1)  # same eligible row order and RNG as the full-table version
    out["overlap_top1000"] = {f"{a}-{b}": len(top[a] & top[b]) for i, a in enumerate(measures) for b in measures[i + 1:]}
    out["spearman"] = sub.corr("spearman").round(3).to_dict()
    del eligible, sub
    gc.collect()
    attrs = selected_metadata(CACHE / "node_attrs.parquet", set().union(*top.values()))
    out["top10"] = {m: attrs.name.reindex(top10[m]).tolist() for m in measures}
    out["top1000_countries"] = {m: attrs.country.reindex(list(top[m])).value_counts().head(5).to_dict() for m in measures}
    out["top1000_fields"] = {m: attrs.field.reindex(list(top[m])).value_counts().head(5).to_dict() for m in measures}
    out["assortativity_degree"] = float(assortativity[0])
    out["assortativity_log_degree"] = float(assortativity[1])
    save_result("b3_hubs", out)
    log({k: out[k] for k in ("overlap_top1000", "assortativity_degree", "assortativity_log_degree")})
    log("spearman", out["spearman"])
    for m in measures:
        log(m, out["top10"][m])


if __name__ == "__main__":
    main()
