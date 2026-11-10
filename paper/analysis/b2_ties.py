"""B2: strong and weak ties, long and short links.

Each link's weight is the sum of 1/(n-1) over the shared works with n authors (Newman 2001c). For every one of
every link: whether it stays inside a community, a country (both ends known) and a field, and its length on
the map (layout coordinates). For a random sample of links, the neighbourhood overlap of its ends (shared
coauthors / coauthors of either, excluding each other; Onnela et al. 2007) against its weight: in a social
network, weak ties are the bridges between groups (Granovetter 1973).
"""
import gc

import numpy as np
from numba import njit, prange

from common import CACHE, G, load_csr, log, save_result
from ties_cache import (BATCH, BOTH_CC, BOTH_FD, IN_CC, IN_COMM, IN_FD, ON_MAP, close_features,
                        describe, load_features, read_edge_rows, selected_values)

SAMPLE = 2_000_000


@njit(parallel=True, cache=True)
def sort_rows(offsets, neighbors):
    out = neighbors.copy()
    for u in prange(offsets.shape[0] - 1):
        out[offsets[u]:offsets[u + 1]].sort()
    return out


@njit(parallel=True, cache=True)
def overlap(offsets, sn, a, b):
    """|N(a) & N(b)| / (|N(a)| - 1 + |N(b)| - 1 - |N(a) & N(b)|) for each pair, from sorted neighbour lists."""
    out = np.zeros(a.shape[0], np.float64)
    for k in prange(a.shape[0]):
        i, ie = offsets[a[k]], offsets[a[k] + 1]
        j, je = offsets[b[k]], offsets[b[k] + 1]
        c = 0
        while i < ie and j < je:
            x, y = sn[i], sn[j]
            if x == y:
                c += 1
                i += 1
                j += 1
            elif x < y:
                i += 1
            else:
                j += 1
        denom = (offsets[a[k] + 1] - offsets[a[k]] - 1) + (offsets[b[k] + 1] - offsets[b[k]] - 1) - c
        out[k] = c / denom if denom > 0 else 0.0
    return out


def summary_counts(flags, selector=None, batch_size=BATCH):
    """Exact conditional denominators and crossing counts in bounded chunks."""
    counts = {key: 0 for key in ("links", "across_communities", "known_country", "across_countries", "known_field", "across_fields")}
    for start in range(0, len(flags), batch_size):
        stop = min(start + batch_size, len(flags))
        values = flags[start:stop]
        if selector is not None:
            values = values[selector(start, stop)]
        counts["links"] += len(values)
        counts["across_communities"] += int(np.count_nonzero((values & IN_COMM) == 0))
        counts["known_country"] += int(np.count_nonzero(values & BOTH_CC))
        counts["across_countries"] += int(np.count_nonzero((values & BOTH_CC != 0) & (values & IN_CC == 0)))
        counts["known_field"] += int(np.count_nonzero(values & BOTH_FD))
        counts["across_fields"] += int(np.count_nonzero((values & BOTH_FD != 0) & (values & IN_FD == 0)))
    return counts


def summarize_ties(arrays, scratch, sample_size=SAMPLE, batch_size=BATCH):
    """All original B2 edge summaries, with exact disk-backed masked vectors."""
    weight, length, flags = (arrays[key] for key in ("weights", "length", "flags"))
    total = len(weight)
    total_weight = weight.sum()
    out = {"links": total}
    map_selector = lambda start, stop: flags[start:stop] & ON_MAP != 0
    with selected_values(weight, map_selector, scratch, batch_size) as values:
        map_count, map_weight = len(values), values.sum()
        out["links_on_map"] = map_count
        out["weight_share_on_map"] = float(map_weight / total_weight)
    quantiles = (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
    with selected_values(weight, None, scratch, batch_size) as values:
        cuts = np.quantile(values, np.concatenate((quantiles, np.linspace(0, 1, 11))), overwrite_input=True)
    out["weight_quantiles"] = {str(q): float(value) for q, value in zip(quantiles, cuts[:len(quantiles)])}
    dec = cuts[len(quantiles):]
    selectors = {
        "inside_community": lambda a,b: flags[a:b] & IN_COMM != 0,
        "across_communities": lambda a,b: flags[a:b] & IN_COMM == 0,
        "same_country": lambda a,b: flags[a:b] & IN_CC != 0,
        "across_countries": lambda a,b: (flags[a:b] & BOTH_CC != 0) & (flags[a:b] & IN_CC == 0),
        "same_field": lambda a,b: flags[a:b] & IN_FD != 0,
        "across_fields": lambda a,b: (flags[a:b] & BOTH_FD != 0) & (flags[a:b] & IN_FD == 0),
    }
    for label, selector in selectors.items():
        with selected_values(weight, selector, scratch, batch_size) as values:
            out[label] = describe(values, total, total_weight)
        log("tie summary", label, out[label]["links"])
    bin_counts = np.zeros((10, 4), np.int64)  # links, across-community, known-country, across-country
    for start in range(0,total,batch_size):
        stop = min(start+batch_size,total)
        bins = np.clip(np.searchsorted(dec,weight[start:stop],side="right")-1,0,9)
        bits = flags[start:stop]
        for column, selected in enumerate((np.ones(len(bits),bool), bits & IN_COMM == 0,
                                           bits & BOTH_CC != 0, (bits & BOTH_CC != 0) & (bits & IN_CC == 0))):
            bin_counts[:,column] += np.bincount(bins[selected],minlength=10)
    out["by_weight_decile"] = []
    for k in range(10):
        selector = lambda a,b,k=k: (np.clip(np.searchsorted(dec,weight[a:b],side="right")-1,0,9)==k) & ~np.isnan(length[a:b])
        with selected_values(length,selector,scratch,batch_size) as values:
            median_length = float(np.median(values,overwrite_input=True)) if len(values) else float("nan")
        count, across, known, foreign = bin_counts[k]
        out["by_weight_decile"].append({"decile":k,"weight_from":float(dec[k]),"weight_to":float(dec[k+1]),
                                        "across_communities":float(across/count) if count else float("nan"),
                                        "across_countries":float(foreign/max(known,1)),"median_length":median_length})
        log(f"tie length median: weight decile {k+1}/10")
    with selected_values(length,None,scratch,batch_size,finite=True) as values:
        lq = np.quantile(values,[.5,.9,.99,.999],overwrite_input=True)
    out["length_quantiles"] = dict(zip(("p50","p90","p99","p999"),map(float,lq)))
    for label, selector in (("shortest_half",lambda a,b:length[a:b]<=lq[0]),
                             ("longest_10pct",lambda a,b:length[a:b]>=lq[1]),
                             ("longest_1pct",lambda a,b:length[a:b]>=lq[2])):
        counts = summary_counts(flags,selector,batch_size)
        with selected_values(weight,selector,scratch,batch_size) as values:
            selected_weight = values.sum()
            stats = describe(values,total,total_weight)
        out[label] = {**stats,"share_map_links":stats["links"]/map_count,
                      "share_map_weight":float(selected_weight/map_weight),
                      "across_communities":counts["across_communities"]/counts["links"] if counts["links"] else float("nan"),
                      "across_countries":counts["across_countries"]/max(counts["known_country"],1),
                      "across_fields":counts["across_fields"]/max(counts["known_field"],1)}
        log("tie summary",label,stats["links"])
    # Exactly the original seed, draw size, and unsorted sample row order.
    pick = np.random.default_rng(20261025).choice(total,sample_size,replace=False)
    bits = np.asarray(flags[pick])
    sample = {"rows":pick,"decile":np.clip(np.searchsorted(dec,weight[pick],side="right")-1,0,9).astype(np.int8),
              "in_community":bits&IN_COMM!=0,"in_country":bits&IN_CC!=0,
              "across_country":(bits&BOTH_CC!=0)&(bits&IN_CC==0)}
    return out,sample


@njit(parallel=True,cache=True)
def sort_rows_inplace(offsets, neighbors):
    for u in prange(len(offsets)-1):
        neighbors[offsets[u]:offsets[u+1]].sort()


def main():
    arrays, manifest = load_features()
    for key in ("community","country","field","x","y"):
        arrays.pop(key)._mmap.close()
    log(f"{manifest['links']:,} links; disk-backed exact tie summaries")
    out,sample = summarize_ties(arrays,CACHE / "ties/selection_scratch.npy")
    out["provenance"] = {"feature_cache":str(CACHE / "ties/manifest.json"),
                         "feature_identity_sha256":manifest["fingerprint"],
                         "feature_code_sha256":manifest["identity"]["feature_code_sha256"]}
    sample_s,sample_d = read_edge_rows(G / "edges_int32.parquet",sample.pop("rows"))
    close_features(arrays)
    del arrays
    gc.collect()
    offsets,neighbors = load_csr()
    sn = np.lib.format.open_memmap(CACHE / "ties/sorted_neighbors.npy",mode="w+",dtype=neighbors.dtype,shape=neighbors.shape)
    for start in range(0,len(neighbors),BATCH):
        sn[start:start+BATCH] = neighbors[start:start+BATCH]
    sn.flush()
    neighbors._mmap.close()
    del neighbors
    log("tie overlap: sorting disk-backed CSR rows")
    sort_rows_inplace(offsets,sn)
    sn.flush()
    ov = overlap(offsets,sn,sample_s.astype(np.int64),sample_d.astype(np.int64))
    wb = sample["decile"]
    out["overlap_by_weight_decile"] = [float(ov[wb==k].mean()) for k in range(10)]
    out["overlap"] = {"inside_community":float(ov[sample["in_community"]].mean()),
                      "across_communities":float(ov[~sample["in_community"]].mean()),
                      "same_country":float(ov[sample["in_country"]].mean()),
                      "across_countries":float(ov[sample["across_country"]].mean())}
    sn._mmap.close()
    save_result("b2_ties",out)
    log({key:value for key,value in out.items() if key != "by_weight_decile"})
    for row in out["by_weight_decile"]:log(row)


if __name__ == "__main__":
    main()
