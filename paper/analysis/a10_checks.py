"""A10: manuscript validation with bounded-memory Parquet scans.

Counts links touching flagged profiles/hubs, weighted Leiden modularity, and
social-science/fallback-topic citation summaries. No full edge table or
20-million-row string DataFrame is materialized. Run after export_deposit.py.
"""
import json

import numpy as np
import pyarrow.parquet as pq

from common import CACHE, D, G, ND, RESULTS, SCRIPT_COMMUNITIES, con, log, name_script, save_result

BATCH_SIZE = 250_000


def edge_checks(batches, community, masks):
    """Aggregate undirected edges once; self-loops count as two link ends."""
    k = int(community.max()) + 1
    strength = np.zeros(k, np.float64)
    inside = np.zeros(k, np.float64)
    touched = dict.fromkeys(masks, 0)
    links, total_weight = 0, 0.0
    for b in batches:
        src = b.column("src").to_numpy()
        dst = b.column("dst").to_numpy()
        weight = b.column("weight").to_numpy().astype(np.float64)
        if len(src) and (min(src.min(), dst.min()) < 0 or max(src.max(), dst.max()) >= len(community)):
            raise ValueError("edge endpoint outside researcher population")
        cs, cd = community[src], community[dst]
        same = cs == cd
        strength += np.bincount(cs, weight, minlength=k)
        strength += np.bincount(cd, weight, minlength=k)
        inside += np.bincount(cs[same], weight[same], minlength=k)
        for name, mask in masks.items():
            touched[name] += int(np.count_nonzero(mask[src] | mask[dst]))
        links += len(src)
        total_weight += float(weight.sum())
    if total_weight <= 0:
        raise ValueError("network has no positive link weight")
    return {"links": links, "touching": touched,
            "modularity": float((inside / total_weight - (strength / (2 * total_weight)) ** 2).sum()),
            "weight_inside_share": float(inside.sum() / total_weight)}


def demographic_checks(batches):
    """Batch additive numerators/denominators, including conditional means."""
    totals = dict.fromkeys(("n", "known", "fallback", "fallback_soc", "soc", "soc_uncited", "soc_fallback",
                           "soc_fallback_uncited", "soc_no_fallback", "soc_no_fallback_uncited",
                           "soc_uncited_native", "soc_latin_inst", "soc_latin_inst_uncited"), 0)
    for batch in batches:
        r = batch.to_pandas()
        fallback = r.fallback.fillna(False).to_numpy(dtype=bool)
        social = (r.field == "Social Sciences").to_numpy()
        totals["n"] += len(r)
        totals["known"] += int(r.country.notna().sum())
        totals["fallback"] += int(fallback.sum())
        totals["fallback_soc"] += int((fallback & social).sum())
        soc = r[social]
        fb = fallback[social]
        uncited = (soc.citations == 0).to_numpy()
        native = (name_script(soc.name) != "Latin").to_numpy()
        latin_inst = ~native & soc.institution.notna().to_numpy()
        for key, mask in (("soc", social), ("soc_uncited", uncited), ("soc_fallback", fb),
                          ("soc_fallback_uncited", fb & uncited), ("soc_no_fallback", ~fb),
                          ("soc_no_fallback_uncited", ~fb & uncited), ("soc_uncited_native", uncited & native),
                          ("soc_latin_inst", latin_inst), ("soc_latin_inst_uncited", latin_inst & uncited)):
            totals[key] += int(mask.sum())
    def ratio(num, den):
        return totals[num] / totals[den] if totals[den] else None
    return {"country_known": ratio("known", "n"), "social_sciences": {
        "n": totals["soc"], "uncited": ratio("soc_uncited", "soc"), "fallback": totals["soc_fallback"],
        "fallback_share": ratio("soc_fallback", "soc"),
        "fallback_uncited": ratio("soc_fallback_uncited", "soc_fallback"),
        "without_fallback_n": totals["soc_no_fallback"],
        "without_fallback_uncited": ratio("soc_no_fallback_uncited", "soc_no_fallback"),
        "uncited_native_script": ratio("soc_uncited_native", "soc_uncited"),
        "latin_with_institution_uncited": ratio("soc_latin_inst_uncited", "soc_latin_inst"),
        "all_fallback": totals["fallback"],
        "all_fallback_social_sciences": ratio("fallback_soc", "fallback")}}


def main():
    offsets = np.load(ND / "pathgraph" / "offsets.npy", mmap_mode="r")
    deg = np.diff(offsets)
    n = len(deg)
    expected_links = int(offsets[-1]) // 2
    researcher_path = ND / "deposit" / "researchers.parquet"
    population = pq.ParquetFile(researcher_path)
    assert population.metadata.num_rows == n
    community = np.empty(n, np.int32)
    masks = {name: np.zeros(n, bool) for name in ("merged", "split", "not_a_person")}
    start = 0
    for b in population.iter_batches(batch_size=BATCH_SIZE, columns=["node", "flag", "community"]):
        r = b.to_pandas()
        stop = start + len(r)
        assert np.array_equal(r.node.to_numpy(), np.arange(start, stop)), "researcher node order changed"
        community[start:stop] = r.community.to_numpy()
        for name in masks:
            masks[name][start:stop] = (r.flag == name).to_numpy()
        start = stop
    community[community < 0] = int(community.max()) + 1
    masks["merged"][:] = False
    masks["merged"][np.load(CACHE / "a7_sets_peak25_strict.npz")["merged"]] = True
    order = np.argsort(-deg, kind="stable")
    for share in (0.01, 0.05):
        mask = np.zeros(n, bool)
        mask[order[:int(round(share * n))]] = True
        masks[f"top_{share:g}"] = mask
    del order
    out = edge_checks(pq.ParquetFile(G / "edges_int32.parquet").iter_batches(
        batch_size=BATCH_SIZE, columns=["src", "dst", "weight"]), community, masks)
    assert out["links"] == expected_links, "edge table and CSR link counts disagree"
    links = out["links"]
    for name, count in out["touching"].copy().items():
        mask = masks[name]
        out["touching"][name] = {"researchers": int(mask.sum()), "links": count, "share_of_links": count / links,
                                  "share_of_link_ends": float(deg[mask].sum() / (2 * links))}
        log(name, out["touching"][name])
    merged = masks["merged"]
    out["merged_in_china_community"] = float((community[merged] == SCRIPT_COMMUNITIES["China (main)"]).mean())
    del deg, community, masks, merged
    log("modularity", out["modularity"], "weight inside", out["weight_inside_share"])
    c = con()
    c.execute("SET memory_limit='6GB'")
    c.execute("SET threads=4")
    c.execute(f"""CREATE TEMP TABLE fallback_nodes AS
        SELECT DISTINCT a.node_idx FROM read_parquet('{CACHE / 'node_attrs.parquet'}') a
        JOIN read_parquet('{D}/authors_extract/*/*.parquet') f ON f.id = a.id
        WHERE f.topics[1].display_name = 'Legal and Regulatory Analysis'""")
    reader = c.execute(f"""SELECT r.citations, r.country, r.institution, r.field, r.name,
                                  f.node_idx IS NOT NULL AS fallback
        FROM read_parquet('{researcher_path}') r LEFT JOIN fallback_nodes f ON r.node = f.node_idx""").to_arrow_reader(BATCH_SIZE)
    out.update(demographic_checks(reader))
    c.close()
    log("social sciences", out["social_sciences"])
    save_result("a10_checks", out)
    b4 = json.loads((RESULTS / "b4_countries.json").read_text())
    print(json.dumps({k: v for k, v in b4.items()
                      if not isinstance(v, (list, dict)) or len(json.dumps(v)) < 400}, indent=1)[:1500])


if __name__ == "__main__":
    main()
