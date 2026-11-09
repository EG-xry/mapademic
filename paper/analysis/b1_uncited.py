"""B1: researchers with many papers and few citations -- how many, where, and where in the network.

Every researcher in the network has at least MIN_WORKS (5, or 15 on the map's network) scholarly works. Here: the share with no citations at all (cited_by_count 0) or
fewer than 10, by name script, country, field and community; the prolific uncited (50+ works, none cited); their
place in the network (estimated mean distance from A2's 1,000 sources, cached by A4; degree); and how many uncited
native-script records are second records of a researcher who is cited under a Latin-script name (A6g twins, at
the thresholds A8 joins them).
"""
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import CACHE, RESULTS, log, name_script, save_result

def load_analysis_attrs(path, closeness_path, batch_size=250_000):
    """Keep numeric/category data; classify names in bounded Parquet batches.

    Only the 15 names that the report prints survive their input batch. Script
    values retain the same object dtype, order and classification precedence as
    the former all-name DataFrame, including value_counts tie behavior.
    """
    columns = ["node_idx", "works", "works_kept", "cited", "country", "field", "community", "degree"]
    a = pq.read_table(path, columns=columns).to_pandas()
    a["closeness"] = np.load(closeness_path, mmap_mode="r")
    a["uncited"] = a.cited == 0
    a["few"] = a.cited < 10
    wanted = a[(a.works_kept >= 50) & a.uncited].nlargest(15, "works_kept").node_idx.to_numpy()
    scripts = np.empty(len(a), dtype=object)
    reported_names, at = {}, 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=["node_idx", "name"]):
        values = batch.to_pandas()
        end = at + len(values)
        assert np.array_equal(values.node_idx.to_numpy(), a.node_idx.to_numpy()[at:end])
        scripts[at:end] = name_script(values.name).to_numpy()
        selected = values[values.node_idx.isin(wanted)]
        reported_names.update(zip(selected.node_idx, selected.name))
        at = end
    assert at == len(a) and len(reported_names) == len(wanted)
    a["script"] = scripts
    return a, reported_names


def summary(g):
    return {"n": int(len(g)), "uncited": float(g.uncited.mean()), "under_10": float(g.few.mean()),
            "median_cited": float(g.cited.median()), "median_works": float(g.works_kept.median())}


def main():
    a, reported_names = load_analysis_attrs(CACHE / "node_attrs.parquet", CACHE / "closeness_1000.npy")
    out = {"all": summary(a), "quantiles_cited": {str(q): float(a.cited.quantile(q)) for q in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)}}
    out["by_script"] = {s: summary(g) for s, g in a.groupby("script")}
    big = a.country.value_counts().head(25).index
    out["by_country"] = {c: summary(a[a.country == c]) for c in big}
    out["no_country"] = summary(a[a.country.isna()])
    out["by_field"] = {f: summary(g) for f, g in a.groupby("field") if len(g) >= 10_000}
    comm = a.community.value_counts()
    out["by_community"] = {int(c): summary(a[a.community == c]) for c in comm[comm >= 1000].index}
    # where they sit in the network
    reach = np.isfinite(a.closeness)
    for lab, g in (("uncited", a.uncited), ("under_10", a.few & ~a.uncited), ("10_to_99", (a.cited >= 10) & (a.cited < 100)),
                   ("100_plus", a.cited >= 100), ("100_to_999", (a.cited >= 100) & (a.cited < 1000)), ("1000_plus", a.cited >= 1000)):
        out.setdefault("position", {})[lab] = {"n": int(g.sum()), "median_distance_to_all": float(a.closeness[g & reach].median()),
                                               "median_degree": float(a.degree[g].median()),
                                               "share_unreachable": float((~reach[g]).mean())}
    prolific = a[(a.works_kept >= 50) & a.uncited]
    reported = prolific.nlargest(15, "works_kept")[["node_idx", "works_kept", "country", "field"]].copy()
    reported["name"] = reported.node_idx.map(reported_names)
    out["prolific_uncited"] = {"n": int(len(prolific)), "by_script": prolific.script.value_counts().head(6).to_dict(),
                               "by_country": prolific.country.value_counts(dropna=False).head(8).to_dict(),
                               "top": reported[["name", "works_kept", "country", "field"]].values.tolist()}
    # uncited native-script records that are second records of a cited Latin-script researcher
    twins = []
    for code in ("zh", "ja"):
        rate = json.loads((RESULTS / f"a6_twins_{code}.json").read_text())["pass_rate"]
        t = min(int(k) for k, r in rate.items() if r["decoys"] <= 0.05)
        tw = pd.read_csv(CACHE / f"a6_twins_{code}.csv")
        twins.append(tw[tw.shared >= t][["focal_idx", "twin_idx"]])
    tw = pd.concat(twins)
    cited = a.set_index("node_idx").cited
    tw = tw[cited.reindex(tw.focal_idx).to_numpy() == 0]
    out["uncited_twins"] = {"uncited_native_script_with_twin": int(len(tw)),
                            "share_of_all_uncited": float(len(tw) / a.uncited.sum()),
                            "twin_median_cited": float(cited.reindex(tw.twin_idx).median()),
                            "share_of_uncited_native_script": float(len(tw) / (a.uncited & (a.script != "Latin")).sum())}
    save_result("b1_uncited", out)
    log({k: out[k] for k in ("all", "quantiles_cited", "position", "uncited_twins")})
    log("by script", {k: (v["n"], round(v["uncited"], 3), v["median_cited"]) for k, v in out["by_script"].items()})
    log("by country", {k: (round(v["uncited"], 3), round(v["under_10"], 3), v["median_cited"]) for k, v in out["by_country"].items()})
    log("by field", {k[:25]: round(v["uncited"], 3) for k, v in sorted(out["by_field"].items(), key=lambda kv: -kv[1]["uncited"])})
    log("prolific uncited", out["prolific_uncited"])


if __name__ == "__main__":
    main()
