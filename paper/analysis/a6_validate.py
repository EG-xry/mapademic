"""A6h: score merge signals against later recorded work-count losses.

Among the 50,000 highest-degree profiles observed in the cached 2026-10-14
comparison snapshot, the label marks a raw works_count below half its original
value. Such a loss is consistent with profile splitting but can reflect other
catalogue revisions; it is not a manually confirmed merge or an adjudication
record from OpenAlex. The legacy output key split_by_openalex denotes this proxy.

Signals measured on the original snapshot are name commonness (records sharing
the name key; retained profiles sharing the full name), degree-adjusted local
clustering (residual log C on log k), flagged-link share, and degree. AUC is the
probability that a labelled profile scores higher, with half credit for ties.
Later affiliation measurements are omitted because they postdate the original
snapshot and can reflect revisions already made to the profiles.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from common import CACHE, ND, PG, con, save_result

TOP = 50_000


def validation_metadata(db, targets, nodes, snapshot, identity, full_names):
    """Independently join each signal to selected validation profiles only.

    Separate small joins avoid retaining all researcher IDs and preserve the
    target order used in the rank-based validation. Uniqueness assertions catch
    accidental one-to-many joins instead of silently repeating validation rows.
    """
    db.register("validation_targets", targets)
    result = targets.copy()
    try:
        for path, condition, columns in (
                (nodes, "CAST(regexp_extract(p.id, 'A([0-9]+)$', 1) AS BIGINT) = t.id", "p.works_count AS works_old"),
                (snapshot, "p.id = t.id", "p.works_count AS works_new"),
                (identity, "p.node_idx = t.node_idx", "p.key_n"),
                (full_names, "p.node_idx = t.node_idx", "p.full_n")):
            values = db.execute(f"""SELECT t.node_idx, {columns} FROM validation_targets t
                LEFT JOIN read_parquet('{path}') p ON {condition}""").fetchdf().set_index("node_idx")
            assert values.index.is_unique and len(values) == len(targets), "validation metadata is not one row per profile"
            for column in values.columns:
                result[column] = values[column].reindex(targets.node_idx).to_numpy()
    finally:
        db.unregister("validation_targets")
    return result


def auc(score, label):
    r = rankdata(score)
    pos = label.sum()
    return float((r[label].sum() - pos * (pos + 1) / 2) / (pos * (len(label) - pos)))


def main():
    offsets = np.load(PG / "offsets.npy", mmap_mode="r")
    deg = np.diff(offsets)
    ids = np.load(PG / "ids.npy", mmap_mode="r")
    top = np.argsort(-deg, kind="stable")[:TOP]
    targets = pd.DataFrame({"node_idx": top, "id": ids[top], "degree": deg[top]})
    del offsets, deg, ids, top
    db = con()
    db.execute("SET memory_limit='6GB'")
    try:
        df = validation_metadata(db, targets, ND / "nodes.parquet", CACHE / "snapshot_20261014.parquet",
                                 PG / "identity_signals.parquet", CACHE / "full_name_counts.parquet")
    finally:
        db.close()
    df = df[df.works_new.notna()].copy()  # IDs absent from the Oct14 census remain excluded.
    df["split"] = df.works_new < 0.5 * df.works_old
    clust = np.load(CACHE / "clustering_all.npy", mmap_mode="r")
    logc, logk = np.log(np.maximum(clust[df.node_idx], 1e-5)), np.log(df.degree.to_numpy())
    fit = np.polyfit(logk, logc, 1)
    df["low_clustering"] = -(logc - np.polyval(fit, logk))
    flagged = np.load(CACHE / "flagged_per_node.npy", mmap_mode="r")
    df["flagged_share"] = flagged[df.node_idx] / df.degree
    label = df.split.to_numpy()
    out = {"profiles": int(len(df)), "split_by_openalex": int(label.sum()), "split_share": float(label.mean()),
           "auc": {f: auc(df[f].to_numpy(dtype=float), label) for f in ("key_n", "full_n", "low_clustering", "flagged_share", "degree")}}
    common = (df.key_n >= 200) & (df.full_n >= 10)
    out["split_share_by_name"] = {"common_key_and_full_name": float(df.split[common].mean()), "other": float(df.split[~common].mean())}
    out["split_share_by_degree_rank"] = {f"top{n}": float(df.sort_values("degree", ascending=False).split.head(n).mean())
                                         for n in (100, 1000, 10000, 50000)}
    save_result("a6_validate", out)
    print(out)


if __name__ == "__main__":
    main()
