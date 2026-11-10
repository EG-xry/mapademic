"""RV remoteness: how much of each field's remoteness does its researchers' degree predict?

In random networks with given degrees, a researcher's mean distance to everyone falls with the logarithm of their
degree (Fronczak, Fronczak and Holyst 2004; Holyst et al. 2005), and a sum of two such terms is additive, like A9c's
field fit. Here each giant-component researcher's estimated mean distance to everyone (A4: the mean distance to A2's
1,000 sources, cache/closeness_1000.npy) is compared with the mean over all researchers of the same degree (exact
degree up to 300, then bins of 0.05 in log degree). For each field (first topic's field; fallback-topic records left
out, as in A9 and A9c):
  - raw: the field's mean distance to everyone;
  - predicted: the mean of its researchers' degree-predicted distances;
  - residual = raw - predicted, the field's remoteness at matched degree.
Then, over the 325 pairs of different fields of A9's matrix (symmetrized as in A9c), a degree-only model
d_ij = alpha + beta (p_i + p_j), p the fields' predicted distances, against A9c's free fit d_ij = a_i + a_j; and each
field's cohesion residual d_ii - 2 a_i, its distance within the field against the additive prediction.
Writes results/rv_remoteness.json.
"""
import numpy as np

EXACT_DEGREE, LOG_BIN = 300, 0.05
FALLBACK_TOPIC = "Legal and Regulatory Analysis"


def degree_bins(degree):
    """Exact degree up to EXACT_DEGREE, then bins of LOG_BIN in natural log degree."""
    d = np.asarray(degree, np.float64)
    out = d.astype(np.int64)
    big = d > EXACT_DEGREE
    out[big] = EXACT_DEGREE + 1 + np.floor((np.log(d[big]) - np.log(EXACT_DEGREE)) / LOG_BIN).astype(np.int64)
    return out


def degree_prediction(degree, distance):
    """Each researcher's predicted distance: the mean distance of all researchers in the same degree bin."""
    b = degree_bins(degree)
    sums = np.bincount(b, weights=distance)
    counts = np.bincount(b)
    return (sums / np.maximum(counts, 1))[b]


def pairs_design(n):
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    X = np.zeros((len(pairs), n))
    for k, (i, j) in enumerate(pairs):
        X[k, i] = X[k, j] = 1
    return pairs, X


def r_squared(y, fit):
    return float(1 - ((y - fit) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def additive_fit(s):
    """A9c's fit: least squares of d_ij = a_i + a_j over pairs of different groups, no intercept."""
    pairs, X = pairs_design(len(s))
    y = np.array([s[i, j] for i, j in pairs])
    a, *_ = np.linalg.lstsq(X, y, rcond=None)
    return a, r_squared(y, X @ a)


def degree_only_fit(s, p):
    """d_ij = alpha + beta (p_i + p_j) over pairs of different groups: (alpha, beta, R2)."""
    pairs, _ = pairs_design(len(s))
    y = np.array([s[i, j] for i, j in pairs])
    X = np.column_stack([np.ones(len(pairs)), [p[i] + p[j] for i, j in pairs]])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[0]), float(coef[1]), r_squared(y, X @ coef)


def main():
    import json

    import pandas as pd
    import pyarrow.parquet as pq

    from common import CACHE, D, RESULTS, con, log, save_result
    distance = np.load(CACHE / "closeness_1000.npy")
    attrs = pq.read_table(CACHE / "node_attrs.parquet", columns=["node_idx", "field", "degree"]).to_pandas()
    assert np.array_equal(attrs.node_idx.to_numpy(), np.arange(len(distance)))
    fallback = con().execute(f"""SELECT n.node_idx FROM read_parquet('{CACHE / 'node_attrs.parquet'}') n
                                 JOIN (SELECT id FROM read_parquet('{D}/authors_extract/*/*.parquet')
                                       WHERE topics[1].display_name = '{FALLBACK_TOPIC}') f USING (id)""").fetchnumpy()
    field = np.array(attrs.field.astype(object).to_numpy(), dtype=object, copy=True)   # writable copy
    field[fallback["node_idx"]] = None
    finite = np.isfinite(distance)
    degree = attrs.degree.to_numpy()
    assert (degree[finite] > 0).all()
    predicted = np.full(len(distance), np.nan)
    predicted[finite] = degree_prediction(degree[finite], distance[finite])
    log(f"{int(finite.sum()):,} researchers with a distance estimate; {len(fallback['node_idx']):,} fallback-topic records")

    a9 = json.loads((RESULTS / "a9_structure.json").read_text())["field"]
    names, m = a9["names"], np.array(a9["mean_distance"])
    s = (m + m.T) / 2
    a, r2 = additive_fit(s)
    a9c = json.loads((RESULTS / "a9c_field_order.json").read_text())
    assert abs(r2 - a9c["additive_r2"]) < 1e-9
    frame = pd.DataFrame({"field": field[finite], "raw": distance[finite], "predicted": predicted[finite]})
    per = frame.groupby("field").agg(n=("raw", "size"), raw=("raw", "mean"), predicted=("predicted", "mean"))
    per = per.reindex(names)
    assert per.n.notna().all()
    per["residual"] = per.raw - per.predicted
    alpha, beta, r2_degree = degree_only_fit(s, per.predicted.to_numpy())
    within = np.diag(m)
    cohesion = within - 2 * a
    imm, arts = names.index("Immunology and Microbiology"), names.index("Arts and Humanities")
    gap = {"remoteness_a": float(a[arts] - a[imm]), "raw": float(per.raw.iloc[arts] - per.raw.iloc[imm]),
           "predicted_by_degree": float(per.predicted.iloc[arts] - per.predicted.iloc[imm]),
           "residual": float(per.residual.iloc[arts] - per.residual.iloc[imm])}
    gap["share_predicted_by_degree"] = gap["predicted_by_degree"] / gap["raw"]
    out = {"researchers": int(finite.sum()), "fallback_records": int(len(fallback["node_idx"])),
           "binning": f"exact degree up to {EXACT_DEGREE}, then {LOG_BIN} in natural log degree",
           "fields": {name: {"n": int(per.n.iloc[i]), "raw": float(per.raw.iloc[i]),
                             "predicted": float(per.predicted.iloc[i]), "residual": float(per.residual.iloc[i]),
                             "remoteness_a": float(a[i]), "within": float(within[i]), "cohesion": float(cohesion[i])}
                      for i, name in enumerate(names)},
           "additive_r2": r2, "degree_only": {"alpha": alpha, "beta": beta, "r2": r2_degree},
           "spread": {"raw_sd": float(per.raw.std()), "residual_sd": float(per.residual.std()),
                      "raw_range": float(per.raw.max() - per.raw.min()),
                      "residual_range": float(per.residual.max() - per.residual.min())},
           "arts_vs_immunology": gap,
           "cohesion_range": [float(cohesion.min()), float(cohesion.max())], "cohesion_median": float(np.median(cohesion))}
    save_result("rv_remoteness", out)
    log({k: out[k] for k in ("additive_r2", "degree_only", "spread", "arts_vs_immunology", "cohesion_range")})


if __name__ == "__main__":
    main()
