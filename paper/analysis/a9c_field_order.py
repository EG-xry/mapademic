"""A9c: is the order of fields expected? (A9's field matrix)

1. One number per field: fit d_ij = a_i + a_j to the symmetrized distances between different fields (least
   squares) and report the share of their variance it explains. A field with a large a_i is far from every other
   field and from itself: paths between its researchers run through the rest of the network.
2. Team size: the Spearman correlation, over the fields, between the distance within a field and its researchers'
   median number of coauthors (fallback-topic records left out, as in A9).
"""
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import CACHE, D, RESULTS, con, log, save_result

f = json.loads((RESULTS / "a9_structure.json").read_text())["field"]
names, m = f["names"], np.array(f["mean_distance"])
n = len(names)
s = (m + m.T) / 2
pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
X = np.zeros((len(pairs), n))
for k, (i, j) in enumerate(pairs):
    X[k, i] = X[k, j] = 1
y = np.array([s[i, j] for i, j in pairs])
a, *_ = np.linalg.lstsq(X, y, rcond=None)
fit = X @ a
r2 = 1 - ((y - fit) ** 2).sum() / ((y - y.mean()) ** 2).sum()
within = np.diag(m)

attrs = pq.read_table(CACHE / "node_attrs.parquet", columns=["id", "field", "degree"]).to_pandas()
fallback = con().execute(f"""SELECT id FROM read_parquet('{D}/authors_extract/*/*.parquet')
                             WHERE topics[1].display_name = 'Legal and Regulatory Analysis'""").fetchdf().id
attrs = attrs[~attrs.id.isin(set(fallback)) & (attrs.degree > 0)]
median_coauthors = attrs.groupby("field").degree.median().reindex(names).to_numpy()
rank = lambda v: pd.Series(v).rank().to_numpy()
rho_within = float(np.corrcoef(rank(within), rank(median_coauthors))[0, 1])
rho_offset = float(np.corrcoef(rank(a), rank(median_coauthors))[0, 1])
residual = sorted(((float(y[k] - fit[k]), names[i], names[j]) for k, (i, j) in enumerate(pairs)))
res = y - fit
grid = np.full((n, n), np.nan)
for k, (i, j) in enumerate(pairs):
    grid[i, j] = grid[j, i] = res[k]
out = {"fields": n, "pairs": len(pairs), "additive_r2": float(r2), "rmse": float(np.sqrt(((y - fit) ** 2).mean())),
       "fit": "unweighted least squares of the 325 symmetrized between-field distances on d_ij = a_i + a_j; no "
              "intercept, no constraints; R2 = 1 - SS_residual / SS_total about the mean; RMSE = sqrt(mean residual^2)",
       "offsets": dict(zip(names, a.round(3).tolist())), "median_coauthors": dict(zip(names, median_coauthors.tolist())),
       "spearman_within_vs_median_coauthors": rho_within, "spearman_offset_vs_median_coauthors": rho_offset,
       "closer_than_fit": residual[:12], "farther_than_fit": residual[-8:],
       "residual_summary": {"mean_abs": float(np.abs(res).mean()), "max_abs": float(np.abs(res).max()),
                            "share_within_0.05": float((np.abs(res) <= 0.05).mean()),
                            "share_within_0.10": float((np.abs(res) <= 0.10).mean())},
       "residual_matrix": [[None if np.isnan(v) else round(float(v), 3) for v in row] for row in grid]}
save_result("a9c_field_order", out)
log({k: v for k, v in out.items() if k not in ("offsets", "median_coauthors", "closer_than_fit", "farther_than_fit")})
log("closer than the fit", [(round(r, 2), x, z) for r, x, z in residual[:8]])
