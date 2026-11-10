"""Parity against the original full-edge B4 formulas and A5 DuckDB query.

Run from the repository root:
  .tmp/venv-paper/bin/python -m unittest discover -s paper/analysis/tests -p test_country_aggregates.py -v

Fixtures include missing countries/fields/communities, true community -1 labels,
within-country edges (once overall, twice by endpoint field), foreign-only fields,
and international links between two countries in the shared 'other' bucket.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import a5_clusters as a5
import b4_countries as b4


ATOL = RTOL = 2e-12


def legacy_b4(attrs, edge, top):
    """The former NumPy implementation, with a smaller configurable top-country list."""
    names = attrs.country.value_counts().head(top).index.tolist()
    top = len(names)
    cc = attrs.country.map(pd.Series(np.arange(top), index=names)).fillna(top).to_numpy(np.int64)
    known = attrs.country.notna().to_numpy()
    country = attrs.country.astype("category").cat.codes.to_numpy()
    fields = attrs.field.astype("category")
    fd, field_names = fields.cat.codes.to_numpy(), list(fields.cat.categories)
    s, d, w = edge.src.to_numpy(), edge.dst.to_numpy(), edge.weight.to_numpy(dtype=np.float64)
    both = known[s] & known[d]
    s, d, w = s[both], d[both], w[both]
    same = country[s] == country[d]
    W = np.zeros((top + 1, top + 1))
    np.add.at(W, (cc[s], cc[d]), w)
    W = W + W.T
    strength = W.sum(axis=1)
    affinity = W / (np.outer(strength, strength) / strength.sum())
    out = {"countries": names, "share_links_known": float(both.mean()),
           "share_weight_abroad_overall": float(1 - np.average(same, weights=w))}
    abroad = {}
    for i, c in enumerate(names):
        inc = (cc[s] == i) | (cc[d] == i)
        abroad[c] = float(1 - np.average(same[inc], weights=w[inc]))
    out["share_abroad"] = abroad
    pairs = [(names[i], names[j], float(affinity[i, j]), float(W[i, j]))
             for i in range(top) for j in range(i + 1, top)]
    out["affinity_top"] = sorted(pairs, key=lambda p: -p[2])[:25]
    out["affinity_bottom"] = sorted(pairs, key=lambda p: p[2])[:15]
    out["affinity_self"] = {c: float(affinity[i, i]) for i, c in enumerate(names)}
    out["affinity_matrix"] = affinity[:top, :top].round(3).tolist()
    Wi = W.copy()
    np.fill_diagonal(Wi, 0)
    si = Wi.sum(axis=1)
    intl = Wi / (np.outer(si, si) / si.sum())
    out["intl_weight_matrix"] = Wi[:top, :top].round(1).tolist()
    out["intl_affinity_matrix"] = intl[:top, :top].round(3).tolist()
    ipairs = [(names[i], names[j], float(intl[i, j]), float(Wi[i, j]))
              for i in range(top) for j in range(i + 1, top)]
    out["intl_affinity_top"] = sorted(ipairs, key=lambda p: -p[2])[:25]
    out["intl_affinity_bottom"] = sorted(ipairs, key=lambda p: p[2])[:15]
    by_field = {}
    for j, f in enumerate(field_names):
        row = {}
        for i, c in enumerate(names[:15]):
            inc_s = (cc[s] == i) & (fd[s] == j)
            inc_d = (cc[d] == i) & (fd[d] == j)
            ww = np.concatenate([w[inc_s], w[inc_d]])
            sc = np.concatenate([same[inc_s], same[inc_d]])
            row[c] = float(1 - np.average(sc, weights=ww)) if ww.sum() > 0 else None
        by_field[f] = row
    out["share_abroad_by_field"] = by_field
    return out, cc, country, fd, field_names


def legacy_a5(attrs, edge):
    """The original two-ended SQL aggregation, suitable for small fixtures only."""
    with duckdb.connect() as c:
        c.register("a", attrs)
        c.register("e", edge)
        c.execute("""CREATE TABLE ends AS
            SELECT a.field, a.country cu, b.country cv, e.weight w, a.community ku, b.community kv
            FROM e JOIN a ON a.node_idx = e.src JOIN a b ON b.node_idx = e.dst
            UNION ALL
            SELECT b.field, b.country, a.country, e.weight, b.community, a.community
            FROM e JOIN a ON a.node_idx = e.src JOIN a b ON b.node_idx = e.dst""")
        within = c.execute("SELECT sum(w) FILTER (WHERE ku = kv) / sum(w) FROM ends").fetchone()[0]
        dom = c.execute("""
            WITH k AS (SELECT * FROM ends WHERE cu IS NOT NULL AND cv IS NOT NULL AND field IS NOT NULL),
                 obs AS (SELECT field, sum(w) FILTER (WHERE cu = cv) / sum(w) AS observed,
                                sum(w) AS weight FROM k GROUP BY 1),
                 mix AS (SELECT field, cu, sum(w) AS wc FROM k GROUP BY 1, 2),
                 expd AS (SELECT field, sum((wc / tot) * (wc / tot)) AS expected
                          FROM (SELECT field, cu, wc, sum(wc) OVER (PARTITION BY field) AS tot FROM mix)
                          GROUP BY 1)
            SELECT field, observed, expected, (observed - expected) / (1 - expected) AS homophily, weight
            FROM obs JOIN expd USING (field) ORDER BY homophily DESC""").fetchdf()
    return np.nan if within is None else within, dom.set_index("field").sort_index()


def hand_fixture():
    attrs = pd.DataFrame({
        "node_idx": np.arange(16),
        "country": ["A", "A", "B", "B", "C", "D", None, "A", "B", "C", "D", "E", None, "A", "F", "F"],
        "field": ["Medicine", "Physics", "Medicine", "Physics", "Medicine", "Physics", "Medicine", None,
                  "ForeignOnly", "Medicine", None, "ForeignOnly", "Medicine", "Medicine", "Unused", "Unused"],
        "community": [0, 0, 1, 1, 2, -1, -1, None, 3, 2, -1, 3, None, 0, 9, 9],
    })
    edge = pd.DataFrame({
        "src": [0, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0],
        "dst": [1, 2, 3, 3, 5, 10, 7, 8, 9, 10, 11, 13, 13, 13],
        "weight": [2., 3., 5., 7., 11., 13., 17., 19., 23., 29., 31., 37., 41., 43.],
    })
    return attrs, edge


class CountryAggregateParity(unittest.TestCase):
    def assert_nested_equal(self, left, right, path=""):
        if isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys(), path)
            for key in left:
                self.assert_nested_equal(left[key], right[key], path + "." + key)
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right), path)
            for i, (x, y) in enumerate(zip(left, right)):
                self.assert_nested_equal(x, y, path + f"[{i}]")
        elif isinstance(left, (float, np.floating)):
            np.testing.assert_allclose(left, right, rtol=RTOL, atol=ATOL, equal_nan=True, err_msg=path)
        else:
            self.assertEqual(left, right, path)

    def check_fixture(self, attrs, edge, batches):
        expected, cc, country, fd, fields = legacy_b4(attrs, edge, top=2)
        expected_within, expected_dom = legacy_a5(attrs, edge)
        community = pd.factorize(attrs.community, sort=False)[0].astype(np.int32)
        countries = len(attrs.country.astype("category").cat.categories)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "edges.parquet"
            pq.write_table(pa.Table.from_pandas(edge, preserve_index=False), file, row_group_size=113)
            for batch_size in batches:
                with self.subTest(batch_size=batch_size):
                    agg = b4.aggregate_edges(file, cc, country, fd, len(expected["countries"]), len(fields), batch_size)
                    self.assert_nested_equal(expected, b4.summarize(agg, expected["countries"], fields, fields))
                    agg = a5.aggregate_edges(file, community, country, fd, countries, len(fields), batch_size)
                    within, dom = a5.summarize(agg, fields)
                    dom = dom.set_index("field").sort_index()
                    np.testing.assert_allclose(within, expected_within, rtol=RTOL, atol=ATOL, equal_nan=True)
                    pd.testing.assert_index_equal(dom.index, expected_dom.index)
                    np.testing.assert_allclose(dom.to_numpy(), expected_dom.to_numpy(), rtol=RTOL, atol=ATOL, equal_nan=True)

    def test_unknowns_incident_counting_and_other_bucket(self):
        self.check_fixture(*hand_fixture(), batches=(1, 3, 7, 100))

    def test_no_within_community_links_retains_sql_null(self):
        attrs, edge = hand_fixture()
        attrs["community"] = np.arange(len(attrs))
        self.check_fixture(attrs, edge, batches=(3, 100))

    def test_random_edges_across_row_groups_and_batches(self):
        rng = np.random.default_rng(442)
        n = 2_000
        attrs = pd.DataFrame({"node_idx": np.arange(n),
                              "country": rng.choice(["A", "A", "B", "B", "C", "D", "E", None], n),
                              "field": rng.choice(["Medicine", "Physics", "Arts", None], n),
                              "community": rng.choice([0, 1, 2, -1, None], n)})
        s = rng.integers(0, n, 50_000, dtype=np.int32)
        d = rng.integers(0, n, 50_000, dtype=np.int32)
        keep = s != d
        edge = pd.DataFrame({"src": s[keep], "dst": d[keep],
                             "weight": rng.uniform(.01, 4, keep.sum()).astype(np.float32)})
        self.check_fixture(attrs, edge, batches=(1024, 100_000))


if __name__ == "__main__":
    unittest.main()
