"""Bounded counts preserve pandas proportions and make category ties deterministic."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from map_labels import FALLBACK_TOPIC, community_names, community_statistics


def old_statistics(attrs, authors, minimum):
    attrs = attrs.copy()
    fallback = {row["id"] for row in authors if row["topics"]
                and row["topics"][0]["display_name"] == FALLBACK_TOPIC}
    attrs.loc[attrs.id.isin(fallback), "field"] = None
    sizes = attrs.community.value_counts()
    sizes = sizes[sizes >= minimum]
    attrs = attrs[attrs.community.isin(sizes.index)]
    world = attrs.country.value_counts(normalize=True).to_dict()
    groups = []
    for name in ("field", "country"):
        series = attrs.dropna(subset=[name]).groupby("community")[name].value_counts(normalize=True)
        groups.append({int(c): list(series[c].items()) for c in series.index.get_level_values(0).unique()})
    return sizes, world, *groups, attrs.groupby("community").is_dust.any().to_dict()


class MapLabelAggregateTests(unittest.TestCase):
    def compare(self, attrs, authors, minimum):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "attrs.parquet"
            attrs.to_parquet(path, index=False)
            author_path = root / "authors.parquet"
            schema = pa.schema([("id", pa.string()),
                                ("topics", pa.list_(pa.struct([("display_name", pa.string())])))])
            pq.write_table(pa.Table.from_pylist(authors, schema=schema), author_path)
            actual = community_statistics(path, author_path, minimum)
            expected = old_statistics(attrs, authors, minimum)
            self.assertEqual(list(actual[0].items()), list(expected[0].items()))
            self.assertEqual(actual[1], expected[1])
            self.assertEqual(actual[4], expected[4])
            reference = attrs.copy()
            fallback = {row["id"] for row in authors if row["topics"]
                        and row["topics"][0]["display_name"] == FALLBACK_TOPIC}
            reference.loc[reference.id.isin(fallback), "field"] = None
            for key, got, want in zip(("field", "country"), actual[2:4], expected[2:4]):
                # Counts/proportions must agree exactly, independently of the
                # old pandas release's incidental ordering of equal counts.
                self.assertEqual({c: dict(rows) for c, rows in got.items()},
                                 {c: dict(rows) for c, rows in want.items()})
                for community, rows in want.items():
                    first = reference.loc[reference.community == community, key].dropna().drop_duplicates().tolist()
                    prescribed = sorted(rows, key=lambda row: (-row[1], first.index(row[0])))
                    self.assertEqual(got[community], prescribed)
                    want[community] = prescribed
            self.assertEqual(community_names(*actual, scripts={}), community_names(*expected, scripts={}))

    def test_ties_exact_ids_missing_values_and_small_community_denominator(self):
        attrs = pd.DataFrame({
            "node_idx": np.arange(17),
            "id": [f"https://openalex.org/A{i}" for i in range(17)],
            "community": [2, 1, 2, 1, 4, 4, 9, 9, 10, 10, 99, 2, 1, 4, 9, 10, 2],
            "field": ["Z", "A", "A", "Z", "Medicine", "Medicine", None, None, None, None,
                      "Medicine", "Z", "Z", "Medicine", None, None, "A"],
            "country": ["JP", "US", "US", "JP", "FR", "FR", None, None, None, None,
                        "US", "JP", "US", "FR", None, None, "US"],
            "is_dust": [False] * 6 + [True] * 4 + [False] * 7})
        authors = [
            {"id": "https://openalex.org/A0", "topics": [{"display_name": FALLBACK_TOPIC}]},
            {"id": "A1", "topics": [{"display_name": FALLBACK_TOPIC}]},  # alias must not match
            {"id": "https://openalex.org/A4", "topics": [{"display_name": "Medicine"},
                                                          {"display_name": FALLBACK_TOPIC}]},
            {"id": "https://openalex.org/A5", "topics": []},
            {"id": "https://openalex.org/A5", "topics": [{"display_name": FALLBACK_TOPIC}]},
            {"id": "https://openalex.org/A6", "topics": None},
        ]
        self.compare(attrs, authors, 2)

    def test_random_count_and_tie_parity(self):
        for seed in range(8):
            rng = np.random.default_rng(seed)
            count = 200
            attrs = pd.DataFrame({"node_idx": np.arange(count), "id": [f"A{i}" for i in range(count)],
                                  "community": rng.integers(0, 12, count),
                                  "field": rng.choice([None, "Medicine", "Z", "A"], count),
                                  "country": rng.choice([None, "US", "JP", "FR"], count),
                                  "is_dust": rng.random(count) < .1})
            authors = [{"id": f"A{i}", "topics": [{"display_name": FALLBACK_TOPIC}]} for i in range(0, count, 7)]
            self.compare(attrs, authors, 15)

    def test_all_unknown_fields_and_countries(self):
        attrs = pd.DataFrame({"node_idx": range(4), "id": ["A1", "A2", "A3", "A4"],
                              "community": [9, 10, 10, 9], "field": [None] * 4,
                              "country": [None] * 4, "is_dust": [False] * 4})
        self.compare(attrs, [{"id": "A1", "topics": []}], 2)


if __name__ == "__main__":
    unittest.main()
