"""Independent source parity for fallback flags and nullable map coordinates."""
from pathlib import Path
import sys
import tempfile
import unittest

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validate_min5 import aligned_batches, check_supplemental_batch, expected_researcher_metadata


def fixture(root):
    attrs = pa.table({'node_idx': np.arange(5, dtype=np.int32),
                      'id': ['https://openalex.org/A104', 'https://openalex.org/A101', 'https://openalex.org/A102', 'A103', 'A105'],
                      'on_map': [True, True, False, True, False]})
    authors = pa.table({'id': ['https://openalex.org/A102', 'https://openalex.org/A102', 'https://openalex.org/A104',
                              'A105', 'https://openalex.org/A101', 'A999', 'A104', 'https://openalex.org/A105', 'A103'],
                        'topics': [[{'display_name': 'Legal and Regulatory Analysis'}],
                                   [{'display_name': 'Legal and Regulatory Analysis'}],
                                   [{'display_name': 'Other'}, {'display_name': 'Legal and Regulatory Analysis'}],
                                   [], None, [{'display_name': 'Legal and Regulatory Analysis'}],
                                   [{'display_name': 'Legal and Regulatory Analysis'}],
                                   [{'display_name': 'Legal and Regulatory Analysis'}],
                                   [{'display_name': 'Legal and Regulatory Analysis'}]]})
    coords = pa.table({'id': ['https://openalex.org/A101', 'A104', 'A103'],
                       'xw': pa.array([1.1234567891234567, None, float('nan')], type=pa.float64()),
                       'yw': pa.array([2.234567891234567, 3.345678912345678, None], type=pa.float64())})
    for name, table in (('attrs', attrs), ('authors', authors), ('coords', coords)):
        pq.write_table(table, root / f'{name}.parquet')
    deposited = pa.table({'node': np.arange(5, dtype=np.int32), 'on_map': [True, True, False, True, False],
                          'is_fallback_topic': [False, False, True, True, False],
                          'xw': pa.array([None, 1.1234567891234567, None, float('nan'), None], type=pa.float64()),
                          'yw': pa.array([3.345678912345678, 2.234567891234567, None, None, None], type=pa.float64())})
    return deposited


def checks(root, deposited):
    db = duckdb.connect()
    expected = expected_researcher_metadata(db, root / 'attrs.parquet', root / 'authors.parquet', root / 'coords.parquet', batch_size=3)
    try:
        counts = {}
        for actual, source in aligned_batches(deposited.to_batches(max_chunksize=2), expected):
            for key, value in check_supplemental_batch(actual, source).items():
                counts[key] = counts.get(key, 0) + value
        return counts
    finally:
        expected.close()
        db.close()


class DepositMetadata(unittest.TestCase):
    def test_independent_source_join_preserves_null_nan_membership_and_first_topic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deposited = fixture(root)
            counts = checks(root, deposited)
            self.assertEqual(counts['coordinate_id_matches'], 3)
            # Exact URL+URL and short+short matches are true; both directions
            # of URL/short mismatch stay false, as in the analysis ID joins.
            self.assertEqual(counts['fallback_topic_records'], 2)
            self.assertEqual(counts['xw_nulls'], 3)
            self.assertEqual(counts['xw_nan_values'], 1)
            self.assertEqual(counts['yw_nulls'], 3)
            self.assertEqual(counts['yw_nan_values'], 0)

    def test_each_metadata_corruption_is_rejected(self):
        for damage in ('fallback', 'mapped_null', 'offmap_coordinate', 'null_to_nan', 'coordinate_precision', 'nullable_boolean'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                deposited = fixture(root)
                if damage in ('fallback', 'mapped_null', 'nullable_boolean'):
                    key = 'is_fallback_topic' if damage == 'fallback' else 'on_map'
                    values = deposited[key].to_pylist()
                    values[0] = True if damage == 'fallback' else False if damage == 'mapped_null' else None
                    array = pa.array(values, type=pa.bool_())
                else:
                    key = 'xw'
                    values = deposited[key].to_pylist()
                    if damage == 'offmap_coordinate':
                        values[2] = 0.0
                    elif damage == 'null_to_nan':
                        values[0] = float('nan')
                    else:
                        values[1] += 1e-10
                    array = pa.array(values, type=pa.float64())
                changed = deposited.set_column(deposited.schema.get_field_index(key), key, array)
                with self.assertRaises(AssertionError):
                    checks(root, changed)

    def test_duplicate_numeric_coordinate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deposited = fixture(root)
            coords = pq.read_table(root / 'coords.parquet')
            pq.write_table(pa.concat_tables([coords, coords.slice(1, 1)]), root / 'coords.parquet')
            with self.assertRaisesRegex(AssertionError, 'one row per deposited node|row counts differ'):
                checks(root, deposited)

    def test_independent_reader_lengths_and_empty_inputs(self):
        a = pa.table({'value': [1, 2, 3]})
        b = pa.table({'value': [1, 2]})
        with self.assertRaisesRegex(AssertionError, 'row counts differ'):
            list(aligned_batches(a.to_batches(max_chunksize=2), b.to_batches(max_chunksize=1)))
        self.assertEqual(list(aligned_batches([], [])), [])


if __name__ == '__main__':
    unittest.main()
