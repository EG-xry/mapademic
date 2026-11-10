"""Independent census fixtures: every qualifying author/work and eligibility rule."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from a6_quality import qualifying_author_census
from validate_min5 import qualifying_work_counts
import identities
import validate_min5


def census_layout(root):
    """The fixture in the extract layout census_provenance reads: works, authors and nodes in their own folders."""
    works, authors, nodes = fixture(root)
    for part in ('works_authorships/p0', 'authors_extract/p0', 'network', 'results'):
        (root / part).mkdir(parents=True)
    works.replace(root / 'works_authorships/p0/works.parquet')
    authors.replace(root / 'authors_extract/p0/authors.parquet')
    nodes.replace(root / 'network/nodes.parquet')
    return census_layout_paths(root)


def census_layout_paths(root):
    return str(root / 'works_authorships/*/*.parquet'), str(root / 'authors_extract/*/*.parquet'), root / 'network/nodes.parquet'


def fixture(root):
    work_rows = []
    for i in range(1000):
        authors = [6, 8] + ([1, 3, 4, 5, 7] if i < 5 else [])
        if i == 0:
            authors += [1, None]  # Same author repeated on a work counts once.
        work_rows.append({'work_id': i, 'author_ids': authors, 'n_authors': 100 if i == 0 else len(authors),
                          'type': 'article', 'is_xpac': False, 'is_paratext': False})
    for wid, aid in ((1000, 2), (1001, 10), (1002, 11)):
        work_rows.append({'work_id': wid, 'author_ids': [aid], 'n_authors': 1,
                          'type': None, 'is_xpac': None, 'is_paratext': None})
    for wid, category in ((2000, 'dataset'), (2001, 'xpac'), (2002, 'paratext')):
        work_rows.append({'work_id': wid, 'author_ids': [9, 1], 'n_authors': 2,
                          'type': 'dataset' if category == 'dataset' else 'article',
                          'is_xpac': category == 'xpac', 'is_paratext': category == 'paratext'})
    works, authors, nodes = root / 'works.parquet', root / 'authors.parquet', root / 'nodes.parquet'
    pq.write_table(pa.Table.from_pylist(work_rows), works)
    # ID4 has no author record; ID3 placeholder, ID7 absent name, ID6 uncited
    # 1000-work bulk. ID5 has raw works_count=1000 but only5 qualifying works,
    # so the pipeline's filter must retain it despite its lack of citations.
    author_rows = [{'id': f'A{i}', 'display_name': 'et al.' if i == 3 else None if i == 7 else f'Person {i}',
                    'works_count': 1000 if i in (5, 6, 8) else 5,
                    'cited_by_count': 0 if i in (5, 6) else 1}
                   for i in (1, 2, 3, 5, 6, 7, 8, 9, 10, 11)]
    pq.write_table(pa.Table.from_pylist(author_rows), authors)
    pd.DataFrame({'id': ['A1', 'A5', 'A8'], 'works_kept': [5, 5, 1000]}).to_parquet(nodes, index=False)
    return works, authors, nodes


class AuthorCensus(unittest.TestCase):
    def test_cache_checksum_and_changed_work_input_force_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            works, authors, nodes = fixture(root)
            cache = root / 'cache'
            db = duckdb.connect()
            try:
                original = qualifying_author_census(db, str(works), str(authors), nodes, cache, 5)
                count_file = cache / 'qualifying_author_counts.parquet'
                counts = pd.read_parquet(count_file)
                counts.loc[counts.author_key == 2, 'works_kept'] = 2
                counts.to_parquet(count_file, index=False)
                repaired = qualifying_author_census(db, str(works), str(authors), nodes, cache, 5)
                self.assertEqual(repaired['authorships_total'], original['authorships_total'])
                extra = pd.read_parquet(works).iloc[[1000]].copy()
                extra['work_id'], extra['author_ids'] = 1003, pd.Series([[12]], index=extra.index)
                pd.concat([pd.read_parquet(works), extra]).to_parquet(works, index=False)
                changed = qualifying_author_census(db, str(works), str(authors), nodes, cache, 5)
                self.assertEqual(changed['author_ids_with_qualifying_work'], 11)
                self.assertEqual(changed['authorships_total'], 2029)
                self.assertNotEqual(changed['counts_fingerprint'], original['counts_fingerprint'])
            finally:
                db.close()

    def test_final_census_provenance_checks_current_sources_and_output_consistency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            works, authors, nodes = fixture(root)
            for part in ('works_authorships/p0', 'authors_extract/p0', 'network', 'results'):
                (root / part).mkdir(parents=True)
            work_file, author_file, node_file = root / 'works_authorships/p0/works.parquet', root / 'authors_extract/p0/authors.parquet', root / 'network/nodes.parquet'
            works.replace(work_file)
            authors.replace(author_file)
            nodes.replace(node_file)
            cache = root / 'cache'
            db = duckdb.connect()
            try:
                result = qualifying_author_census(db, str(root / 'works_authorships/*/*.parquet'),
                                                   str(root / 'authors_extract/*/*.parquet'), node_file, cache, 5)
            finally:
                db.close()
            (root / 'results/a6_quality.json').write_text(json.dumps({'qualifying_author_census': result}))
            with mock.patch.multiple(validate_min5, D=root, ND=root / 'network', CACHE=cache, RESULTS=root / 'results', EXPECTED_N=3):
                checked = validate_min5.census_provenance()
                self.assertTrue(checked['current_input_provenance_verified'])
                self.assertEqual(checked['retained_author_ids'], 3)
                frame = pd.read_parquet(author_file)
                frame.loc[0, 'display_name'] = 'Changed name'
                frame.to_parquet(author_file, index=False)
                with self.assertRaises(AssertionError):
                    validate_min5.census_provenance()

    def test_moved_project_reuses_the_census_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            before, after = Path(directory) / 'before', Path(directory) / 'after'
            before.mkdir()
            works, authors, nodes = census_layout(before)
            db = duckdb.connect()
            try:
                original = qualifying_author_census(db, works, authors, nodes, before / 'cache', 5)
                written = (before / 'cache/qualifying_author_counts.parquet').stat().st_mtime_ns
                os.rename(before, after)
                with mock.patch.object(identities, 'ROOT', after):
                    moved = qualifying_author_census(db, *census_layout_paths(after), after / 'cache', 5)
            finally:
                db.close()
            self.assertEqual((after / 'cache/qualifying_author_counts.parquet').stat().st_mtime_ns, written)
            self.assertEqual(moved['counts_fingerprint'], original['counts_fingerprint'])

    def test_final_census_check_passes_after_moving_the_project(self):
        # The 4 November failure: provenance recorded in ~/Downloads, validated in ~/Desktop.
        with tempfile.TemporaryDirectory() as directory:
            before, after = Path(directory) / 'before', Path(directory) / 'after'
            before.mkdir()
            works, authors, nodes = census_layout(before)
            db = duckdb.connect()
            try:
                result = qualifying_author_census(db, works, authors, nodes, before / 'cache', 5)
            finally:
                db.close()
            (before / 'results/a6_quality.json').write_text(json.dumps({'qualifying_author_census': result}))
            os.rename(before, after)
            with mock.patch.object(identities, 'ROOT', after), mock.patch.multiple(
                    validate_min5, D=after, ND=after / 'network', CACHE=after / 'cache', RESULTS=after / 'results',
                    EXPECTED_N=3):
                self.assertTrue(validate_min5.census_provenance()['current_input_provenance_verified'])

    def test_counts_coverage_exclusions_and_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            works, authors, nodes = fixture(root)
            db = duckdb.connect()
            try:
                result = qualifying_author_census(db, str(works), str(authors), nodes, root / 'cache', 5)
                repeated = qualifying_author_census(db, str(works), str(authors), nodes, root / 'cache', 5)
                uniqueness = qualifying_work_counts(db, works)
            finally:
                db.close()
            self.assertEqual(result, repeated)
            self.assertEqual(result['author_ids_with_qualifying_work'], 10)
            self.assertEqual(result['authorships_total'], 2028)
            self.assertEqual(result['exactly_one_work']['author_ids'], 3)
            self.assertEqual(result['work_count_bands']['2_to_4']['author_ids'], 0)
            self.assertEqual(result['work_count_bands']['5_to_14']['author_ids'], 5)
            self.assertEqual(result['work_count_bands']['15_plus']['authorships'], 2000)
            threshold = result['threshold']
            self.assertEqual(threshold['eligible_author_ids'], 7)
            self.assertEqual(threshold['eligible_authorships'], 2025)
            self.assertEqual(threshold['missing_author_records'], 1)
            self.assertEqual(threshold['excluded_nonperson_records'], 3)
            self.assertEqual(threshold['retained_author_ids'], 3)
            self.assertEqual(threshold['retained_authorships'], 1010)
            self.assertEqual(threshold['retained_author_share'], .3)
            self.assertEqual(threshold['retained_authorship_share'], 1010 / 2028)
            self.assertEqual(uniqueness['qualifying_work_rows'], 1003)
            self.assertEqual(uniqueness['qualifying_multi_author_rows'], 1000)
            self.assertTrue(uniqueness['includes_solo_authored_works'])
            manifest = json.loads((root / 'cache/qualifying_author_counts.provenance.json').read_text())
            self.assertEqual(manifest['cache']['author_rows'], 10)

    def test_delivered_works_counts_and_id_set_must_match(self):
        for damage in ('works', 'missing_node', 'unexpected_node'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                works, authors, nodes = fixture(root)
                frame = pd.read_parquet(nodes)
                if damage == 'works':
                    frame.loc[0, 'works_kept'] += 1
                elif damage == 'missing_node':
                    frame = frame.iloc[1:]
                else:
                    frame = pd.concat([frame, pd.DataFrame({'id': ['A2'], 'works_kept': [1]})])
                frame.to_parquet(nodes, index=False)
                db = duckdb.connect()
                try:
                    with self.assertRaisesRegex(AssertionError, 'census disagrees'):
                        qualifying_author_census(db, str(works), str(authors), nodes, root / 'cache', 5)
                finally:
                    db.close()

    def test_duplicate_eligible_author_records_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            works, authors, nodes = fixture(root)
            frame = pd.read_parquet(authors)
            pd.concat([frame, frame.iloc[:1]]).to_parquet(authors, index=False)
            db = duckdb.connect()
            try:
                with self.assertRaisesRegex(AssertionError, 'duplicate author records'):
                    qualifying_author_census(db, str(works), str(authors), nodes, root / 'cache', 5)
            finally:
                db.close()

    def test_final_uniqueness_dependency_includes_solo_works(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            works, _, _ = fixture(root)
            frame = pd.read_parquet(works)
            pd.concat([frame, frame[frame.work_id == 1000]]).to_parquet(works, index=False)
            db = duckdb.connect()
            try:
                with self.assertRaisesRegex(AssertionError, 'duplicate or null qualifying work IDs'):
                    qualifying_work_counts(db, works)
            finally:
                db.close()


if __name__ == '__main__':
    unittest.main()
