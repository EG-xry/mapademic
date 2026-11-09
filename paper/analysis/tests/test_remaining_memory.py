"""Small-fixture parity for the remaining-run memory reductions."""
import ast
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import weakref

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ANALYSIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS))
import b1_uncited
import a8b_communities
import a6_validate
from common import name_script
from kernels import contract, highest_degree_members, histograms, summary


def ast_segment(file, start, end, environment):
    body = ast.parse((ANALYSIS / file).read_text()).body
    begin = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign) and
                 isinstance(node.targets[0], ast.Name) and node.targets[0].id == start)
    finish = next(i for i, node in enumerate(body[begin + 1:], begin + 1) if isinstance(node, ast.Assign) and
                  isinstance(node.targets[0], ast.Name) and node.targets[0].id == end)
    exec(compile(ast.Module(body=body[begin:finish], type_ignores=[]), file, 'exec'), environment)


class RemainingMemory(unittest.TestCase):
    def test_a6_selected_validation_metadata_matches_full_pandas_lookups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = pd.DataFrame({'id': ['A10', 'A11', 'https://openalex.org/A12', 'A13'], 'works_count': [100, 0, 60, 20]})
            new = pd.DataFrame({'id': [10, 11, 12], 'works_count': [30, 0, 55]})
            identity = pd.DataFrame({'node_idx': [0, 1, 2, 3], 'key_n': [200, 3, 40, 400]})
            full = pd.DataFrame({'node_idx': [0, 1, 2, 3], 'full_n': [10, 2, 6, 20]})
            targets = pd.DataFrame({'node_idx': [3, 0, 2, 1], 'id': [13, 10, 12, 11], 'degree': [30, 25, 20, 5]})
            files = []
            for name, values in (('old', old), ('new', new), ('identity', identity), ('full', full)):
                files.append(root / f'{name}.parquet')
                values.to_parquet(files[-1], index=False)
            expected = targets.copy()
            expected['works_old'] = pd.Series(old.works_count.to_numpy(), index=old.id.str.extract(r'A(\d+)$')[0].astype('int64')).reindex(targets.id).to_numpy()
            expected['works_new'] = new.set_index('id').works_count.reindex(targets.id).to_numpy()
            expected['key_n'] = identity.set_index('node_idx').key_n.reindex(targets.node_idx).to_numpy()
            expected['full_n'] = full.set_index('node_idx').full_n.reindex(targets.node_idx).to_numpy()
            db = duckdb.connect()
            try:
                actual = a6_validate.validation_metadata(db, targets, *files)
            finally:
                db.close()
            pd.testing.assert_frame_equal(actual, expected)

    def test_b1_batched_name_classification_and_reported_name_ties(self):
        names = ['王小明', '山田 太郎', '김민수', 'Иван Иванов', 'علي', 'Ada Lovelace', None, '王小明 Cyrillic Ѐ'] * 5
        n = len(names)
        attrs = pd.DataFrame({'node_idx': np.arange(n), 'name': names, 'works': np.arange(n) + 5,
                              'works_kept': np.repeat([50, 51, 52, 53], 10), 'cited': np.arange(n) % 4,
                              'country': ['US', None, 'CN', 'JP'] * 10, 'field': ['A', 'B', None, 'B'] * 10,
                              'community': np.arange(n) % 3, 'degree': np.arange(n) % 7})
        closeness = np.arange(n, dtype=np.float32) / 10
        expected = attrs.drop(columns='name').copy()
        expected['closeness'], expected['uncited'], expected['few'] = closeness, attrs.cited == 0, attrs.cited < 10
        expected['script'] = name_script(attrs.name)
        top = attrs[(attrs.works_kept >= 50) & (attrs.cited == 0)].nlargest(15, 'works_kept')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attrs.to_parquet(root / 'attrs.parquet', index=False)
            np.save(root / 'closeness.npy', closeness)
            for batch in (1, 7, 100):
                actual, selected = b1_uncited.load_analysis_attrs(root / 'attrs.parquet', root / 'closeness.npy', batch)
                pd.testing.assert_frame_equal(actual, expected)
                self.assertEqual(selected, dict(zip(top.node_idx, top.name)))
                self.assertEqual(actual.script.value_counts().to_dict(), expected.script.value_counts().to_dict())

    def test_a7b_delivered_fast_path_matches_identity_contraction(self):
        off = np.array([0, 2, 4, 6], np.int64)
        nb = np.array([1, 2, 0, 2, 0, 1], np.int32)
        sources = np.array([0, 2], np.int64)
        body = ast.parse((ANALYSIS / 'a7b_paired.py').read_text()).body
        node = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == 'per_source')
        forbidden = mock.Mock(side_effect=AssertionError('baseline allocated a duplicate CSR'))
        env = dict(np=np, offsets=off, neighbors=nb, sources=sources, histograms=histograms, contract=forbidden)
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'a7b_paired.py', 'exec'), env)
        result = env['per_source'](np.zeros(3, bool), False)
        off2, nb2 = contract(off, nb, np.zeros(len(nb), bool), np.arange(3), 3)
        h = histograms(off2, nb2, sources, 64)
        np.testing.assert_array_equal(result, (h * np.arange(65)).sum(axis=1) / h.sum(axis=1))
        forbidden.assert_not_called()

    def test_a8b_staged_graphs_match_former_three_graph_results(self):
        off = np.arange(0, 17, 2, dtype=np.int64)
        nb = np.array([7, 1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 0], np.int32)
        skip = np.zeros(len(nb), bool)
        rep = np.array([0, 0, 1, 2, 3, 4, -1, 5], np.int64)
        comm = np.array([0, 1, 0, 0, 1, 1, 1, 1], np.int64)
        communities = {'first': 0, 'second': 1}
        keep, deg = rep >= 0, np.diff(off)
        rep_rm = np.where(keep, np.cumsum(keep) - 1, -1)
        off_rm, nb_rm = contract(off, nb, skip, rep_rm, keep.sum())
        off2, nb2 = contract(off, nb, skip, rep, rep.max() + 1)
        lab_rm = comm[keep]
        lab2 = comm[highest_degree_members(rep, np.argsort(-deg, kind='stable'), rep.max() + 1)]
        rng, expected = np.random.default_rng(20261024), {}
        for label, c in communities.items():
            members = np.flatnonzero((comm == c) & keep & (deg > 0))
            src = rng.choice(members, size=min(100, len(members)), replace=False)
            expected[label] = {'members': len(members), 'absorbed_by_other_community': float(np.mean(lab2[rep[members]] != c)),
                               'inside_share': {'delivered': float(a8b_communities.inside_share(off, nb, comm, c)),
                                                'removals_only': float(a8b_communities.inside_share(off_rm, nb_rm, lab_rm, c)),
                                                'repaired': float(a8b_communities.inside_share(off2, nb2, lab2, c))},
                               'mean_distance_removals_only': summary(histograms(off_rm, nb_rm, rep_rm[src], 64))['mean']}
        previous, outputs = [], []
        def monitored_contract(*args):
            if previous:
                self.assertIsNone(previous[-1](), 'the previous transformed adjacency is still retained')
            result = contract(*args)
            previous.append(weakref.ref(result[1]))
            return result
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            pq.write_table(pa.table({'community': comm}), cache / 'node_attrs.parquet')
            np.save(cache / 'repair_rep.npy', rep)
            with mock.patch.multiple(a8b_communities, CACHE=cache, SCRIPT_COMMUNITIES=communities,
                                     load_csr=lambda **_: (off.copy(), nb.copy()), repair_link_mask=lambda *_: skip,
                                     contract=monitored_contract, log=lambda *_: None,
                                     save_result=lambda name, value: outputs.append(value)):
                a8b_communities.main()
        self.assertEqual(len(previous), 2)
        self.assertEqual(outputs, [expected])

    def test_a7_strict_selected_count_join_matches_full_reindex(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({'node_idx': [0, 1, 2], 'key_n': [250, 300, 100]}).to_parquet(root / 'identity_signals.parquet')
            pd.DataFrame({'node_idx': [0, 1, 2], 'full_n': [10, 9, 20]}).to_parquet(root / 'full_name_counts.parquet')
            top = pd.DataFrame({'node_idx': [3, 1, 2, 0], 'peak_insts': [30, 4, 40, 30]})
            body = ast.parse((ANALYSIS / 'a7_remedy.py').read_text()).body
            block = next(node for node in body if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == 'STRICT')
            env = dict(top=top.copy(), con=duckdb.connect, CACHE=root, PG=root)
            exec(compile(ast.Module(body=block.body, type_ignores=[]), 'a7_remedy.py', 'exec'), env)
            pd.testing.assert_frame_equal(env['top'], top.iloc[[1, 3]])

    def test_a9b_fallback_semijoin_matches_original_id_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extract = root / 'authors_extract' / 'part'
            extract.mkdir(parents=True)
            attrs = pd.DataFrame({'node_idx': np.arange(4), 'id': ['A1', 'A2', 'A3', 'A4'],
                                  'field': ['F1', 'F1', None, 'F2'], 'country': ['X', None, 'Y', 'X']})
            attrs.to_parquet(root / 'node_attrs.parquet', index=False)
            authors = pa.table({'id': ['A1', 'A1', 'A4', 'A5'], 'topics': [
                [{'display_name': 'Legal and Regulatory Analysis'}], [{'display_name': 'Legal and Regulatory Analysis'}],
                [{'display_name': 'Other'}], [{'display_name': 'Legal and Regulatory Analysis'}]]})
            pq.write_table(authors, extract / 'authors.parquet')
            env = dict(pq=pq, CACHE=root, D=root, con=duckdb.connect)
            ast_segment('a9b_repaired.py', 'attrs', 'first', env)
            expected = attrs[['node_idx', 'field', 'country']].copy()
            expected.loc[attrs.id.isin({'A1', 'A5'}), 'field'] = None
            pd.testing.assert_frame_equal(env['attrs'].astype({'field': object, 'country': object}).fillna('MISSING'),
                                          expected.astype({'field': object, 'country': object}).fillna('MISSING'))


if __name__ == '__main__':
    unittest.main()
