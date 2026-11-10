"""Application identity IDs must survive the actual graph and viewer exports."""
import json

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipeline.build_edges import build_edges
from pipeline.build_index import build_id_shards
from pipeline.pathgraph import build_pathgraph
from pipeline.prep_graph import prep_graph
from scripts.export_ids_bin import export_ids
from scripts.export_paths_bin import ID_BASE, export_paths, read_neighbors
from tests.conftest import write_extract_output


def test_signed_author_keys_preserve_namespace_and_work_ids():
    from pipeline.identifiers import author_key, encode_author_id, decode_author_id, IDENTITY_TAG
    from pipeline.works import id_key_sql

    assert author_key('A5000000001') == 5000000001
    assert author_key('M5000000001') == -5000000001
    assert encode_author_id('M5000000001') == IDENTITY_TAG | 5000000001
    for aid in ['A0', 'A5000000001', 'M5000000001', 'M9007199254740991']:
        assert decode_author_id(encode_author_id(aid)) == aid
    con = duckdb.connect()
    sql = id_key_sql('id', repaired=True)
    assert con.execute(f"SELECT {sql} FROM (VALUES ('A5000000001'), ('M5000000001'), ('W42')) t(id)").fetchall() == [
        (5000000001,), (-5000000001,), (42,),
    ]
    for aid in ['M0', 'M9007199254740992', 'not-an-author', 'https://evil.test/A123']:
        with pytest.raises(ValueError):
            encode_author_id(aid)


def _web(path, ids):
    pq.write_table(pa.table({
        'id': ids, 'display_name': ['Name ' + aid for aid in ids],
        'xw': [0.1 + i * 0.2 for i in range(len(ids))], 'yw': [0.4] * len(ids),
        'community': [1] * len(ids), 'works_count': [20] * len(ids),
        'cited_by_count': [10 + i for i in range(len(ids))],
        'institution': ['Institution'] * len(ids), 'field': ['Field'] * len(ids),
        'is_ring': [False] * len(ids),
    }), path)


def test_repaired_identity_roundtrip_edges_csr_paths_search_and_cloud(tmp_path):
    from pipeline.identifiers import decode_author_id

    # Equal numeric suffixes must not collapse A and M into the same person.
    ids = ['A5000000001', 'M5000000001', 'A5000000002']
    nodes = tmp_path / 'nodes.parquet'
    pq.write_table(pa.table({'id': ids, 'display_name': ['Original', 'Partition', 'Collaborator']}), nodes)
    works = tmp_path / 'works.parquet'
    write_extract_output(works, [('W1', 2, ['A5000000001', 'A5000000002'])])
    # The resolver writes signed numeric keys in its repaired extract.
    con = duckdb.connect()
    con.execute(f"COPY (SELECT * REPLACE ([-5000000001, 5000000002]::BIGINT[] AS author_ids) FROM '{works}') TO '{tmp_path / 'repaired.parquet'}' (FORMAT PARQUET)")
    edge_path = tmp_path / 'edges.parquet'
    assert build_edges(str(tmp_path / 'repaired.parquet'), str(nodes), str(edge_path)) == 1
    edge = con.execute(f"SELECT src, dst FROM '{edge_path}'").fetchone()
    assert set(edge) == {'M5000000001', 'A5000000002'}
    graph = tmp_path / 'graph'
    prep_graph(str(nodes), str(edge_path), graph)
    build_pathgraph(str(graph / 'nodes_int32.parquet'), str(graph / 'edges_int32.parquet'), tmp_path / 'pg')
    encoded = np.load(tmp_path / 'pg/ids.npy')
    assert set(map(lambda x: decode_author_id(int(x)), encoded)) == set(ids)
    web = tmp_path / 'web.parquet'
    _web(web, ids)
    meta = export_paths(tmp_path / 'pg', str(web), tmp_path / 'paths', chunk_bytes=64)
    assert meta['version'] == 3
    assert meta['ids_record_bytes'] == 8
    assert meta['id_encoding'] == 'tagged-u64'
    raw_ids = np.fromfile(tmp_path / 'paths' / meta['files']['ids'], dtype='<u8')
    path_ids = [decode_author_id(int(x)) for x in raw_ids]
    assert set(path_ids) == set(ids)
    by_id = {aid: i for i, aid in enumerate(path_ids)}
    assert read_neighbors(tmp_path / 'paths', by_id['A5000000001']) == []
    assert read_neighbors(tmp_path / 'paths', by_id['M5000000001']) == [(by_id['A5000000002'], False)]
    build_id_shards(str(web), tmp_path / 'index', tmp_path / 'paths')
    same_bucket = json.loads((tmp_path / 'index/ids/1.json').read_text())
    assert same_bucket['A5000000001'][2] == by_id['A5000000001']
    assert same_bucket['M5000000001'][2] == by_id['M5000000001']
    assert same_bucket['A5000000001'][:2] != same_bucket['M5000000001'][:2]
    cloud = tmp_path / 'cloud'
    cloud_meta = export_ids(str(web), cloud)
    assert cloud_meta['ids']['id_encoding'] == 'tagged-u64'
    assert [decode_author_id(int(x)) for x in np.fromfile(cloud / 'ids.bin', dtype='<u8')] == ids[::-1]


def test_legacy_release_retains_v2_uint32_ids_and_cloud_a_prefix(tmp_path):
    ids = ['A5000000001', 'A5000000002']
    pq.write_table(pa.table({'node_idx': pa.array([0, 1], pa.int32()), 'id': ids,
                             'display_name': ['A', 'B']}), tmp_path / 'nodes.parquet')
    pq.write_table(pa.table({'src': pa.array([0], pa.int32()), 'dst': pa.array([1], pa.int32()),
                             'weight': pa.array([1], pa.float32())}), tmp_path / 'edges.parquet')
    build_pathgraph(str(tmp_path / 'nodes.parquet'), str(tmp_path / 'edges.parquet'), tmp_path / 'pg')
    web = tmp_path / 'web.parquet'
    _web(web, ids)
    meta = export_paths(tmp_path / 'pg', str(web), tmp_path / 'paths')
    assert meta['version'] == 2
    assert 'ids_record_bytes' not in meta
    raw = (tmp_path / 'paths' / meta['files']['ids']).read_bytes()
    assert raw == np.array([1, 2], dtype='<u4').tobytes()
    assert np.load(tmp_path / 'pg/ids.npy').tolist() == [ID_BASE + 1, ID_BASE + 2]
    cloud_meta = export_ids(str(web), tmp_path / 'cloud')
    assert cloud_meta['ids']['id_prefix'] == 'A'
    assert 'id_encoding' not in cloud_meta['ids']
