import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipeline.identifiers import encode_author_id


def fixture(tmp_path):
    repair = tmp_path / 'repair'; repair.mkdir()
    profiles = {
        'M101': {'id': 'M101', 'source_ids': ['A5000000001'], 'display_name': 'Michael Boyle',
                 'orcid': None, 'institution': 'Rutgers', 'field': 'Politics', 'works_count': 2,
                 'cited_by_count': 11, 'status': 'split', 'work_ids': ['W1', 'W2'], 'institution_ids': ['https://openalex.org/I123']},
        'M102': {'id': 'M102', 'source_ids': ['A5000000001'], 'display_name': 'Michael Boyle',
                 'orcid': '0000-0002-7287-1353', 'institution': 'Smithsonian', 'field': 'Biology',
                 'works_count': 1, 'cited_by_count': 7, 'status': 'split', 'work_ids': ['W3'], 'institution_ids': ['I456']},
        'M103': {'id': 'M103', 'source_ids': ['A5000000001'], 'display_name': 'Michael Boyle',
                 'orcid': None, 'institution': None, 'field': None, 'works_count': 1,
                 'cited_by_count': 2, 'status': 'quarantined', 'work_ids': ['W4']},
        'M104': {'id': 'M104', 'source_ids': ['A5000000004', 'A5000000005'], 'display_name': 'Paul Miller',
                 'orcid': '0000-0003-0443-3957', 'institution': 'Georgetown', 'field': 'Politics',
                 'works_count': 1, 'cited_by_count': 5, 'status': 'merged', 'work_ids': ['W5']},
    }
    aliases = {'A5000000001': ['M101', 'M102', 'M103'], 'A5000000004': ['M104'], 'A5000000005': ['M104'],
               'A5000000009': []}
    (repair / 'profiles.json').write_text(json.dumps({'version': 1, 'profiles': profiles, 'aliases': aliases}))
    keys = [[-101, 5000000002], [-101, 5000000002], [-102, 5000000003], [-103, 5000000002, 5000000003], [-104, 5000000002]]
    pq.write_table(pa.table({'work_id': [1, 2, 3, 4, 5], 'n_authors': [2, 2, 2, 3, 2], 'author_ids': keys,
        'edge_author_ids': [keys[0], keys[1], keys[2], [], keys[4]],
        'title': ['Political paper', 'Political followup', 'Marine biology paper', 'Book reviews', 'Strategy report'],
        'doi': ['https://doi.org/10.1234/politics', None, 'https://doi.org/10.1234/biology', None, None],
        'year': [2021, 2022, 2021, 2021, 2014], 'cited_by_count': [10, 1, 7, 2, 5],
        'type': ['article'] * 5, 'is_xpac': [False] * 5, 'is_paratext': [False] * 5}), repair / 'works.parquet')
    memberships = [{'work_id': i + 1, 'identity_key': key, 'identity_id': 'M' + str(-key) if key < 0 else 'A' + str(key),
                    'source_author_id': 5000000001 if key in (-101, -102, -103) else 5000000004 if key == -104 else key}
                   for i, row in enumerate(keys) for key in row]
    pq.write_table(pa.Table.from_pylist(memberships), repair / 'memberships.parquet')
    edge_path = tmp_path / 'edges.parquet'
    pq.write_table(pa.table({'src': ['M101', 'M102', 'M104'], 'dst': ['A5000000002', 'A5000000003', 'A5000000002'],
                             'weight': [2.0, 1.0, 1.0]}), edge_path)
    ids = ['M101', 'M102', 'M103', 'M104', 'A5000000002', 'A5000000003', 'A5000000008']
    web = tmp_path / 'coords_web.parquet'
    pq.write_table(pa.table({'id': ids, 'display_name': ['Michael Boyle'] * 3 + ['Paul Miller', 'Political colleague', 'Marine colleague', 'Unaffected'],
        'works_count': [2, 1, 1, 1, 22, 33, 40], 'cited_by_count': [11, 7, 2, 5, 220, 330, 400],
        'institution': ['Rutgers', 'Smithsonian', None, 'Georgetown', 'Political institute', 'Marine institute', 'Other'],
        'field': ['Politics', 'Biology', None, 'Politics', 'Politics', 'Biology', 'Other'],
        'xw': [0.1] * 7, 'yw': [0.2] * 7}), web)
    paths = tmp_path / 'paths'; paths.mkdir()
    (paths / 'manifest.json').write_text(json.dumps({'version': 3, 'id_encoding': 'tagged-u64', 'ids_record_bytes': 8,
        'files': {'ids': 'ids-test.bin'}}))
    np.array([encode_author_id(aid) for aid in ids], dtype='<u8').tofile(paths / 'ids-test.bin')
    return repair, edge_path, web, paths


def test_static_profiles_use_actual_repaired_work_memberships_and_keep_aliases(tmp_path):
    from scripts.export_identities import export_identities
    repair, edges, web, paths = fixture(tmp_path)
    out = tmp_path / 'identities'
    manifest = export_identities(repair, str(edges), str(web), paths, out)
    assert manifest['version'] == 1 and manifest['graph_repaired']
    assert manifest['paths_ids_file'] == 'ids-test.bin'
    political = json.loads((out / 'profiles/M101.json').read_text())
    biology = json.loads((out / 'profiles/M102.json').read_text())
    assert political['papers']['A5000000002']['title'] == 'Political paper'
    assert political['papers']['A5000000002']['work_id'] == 'W1'
    assert political['coauthors'] == [{'id': 'A5000000002', 'name': 'Political colleague', 'count': 2}]
    assert 'A5000000003' not in political['papers']
    assert biology['papers']['A5000000003']['title'] == 'Marine biology paper'
    assert 'A5000000002' not in biology['papers']
    assert json.loads((out / 'profiles/A5000000002.json').read_text())['cited_by_count'] == 220
    assert not (out / 'profiles/A5000000008.json').exists()
    assert json.loads((out / 'aliases/4.json').read_text())['A5000000004'] == ['M104']
    assert json.loads((out / 'aliases/5.json').read_text())['A5000000005'] == ['M104']
    assert json.loads((out / 'aliases/9.json').read_text())['A5000000009'] == []


def test_collections_never_create_static_coauthors_and_quarantine_is_routing_flagged(tmp_path):
    from scripts.export_identities import export_identities
    repair, edges, web, paths = fixture(tmp_path)
    out = tmp_path / 'identities'
    export_identities(repair, str(edges), str(web), paths, out)
    quarantine = json.loads((out / 'profiles/M103.json').read_text())
    assert quarantine['coauthors'] == [] and quarantine['papers'] == {}
    assert quarantine['work_ids'] == ['W4']
    flags = json.loads((out / 'flags.json').read_text())
    assert flags['paths_ids_file'] == 'ids-test.bin'
    assert flags['entries'] == [{'id': 'M103', 'n': 2, 'kind': 'quarantined'}]


def test_old_flags_are_remapped_by_id_never_by_stale_node_number(tmp_path):
    from scripts.export_identities import export_identities
    repair, edges, web, paths = fixture(tmp_path)
    old = tmp_path / 'legacy-flags.json'
    old.write_text(json.dumps({'flags': {'merged': {'id': [2, 1], 'n': [999, 4], 'peak': [30, 35], 'year': [2020, 2020]}}}))
    out = tmp_path / 'identities'
    export_identities(repair, str(edges), str(web), paths, out, flags_path=old)
    rows = json.loads((out / 'flags.json').read_text())['entries']
    assert {'id': 'A5000000002', 'n': 4, 'kind': 'merged', 'peak': 30, 'year': 2020} in rows
    assert all(row['n'] != 999 and row['id'] != 'A5000000001' for row in rows)


def test_failed_generation_preserves_previously_published_directory(tmp_path, monkeypatch):
    import scripts.export_identities as exporter
    repair, edges, web, paths = fixture(tmp_path)
    out = tmp_path / 'identities'; out.mkdir()
    (out / 'sentinel').write_text('previous generation')
    monkeypatch.setattr(exporter, '_write_json', lambda *args: (_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(OSError, match='disk full'):
        exporter.export_identities(repair, str(edges), str(web), paths, out)
    assert (out / 'sentinel').read_text() == 'previous generation'
    assert list(out.iterdir()) == [out / 'sentinel']


def test_absent_legacy_flags_is_valid_and_rebuild_replaces_old_generation(tmp_path):
    from scripts.export_identities import export_identities
    repair, edges, web, paths = fixture(tmp_path)
    out = tmp_path / 'identities'; out.mkdir()
    (out / 'old').write_text('old')
    export_identities(repair, str(edges), str(web), paths, out, flags_path=tmp_path / 'does-not-exist.json')
    assert not (out / 'old').exists()
    assert (out / 'manifest.json').exists()


def test_equal_a_and_m_numeric_suffixes_never_share_static_papers_or_counts(tmp_path):
    from scripts.export_identities import export_identities
    repair, edges, web, paths = fixture(tmp_path)
    replacement = 'M5000000002'
    metadata = (repair / 'profiles.json').read_text().replace('M102', replacement)
    (repair / 'profiles.json').write_text(metadata)
    for name in ['works.parquet', 'memberships.parquet']:
        table = pq.read_table(repair / name).to_pylist()
        for row in table:
            for key in ['author_ids', 'edge_author_ids']:
                if key in row:
                    row[key] = [-5000000002 if value == -102 else value for value in row[key]]
            if row.get('identity_key') == -102:
                row['identity_key'] = -5000000002
                row['identity_id'] = replacement
        pq.write_table(pa.Table.from_pylist(table), repair / name)
    for path, fields in [(web, ['id']), (edges, ['src', 'dst'])]:
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            for field in fields:
                if row[field] == 'M102':
                    row[field] = replacement
        pq.write_table(pa.Table.from_pylist(rows), path)
    binary = np.fromfile(paths / 'ids-test.bin', dtype='<u8')
    binary[binary == encode_author_id('M102')] = encode_author_id(replacement)
    binary.tofile(paths / 'ids-test.bin')
    out = tmp_path / 'identities'
    export_identities(repair, str(edges), str(web), paths, out)
    source = json.loads((out / 'profiles/A5000000002.json').read_text())
    repaired = json.loads((out / 'profiles/M5000000002.json').read_text())
    assert source['display_name'] == 'Political colleague'
    assert repaired['display_name'] == 'Michael Boyle'
    assert source['cited_by_count'] == 220 and repaired['cited_by_count'] == 7
    assert source['papers']['M101']['title'] == 'Political paper'
    assert repaired['papers']['A5000000003']['title'] == 'Marine biology paper'


def test_flag_lookup_supports_legacy_uint32_path_ids_without_reusing_old_numbers(tmp_path):
    from scripts.export_identities import _flag_numbers
    np.array([2, 1], dtype='<u4').tofile(tmp_path / 'ids-legacy.bin')
    manifest = {'version': 2, 'id_base': 5000000000, 'files': {'ids': 'ids-legacy.bin'}}
    assert _flag_numbers(tmp_path, manifest, {'A5000000002', 'M2'}) == {'A5000000002': 0}


def test_institution_candidates_follow_partition_evidence_not_parent_affiliation(tmp_path):
    from scripts.export_identities import export_identities
    repair, edges, web, paths = fixture(tmp_path)
    out = tmp_path / 'identities'
    manifest = export_identities(repair, str(edges), str(web), paths, out)
    assert manifest['institutions_path'] == 'institutions/{id}.json'
    rutgers = json.loads((out / 'institutions/I123.json').read_text())
    marine = json.loads((out / 'institutions/I456.json').read_text())
    assert [r['id'] for r in rutgers] == ['M101']
    assert [r['id'] for r in marine] == ['M102']
    assert rutgers[0]['institution_ids'] == ['I123']
    assert marine[0]['source_ids'] == ['A5000000001']


def test_missing_repaired_id_in_path_graph_cannot_publish_metadata(tmp_path):
    from scripts.export_identities import export_identities
    repair, edges, web, paths = fixture(tmp_path)
    binary = np.fromfile(paths / 'ids-test.bin', dtype='<u8')
    binary[1:].tofile(paths / 'ids-test.bin')
    out = tmp_path / 'identities'; out.mkdir()
    (out / 'old').write_text('complete old generation')
    with pytest.raises(ValueError, match='missing from the current path graph'):
        export_identities(repair, str(edges), str(web), paths, out)
    assert (out / 'old').read_text() == 'complete old generation'


def test_publish_rename_failure_rolls_back_old_generation(tmp_path, monkeypatch):
    import scripts.export_identities as exporter
    stage = tmp_path / 'stage'; stage.mkdir(); (stage / 'new').write_text('new')
    output = tmp_path / 'published'; output.mkdir(); (output / 'old').write_text('old')
    replace = exporter.os.replace
    def fail_stage(source, destination):
        if source == stage:
            raise OSError('rename failed')
        return replace(source, destination)
    monkeypatch.setattr(exporter.os, 'replace', fail_stage)
    with pytest.raises(OSError, match='rename failed'):
        exporter._publish(stage, output)
    assert (output / 'old').read_text() == 'old'
    assert not (output / 'new').exists()
