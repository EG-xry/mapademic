import json

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.resolve_identities import Authorship, partition_profile, resolve_identities


def record(work, *, author=5023477610, orcid=None, institution="I1", field="biology", year=2020, coauthors=(5000000001,)):
    return Authorship(work, author, "Michael J. Boyle", orcid,
                      (institution,) if institution else (), field, year, tuple(coauthors), 1)


def test_mixed_boyle_splits_automatically_without_requiring_a_political_orcid():
    records = [record(1, orcid="0000-0002-7287-1353"), record(2, orcid="0000-0002-7287-1353"),
               record(3, institution="I2", field="politics", coauthors=(5000000002,)),
               record(4, institution="I2", field="politics", coauthors=(5000000003,))]
    plan = partition_profile(records)
    assert len(plan.groups) == 2
    assert plan.assignments[1] == plan.assignments[2]
    assert plan.assignments[3] == plan.assignments[4]
    assert plan.assignments[1] != plan.assignments[3]


def test_multidisciplinary_person_and_career_move_are_not_split_by_topic():
    records = [record(1), record(2), record(3, field="politics"), record(4, field="politics")]
    assert len(partition_profile(records).groups) == 1
    records = [record(1, year=2000), record(2, year=2001),
               record(3, institution="I2", field="politics", year=2010, coauthors=(5000000002,)),
               record(4, institution="I2", field="politics", year=2011, coauthors=(5000000003,))]
    assert len(partition_profile(records).groups) == 1


def test_unanchored_profiles_remain_visible_and_can_split_on_repeated_corroboration():
    records = [record(1), record(2), record(3, institution="I2", field="politics", coauthors=(5000000002,)),
               record(4, institution="I2", field="politics", coauthors=(5000000003,))]
    plan = partition_profile(records)
    assert len(plan.groups) == 2
    assert set(plan.assignments) == {1, 2, 3, 4}


def test_same_deposited_orcid_protects_one_person_with_joint_appointments_and_fields():
    oid = "0000-0002-7287-1353"
    records = [record(1, orcid=oid), record(2, orcid=oid),
               record(3, orcid=oid, institution="I2", field="politics", coauthors=(5000000002,)),
               record(4, orcid=oid, institution="I2", field="politics", coauthors=(5000000003,))]
    # Raw byline evidence, unlike an inherited profile ORCID, is an identity
    # anchor. Topic/institution divergence cannot split it by itself.
    assert len(partition_profile(records).groups) == 1


def test_ambiguous_work_is_quarantined_instead_of_bridging_partitions():
    records = [record(1), record(2), record(3, institution="I2", field="politics", coauthors=(5000000002,)),
               record(4, institution="I2", field="politics", coauthors=(5000000003,)),
               record(5, institution=None, field=None, coauthors=())]
    plan = partition_profile(records)
    assert len(plan.groups) == 3
    ambiguous = plan.groups[plan.assignments[5]]
    assert ambiguous["status"] == "quarantined"


def make_inputs(tmp_path, duplicate=False):
    author = 5023477610
    rows = []
    for work, inst, field, coauthor in [(1, "I1", "biology", 5000000001), (2, "I1", "biology", 5000000001),
                                       (3, "I2", "politics", 5000000002), (4, "I2", "politics", 5000000003)]:
        rows.append({"work_id": work, "n_authors": 2, "author_ids": [author, coauthor], "year": 2020,
                     "type": "article", "is_xpac": False, "is_paratext": False, "is_retracted": False,
                     "cited_by_count": work,
                     "authorship_evidence": [{"author_id": author, "raw_name": "Michael J. Boyle", "raw_orcid": None,
                                              "institution_ids": [inst], "field_id": field}]})
    works = tmp_path / "works.parquet"
    pq.write_table(pa.Table.from_pylist(rows), works)
    nodes = tmp_path / "nodes.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"id": f"https://openalex.org/A{a}", "display_name": "Michael J. Boyle" if a == author else f"Researcher {a}",
         "works_count": 10, "works_kept": 10, "cited_by_count": 100, "institution": "Old institution", "field": "Old field"}
        for a in [author, 5000000001, 5000000002, 5000000003]
    ]), nodes)
    return works, nodes


def test_stage_preserves_graph_contracts_and_partitions_counts_without_duplication(tmp_path):
    works, nodes = make_inputs(tmp_path)
    out = tmp_path / "resolved"
    report = resolve_identities(str(works), str(nodes), out)
    assert report["split_profiles"] == 1
    repaired = pq.read_table(out / "works.parquet").to_pylist()
    repaired_nodes = pq.read_table(out / "nodes.parquet").to_pylist()
    assert len(repaired) == 4
    assert len(repaired_nodes) == 5
    changed = [n for n in repaired_nodes if n["id"].startswith("M")]
    assert len(changed) == 2
    assert sum(n["works_count"] for n in changed) == 4
    assert sum(n["cited_by_count"] for n in changed) == 10
    assert all(r["author_ids"][0] < 0 for r in repaired)
    assert all(r["source_author_ids"][0] == 5023477610 for r in repaired)
    metadata = json.loads((out / "profiles.json").read_text())
    assert len(metadata["aliases"]["A5023477610"]) == 2
    assert all(p["source_ids"] == ["A5023477610"] for p in metadata["profiles"].values())
    assert all(p["work_ids"] for p in metadata["profiles"].values())


def test_same_inputs_and_registry_produce_stable_identity_ids(tmp_path):
    works, nodes = make_inputs(tmp_path)
    resolve_identities(str(works), str(nodes), tmp_path / "one")
    resolve_identities(str(works), str(nodes), tmp_path / "two", registry_path=tmp_path / "one" / "identity_registry.json")
    a = pq.read_table(tmp_path / "one" / "nodes.parquet").column("id").to_pylist()
    b = pq.read_table(tmp_path / "two" / "nodes.parquet").column("id").to_pylist()
    assert a == b


def test_missing_evidence_fails_clearly_instead_of_claiming_a_repair(tmp_path):
    works, nodes = make_inputs(tmp_path)
    pq.write_table(pq.read_table(works).drop(["authorship_evidence"]), works)
    import pytest
    with pytest.raises(ValueError, match="authorship_evidence"):
        resolve_identities(str(works), str(nodes), tmp_path / "resolved")


def duplicate_inputs(tmp_path, *, name_b="Paul D. Miller", inst_b="I184565670", orcid_b="0000-0003-0443-3957"):
    oid = "0000-0003-0443-3957"
    rows = []
    for i, aid, name, inst, raw in [(1, 5088470719, "Paul D. Miller", "I184565670", oid),
                                   (2, 5088470719, "Paul D. Miller", "I184565670", oid),
                                   (3, 5088470720, name_b, inst_b, orcid_b),
                                   (4, 5088470720, name_b, inst_b, orcid_b)]:
        rows.append({"work_id": i, "n_authors": 2, "author_ids": [aid, 5000000001], "year": 2020,
                     "type": "article", "is_xpac": False, "is_paratext": False, "is_retracted": False,
                     "cited_by_count": i, "authorship_evidence": [{"author_id": aid, "raw_name": name,
                     "raw_orcid": raw, "institution_ids": [inst] if inst else [], "field_id": "politics"}]})
    works = tmp_path / "works.parquet"
    pq.write_table(pa.Table.from_pylist(rows), works)
    nodes = tmp_path / "nodes.parquet"
    pq.write_table(pa.Table.from_pylist([{"id": f"https://openalex.org/A{aid}", "display_name": name,
                                        "works_count": 2, "works_kept": 2, "cited_by_count": 10,
                                        "institution": inst, "field": "politics"}
                                       for aid, name, inst in [(5088470719, "Paul D. Miller", "Georgetown"),
                                                              (5088470720, name_b, "Georgetown"),
                                                              (5000000001, "Other author", "Elsewhere")]]), nodes)
    return works, nodes


def test_duplicate_profiles_merge_only_with_deposited_orcid_and_additional_evidence(tmp_path):
    works, nodes = duplicate_inputs(tmp_path)
    out = tmp_path / "resolved"
    report = resolve_identities(str(works), str(nodes), out)
    assert report["merged_profiles"] == 1
    metadata = json.loads((out / "profiles.json").read_text())
    assert len(metadata["profiles"]) == 1
    identity = next(iter(metadata["profiles"].values()))
    assert identity["source_ids"] == ["A5088470719", "A5088470720"]
    assert identity["works_count"] == 4
    assert identity["cited_by_count"] == 10
    assert len(pq.read_table(out / "nodes.parquet")) == 2


def test_same_orcid_but_contradictory_middle_names_does_not_merge(tmp_path):
    works, nodes = duplicate_inputs(tmp_path, name_b="Paul A. Miller")
    report = resolve_identities(str(works), str(nodes), tmp_path / "resolved")
    assert report["merged_profiles"] == 0


def test_same_orcid_without_shared_affiliation_or_network_does_not_merge(tmp_path):
    works, nodes = duplicate_inputs(tmp_path, inst_b="I999")
    report = resolve_identities(str(works), str(nodes), tmp_path / "resolved")
    assert report["merged_profiles"] == 0


def test_invalid_source_orcids_never_merge_profiles(tmp_path):
    works, nodes = duplicate_inputs(tmp_path, orcid_b="0000-0003-0443-3958")
    report = resolve_identities(str(works), str(nodes), tmp_path / "resolved")
    assert report["merged_profiles"] == 0


def test_generic_collection_titles_have_no_coauthor_edges_and_cannot_join_partitions(tmp_path):
    works, nodes = make_inputs(tmp_path)
    rows = pq.read_table(works).to_pylist()
    for r in rows:
        r["title"] = "A normal article"
    row = dict(rows[0], work_id=5, n_authors=3, author_ids=[5023477610, 5000000001, 5000000002], title="Book reviews")
    row["authorship_evidence"] = [{"author_id": 5023477610, "raw_name": "Michael J. Boyle", "raw_orcid": None,
                                  "institution_ids": [], "field_id": None}]
    rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows), works)
    out = tmp_path / "resolved"
    report = resolve_identities(str(works), str(nodes), out)
    assert report["split_profiles"] == 1
    assert report["collection_works_without_edges"] == 1
    repaired = pq.read_table(out / "works.parquet").to_pylist()
    collection = next(r for r in repaired if r["work_id"] == 5)
    assert len(collection["author_ids"]) == 3
    assert collection["edge_author_ids"] == []
    assert report["quarantined_identities"] == 1


def test_source_inputs_are_unchanged_and_existing_repaired_inputs_are_rejected(tmp_path):
    works, nodes = make_inputs(tmp_path)
    before_works, before_nodes = works.read_bytes(), nodes.read_bytes()
    out = tmp_path / "resolved"
    resolve_identities(str(works), str(nodes), out)
    assert works.read_bytes() == before_works and nodes.read_bytes() == before_nodes
    import pytest
    with pytest.raises(ValueError, match="already identity-resolved"):
        resolve_identities(str(out / "works.parquet"), str(out / "nodes.parquet"), tmp_path / "twice")


def test_profile_size_bound_is_reported_without_dropping_nodes_or_works(tmp_path):
    works, nodes = make_inputs(tmp_path)
    out = tmp_path / "resolved"
    report = resolve_identities(str(works), str(nodes), out, max_profile_works=3)
    assert report["oversized_profiles_unchanged"] == 1
    assert len(pq.read_table(out / "nodes.parquet")) == 4
    assert len(pq.read_table(out / "works.parquet")) == 4


def test_distinct_namesakes_in_two_bylines_of_same_work_are_quarantined(tmp_path):
    works, nodes = make_inputs(tmp_path)
    rows = pq.read_table(works).to_pylist()
    # A genuinely ambiguous same-profile paper cannot choose either namesake.
    rows[0]["authorship_evidence"].append({"author_id": 5023477610, "raw_name": "Michael Boyle",
                                         "raw_orcid": "0000-0003-0443-3957", "institution_ids": ["I9"], "field_id": "biology"})
    rows[0]["authorship_evidence"][0]["raw_orcid"] = "0000-0002-7287-1353"
    pq.write_table(pa.Table.from_pylist(rows), works)
    out = tmp_path / "resolved"
    report = resolve_identities(str(works), str(nodes), out)
    assert report["ambiguous_profiles"] == 1
    metadata = json.loads((out / "profiles.json").read_text())
    owner = next(p for p in metadata["profiles"].values() if "W1" in p["work_ids"])
    assert owner["identity_status"] == "quarantined"


def test_repeated_source_orcid_on_multiple_authors_is_not_a_merge_anchor(tmp_path):
    works, nodes = duplicate_inputs(tmp_path)
    rows = pq.read_table(works).to_pylist()
    for row in rows:
        row["authorship_evidence"].append({"author_id": 5000000001, "raw_name": "Other author",
                                           "raw_orcid": "0000-0003-0443-3957", "institution_ids": ["I184565670"], "field_id": "politics"})
    pq.write_table(pa.Table.from_pylist(rows), works)
    assert resolve_identities(str(works), str(nodes), tmp_path / "resolved")["merged_profiles"] == 0


def test_profile_level_orcid_without_deposited_evidence_is_not_a_merge_anchor(tmp_path):
    works, nodes = duplicate_inputs(tmp_path)
    rows = pq.read_table(works).to_pylist()
    for row in rows:
        row["authorship_evidence"][0]["raw_orcid"] = None
    pq.write_table(pa.Table.from_pylist(rows), works)
    ns = pq.read_table(nodes).to_pylist()
    for n in ns:
        n["orcid"] = "0000-0003-0443-3957"
    pq.write_table(pa.Table.from_pylist(ns), nodes)
    assert resolve_identities(str(works), str(nodes), tmp_path / "resolved")["merged_profiles"] == 0


def test_invalid_registry_collisions_are_rejected(tmp_path):
    works, nodes = make_inputs(tmp_path)
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"tokens": {"one": -1, "two": -1}}))
    import pytest
    with pytest.raises(ValueError, match="colliding keys"):
        resolve_identities(str(works), str(nodes), tmp_path / "resolved", registry_path=registry)


def test_partition_metadata_uses_its_own_evidence_labels(tmp_path):
    works, nodes = make_inputs(tmp_path)
    rows = pq.read_table(works).to_pylist()
    for row in rows:
        e = row["authorship_evidence"][0]
        political = e["field_id"] == "politics"
        e["institution_names"] = ["Rutgers University" if political else "Smithsonian Marine Station"]
        e["field_name"] = "Political Science" if political else "Biology"
    pq.write_table(pa.Table.from_pylist(rows), works)
    out = tmp_path / "resolved"
    resolve_identities(str(works), str(nodes), out)
    profiles = json.loads((out / "profiles.json").read_text())["profiles"].values()
    political = next(p for p in profiles if p["field_id"] == "politics")
    assert political["institution"] == "Rutgers University"
    assert political["institution_ids"] == ["I2"]
    assert political["field"] == "Political Science"


def test_hash_collision_is_disambiguated_without_aliasing_identities(monkeypatch):
    import pipeline.resolve_identities as module

    class Digest:
        def __init__(self, value):
            self.value = value

        def digest(self):
            return (2 if self.value.endswith(b":1") else 1).to_bytes(7, "big")

    monkeypatch.setattr(module.hashlib, "sha256", Digest)
    registry = {}
    assert module._identity_key("one", registry) == -1
    assert module._identity_key("two", registry) == -2
    assert module._identity_key("one", registry) == -1


def test_same_orcid_name_but_disjoint_merge_components_get_distinct_ids(tmp_path):
    works, nodes = duplicate_inputs(tmp_path)
    rows, ns = pq.read_table(works).to_pylist(), pq.read_table(nodes).to_pylist()
    extra = []
    for row in rows:
        r = dict(row, work_id=row["work_id"] + 10,
                 author_ids=[row["author_ids"][0] + 100, 5000000002])
        e = dict(row["authorship_evidence"][0], author_id=row["author_ids"][0] + 100, institution_ids=["I999"])
        r["authorship_evidence"] = [e]
        extra.append(r)
    pq.write_table(pa.Table.from_pylist(rows + extra), works)
    for n in list(ns[:2]):
        ns.append(dict(n, id=f"https://openalex.org/A{int(n['id'].split('A')[-1]) + 100}"))
    pq.write_table(pa.Table.from_pylist(ns), nodes)
    out = tmp_path / "resolved"
    assert resolve_identities(str(works), str(nodes), out)["merged_profiles"] == 2
    profiles = json.loads((out / "profiles.json").read_text())["profiles"]
    assert len(profiles) == 2
    assert len({p["id"] for p in profiles.values()}) == 2
    assert sum(p["works_count"] for p in profiles.values()) == 8


def test_duplicate_work_partitions_do_not_inflate_stats_or_memberships(tmp_path):
    works, nodes = make_inputs(tmp_path)
    rows = pq.read_table(works).to_pylist()
    pq.write_table(pa.Table.from_pylist(rows + [rows[0], rows[2]]), works)
    out = tmp_path / "resolved"
    report = resolve_identities(str(works), str(nodes), out)
    assert report["duplicate_works_removed"] == 2
    assert len(pq.read_table(out / "works.parquet")) == 4
    profiles = json.loads((out / "profiles.json").read_text())["profiles"]
    assert sum(p["works_count"] for p in profiles.values()) == 4
    assert sum(p["cited_by_count"] for p in profiles.values()) == 10


def test_conflicting_versions_of_one_work_fail_instead_of_picking_arbitrary_authorship(tmp_path):
    works, nodes = make_inputs(tmp_path)
    rows = pq.read_table(works).to_pylist()
    rows.append(dict(rows[0], author_ids=[5023477610, 5000000009]))
    pq.write_table(pa.Table.from_pylist(rows), works)
    import pytest
    with pytest.raises(ValueError, match="conflicting versions"):
        resolve_identities(str(works), str(nodes), tmp_path / "resolved")


def test_orcid_anchors_do_not_pull_a_different_namesake_work_into_whole_profile_merge(tmp_path):
    works, nodes = duplicate_inputs(tmp_path)
    rows = pq.read_table(works).to_pylist()
    rows.append(dict(rows[0], work_id=5, authorship_evidence=[{
        "author_id": 5088470719, "raw_name": "Paul A. Miller", "raw_orcid": None,
        "institution_ids": ["I999"], "field_id": "biology"}]))
    pq.write_table(pa.Table.from_pylist(rows), works)
    out = tmp_path / "resolved"
    report = resolve_identities(str(works), str(nodes), out)
    assert report["merged_profiles"] == 0
    assert len(pq.read_table(out / "works.parquet")) == 5


def test_unsupported_conflicting_affiliation_and_field_prevent_whole_profile_merge(tmp_path):
    works, nodes = duplicate_inputs(tmp_path)
    rows = pq.read_table(works).to_pylist()
    rows.append(dict(rows[0], work_id=5, author_ids=[5088470719, 5000000010], authorship_evidence=[{
        "author_id": 5088470719, "raw_name": "Paul D. Miller", "raw_orcid": None,
        "institution_ids": ["I999"], "field_id": "biology"}]))
    pq.write_table(pa.Table.from_pylist(rows), works)
    assert resolve_identities(str(works), str(nodes), tmp_path / "resolved")["merged_profiles"] == 0


def test_singleton_misdeposited_orcid_cannot_reconnect_supported_clusters():
    oid = '0000-0002-7287-1353'
    records = [record(1, orcid=oid), record(2, orcid=oid),
               record(3, institution='I2', field='politics', coauthors=()),
               record(4, institution='I2', field='politics', coauthors=()),
               record(5, orcid=oid, institution=None, field='politics', coauthors=(5000000004,))]
    plan = partition_profile(records)
    assert plan.assignments[1] == plan.assignments[2]
    assert plan.assignments[3] == plan.assignments[4]
    assert plan.assignments[1] != plan.assignments[3]
    assert plan.groups[plan.assignments[5]]['status'] == 'quarantined'
    assert plan.groups[plan.assignments[3]]['orcid'] is None


def homogeneous_inputs(tmp_path):
    works, nodes = make_inputs(tmp_path)
    rows = pq.read_table(works).to_pylist()
    for row in rows:
        row['authorship_evidence'][0].update(raw_name='David J. Chalmers', raw_orcid=None,
                                            institution_ids=[], field_id='psychology')
    return works, nodes, rows


def test_compatible_duplicate_bylines_do_not_quarantine_a_researcher(tmp_path):
    works, nodes, rows = homogeneous_inputs(tmp_path)
    rows[0]['authorship_evidence'].append(dict(rows[0]['authorship_evidence'][0], raw_name='David Chalmers'))
    pq.write_table(pa.Table.from_pylist(rows), works)
    out = tmp_path / 'resolved'
    report = resolve_identities(str(works), str(nodes), out)
    assert report['repaired_identities'] == 0
    assert report['ambiguous_profiles'] == 0
    assert len(pq.read_table(out / 'works.parquet')) == 4


def test_ambiguous_bylines_quarantine_only_the_affected_work_and_preserve_clean_coauthors(tmp_path):
    works, nodes, rows = homogeneous_inputs(tmp_path)
    rows[0]['authorship_evidence'].append(dict(rows[0]['authorship_evidence'][0], raw_name='David A. Chalmers'))
    pq.write_table(pa.Table.from_pylist(rows), works)
    out = tmp_path / 'resolved'
    resolve_identities(str(works), str(nodes), out)
    profiles = json.loads((out / 'profiles.json').read_text())['profiles'].values()
    ambiguous = next(p for p in profiles if p['status'] == 'quarantined')
    retained = next(p for p in profiles if p['status'] == 'retained')
    assert ambiguous['work_ids'] == ['W1']
    assert retained['work_ids'] == ['W2', 'W3', 'W4']
    assert sum(p['works_count'] for p in profiles) == 4
    assert sum(p['cited_by_count'] for p in profiles) == 10
    memberships = pq.read_table(out / 'works.parquet').to_pylist()
    assert all(len(w['edge_author_ids']) == 2 for w in memberships)


def test_name_compatibility_handles_diacritics_and_initials_but_rejects_conflicting_full_names():
    from pipeline.resolve_identities import _names_compatible
    assert _names_compatible('José D. Miller', 'Jose David Miller')
    assert _names_compatible('D. J. Chalmers', 'David Chalmers')
    assert not _names_compatible('Paul David Miller', 'Paul Daniel Miller')
    assert not _names_compatible('David J. Chalmers', 'David A. Chalmers')
