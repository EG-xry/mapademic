import json

import duckdb
import pytest

from pipeline import snapshot

ROOT = "s3://openalex/data/parquet"


def manifest_for(files, date="2026-09-23"):
    """files: {entity: [(relative path under the entity, size, rows)]}"""
    return {
        "date": date,
        "format": "parquet",
        "entities": [
            {"entity": entity, "files": [
                {"url": f"{ROOT}/{entity}/{rel}",
                 "meta": {"content_length": size, "record_count": rows}}
                for rel, size, rows in entries]}
            for entity, entries in files.items()
        ],
    }


def write_parquet(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    duckdb.sql(f"COPY (SELECT range AS id FROM range({rows})) TO '{path}' (FORMAT PARQUET)")
    return path.stat().st_size


def test_entity_files_map_urls_to_local_paths(tmp_path):
    m = manifest_for({"authors": [("updated_date=2026-04-08/part_0000.parquet", 10, 3)]})
    (f,) = snapshot.entity_files(m, "authors")
    assert f.partition == "updated_date=2026-04-08"
    assert f.local(tmp_path) == (
        tmp_path / "authors" / "updated_date=2026-04-08" / "part_0000.parquet")
    assert (f.size, f.rows) == (10, 3)


def test_unknown_entity_is_an_error():
    with pytest.raises(KeyError, match="no 'works' entity"):
        snapshot.entity_files(manifest_for({"authors": []}), "works")


def test_partition_rows_sums_part_files():
    m = manifest_for({"works": [
        ("updated_date=2026-01-01/part_0000.parquet", 1, 4),
        ("updated_date=2026-01-01/part_0001.parquet", 1, 6),
        ("updated_date=2026-02-01/part_0000.parquet", 1, 2),
    ]})
    assert snapshot.partition_rows(m, "works") == {
        "updated_date=2026-01-01": 10, "updated_date=2026-02-01": 2}


def test_release_stamp_written_then_enforced(tmp_path):
    snapshot.ensure_release(tmp_path, "2026-09-23")
    assert (tmp_path / "RELEASE").read_text().strip() == "2026-09-23"
    snapshot.ensure_release(tmp_path, "2026-09-23")          # same release: fine
    with pytest.raises(SystemExit, match="holds OpenAlex release 2026-09-23"):
        snapshot.ensure_release(tmp_path, "2026-12-09")
    snapshot.ensure_release(tmp_path, "2026-12-09", force=True)
    assert (tmp_path / "RELEASE").read_text().strip() == "2026-12-09"


def test_verify_accepts_a_complete_mirror(tmp_path):
    rel = "updated_date=2026-04-08/part_0000.parquet"
    size = write_parquet(tmp_path / "authors" / rel, 3)
    m = manifest_for({"authors": [(rel, size, 3)]})
    assert snapshot.verify_entity(m, "authors", tmp_path) == []


def test_verify_reports_missing_truncated_extra_and_miscounted(tmp_path):
    ok = "updated_date=2026-01-01/part_0000.parquet"
    short = "updated_date=2026-01-02/part_0000.parquet"
    gone = "updated_date=2026-01-03/part_0000.parquet"
    miscount = "updated_date=2026-01-04/part_0000.parquet"
    ok_size = write_parquet(tmp_path / "authors" / ok, 3)
    short_size = write_parquet(tmp_path / "authors" / short, 3)
    mc_size = write_parquet(tmp_path / "authors" / miscount, 3)
    write_parquet(tmp_path / "authors" / "updated_date=2025-01-01/part_0000.parquet", 1)
    m = manifest_for({"authors": [
        (ok, ok_size, 3), (short, short_size + 100, 3), (gone, 50, 1),
        (miscount, mc_size, 4),
    ]})
    problems = snapshot.verify_entity(m, "authors", tmp_path)
    text = "\n".join(problems)
    assert len(problems) == 4
    assert f"missing: authors/{gone}" in text
    assert f"size {short_size} != {short_size + 100}: authors/{short}" in text
    assert "not in release: authors/updated_date=2025-01-01/part_0000.parquet" in text
    assert f"rows 3 != 4: authors/{miscount}" in text


def test_load_manifest_reads_local_json(tmp_path):
    m = manifest_for({"authors": []})
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(m))
    assert snapshot.load_manifest(duckdb.connect(), str(p)) == m


def test_deep_verify_catches_damaged_data_pages(tmp_path):
    # size and footer row counts survive a flash drive that silently rewrote
    # the middle of a file; only decoding the pages notices
    rel = "updated_date=2026-01-01/part_0000.parquet"
    p = tmp_path / "authors" / rel
    p.parent.mkdir(parents=True)
    duckdb.sql(f"COPY (SELECT range AS id, 'name number ' || range AS display_name"
               f" FROM range(20000)) TO '{p}' (FORMAT PARQUET)")
    size = p.stat().st_size
    m = manifest_for({"authors": [(rel, size, 20000)]})
    assert snapshot.verify_entity(m, "authors", tmp_path, deep=True) == []
    data = bytearray(p.read_bytes())
    mid = len(data) // 3
    data[mid:mid + 4096] = bytes(4096)
    p.write_bytes(bytes(data))
    assert snapshot.verify_entity(m, "authors", tmp_path) == []   # shallow: blind
    problems = snapshot.verify_entity(m, "authors", tmp_path, deep=True)
    assert len(problems) == 1 and problems[0].startswith(f"unreadable: authors/{rel}")
