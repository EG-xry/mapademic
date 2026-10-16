from pathlib import Path

from pipeline import download


def test_sync_command_is_anonymous_authors_parquet_sync():
    cmd = download.sync_command(Path("/data"))
    assert cmd[:3] == ["aws", "s3", "sync"]
    assert "s3://openalex/data/parquet/authors/" in cmd
    assert "--no-sign-request" in cmd
    assert "--delete" in cmd
    assert str(Path("/data/snapshot/authors")) in cmd


def test_sync_command_takes_other_entities():
    cmd = download.sync_command(Path("/data"), "works")
    assert "s3://openalex/data/parquet/works/" in cmd
    assert str(Path("/data/snapshot/works")) in cmd


def test_parser_defaults_to_authors_only():
    import argparse
    p = argparse.ArgumentParser()
    download.add_parser(p)
    args = p.parse_args([])
    assert args.entities == ["authors"]
    assert p.parse_args(["--entities", "authors", "works"]).entities == ["authors", "works"]
