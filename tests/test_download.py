from pathlib import Path

from pipeline import download


def test_sync_command_is_anonymous_authors_parquet_sync():
    cmd = download.sync_command(Path("/data"))
    assert cmd[:3] == ["aws", "s3", "sync"]
    assert "s3://openalex/data/parquet/authors/" in cmd
    assert "--no-sign-request" in cmd
    assert str(Path("/data/snapshot/authors")) in cmd
