import duckdb

from pipeline.extract_works import connect, extract_partition, partitions
from tests.conftest import write_works_partition


def make_src(tmp_path):
    src = tmp_path / "works"
    write_works_partition(
        src / "updated_date=2026-01-01",
        [("W1", 2, ["A1", "A2"]), ("W2", 1, ["A1"])],
    )
    write_works_partition(
        src / "updated_date=2026-02-01",
        [("W3", 3, ["A1", "A2", "A3"])],
    )
    return str(src)


def test_partitions_sorted_from_glob(tmp_path):
    src = make_src(tmp_path)
    con = connect(src)
    assert partitions(con, src) == [
        "updated_date=2026-01-01", "updated_date=2026-02-01"
    ]


def test_extract_partition_writes_authorship_columns(tmp_path):
    src = make_src(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    con = connect(src)
    assert extract_partition(con, src, "updated_date=2026-01-01", out_dir) is True
    rows = duckdb.sql(
        f"SELECT work_id, n_authors, author_ids "
        f"FROM '{out_dir / 'updated_date=2026-01-01.parquet'}' ORDER BY work_id"
    ).fetchall()
    assert rows == [("W1", 2, ["A1", "A2"]), ("W2", 1, ["A1"])]


def test_extract_partition_skips_existing_checkpoint(tmp_path):
    src = make_src(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    con = connect(src)
    assert extract_partition(con, src, "updated_date=2026-01-01", out_dir) is True
    assert extract_partition(con, src, "updated_date=2026-01-01", out_dir) is False


def test_no_leftover_tmp_files(tmp_path):
    src = make_src(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    con = connect(src)
    for part in partitions(con, src):
        extract_partition(con, src, part, out_dir)
    assert not list(out_dir.glob("*.tmp"))
    assert len(list(out_dir.glob("*.parquet"))) == 2
