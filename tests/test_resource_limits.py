import duckdb
import pytest

from pipeline.config import apply_resource_limits


def current(con, name):
    return con.execute(f"SELECT current_setting('{name}')").fetchone()[0]


def test_no_env_leaves_defaults(monkeypatch):
    monkeypatch.delenv("MAPADEMIC_THREADS", raising=False)
    monkeypatch.delenv("MAPADEMIC_MEMORY_LIMIT", raising=False)
    con = duckdb.connect()
    before = (current(con, "threads"), current(con, "memory_limit"))
    apply_resource_limits(con)
    assert (current(con, "threads"), current(con, "memory_limit")) == before


def test_env_caps_threads_and_memory(monkeypatch):
    monkeypatch.setenv("MAPADEMIC_THREADS", "4")
    monkeypatch.setenv("MAPADEMIC_MEMORY_LIMIT", "1GB")
    con = duckdb.connect()
    apply_resource_limits(con)
    assert current(con, "threads") == 4
    assert current(con, "memory_limit") == "953.6 MiB"  # duckdb normalizes 1GB


def test_bad_memory_limit_rejected(monkeypatch):
    monkeypatch.setenv("MAPADEMIC_MEMORY_LIMIT", "1GB'; DROP TABLE x; --")
    con = duckdb.connect()
    with pytest.raises(ValueError):
        apply_resource_limits(con)
