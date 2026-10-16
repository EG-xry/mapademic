from pathlib import Path

from pipeline import config


def test_data_dir_defaults_to_local_data(monkeypatch):
    monkeypatch.delenv("MAPADEMIC_DATA", raising=False)
    assert config.data_dir() == Path("data")


def test_data_dir_env_override(monkeypatch):
    monkeypatch.setenv("MAPADEMIC_DATA", "/Volumes/Untitled/mapademic")
    assert config.data_dir() == Path("/Volumes/Untitled/mapademic")


def test_spill_dir_defaults_next_to_the_output(monkeypatch):
    monkeypatch.delenv("MAPADEMIC_TMP", raising=False)
    assert config.spill_dir(Path("/d/out")) == Path("/d/out/.duckdb_tmp")


def test_spill_dir_env_override(monkeypatch):
    # an external output drive is often the slowest disk on the machine;
    # DuckDB's spill traffic can go to the internal one
    monkeypatch.setenv("MAPADEMIC_TMP", "/fast/tmp")
    assert config.spill_dir(Path("/d/out")) == Path("/fast/tmp")
