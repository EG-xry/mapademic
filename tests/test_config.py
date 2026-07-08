from pathlib import Path

from pipeline import config


def test_data_dir_defaults_to_local_data(monkeypatch):
    monkeypatch.delenv("MAPADEMIC_DATA", raising=False)
    assert config.data_dir() == Path("data")


def test_data_dir_env_override(monkeypatch):
    monkeypatch.setenv("MAPADEMIC_DATA", "/Volumes/Untitled/mapademic")
    assert config.data_dir() == Path("/Volumes/Untitled/mapademic")
