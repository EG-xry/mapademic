"""Unit tests for latest_checkpoint() in slurm/run_layout.py (loaded by path; slurm/ is not a package)."""
import importlib.util
from pathlib import Path

RUN_LAYOUT = Path(__file__).resolve().parent.parent / "slurm" / "run_layout.py"

spec = importlib.util.spec_from_file_location("run_layout", RUN_LAYOUT)
run_layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_layout)


def test_empty_dir_returns_zero_none(tmp_path):
    assert run_layout.latest_checkpoint(tmp_path) == (0, None)


def test_picks_highest_iteration_checkpoint(tmp_path):
    (tmp_path / "pos_25.parquet").touch()
    (tmp_path / "pos_50.parquet").touch()
    done, ckpt = run_layout.latest_checkpoint(tmp_path)
    assert done == 50
    assert ckpt == tmp_path / "pos_50.parquet"


def test_ignores_junk_names(tmp_path):
    (tmp_path / "pos_abc.parquet").touch()
    (tmp_path / "other.parquet").touch()
    (tmp_path / "pos_10.parquet.tmp").touch()
    assert run_layout.latest_checkpoint(tmp_path) == (0, None)
    (tmp_path / "pos_10.parquet").touch()
    done, ckpt = run_layout.latest_checkpoint(tmp_path)
    assert done == 10
    assert ckpt == tmp_path / "pos_10.parquet"
