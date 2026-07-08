import subprocess
import sys

import pytest

from pipeline.__main__ import main


def test_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_module_is_runnable():
    proc = subprocess.run(
        [sys.executable, "-m", "pipeline", "--help"], capture_output=True
    )
    assert proc.returncode == 0
