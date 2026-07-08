"""Shared paths for pipeline stages."""
import os
from pathlib import Path


def data_dir() -> Path:
    """Checkpoint root; override with MAPADEMIC_DATA (e.g. an external drive)."""
    return Path(os.environ.get("MAPADEMIC_DATA", "data")).expanduser()
