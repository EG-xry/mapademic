"""File identities that stay valid when the project folder moves.

A file identity is [path, bytes, mtime_ns], and caches are reused only while every input's identity is unchanged.
New identities record paths relative to the project root, so a move (which keeps sizes and mtimes) leaves them
valid. Records written before 4 November 2026, when the project moved from ~/Downloads to ~/Desktop, hold
absolute paths in the old folder; same_identity accepts one of those when the path below that folder agrees,
and locate finds the file in the current project. Sizes and mtimes must still match exactly, so an edited or
replaced file is never mistaken for the recorded one. Files outside the project keep absolute paths.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def file_identity(path, root=None):
    path = Path(path)
    stat = path.stat()
    return [_name(path, Path(ROOT if root is None else root)), stat.st_size, stat.st_mtime_ns]


def _name(path, root):
    for candidate, base in ((path, root), (path.resolve(), root.resolve())):
        try:
            return candidate.relative_to(base).as_posix()
        except ValueError:
            pass
    return str(path)


def same_identity(saved, current):
    """Equal records, except that a file identity's recorded path may come from where the project used to be."""
    if _is_file_identity(saved) and _is_file_identity(current):
        return saved[1:] == current[1:] and _same_path(saved[0], current[0])
    if isinstance(saved, dict) and isinstance(current, dict):
        return saved.keys() == current.keys() and all(same_identity(saved[key], current[key]) for key in saved)
    if isinstance(saved, list) and isinstance(current, list):
        return len(saved) == len(current) and all(map(same_identity, saved, current))
    return saved == current


def _is_file_identity(value):
    return (isinstance(value, list) and len(value) == 3 and isinstance(value[0], str)
            and all(isinstance(number, int) and not isinstance(number, bool) for number in value[1:]))


def _same_path(saved, current):
    return saved == current or (Path(saved).is_absolute() and not Path(current).is_absolute()
                                and saved.endswith("/" + current))


def locate(recorded, root=None):
    """The current file for a recorded path: under the project root, at the longest tail of an old absolute path
    that exists there; otherwise the recorded path itself."""
    root, path = Path(ROOT if root is None else root), Path(recorded)
    if not path.is_absolute():
        return root / path
    for start in range(1, len(path.parts)):
        candidate = root.joinpath(*path.parts[start:])
        if candidate.exists():
            return candidate
    return path
