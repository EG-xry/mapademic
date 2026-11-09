"""File identities that survive moving the project folder, as from ~/Downloads to ~/Desktop on 4 November 2026."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import identities


class MovedProjectIdentities(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        base = Path(self.temp.name)
        self.before, self.after = base / "before", base / "after"
        (self.before / "data").mkdir(parents=True)
        (self.before / "data/works.parquet").write_bytes(b"works")
        (self.before / "data/other.parquet").write_bytes(b"other")

    def tearDown(self):
        self.temp.cleanup()

    def move(self):
        os.rename(self.before, self.after)  # like mv or a Finder move on one disk: sizes and mtimes are kept

    def test_a_file_inside_the_project_is_recorded_relative_to_it(self):
        row = identities.file_identity(self.before / "data/works.parquet", root=self.before)
        self.assertEqual(row[0], "data/works.parquet")
        self.move()
        current = identities.file_identity(self.after / "data/works.parquet", root=self.after)
        self.assertTrue(identities.same_identity(row, current))

    def test_a_record_made_before_the_move_matches_the_moved_file(self):
        # Records written before 4 November 2026 hold absolute paths in the folder the project was in then.
        path = self.before / "data/works.parquet"
        stat = path.stat()
        legacy = {"version": 1, "inputs": [[str(path), stat.st_size, stat.st_mtime_ns]], "query": "SELECT 1"}
        self.move()
        current = {"version": 1, "inputs": [identities.file_identity(self.after / "data/works.parquet", root=self.after)],
                   "query": "SELECT 1"}
        self.assertTrue(identities.same_identity(legacy, current))
        self.assertEqual(identities.locate(legacy["inputs"][0][0], root=self.after), self.after / "data/works.parquet")

    def test_a_changed_or_different_file_does_not_match(self):
        path, other = self.before / "data/works.parquet", self.before / "data/other.parquet"
        stat = path.stat()
        os.utime(other, ns=(stat.st_atime_ns, stat.st_mtime_ns))  # same size and mtime, different file
        legacy = [str(path), stat.st_size, stat.st_mtime_ns]
        self.move()
        renamed = identities.file_identity(self.after / "data/other.parquet", root=self.after)
        self.assertFalse(identities.same_identity(legacy, renamed))
        edited = self.after / "data/works.parquet"
        edited.write_bytes(b"works, edited")
        self.assertFalse(identities.same_identity(legacy, identities.file_identity(edited, root=self.after)))

    def test_only_the_path_of_a_file_identity_may_differ(self):
        self.assertFalse(identities.same_identity({"query": "/old/root/a"}, {"query": "a"}))
        self.assertFalse(identities.same_identity({"version": 1}, {"version": 2}))
        self.assertFalse(identities.same_identity({"rows": [["/old/root/a", 5, 7]]}, {"rows": [["a", 5, 8]]}))
        self.assertTrue(identities.same_identity({"rows": [["/old/root/a", 5, 7]]}, {"rows": [["a", 5, 7]]}))


if __name__ == "__main__":
    unittest.main()
