"""Small-input checks for figure selection, map swap recovery and manuscript preflight.

No graph data, map rendering or manuscript build is run.
  .tmp/venv-paper/bin/python -m unittest discover -s paper/analysis/tests -p test_figure_preparation.py -v
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

import numpy as np
import pandas as pd
from PIL import Image

ANALYSIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS))
import figures
import map_recolour

BUILD = ANALYSIS.parent / "build.py"  # the manuscript build script, which is not released with this code
build = None
if BUILD.exists():
    spec = importlib.util.spec_from_file_location("paper_build", BUILD)
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)


class FigureSelectionTests(unittest.TestCase):
    def test_affinity_scale_encloses_new_extremes_and_centers_on_one(self):
        lo, hi, ticks, labels = figures.affinity_log_scale(np.array([0.134, 19.58, np.nan, 0.]))
        self.assertEqual((lo, hi), (1 / 32, 32))
        self.assertEqual(lo * hi, 1)
        self.assertIn(1., ticks)
        self.assertEqual(labels[ticks.index(1.)], "1")
        self.assertEqual(figures.affinity_log_scale(np.array([np.nan, 0.]))[:2], (0.5, 2.))

    def test_region_order_covers_new_countries_once(self):
        names = ["PK", "US", "EG", "DE", "KR", "ID"]
        blocks = figures.country_region_blocks(names)
        self.assertEqual(blocks, [["US"], ["DE"], ["ID"], ["KR"], ["PK", "EG"]])
        flat = [name for block in blocks for name in block]
        self.assertCountEqual(flat, names)
        self.assertEqual(len(flat), len(set(flat)))
        self.assertEqual(figures.country_region_blocks([]), [])
        self.assertEqual(figures.country_region_blocks(["UA", "RU", "PL", "JP"]), [["PL", "RU", "UA"], ["JP"]])

    def test_heatmap_groups_include_missing_data_and_threshold_boundary(self):
        frame = pd.DataFrame({"high": [0.2, np.nan], "boundary": [0.005, 0],
                              "small_nonzero": [0.003, 0], "zero": [0., 0.], "missing": [np.nan, np.nan]})
        self.assertEqual(figures.national_country_groups(frame),
                         (["high", "boundary"], ["small_nonzero", "zero"], ["missing"]))

    def test_heatmap_handles_each_empty_group(self):
        self.assertEqual(figures.national_country_groups(pd.DataFrame({"A": [0.2], "B": [0.7]})),
                         (["A", "B"], [], []))
        self.assertEqual(figures.national_country_groups(pd.DataFrame({"A": [0.], "B": [0.001]})),
                         ([], ["A", "B"], []))
        self.assertEqual(figures.national_country_groups(pd.DataFrame({"A": [np.nan]})), ([], [], ["A"]))


class MapPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.destination = self.root / "map"
        self.stage = self.root / ".map-recolour-stage-fixture"
        self.backup = self.root / ".map-recolour-backup-fixture"
        self.journal = self.root / ".map-recolour-swap.json"
        self.stage.mkdir()
        (self.stage / "tile.txt").write_text("new")
        (self.stage / map_recolour.COMPLETE).write_text(json.dumps({"token": "fixture-token"}))
        self.destination.mkdir()
        (self.destination / "tile.txt").write_text("old")

    def tearDown(self):
        self.temp.cleanup()

    def write_journal(self):
        self.journal.write_text(json.dumps({"destination": "map", "stage": self.stage.name,
                                           "backup": self.backup.name, "token": "fixture-token"}))

    def test_success_removes_only_owned_backup(self):
        unrelated = self.root / "user-backup"
        unrelated.mkdir()
        (unrelated / "keep.txt").write_text("keep")
        map_recolour.publish_map(self.stage, self.destination)
        self.assertEqual((self.destination / "tile.txt").read_text(), "new")
        self.assertFalse(self.stage.exists())
        self.assertFalse(self.journal.exists())
        self.assertEqual(list(self.root.glob(".map-recolour-backup-*")), [])
        self.assertEqual((unrelated / "keep.txt").read_text(), "keep")

    def test_failed_promotion_restores_previous_map(self):
        real_rename = Path.rename
        stage = self.stage

        def fail_stage(path, target):
            if path == stage:
                raise OSError("simulated promotion failure")
            return real_rename(path, target)

        with patch.object(Path, "rename", fail_stage):
            with self.assertRaisesRegex(OSError, "simulated"):
                map_recolour.publish_map(self.stage, self.destination)
        self.assertEqual((self.destination / "tile.txt").read_text(), "old")
        self.assertTrue(self.stage.exists())
        self.assertFalse(self.journal.exists())

    def test_restart_between_renames_restores_previous_map(self):
        self.write_journal()
        self.destination.rename(self.backup)
        map_recolour.recover_publish(self.destination)
        self.assertEqual((self.destination / "tile.txt").read_text(), "old")
        self.assertTrue(self.stage.exists())
        self.assertFalse(self.backup.exists())
        self.assertFalse(self.journal.exists())

    def test_restart_after_promotion_cleans_verified_backup(self):
        self.write_journal()
        self.destination.rename(self.backup)
        self.stage.rename(self.destination)
        map_recolour.recover_publish(self.destination)
        self.assertEqual((self.destination / "tile.txt").read_text(), "new")
        self.assertFalse(self.backup.exists())
        self.assertFalse(self.journal.exists())

    def test_missing_completion_marker_keeps_previous_map(self):
        (self.stage / map_recolour.COMPLETE).unlink()
        with self.assertRaisesRegex(ValueError, "completion marker"):
            map_recolour.publish_map(self.stage, self.destination)
        self.assertEqual((self.destination / "tile.txt").read_text(), "old")


@unittest.skipIf(build is None, "needs paper/build.py, the manuscript build script")
class ManuscriptPreflightTests(unittest.TestCase):
    def test_numerical_placeholders_refuse_build_source(self):
        for text in ("Mean {{TODO a8}}", "{{ TODO new result}}"):
            with self.assertRaisesRegex(ValueError, "numerical"):
                build.validate_source(text)
        text = "[TODO, authors: release code.]"
        self.assertEqual(build.validate_source(text), text)

    def test_date_uses_los_angeles_at_utc_date_boundary(self):
        self.assertEqual(build.draft_date(datetime(2026, 11, 4, 6, 30, tzinfo=timezone.utc)),
                         "Draft, 3 November 2026")

    def test_mac_locale_is_installed_and_changes_only_child_copy(self):
        parent = {"LANG": "C.UTF-8", "LC_ALL": "C", "PATH": "/fixture/bin", "KEEP": "unchanged"}
        before = dict(os.environ)
        with patch.object(build.subprocess, "run") as run:
            run.return_value.stdout = "C\nPOSIX\nen_US.UTF-8\nfr_FR.UTF-8\n"
            child = build.build_environment(parent, "darwin")
        self.assertEqual(child, {**parent, "LANG": "en_US.UTF-8", "LC_ALL": "en_US.UTF-8"})
        self.assertEqual(parent["LANG"], "C.UTF-8")
        self.assertEqual(dict(os.environ), before)
        self.assertEqual(run.call_args.kwargs["env"], {**parent, "LANG": "C", "LC_ALL": "C"})
        self.assertEqual(run.call_args.args[0], ["/usr/bin/locale", "-a"])

    def test_mac_fallback_missing_locale_and_nonmac_inheritance(self):
        parent = {"LANG": "fr_FR.UTF-8", "LC_ALL": "C", "KEEP": "unchanged"}
        with patch.object(build.subprocess, "run") as run:
            run.return_value.stdout = "C\nfr_FR.UTF-8\n"
            self.assertEqual(build.build_environment(parent, "darwin")["LC_ALL"], "fr_FR.UTF-8")
            run.return_value.stdout = "C\nPOSIX\n"
            with self.assertRaisesRegex(RuntimeError, "en_US.UTF-8.*missing"):
                build.build_environment(parent, "darwin")
        with patch.object(build.subprocess, "run") as run:
            inherited = build.build_environment(parent, "linux")
            self.assertEqual(inherited, parent)
            self.assertIsNot(inherited, parent)
            run.assert_not_called()

    def test_all_pandoc_children_and_receipt_use_one_environment(self):
        # Simulated Pandoc only: no TeX engine or production build is launched.
        with tempfile.TemporaryDirectory() as temporary:
            paper = Path(temporary)
            (paper / "draft2.md").write_text("# Title\n\nAuthor\n\nAffiliation\n\n## Abstract\n\nSee Fig. 1.\n\n"
                                              "## Online Resource 1\n\nSupplementary text.\n\n"
                                              "## Figure captions\n\n**Fig. 1** Caption.\n\n## References\nReference.\n")
            child = {"LANG": "en_US.UTF-8", "LC_ALL": "en_US.UTF-8", "KEEP": "unchanged"}
            with patch.object(build, "PAPER", paper), patch.object(build, "OUT", paper / "build"), \
                    patch.object(build, "FIGURES", {"1": "fixture.png"}), \
                    patch.object(build, "build_inputs", return_value=[]), \
                    patch.object(build, "build_environment", return_value=child) as environment, \
                    patch.object(build, "write_build_receipt") as receipt, \
                    patch.object(build.subprocess, "run") as run:
                run.return_value.stdout = "Author or affiliation"
                build.main()
            environment.assert_called_once_with()
            self.assertEqual(run.call_count, 8)  # two DOCX, two author paragraphs, then TEX and PDF for each document
            for call in run.call_args_list:
                self.assertIs(call.kwargs["env"], child)
                self.assertEqual(call.args[0][0], "pandoc")
            self.assertIs(receipt.call_args.kwargs["locale"], child)

    @unittest.skipUnless(shutil.which("pandoc"), "needs Pandoc")
    def test_docx_embeds_each_figure_png_after_the_paragraph_that_first_cites_it(self):
        # Real Pandoc on a small fixture. Word cannot show the review copy's PDF figures, so the DOCX must carry
        # each figure's PNG in place, with its caption below; a bracket in a caption must not unhook its image.
        w = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        blip = "{http://schemas.openxmlformats.org/drawingml/2006/main}blip"
        embed = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
        with tempfile.TemporaryDirectory() as temporary:
            paper = Path(temporary)
            (paper / "figures").mkdir()
            for stem, colour in (("vector", "red"), ("raster", "blue")):
                Image.new("RGB", (60, 30), colour).save(paper / "figures" / f"{stem}.png", dpi=(300, 300))
            (paper / "figures/vector.pdf").write_bytes(b"%PDF-1.5\nfixture\n%%EOF\n")
            png = {stem: (paper / "figures" / f"{stem}.png").read_bytes() for stem in ("vector", "raster")}
            body = ("## Results\n\nFirst Fig. 2, then Fig. 1.\n\n: **Table 1a.** Counts.\n\n| a |\n|---|\n| 1 |\n\n"
                    "Fig. 1 again.\n\n")
            captions = {"1": "Vector, on [0, 1).", "2": "Raster."}
            with patch.object(build, "PAPER", paper), \
                    patch.object(build, "FIGURES", {"1": "vector.pdf", "2": "raster.png"}):
                build.pandoc(build.with_figures(body, captions, docx=True), paper / "out.docx", env=dict(os.environ))
            with zipfile.ZipFile(paper / "out.docx") as archive:
                targets = {rel.get("Id"): rel.get("Target")
                           for rel in ElementTree.fromstring(archive.read("word/_rels/document.xml.rels"))}
                blocks = []
                for paragraph in ElementTree.fromstring(archive.read("word/document.xml")).iter(w + "p"):
                    images = [archive.read("word/" + targets[node.get(embed)]) for node in paragraph.iter(blip)]
                    blocks.extend(images or ["".join(text.text or "" for text in paragraph.iter(w + "t"))])
        self.assertEqual(blocks, ["Results", "First Fig. 2, then Fig. 1.", png["raster"], "Fig. 2 Raster.",
                                  png["vector"], "Fig. 1 Vector, on [0, 1).", "Table 1a. Counts.", "a", "1",
                                  "Fig. 1 again."])


if __name__ == "__main__":
    unittest.main()
