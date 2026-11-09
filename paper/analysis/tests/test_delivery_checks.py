"""Small delivery fixtures; never load the production graph or render the paper."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image

ANALYSIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS))
import delivery_checks as delivery
import record_min5_provenance as provenance
from run_min5 import STEPS

BUILD = ANALYSIS.parent / "build.py"  # the manuscript build script, which is not released with this code
build = None
if BUILD.exists():
    spec = importlib.util.spec_from_file_location("delivery_paper_build", BUILD)
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)


def put(path, text="fixture"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def artifact(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix in {".png", ".tif"}:
        Image.new("RGB", (8, 8)).save(path, dpi=(600, 600))
    elif path.suffix == ".pdf":
        path.write_bytes(b"%PDF-1.5\nfixture-container-only\n%%EOF\n")
    elif path.suffix == ".eps":
        path.write_text("%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 8 8\n%%EOF\n")
    elif path.suffix == ".docx":
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("[Content_Types].xml", "<Types/>")
            archive.writestr("word/document.xml", "<document/>")
    else:
        put(path)


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for z in range(6):
            artifact(self.root / f"paper/cache/map/tiles/{z}/0/0.png")
        put(self.root / "paper/cache/map/.recolour-complete.json",
            json.dumps({"network": "min5", "tiles_by_zoom": {str(z): 1 for z in range(6)}}))
        for number in delivery.FIGURES:
            for path in delivery.figure_inputs(self.root, number):
                if not path.exists():
                    put(path)
        for path in delivery.figure_paths(self.root):
            artifact(path)
        for name in ("paper/draft2.md", "paper/build.py", "paper/analysis/delivery_checks.py", "pyproject.toml",
                     "paper/results/a1_basics.json", "data/2026-09-23/min5/deposit/SHA256SUMS",
                     "data/2026-09-23/min5/deposit/README.md", "data/2026-09-23/min5/deposit/links.parquet",
                     "data/2026-09-23/min5/pathgraph/neighbors.npy", "data/2026-09-23/coords_web.parquet",
                     "paper/cache/jmnedict_readings.json", "paper/cache/a2_sources.npy",
                     "paper/cache/repair_rep.npy", "paper/cache/ext/jmnedict/JMnedict.xml.gz"):
            put(self.root / name)
        for name in delivery.BUILD_OUTPUTS:
            artifact(self.root / "paper/build" / name)
        self.receipt = delivery.write_build_receipt(self.root, delivery.build_inputs(self.root), "2026-11-03T00:00:00-08:00")

    def tearDown(self):
        self.temp.cleanup()

    def test_all_formats_and_receipt_are_required(self):
        # 12 figures: PDF, PNG and TIFF each, and an EPS for the 8 that are not mostly raster
        self.assertEqual(delivery.validate_figures(self.root)["files"], 44)
        # the article and Online Resource 1: DOCX, PDF and TEX each, and the shared LaTeX header
        self.assertEqual(delivery.validate_build_receipt(self.root)["output_hashes_verified"], 7)
        (self.root / "paper/figures/submission/FigA2.eps").unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            delivery.validate_figures(self.root)

    def test_receipt_records_only_locale_keys_from_child_environment(self):
        receipt = delivery.write_build_receipt(self.root, delivery.build_inputs(self.root), "2026-11-03T00:00:00-08:00",
                                               locale={"LANG": "en_US.UTF-8", "LC_ALL": "en_US.UTF-8",
                                                       "PRIVATE_TEST_VALUE": "not-for-the-receipt"})
        self.assertEqual(receipt["subprocess_locale"], {"LANG": "en_US.UTF-8", "LC_ALL": "en_US.UTF-8"})
        self.assertNotIn("not-for-the-receipt", (self.root / "paper/build" / delivery.RECEIPT).read_text())
        delivery.validate_build_receipt(self.root)

    def test_each_variant_must_follow_its_data_and_tiff_must_be_600_dpi(self):
        path = self.root / "paper/results/a2_distances.json"
        newer = time.time_ns() + 1_000_000_000
        os.utime(path, ns=(newer, newer))
        with self.assertRaisesRegex(ValueError, "predates"):
            delivery.validate_figures(self.root)
        path = self.root / "paper/figures/submission/Fig2.tif"
        Image.new("RGB", (8, 8)).save(path, dpi=(72, 72))
        with self.assertRaisesRegex(ValueError, "600 dpi"):
            delivery.validate_format(path)

    def test_recolour_receipt_detects_missing_tile(self):
        (self.root / "paper/cache/map/tiles/5/0/0.png").unlink()
        with self.assertRaisesRegex(ValueError, "tile inventory"):
            delivery.validate_figures(self.root)

    def test_receipt_catches_content_change_even_with_preserved_size_and_mtime(self):
        path = self.root / "paper/draft2.md"
        stat = path.stat()
        path.write_text("changed")  # same seven bytes as fixture
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            delivery.validate_build_receipt(self.root)

    def test_receipt_binds_the_png_that_only_the_docx_embeds(self):
        # The review copy embeds Fig. 2 as a PDF; the DOCX embeds its PNG, so a changed PNG must stale the build.
        Image.new("RGB", (8, 8), "white").save(self.root / "paper/figures/fig_distances.png", dpi=(600, 600))
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            delivery.validate_build_receipt(self.root)

    def test_receipt_rejects_missing_output_hash_and_corrupt_container(self):
        path = self.root / "paper/build" / delivery.RECEIPT
        receipt = json.loads(path.read_text())
        receipt["outputs"][0].pop("sha256")
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "output content hashes"):
            delivery.validate_build_receipt(self.root)
        target = self.root / "paper/build/manuscript.pdf"
        target.write_text("not a pdf")
        with self.assertRaisesRegex(ValueError, "PDF"):
            delivery.validate_format(target)

    @unittest.skipIf(build is None, "needs paper/build.py, the manuscript build script")
    def test_failed_build_discards_previous_success_receipt(self):
        source = "# Title\n\nAuthor\n\n## Abstract\n\n" + " ".join(f"Fig. {n}" for n in delivery.FIGURES if n[0] != "A")
        source += "\n\n## Online Resource 1\n\n## Appendix A\n\n" + " ".join(f"Fig. {n}" for n in delivery.FIGURES if n[0] == "A")
        source += "\n\n## Figure captions\n\n" + "\n".join(f"**Fig. {n}** Caption." for n in delivery.FIGURES)
        source += "\n\n## References\nReference.\n"
        (self.root / "paper/draft2.md").write_text(source)
        with patch.object(build, "PAPER", self.root / "paper"), patch.object(build, "OUT", self.root / "paper/build"), \
                patch.object(build, "pandoc", side_effect=RuntimeError("simulated pandoc failure")):
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                build.main()
        self.assertFalse((self.root / "paper/build" / delivery.RECEIPT).exists())

    def test_fingerprint_ignores_progress_but_binds_science_and_large_file_identity(self):
        previous = delivery.validation_fingerprint(self.root)
        for name in ("paper/results/validation_min5.json", "paper/VALIDATION_MIN5.md",
                     "paper/runs/min5-2026-11-03/state.json", "paper/runs/min5-2026-11-03/README.md"):
            put(self.root / name, "updated progress")
        delivery.verify_fingerprint(self.root, previous)
        hashed = {r["path"] for r in previous["content_hashed"]}
        identities = {r["path"]: r for r in previous["metadata_only"]}
        self.assertIn("paper/cache/a2_sources.npy", hashed)
        for name in ("data/2026-09-23/coords_web.parquet", "paper/cache/repair_rep.npy",
                     "paper/cache/jmnedict_readings.json", "paper/cache/ext/jmnedict/JMnedict.xml.gz"):
            self.assertIn(name, identities)
            self.assertNotIn("sha256", identities[name])
        put(self.root / "data/2026-09-23/min5/pathgraph/neighbors.npy", "changed graph identity")
        with self.assertRaisesRegex(ValueError, "neighbors.npy"):
            delivery.verify_fingerprint(self.root, previous)

    def test_final_provenance_requires_full_checks_fresh_review_and_current_fingerprint(self):
        state = {"steps": {step["name"]: {"status": "done"} for step in STEPS}}
        validation = {"status": "passed", "network": "min5", "quick": False,
                      "reconstruction_requested": True, "manuscript_required": True,
                      "checks": [{"name": name, "status": "pass"} for name in delivery.REQUIRED_FINAL_CHECKS],
                      "delivery_fingerprint": delivery.validation_fingerprint(self.root)}
        reviewed = self.root / "paper/review/min5-figure-review.md"
        put(reviewed, "Inspected the final figures and built PDF.")
        with patch.object(provenance, "ROOT", self.root), patch.object(provenance, "PAPER", self.root / "paper"):
            provenance.assert_final(state, validation)
            os.utime(reviewed, ns=(1, 1))
            with self.assertRaisesRegex(ValueError, "review record predates"):
                provenance.assert_final(state, validation)
            reviewed.touch()
            validation["checks"].pop()
            with self.assertRaisesRegex(ValueError, "checks are missing"):
                provenance.assert_final(state, validation)
            validation["checks"] = [{"name": name, "status": "pass"} for name in delivery.REQUIRED_FINAL_CHECKS]
            put(self.root / "paper/analysis/new_scientific_code.py")
            with self.assertRaisesRegex(ValueError, "new_scientific_code"):
                provenance.assert_final(state, validation)


if __name__ == "__main__":
    unittest.main()
