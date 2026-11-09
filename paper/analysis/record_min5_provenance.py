"""Record local code, environment and artifact provenance for the five-work run.

Use --final only after the complete validator and visual/editorial review. The
final gate requires unchanged validated scientific/delivery fingerprints and a
successful exact-input manuscript build receipt. Large
source files are inventoried by path/size/mtime, not silently claimed as content
hashed. Deposit content hashes come from its fully validated SHA256SUMS file.
This command neither uploads files nor changes runner completion status.
"""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from delivery_checks import (BUILD_OUTPUTS, RECEIPT, REQUIRED_FINAL_CHECKS, code_paths,
                             figure_paths, large_source_paths, supporting_paths,
                             validate_build_receipt, verify_fingerprint)

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
RUN = PAPER / "runs/min5-2026-11-03"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def record(path, content_hash=False):
    stat = path.stat()
    value = {"path": str(path.relative_to(ROOT)), "bytes": stat.st_size,
             "mtime_ns": stat.st_mtime_ns}
    if content_hash:
        value["sha256"] = digest(path)
    return value


def command(args):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def assert_final(state, validation):
    from run_min5 import STEPS
    incomplete = [step["name"] for step in STEPS
                  if state["steps"].get(step["name"], {}).get("status") != "done"]
    if incomplete:
        raise ValueError(f"unfinished continuation steps: {incomplete}")
    if not (validation.get("status") == "passed" and not validation.get("quick")
            and validation.get("reconstruction_requested") and validation.get("manuscript_required")):
        raise ValueError("a passing full --require-manuscript validation is required")
    checks = validation.get("checks", [])
    passed = {check["name"] for check in checks if check.get("status") == "pass"}
    if REQUIRED_FINAL_CHECKS - passed or any(check.get("status") != "pass" for check in checks):
        raise ValueError(f"full validation checks are missing or failed: {sorted(REQUIRED_FINAL_CHECKS - passed)}")
    if validation.get("network") != "min5" or not validation.get("delivery_fingerprint"):
        raise ValueError("validation does not bind the five-work scientific/delivery state")
    verify_fingerprint(ROOT, validation["delivery_fingerprint"])
    validate_build_receipt(ROOT)
    reviewed = PAPER / "review/min5-figure-review.md"
    if not reviewed.is_file() or not reviewed.read_text().strip():
        raise ValueError("the figure/manuscript visual review record is missing")
    visual_inputs = [*figure_paths(ROOT), *(PAPER / "build" / name for name in (*BUILD_OUTPUTS, RECEIPT))]
    if reviewed.stat().st_mtime_ns < max(path.stat().st_mtime_ns for path in visual_inputs):
        raise ValueError("visual review record predates a final figure or manuscript build")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    state = json.loads((RUN / "state.json").read_text())
    validation_path = PAPER / "results/validation_min5.json"
    validation = json.loads(validation_path.read_text()) if validation_path.exists() else {}
    if args.final:
        assert_final(state, validation)

    code = set(code_paths(ROOT))
    artifacts = set((PAPER / "results").glob("*.json"))
    artifacts.update((PAPER / "results_min15").glob("*.json"))
    artifacts.update(path for path in (PAPER / "figures").rglob("*")
                     if path.is_file() and path.suffix in {".png", ".pdf", ".eps", ".tif"})
    artifacts.update((PAPER / "build").glob("*"))
    artifacts.update((PAPER / "cache").glob("*.provenance.json"))
    artifacts.update((PAPER / "review").glob("min5-*.md"))
    artifacts.update(PAPER / name for name in (
        "draft2.md", "RESULTS.md", "PLAN.md", "VALIDATION_MIN5.md", "deposit_README.md", "THRESHOLD_COMPARISON.md"))
    artifacts.update((PAPER / "analysis/README.md", RUN / "README.md",
                      PAPER / "results/quoted_numbers.txt"))
    artifacts.update(RUN.glob("tests*.log"))
    artifacts.update(RUN.glob("install-*.log"))
    artifacts.update((RUN / "logs").glob("*.log"))
    artifacts.update(RUN.glob("cap*-validation.json"))
    artifacts.update(path for path in (RUN / "final_validation").glob("*/*")
                     if path.is_file() and path.suffix in {".json", ".jsonl", ".log"})
    artifacts.update((RUN / "snapshot_preflight.json", RUN / "early-cap50-baseline.json", RUN / "cap50-validation.json",
                      RUN / "repair-safeguards-code.json", RUN / "team-cap-validation.json"))
    deposit = ROOT / "data/2026-09-23/min5/deposit"
    artifacts.update((deposit / "README.md", deposit / "SHA256SUMS"))
    artifacts = {path for path in artifacts if path.is_file()}

    sources = set(large_source_paths(ROOT))
    # These precise small selections/receipts are content-hashed. In particular
    # the dictionary, large npy/npz arrays and API-part caches are metadata-only.
    manifests = [record(path, True) for path in supporting_paths(ROOT)]

    deposit_hashes = {}
    if (deposit / "SHA256SUMS").exists():
        for line in (deposit / "SHA256SUMS").read_text().splitlines():
            expected, filename = line.split(maxsplit=1)
            deposit_hashes[filename.lstrip("*")] = expected

    value = {
        "recorded": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds"),
        "network": "min5", "final": args.final,
        "local_only": True,
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(),
        "git_head": command(["git", "rev-parse", "HEAD"]),
        "git_status": command(["git", "status", "--short"]),
        "packages": dict(sorted((dist.metadata["Name"], dist.version) for dist in metadata.distributions())),
        "build_tools": {"pandoc": command(["pandoc", "--version"]).splitlines()[0],
                        "lualatex": command(["lualatex", "--version"]).splitlines()[0]},
        "parallelism": state.get("parallelism"),
        "execution_steps_at_capture": state.get("steps"),
        "validation": validation,
        "code": [record(path, True) for path in sorted(code)],
        "artifacts": [record(path, True) for path in sorted(artifacts)],
        "large_sources_metadata_only": [record(path) for path in sorted(sources)],
        "supporting_manifest_hashes": manifests,
        "deposit_content_hashes": deposit_hashes,
        "limits": [
            "The working tree is not a release commit; the code hashes identify its local state.",
            "Large source files are inventoried by path, size and mtime; these fields are not content hashes.",
            "Original map coordinates, census/extract inputs, institutions/topics, repair arrays, source selection, dictionary and named caches are explicitly inventoried; arbitrary recursive cache trees are not hashed.",
            "A final receipt verifies the validation fingerprint and build receipt; later scientific code, data identity, deposited payload, manuscript or figure changes require validation again.",
            "Historical cache acquisition limitations are recorded in their provenance sidecars.",
            "Runner state/events continue changing until the separate completion event; they are not self-hashed here.",
            "No artifact or dataset is uploaded or published by this command."]}
    if args.final:
        # Reject a concurrent edit made while content hashes were being read.
        assert_final(state, validation)
    target = RUN / ("provenance-final.json" if args.final else "provenance-intermediate.json")
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(target)
    print(f"Recorded {len(code)} code files, {len(artifacts)} artifacts and {len(sources)} source-file identities in {target}")


if __name__ == "__main__":
    main()
