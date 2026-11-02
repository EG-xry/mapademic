"""Exercise release staging and upload ordering without network credentials."""
import gzip
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/upload_r2.sh"


def fixture(tmp_path, *, repaired=True, regions=True):
    root = tmp_path / "uploader"
    (root / "scripts").mkdir(parents=True)
    shutil.copy2(SCRIPT, root / "scripts/upload_r2.sh")
    python = root / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    release = root / "data/test-release"
    for directory in ("tiles", "index", "paths", "vector_experiment"):
        (release / directory).mkdir(parents=True)
    (release / "tiles/0.png").write_bytes(b"tile")
    (release / "index/1.json").write_text('{"A1":[0.2,0.3]}')
    if regions:
        (release / "regions.json").write_text('{}')
    (release / "paths/ids-test.bin").write_bytes(b"ids")
    (release / "paths/manifest.json").write_text(json.dumps({
        "version": 3 if repaired else 2, "files": {"ids": "ids-test.bin"},
    }))
    (release / "vector_experiment/positions.bin").write_bytes(b"positions")
    if repaired:
        identities = release / "identities"
        (identities / "profiles").mkdir(parents=True)
        (identities / "aliases").mkdir()
        (identities / "institutions").mkdir()
        (identities / "manifest.json").write_text(json.dumps({
            "version": 1, "graph_repaired": True, "paths_ids_file": "ids-test.bin", "flags_file": "flags.json",
        }))
        (identities / "flags.json").write_text(json.dumps({
            "version": 1, "paths_ids_file": "ids-test.bin", "entries": [{"id": "M1", "n": 0, "kind": "quarantined"}],
        }))
        (identities / "profiles/M1.json").write_text('{"id":"M1","status":"quarantined"}')
        (identities / "aliases/1.json").write_text('{"A1":["M1"]}')
        (identities / "institutions/I1.json").write_text('[{"id":"M1"}]')
    binary = root / "mock-bin/rclone"
    binary.parent.mkdir()
    binary.write_text(f"#!{sys.executable}\n" +
        "import json, os, sys\n"
        "with open(os.environ['RCLONE_LOG'], 'a') as f: f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if os.environ.get('RCLONE_FAIL_IDENTITIES') and sys.argv[1] == 'copy' and sys.argv[2].endswith('/identities'): sys.exit(3)\n"
        "if sys.argv[1] == 'listremotes': print('r2:')\n")
    binary.chmod(0o755)
    log = root / "rclone.log"
    env = {**os.environ, "PATH": str(binary.parent) + os.pathsep + os.environ["PATH"],
           "MAPADEMIC_DATA": str(release), "BUCKET": "r2:mapademic", "RCLONE_LOG": str(log),
           "FREE_BUDGET_GB": "9.5"}
    return root, release, log, env


def run(root, log, env):
    process = subprocess.run(["bash", str(root / "scripts/upload_r2.sh")], env=env,
                             capture_output=True, text=True)
    commands = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return process, commands


def test_repaired_json_is_gzipped_and_uploaded_before_graph_manifest_without_deletions(tmp_path):
    root, release, log, env = fixture(tmp_path)
    stale = root / ".tmp/r2-stage/test-release/identities/profiles/M999.json"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale generation")
    process, commands = run(root, log, env)
    assert process.returncode == 0, process.stderr
    stage = root / ".tmp/r2-stage/test-release"
    stage_arg = stage.relative_to(root)
    for source in (release / "identities").rglob("*.json"):
        staged = stage / source.relative_to(release)
        assert json.loads(gzip.decompress(staged.read_bytes())) == json.loads(source.read_text())
    assert not stale.exists()
    assert not any(command[0] in {"sync", "delete", "purge"} for command in commands)
    uploads = [command for command in commands if command[0] in {"copy", "copyto"}]
    final = uploads[-1]
    assert final[:3] == ["copyto", str(release / "paths/manifest.json"), "r2:mapademic/test-release/data/paths/manifest.json"]
    identity_manifest = next(command for command in uploads if command[0] == "copyto" and command[1] == str(stage_arg / "identities/manifest.json"))
    assert "Content-Encoding: gzip" in identity_manifest
    assert "Cache-Control: public, max-age=300" in identity_manifest
    json_upload = next(command for command in uploads if command[:2] == ["copy", str(stage_arg)])
    assert "Content-Encoding: gzip" in json_upload
    assert "identities/**" in json_upload
    identity_upload = next(command for command in uploads if command[:2] == ["copy", str(stage_arg / "identities")])
    assert "Content-Encoding: gzip" in identity_upload
    assert "Cache-Control: public, max-age=300" in identity_upload


@pytest.mark.parametrize("missing", ["manifest.json", "flags.json"])
def test_repaired_graph_missing_required_metadata_fails_before_upload(tmp_path, missing):
    root, release, log, env = fixture(tmp_path)
    (release / "identities" / missing).unlink()
    process, commands = run(root, log, env)
    assert process.returncode != 0
    assert not any(command[0] in {"copy", "copyto", "sync"} for command in commands)
    assert "identit" in process.stderr.lower()


@pytest.mark.parametrize("filename", ["manifest.json", "flags.json"])
def test_repaired_metadata_for_another_graph_cannot_upload(tmp_path, filename):
    root, release, log, env = fixture(tmp_path)
    flags = release / "identities" / filename
    data = json.loads(flags.read_text())
    data["paths_ids_file"] = "other-graph.bin"
    flags.write_text(json.dumps(data))
    process, commands = run(root, log, env)
    assert process.returncode != 0
    assert not any(command[0] in {"copy", "copyto", "sync"} for command in commands)


def test_failed_identity_upload_does_not_publish_graph_manifest(tmp_path):
    root, release, log, env = fixture(tmp_path)
    env["RCLONE_FAIL_IDENTITIES"] = "1"
    process, commands = run(root, log, env)
    assert process.returncode != 0
    assert any(command[0] == "copy" and command[1].endswith("/identities") for command in commands)
    assert not any(command[0] == "copyto" and command[1] == str(release / "paths/manifest.json") for command in commands)


def test_identity_json_bytes_count_towards_upload_budget(tmp_path):
    root, release, log, env = fixture(tmp_path)
    noise = random.Random(23).randbytes(8192).hex()
    (release / "identities/profiles/M1.json").write_text(json.dumps({"id": "M1", "evidence": noise}))
    env["FREE_BUDGET_GB"] = "0.000001"
    process, commands = run(root, log, env)
    assert process.returncode != 0
    assert "over the free budget" in process.stderr
    assert not any(command[0] in {"copy", "copyto", "sync"} for command in commands)


def test_legacy_release_needs_no_identity_metadata_or_optional_regions(tmp_path):
    root, release, log, env = fixture(tmp_path, repaired=False, regions=False)
    process, commands = run(root, log, env)
    assert process.returncode == 0, process.stderr
    assert not (root / ".tmp/r2-stage/test-release/identities").exists()
    assert commands[-1][0] == "size"
