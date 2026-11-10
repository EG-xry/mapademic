"""Run full manuscript-required min5 validation under the existing serial-process watchdog.

Run only after the analysis runner and manuscript build finish. This acquires the same runner.lock
without changing its contents, launches only validate_min5.py --require-manuscript, and saves its
own command, log, events, resource receipt and fresh validation report under final_validation/.
It never changes the parent runner's state/events or completion markers and never publishes.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime

import run_min5 as runner
from delivery_checks import REQUIRED_FINAL_CHECKS
from run_resources import (FOOTPRINT_LIMIT, FREE_PERCENT_LIMIT, POLL_SECONDS, SWAP_GROWTH_LIMIT_MB,
                           SWAP_LIMIT_MB, launch_owned, terminate_owned)

STEP = "final_validation"
CODE_FILES = ("run_final_validation.py", "validate_min5.py", "run_min5.py", "run_resources.py", "delivery_checks.py")


@contextmanager
def runner_lock(run_dir):
    """Use the existing flock inode without truncating or replacing the live owner's text."""
    with (run_dir / "runner.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("The analysis runner or another final validation still holds runner.lock") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def require_finished_runner(run_dir):
    parent = json.loads((run_dir / "state.json").read_text())
    incomplete = [step["name"] for step in runner.STEPS
                  if parent.get("steps", {}).get(step["name"], {}).get("status") != "done"]
    if incomplete or parent.get("current") is not None or parent.get("status") == "running":
        raise ValueError(f"Finish the main analysis runner before final validation; unfinished steps: {incomplete}")


def fresh_full_report(path, started_ns):
    """A zero subprocess exit alone cannot certify a stale, partial or failed validation."""
    before = path.stat()
    if before.st_mtime_ns < started_ns:
        raise ValueError("Validation report predates this invocation")
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("Validation report changed while its receipt was read")
    report = json.loads(raw)
    if not (report.get("network") == "min5" and report.get("status") == "passed"
            and report.get("quick") is False and report.get("reconstruction_requested") is True
            and report.get("manuscript_required") is True):
        raise ValueError("A fresh passing full --require-manuscript validation report is required")
    checks = report.get("checks", [])
    passed = {check["name"] for check in checks if check.get("status") == "pass"}
    if REQUIRED_FINAL_CHECKS - passed or any(check.get("status") != "pass" for check in checks):
        raise ValueError(f"Required validation checks are absent or failed: {sorted(REQUIRED_FINAL_CHECKS - passed)}")
    fingerprint = report.get("delivery_fingerprint")
    if not isinstance(fingerprint, dict) or not fingerprint.get("sha256"):
        raise ValueError("Validation report lacks its scientific/delivery fingerprint")
    return raw, {"bytes": before.st_size, "mtime_ns": before.st_mtime_ns,
                 "sha256": hashlib.sha256(raw).hexdigest(), "passed_checks": len(checks),
                 "delivery_fingerprint_sha256": fingerprint["sha256"]}


def run_validation(root=runner.ROOT, run_dir=runner.RUN, python=sys.executable, min_free=35):
    root, run_dir = Path(root), Path(run_dir)
    analysis = root / "paper/analysis"
    with runner_lock(run_dir):
        attempt = run_dir / "final_validation" / f"{datetime.now(runner.TZ):%Y%m%dT%H%M%S}_{time.time_ns()}_{os.getpid()}"
        attempt.mkdir(parents=True)
        info = {"status": "preflight"}
        state = {"network": "min5", "status": "preflight", "started": runner.now(), "pid": os.getpid(),
                 "parent_runner_state_unchanged": True, "steps": {STEP: info},
                 "resource_watchdog": {"poll_seconds": POLL_SECONDS, "footprint_limit_bytes": FOOTPRINT_LIMIT,
                                        "minimum_free_percent": FREE_PERCENT_LIMIT, "swap_absolute_limit_mb": SWAP_LIMIT_MB,
                                        "swap_growth_limit_mb": SWAP_GROWTH_LIMIT_MB,
                                        "accounting": "macOS libproc physical footprint, not RSS"}}

        def emit(current, kind, **details):
            event = {"time": runner.now(), "event": kind, **details}
            with (attempt / "events.jsonl").open("a") as events:
                events.write(json.dumps(event) + "\n")
            current["updated"], current["last_event"] = event["time"], event
            runner.atomic_json(attempt / "receipt.json", current)
            print(event["time"], kind, details, flush=True)

        owned = None
        started = time.monotonic()
        try:
            emit(state, "preflight", receipt=str(attempt / "receipt.json"))
            require_finished_runner(run_dir)
            while True:
                baseline = runner.memory()
                if baseline["free_percent"] >= min_free:
                    break
                emit(state, "memory_wait", resources=baseline, required_free_percent=min_free)
                time.sleep(30)
            if baseline.get("disk_free_gb", 100) < 15:
                raise RuntimeError("Less than 15GB disk space remains for full validation")
            env = {**os.environ, "NET": "min5", "PYTHONUNBUFFERED": "1", "TZ": "America/Los_Angeles",
                   "NUMBA_NUM_THREADS": "2", "MAPADEMIC_THREADS": "4",
                   **{key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}
            command = ["/usr/bin/time", "-l", python, "validate_min5.py", "--require-manuscript"]
            command_record = {"command": command, "cwd": str(analysis),
                              "parallelism": {"numba": 2, "duckdb": 4, "blas_openmp": 1},
                              "source_sha256": {name: hashlib.sha256((analysis / name).read_bytes()).hexdigest()
                                                for name in CODE_FILES}}
            runner.atomic_json(attempt / "command.json", command_record)
            started, started_ns = time.monotonic(), time.time_ns()
            state["status"] = "running"
            info.update(status="running", started=runner.now(), started_ns=started_ns, **command_record,
                        log=str(attempt / "validation.log"))
            emit(state, "start", step=STEP, command=command, resources=baseline)
            with (attempt / "validation.log").open("w") as log:
                owned = launch_owned(command, cwd=analysis, env=env, stdout=log, stderr=subprocess.STDOUT)
                info.update(pid=owned.process.pid, pgid=owned.pgid, leader_start_abstime=owned.leader_start)
                reasons = runner.monitor_step(owned, state, STEP, started, baseline, emit=emit)
            info.update(exit_code=owned.process.returncode, elapsed_seconds=round(time.monotonic() - started, 1))
            if reasons or owned.process.returncode != 0:
                raise RuntimeError(f"Full validation failed; inspect {attempt / 'validation.log'}")
            report_path = root / "paper/results/validation_min5.json"
            raw, receipt = fresh_full_report(report_path, started_ns)
            (attempt / "validation_min5.json").write_bytes(raw)
            info["validation_report"] = {"path": str(report_path), "archived_copy": "validation_min5.json", **receipt}
            info.update(status="done", finished=runner.now())
            state["status"] = "passed"
            emit(state, "passed", step=STEP, report_sha256=receipt["sha256"])
            return attempt
        except BaseException as error:
            state["status"] = info["status"] = "failed"
            if owned is not None:
                try:
                    cleanup = terminate_owned(owned)
                    info["final_cleanup_verification" if "cleanup" in info else "cleanup"] = cleanup
                except Exception as cleanup_error:
                    info["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
            info.update(error=f"{type(error).__name__}: {error}", finished=runner.now(),
                        elapsed_seconds=round(time.monotonic() - started, 1))
            emit(state, "failed", step=STEP, error=info["error"], cleanup=info.get("cleanup"),
                 cleanup_error=info.get("cleanup_error"))
            raise


def interrupted(signum, frame):
    raise InterruptedError(f"final validation received signal {signum}; stopping its owned group")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-free", type=int, choices=range(16, 101), default=35,
                        help="minimum free-memory percentage before launch (default: 35)")
    args = parser.parse_args()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        path = run_validation(min_free=args.min_free)
    except (Exception, KeyboardInterrupt) as error:
        raise SystemExit(f"Final validation stopped: {type(error).__name__}: {error}") from error
    print(f"Full validation passed; resource and output receipt: {path / 'receipt.json'}")


if __name__ == "__main__":
    main()
