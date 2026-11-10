"""Resume the five-work paper analyses, serially, with durable logs and receipts.

Run with the paper environment from anywhere: python paper/analysis/run_min5.py.
Use --status for a read-only report, or --through STEP to stop at a step boundary.
This stages local outputs only; manuscript editing and publication are separate.
Every step owns a new process session/group. The watchdog samples its macOS
physical footprint each second, independently of RSS, and stops only that group
at unsafe footprint/free-memory/swap-growth levels (run_resources.py).
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from run_resources import (FOOTPRINT_LIMIT, FREE_PERCENT_LIMIT, GIB, POLL_SECONDS, SWAP_GROWTH_LIMIT_MB,
                           SWAP_LIMIT_MB, launch_owned, limit_reasons, owned_usage, terminate_owned)

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "paper/analysis"
CACHE = ROOT / "paper/cache"
RESULTS = ROOT / "paper/results"
RUN = ROOT / "paper/runs/min5-2026-11-03"
TZ = ZoneInfo("America/Los_Angeles")
# Each success requires both a zero exit and the declared nonempty output files.
# Earlier completed steps are audited at startup without recomputing them.
EARLIER = {
    "signals": ["data/2026-09-23/min5/pathgraph/link_signals.parquet", "paper/cache/suspect_mask.npy"],
    "a1": ["paper/results/a1_basics.json", "paper/cache/lcc_all.npy"],
    "attrs": ["paper/cache/node_attrs.parquet"],
    "a6_nonperson": ["paper/results/a6_nonperson.json", "paper/cache/nonperson_idx.npy"],
    "a2": ["paper/results/a2_distances.json", "paper/cache/a2_sources.npy"],
    "a4": ["paper/results/a4_hubs.json", "paper/cache/hubs.parquet", "paper/cache/closeness_1000.npy"],
    "a1b": ["paper/results/a1b_clustering.json"],
    "a3": ["paper/results/a3_farthest.json"],
    "a3b": ["paper/results/a3b_far_edge.json"],
    "a9": ["paper/results/a9_structure.json"],
    "a6_affiliations": ["paper/results/a6_affiliations.json", "paper/cache/a6_affiliations.csv"],
    "a6_live_works": ["paper/results/a6_live_works.json", "paper/cache/a6_live_works.csv"],
    "a6_churn_census": ["paper/results/a6_churn_census.json", "paper/cache/snapshot_20261014.parquet"],
}


def step(name, script, result=None, args=(), extra=()):
    return {"name": name, "command": [script, *args],
            "outputs": ([f"paper/results/{result}.json"] if result else []) + list(extra)}


STEPS = [
    step("a9c", "a9c_field_order.py", "a9c_field_order"),
    *[step(f"a6_twins_{s}", "a6_twins.py", f"a6_twins_{s}", [s], [f"paper/cache/a6_twins_{s}.csv"])
      for s in ("zh", "ja", "ko", "ru")],
    step("a6_splits", "a6_splits.py", "a6_splits", extra=["paper/cache/split_pairs.parquet"]),
    step("flags", "flags.py", extra=["paper/cache/flags.npz"]),
    step("b3", "b3_hubs.py", "b3_hubs"),
    step("a4b", "a4b_hubs_analysis.py", "a4b_hubs"),
    step("a5", "a5_clusters.py", "a5_clusters"),
    step("a5b", "a5b_communities.py", "a5b_communities", extra=["paper/cache/a5b_communities.csv"]),
    step("b4", "b4_countries.py", "b4_countries"),
    step("a7_peak25_strict", "a7_remedy.py", "a7_remedy_peak25_strict", ["25", "strict"],
         ["paper/cache/a7_sets_peak25_strict.npz"]),
    step("a7b_strict", "a7b_paired.py", "a7b_paired_peak25_strict", ["peak25_strict"]),
    step("b2", "b2_ties.py", "b2_ties"),
    step("b2b", "b2b_long_links.py", "b2b_long_links"),
    step("b1", "b1_uncited.py", "b1_uncited"),
    step("a2c", "a2c_team_cap.py", "a2c_team_cap", ["1000"], ["paper/results/a2c_team_cap_100.json"]),
    step("a8", "a8_repair.py", "a8_repair", extra=["paper/cache/repair_rep.npy"]),
    step("a8b", "a8b_communities.py", "a8b_communities"),
    step("a8c", "a8c_center.py", "a8c_center"),
    step("a9b", "a9b_repaired.py", "a9b_repaired"),
    step("a6_validate", "a6_validate.py", "a6_validate"),
    step("a6_institutions", "a6_institutions.py", "a6_institutions"),
    step("a6_quality", "a6_quality.py", "a6_quality"),
    step("churn_provenance", "a6_churn_snapshot.py", "a6_churn_census"),
    step("map_recolour", "map_recolour.py", extra=["paper/cache/map/positions.parquet", "paper/cache/map/coords_web.parquet"]),
    step("map_labels", "map_labels.py", extra=["paper/cache/community_labels.json", "paper/cache/map/regions.json"]),
    step("export_deposit", "export_deposit.py", extra=["data/2026-09-23/min5/deposit/SHA256SUMS"]),
    step("a10", "a10_checks.py", "a10_checks"),
    step("figures", "figures.py", extra=[f"paper/figures/fig_{s}.pdf" for s in
         ("map", "distances", "far_edge", "field_distances", "national_heatmap", "countries", "hub_definitions", "ties", "uncited", "affiliations", "twins")]),
]


def now():
    return datetime.now(TZ).isoformat(timespec="seconds")


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def valid(outputs):
    for rel in outputs:
        p = ROOT / rel
        if not p.is_file() or not p.stat().st_size:
            return False
        if p.suffix == ".json":
            try:
                json.loads(p.read_text())
            except (ValueError, OSError):
                return False
    return bool(outputs)


def memory():
    if sys.platform != "darwin":
        return {"free_percent": 100}
    pressure = subprocess.check_output(["memory_pressure"], text=True, timeout=2)
    swap = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True, timeout=2)
    return {"free_percent": int(re.search(r"free percentage: (\d+)%", pressure)[1]),
            "swap_used_mb": float(re.search(r"used = ([\d.]+)M", swap)[1]),
            "disk_free_gb": round(shutil.disk_usage(ROOT).free / 1e9, 2)}


def event(state, kind, **details):
    record = {"time": now(), "event": kind, **details}
    with (RUN / "events.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    state["updated"] = record["time"]
    state["last_event"] = record
    atomic_json(RUN / "state.json", state)
    lines = ["# Five-work run progress", "", f"Updated: {record['time']}", "",
             f"State: **{state.get('status', 'running')}**; current step: **{state.get('current', 'none')}**.", "",
             f"Continuation steps completed: **{sum(state['steps'].get(s['name'], {}).get('status') == 'done' for s in STEPS)}/{len(STEPS)}**.", "",
             "| Step | Status | Elapsed seconds |", "|---|---|---:|"]
    for name in [*EARLIER, *[s["name"] for s in STEPS]]:
        v = state["steps"].get(name, {"status": "pending"})
        lines.append(f"| {name} | {v['status']} | {v.get('elapsed_seconds', '')} |")
    lines += ["", "Commands, output receipts, resource measurements and failures are in `state.json` and `events.jsonl`.",
              "Per-step stdout/stderr logs are in `logs/`; successful checkpoints are checked against their output files.",
              "All timestamps use America/Los_Angeles. This run stages files locally and does not publish.", ""]
    (RUN / "PROGRESS.md").write_text("\n".join(lines))
    if kind != "heartbeat":
        print(record["time"], kind, details, flush=True)


def monitor_step(owned, state, name, started, baseline, emit=None):
    """One-second owned-group watchdog; errors/limits never become successful steps."""
    emit = event if emit is None else emit
    info = state["steps"][name]
    peaks = {"group_phys_footprint_bytes": 0, "max_member_peak_phys_footprint_bytes": 0,
             "group_resident_bytes": 0, "swap_used_mb": baseline.get("swap_used_mb", 0),
             "minimum_free_percent": baseline["free_percent"]}
    info["resource_baseline"] = baseline
    info["resource_peaks"] = peaks
    last_heartbeat = 0.0
    stop_reasons = []
    try:
        while owned.process.poll() is None:
            try:
                usage = owned_usage(owned)
            except Exception as error:
                usage = {"pgid": owned.pgid, "monitor_error": f"{type(error).__name__}: {error}"}
            # Evaluate footprint first, before any system diagnostic subprocess.
            resources = baseline
            reasons = limit_reasons(resources, baseline, usage)
            if not reasons:
                try:
                    resources = memory()
                    reasons = limit_reasons(resources, baseline, usage)
                except Exception as error:
                    reasons = [{"metric": "system_memory_monitor", "error": f"{type(error).__name__}: {error}"}]
            for key in ("group_phys_footprint_bytes", "max_member_peak_phys_footprint_bytes", "group_resident_bytes"):
                peaks[key] = max(peaks[key], usage.get(key, 0))
            peaks["swap_used_mb"] = max(peaks["swap_used_mb"], resources.get("swap_used_mb", 0))
            peaks["minimum_free_percent"] = min(peaks["minimum_free_percent"], resources["free_percent"])
            if reasons:
                stop_reasons = reasons
                info.update(status="resource limit", resource_limit_reasons=reasons)
                state["status"] = "failed"
                emit(state, "resource_limit", step=name, pid=owned.process.pid, pgid=owned.pgid,
                      reasons=reasons, baseline=baseline, resources=resources, process_usage=usage)
                cleanup = terminate_owned(owned)
                info["cleanup"] = cleanup
                emit(state, "owned_group_terminated", step=name, **cleanup)
                break
            elapsed = time.monotonic() - started
            if not last_heartbeat or elapsed - last_heartbeat >= 30:
                emit(state, "heartbeat", step=name, pid=owned.process.pid, pgid=owned.pgid,
                      elapsed_seconds=round(elapsed), resources=resources, process_usage=usage)
                last_heartbeat = elapsed or 1e-9
            try:
                owned.process.wait(timeout=POLL_SECONDS)
            except subprocess.TimeoutExpired:
                pass
        # A wrapper may exit while an unexpected descendant continues. Such a
        # step is failed and its owned group is drained before another starts.
        remaining = owned_usage(owned)["members"]
        if remaining:
            stop_reasons.append({"metric": "lingering_owned_descendants", "pids": [p["pid"] for p in remaining]})
            info.update(status="failed", resource_limit_reasons=stop_reasons)
            state["status"] = "failed"
            emit(state, "owned_descendants_after_exit", step=name, pgid=owned.pgid, members=remaining)
            info["cleanup"] = terminate_owned(owned)
            emit(state, "owned_group_terminated", step=name, **info["cleanup"])
        return stop_reasons
    except BaseException as error:
        info["status"] = "failed"
        state["status"] = "failed"
        try:
            info["cleanup"] = terminate_owned(owned)
        except Exception as cleanup_error:
            info["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
        emit(state, "step_interrupted", step=name, pgid=owned.pgid, error=f"{type(error).__name__}: {error}",
              cleanup=info.get("cleanup"), cleanup_error=info.get("cleanup_error"))
        raise


def stop_runner(signum, frame):
    raise InterruptedError(f"runner received signal {signum}; stopping its owned step")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--through", choices=[s["name"] for s in STEPS])
    parser.add_argument("--min-free", type=int, default=35)
    parser.add_argument("--bfs-threads", type=int, default=8, choices=range(1, (os.cpu_count() or 1) + 1),
                        help="Numba workers; DuckDB stays at four and BLAS/OpenMP libraries at one")
    args = parser.parse_args()
    state_path = RUN / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"network": "min5", "started": now(), "steps": {}}
    if args.status:
        print(json.dumps(state, indent=2))
        return
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "logs").mkdir(exist_ok=True)
    lock = (RUN / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()))
    lock.flush()
    signal.signal(signal.SIGTERM, stop_runner)
    env = {**os.environ, "NET": "min5", "PYTHONUNBUFFERED": "1", "TZ": "America/Los_Angeles",
           "NUMBA_NUM_THREADS": str(args.bfs_threads), "MAPADEMIC_THREADS": "4",
           **{k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}
    state.update(status="running", pid=os.getpid(), python=sys.executable,
                 threads=args.bfs_threads, parallelism={"numba": args.bfs_threads, "duckdb": 4, "blas_openmp": 1},
                 resource_watchdog={"poll_seconds": POLL_SECONDS, "footprint_limit_bytes": FOOTPRINT_LIMIT,
                    "minimum_free_percent": FREE_PERCENT_LIMIT, "swap_absolute_limit_mb": SWAP_LIMIT_MB,
                    "swap_growth_limit_mb": SWAP_GROWTH_LIMIT_MB, "accounting": "macOS libproc physical footprint, not RSS"},
                 population="at least 5 quality-filtered works; coauthor edges from teams of at most 50")
    for name, outputs in EARLIER.items():
        if not valid(outputs):
            raise RuntimeError(f"Missing prerequisite outputs: {name}: {outputs}")
        state["steps"][name] = {"status": "verified prior completion", "outputs": outputs}
    event(state, "resume", through=args.through)
    for s in STEPS:
        name = s["name"]
        old = state["steps"].get(name, {})
        marker = CACHE / f".done_{name}"
        if (old.get("status") == "done" or marker.exists()) and valid(s["outputs"]):
            state["steps"][name] = {**old, "status": "done", "outputs": s["outputs"]}
            event(state, "skip_verified", step=name)
        else:
            if marker.exists():
                (RUN / "invalid_markers").mkdir(exist_ok=True)
                marker.replace(RUN / "invalid_markers" / f"{name}_{time.time_ns()}")
                event(state, "invalid_marker", step=name, reason="required output absent or invalid")
            state["current"] = name
            state["steps"][name] = {**s, "status": "waiting for memory"}
            while True:
                resources = memory()
                if resources["free_percent"] >= args.min_free:
                    break
                event(state, "memory_wait", step=name, resources=resources)
                time.sleep(30)
            if resources.get("disk_free_gb", 100) < 15:
                raise RuntimeError("Less than 15GB disk space remains")
            started = time.monotonic()
            started_ns = time.time_ns()
            log_path = RUN / "logs" / f"{name}_{datetime.now(TZ):%Y%m%dT%H%M%S}.log"
            cmd = ["/usr/bin/time", "-l", sys.executable, *s["command"]]
            info = state["steps"][name]
            info.update(status="running", started=now(), log=str(log_path.relative_to(ROOT)),
                        script_sha256=hashlib.sha256((ANALYSIS / s["command"][0]).read_bytes()).hexdigest(),
                        shared_sha256={f: hashlib.sha256((ANALYSIS / f).read_bytes()).hexdigest()
                                       for f in ("common.py", "kernels.py", "center_cache.py", "run_resources.py",
                                                 "repair_inputs.py", "ties_cache.py")})
            event(state, "start", step=name, command=cmd, resources=resources,
                  script_sha256=info["script_sha256"], shared_sha256=info["shared_sha256"],
                  resource_watchdog=state["resource_watchdog"], parallelism=state["parallelism"])
            with log_path.open("w") as log:
                try:
                    owned = launch_owned(cmd, cwd=ANALYSIS, env=env, stdout=log, stderr=subprocess.STDOUT)
                except Exception as error:
                    info["status"] = "failed"
                    state["status"] = "failed"
                    event(state, "resource_limit", step=name, reasons=[{"metric": "owned_group_launch_or_monitor",
                          "error": f"{type(error).__name__}: {error}"}], baseline=resources)
                    raise
                p = owned.process
                info.update(pid=p.pid, pgid=owned.pgid, leader_start_abstime=owned.leader_start)
                stop_reasons = monitor_step(owned, state, name, started, resources)
            info.update(exit_code=p.returncode, elapsed_seconds=round(time.monotonic() - started, 1), finished=now())
            fresh = all((ROOT / rel).exists() and (ROOT / rel).stat().st_mtime_ns >= started_ns for rel in s["outputs"])
            if stop_reasons or p.returncode or not valid(s["outputs"]) or not fresh:
                info["status"] = "failed"
                state["status"] = "failed"
                event(state, "failure", step=name, exit_code=p.returncode, outputs_valid=valid(s["outputs"]), fresh=fresh,
                      resource_limit_reasons=stop_reasons)
                raise RuntimeError(f"{name} failed; inspect {log_path}")
            info["status"] = "done"
            info["receipts"] = [{"path": rel, "bytes": (ROOT / rel).stat().st_size,
                                  "mtime_ns": (ROOT / rel).stat().st_mtime_ns} for rel in s["outputs"]]
            marker.touch()
            shutil.copy2(log_path, RESULTS / f"{name}.log")
            event(state, "done", step=name, elapsed_seconds=info["elapsed_seconds"])
        if args.through == name:
            state.update(status="stopped at requested boundary", current=None)
            event(state, "boundary", step=name)
            return
    state.update(status="analyses and figures complete; manuscript review pending", current=None)
    event(state, "pipeline_complete")


if __name__ == "__main__":
    main()
