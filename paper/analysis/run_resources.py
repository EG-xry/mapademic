"""Watch only a step launched into its own process group/session.

macOS ABI verified against the local MacOSX SDK's sys/resource.h and libproc.h:
proc_pid_rusage(RUSAGE_INFO_V4) reports ri_phys_footprint separately from RSS,
including the task's compressed-memory accounting. RSS is diagnostic only.
proc_listpids(PROC_PGRP_ONLY) enumerates only the owned group. A missing native
measurement for a still-live member is a monitor failure, never a zero estimate.

Limits: 22 GiB current group footprint or one member's lifetime footprint peak;
15% system free memory; or swap above 18 GiB and 6 GiB above this step's baseline.
Retained swap from earlier work alone does not prevent a recovered machine from
running. Signals target only a group created by launch_owned; unrelated groups
are never signalled. Normal descendants inherit this session/group.
"""
import ctypes
from dataclasses import dataclass
import errno
import os
import signal
import subprocess
import sys
import time

GIB = 1 << 30
FOOTPRINT_LIMIT = 22 * GIB
FREE_PERCENT_LIMIT = 15
SWAP_LIMIT_MB = 18 * 1024
SWAP_GROWTH_LIMIT_MB = 6 * 1024
POLL_SECONDS = 1.0

# Exact rusage_info_v4 layout in the local SDK: 16-byte UUID then 35 uint64 fields.
_RUSAGE_FIELDS = """ri_user_time ri_system_time ri_pkg_idle_wkups ri_interrupt_wkups ri_pageins
ri_wired_size ri_resident_size ri_phys_footprint ri_proc_start_abstime ri_proc_exit_abstime
ri_child_user_time ri_child_system_time ri_child_pkg_idle_wkups ri_child_interrupt_wkups
ri_child_pageins ri_child_elapsed_abstime ri_diskio_bytesread ri_diskio_byteswritten
ri_cpu_time_qos_default ri_cpu_time_qos_maintenance ri_cpu_time_qos_background ri_cpu_time_qos_utility
ri_cpu_time_qos_legacy ri_cpu_time_qos_user_initiated ri_cpu_time_qos_user_interactive
ri_billed_system_time ri_serviced_system_time ri_logical_writes ri_lifetime_max_phys_footprint
ri_instructions ri_cycles ri_billed_energy ri_serviced_energy ri_interval_max_phys_footprint ri_runnable_time""".split()


class RusageInfoV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in _RUSAGE_FIELDS]


_LIBPROC = None


def libproc():
    global _LIBPROC
    if sys.platform != "darwin":
        raise RuntimeError("this watchdog requires macOS libproc footprint accounting")
    if _LIBPROC is None:
        library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        # The C header uses rusage_info_t* (typedef void*); the buffer address
        # is the address of the complete output struct, not a pointer allocation.
        library.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        library.proc_pid_rusage.restype = ctypes.c_int
        library.proc_listpids.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int]
        library.proc_listpids.restype = ctypes.c_int
        _LIBPROC = library
    return _LIBPROC


def pid_usage(pid):
    data = RusageInfoV4()
    if libproc().proc_pid_rusage(int(pid), 4, ctypes.byref(data)) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(pid))
    return {"pid": int(pid), "resident_bytes": int(data.ri_resident_size),
            "phys_footprint_bytes": int(data.ri_phys_footprint),
            "peak_phys_footprint_bytes": int(data.ri_lifetime_max_phys_footprint),
            "start_abstime": int(data.ri_proc_start_abstime), "exited": bool(data.ri_proc_exit_abstime)}


def _native_group_pids(pgid):
    library = libproc()
    needed = library.proc_listpids(2, int(pgid), None, 0)  # PROC_PGRP_ONLY
    if needed < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    capacity = max(32, needed // ctypes.sizeof(ctypes.c_int) + 32)
    for _ in range(4):
        array = (ctypes.c_int * capacity)()
        written = library.proc_listpids(2, int(pgid), array, ctypes.sizeof(array))
        if written < 0 or written % ctypes.sizeof(ctypes.c_int):
            raise RuntimeError("invalid proc_listpids byte count")
        if written < ctypes.sizeof(array):
            return sorted({int(pid) for pid in array[:written // ctypes.sizeof(ctypes.c_int)] if pid > 0})
        capacity *= 2
    raise RuntimeError("owned process group outgrew monitoring buffer")


def group_pids(pgid):
    try:
        return _native_group_pids(pgid)
    except (OSError, RuntimeError, AttributeError):
        # An accounting/enumeration failure must not prevent cleanup of our own
        # created group. ps supplies only PID/group IDs; membership is rechecked
        # with getpgid/getsid immediately before any signal.
        output = subprocess.check_output(["ps", "-axo", "pid=,pgid="], text=True, timeout=2)
        return sorted(int(parts[0]) for line in output.splitlines()
                      if len(parts := line.split()) == 2 and int(parts[1]) == pgid)


@dataclass
class OwnedProcess:
    process: subprocess.Popen
    pgid: int
    leader_start: int | None


def launch_owned(command, **kwargs):
    if "start_new_session" in kwargs:
        raise ValueError("launch_owned controls process session creation")
    process = subprocess.Popen(command, start_new_session=True, **kwargs)
    try:
        usage = pid_usage(process.pid)
        return OwnedProcess(process, process.pid, usage["start_abstime"])
    except OSError as error:
        # A command can finish before the first query. The group still belongs
        # to this Popen creation; monitor/cleanup checks all surviving members.
        if error.errno not in (errno.ESRCH, errno.ENOENT):
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
            raise
        return OwnedProcess(process, process.pid, None)
    except Exception:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
        raise


def owned_usage(owned):
    if owned.pgid <= 1 or owned.pgid != owned.process.pid or owned.pgid == os.getpgrp():
        raise RuntimeError("refusing a group not created for this step")
    members = []
    for pid in group_pids(owned.pgid):
        try:
            if os.getpgid(pid) != owned.pgid or os.getsid(pid) != owned.pgid:
                raise RuntimeError("owned process group/session identity changed")
            item = pid_usage(pid)
        except OSError as error:
            if error.errno in (errno.ESRCH, errno.ENOENT):
                continue
            raise
        if pid == owned.pgid and owned.leader_start is not None and item["start_abstime"] != owned.leader_start:
            raise RuntimeError("step leader PID was reused; refusing to signal it")
        if not item["exited"]:
            members.append(item)
    return {"pgid": owned.pgid, "members": members,
            "group_phys_footprint_bytes": sum(p["phys_footprint_bytes"] for p in members),
            "group_resident_bytes": sum(p["resident_bytes"] for p in members),
            "max_member_peak_phys_footprint_bytes": max((p["peak_phys_footprint_bytes"] for p in members), default=0)}


def limit_reasons(resources, baseline, usage):
    """Pure decision function; a high but unchanged swap baseline is allowed."""
    reasons = []
    if usage.get("monitor_error"):
        reasons.append({"metric": "footprint_monitor", "error": usage["monitor_error"]})
    for metric in ("group_phys_footprint_bytes", "max_member_peak_phys_footprint_bytes"):
        if usage.get(metric, 0) > FOOTPRINT_LIMIT:
            reasons.append({"metric": metric, "value": usage[metric], "limit": FOOTPRINT_LIMIT})
    if resources["free_percent"] < FREE_PERCENT_LIMIT:
        reasons.append({"metric": "free_percent", "value": resources["free_percent"], "minimum": FREE_PERCENT_LIMIT})
    swap = resources.get("swap_used_mb", 0)
    growth = swap - baseline.get("swap_used_mb", swap)
    if swap > SWAP_LIMIT_MB and growth > SWAP_GROWTH_LIMIT_MB:
        reasons.append({"metric": "swap_growth", "value_mb": swap, "growth_mb": growth,
                        "absolute_limit_mb": SWAP_LIMIT_MB, "growth_limit_mb": SWAP_GROWTH_LIMIT_MB})
    return reasons


def terminate_owned(owned, grace_seconds=0.5):
    """TERM then KILL only this created session/group; reap its direct child.

    If native accounting fails, OS group/session identity checks still authorize
    signalling our known process group. Never use process-name matching or a PID
    recovered from a previous run's state file.
    """
    if owned.pgid <= 1 or owned.pgid != owned.process.pid or owned.pgid == os.getpgrp():
        raise RuntimeError("refusing to terminate an unowned process group")
    try:
        if os.getpgid(owned.process.pid) != owned.pgid or os.getsid(owned.process.pid) != owned.pgid:
            raise RuntimeError("step leader no longer belongs to its created session")
    except ProcessLookupError:
        pass  # leader may have exited, leaving owned descendants in the session
    signalled = []
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            # Validate every remaining member before sending a group signal.
            # pid_usage can fail precisely when the watchdog needs to stop; the
            # group/session checks themselves do not depend on that accounting.
            members = group_pids(owned.pgid)
            if not members:
                break
            for pid in members:
                try:
                    if os.getpgid(pid) != owned.pgid or os.getsid(pid) != owned.pgid:
                        raise RuntimeError("refusing to signal a different process session")
                    if pid == owned.pgid and owned.leader_start is not None:
                        try:
                            current = pid_usage(pid)
                        except OSError:
                            current = None
                        if current is not None and current["start_abstime"] != owned.leader_start:
                            raise RuntimeError("refusing to signal a reused step leader PID")
                except ProcessLookupError:
                    continue
            os.killpg(owned.pgid, sig)
            signalled.append(sig.name)
        except ProcessLookupError:
            break
        except PermissionError:
            # macOS can report EPERM for a group that became empty/zombie-only
            # between enumeration and killpg. Never suppress it for a live task.
            if owned_usage(owned)["members"]:
                raise
            break
        deadline = time.monotonic() + (grace_seconds if sig == signal.SIGTERM else 2.0)
        while time.monotonic() < deadline:
            owned.process.poll()  # reap the direct child promptly
            if not group_pids(owned.pgid):
                break
            time.sleep(0.05)
        if not group_pids(owned.pgid):
            break
    try:
        owned.process.wait(timeout=3)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("owned step did not exit after group termination") from error
    remaining = owned_usage(owned)["members"]
    if remaining:
        raise RuntimeError(f"owned descendants remain after termination: {[p['pid'] for p in remaining]}")
    return {"pgid": owned.pgid, "signals": signalled, "returncode": owned.process.returncode, "live_descendants": 0}
