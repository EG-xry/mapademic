"""Small watchdog tests; only newly created synthetic process groups are signalled."""
import ctypes
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_resources as resources
import run_min5 as runner


class ResourcePolicyTests(unittest.TestCase):
    def test_large_retained_swap_alone_is_safe(self):
        baseline = {"free_percent": 73, "swap_used_mb": 23 * 1024}
        current = {"free_percent": 65, "swap_used_mb": 24 * 1024}
        self.assertEqual(resources.limit_reasons(current, baseline, {}), [])
        current["swap_used_mb"] = 30 * 1024
        self.assertEqual(resources.limit_reasons(current, baseline, {})[0]["metric"], "swap_growth")

    def test_footprint_not_rss_drives_limit_and_peak_catches_short_spike(self):
        base = {"free_percent": 36, "swap_used_mb": 12 * 1024}
        usage = {"group_resident_bytes": 9 * resources.GIB, "group_phys_footprint_bytes": 59 * resources.GIB}
        self.assertEqual(resources.limit_reasons(base, base, usage)[0]["metric"], "group_phys_footprint_bytes")
        usage = {"group_phys_footprint_bytes": 10 * resources.GIB,
                 "max_member_peak_phys_footprint_bytes": 24 * resources.GIB}
        self.assertEqual(resources.limit_reasons(base, base, usage)[0]["metric"], "max_member_peak_phys_footprint_bytes")

    def test_boundaries_and_monitor_failure(self):
        base = {"free_percent": 70, "swap_used_mb": 12 * 1024}
        self.assertEqual(resources.limit_reasons({"free_percent": 15, "swap_used_mb": 18 * 1024}, base,
                                               {"group_phys_footprint_bytes": 22 * resources.GIB}), [])
        self.assertEqual(resources.limit_reasons({"free_percent": 14, "swap_used_mb": 12 * 1024}, base, {})[0]["metric"], "free_percent")
        self.assertEqual(resources.limit_reasons({"free_percent": 70, "swap_used_mb": 18.1 * 1024}, base, {})[0]["metric"], "swap_growth")
        self.assertEqual(resources.limit_reasons(base, base, {"monitor_error": "cannot account owned PID"})[0]["metric"], "footprint_monitor")

    def test_current_group_can_never_be_terminated(self):
        fake = resources.OwnedProcess(SimpleNamespace(pid=os.getpgrp()), os.getpgrp(), None)
        with mock.patch.object(resources.os, "killpg", side_effect=AssertionError("must not signal")):
            with self.assertRaisesRegex(RuntimeError, "unowned"):
                resources.terminate_owned(fake)


@unittest.skipUnless(sys.platform == "darwin", "local libproc ABI tests require macOS")
class NativeOwnedProcessTests(unittest.TestCase):
    def test_sdk_layout_and_live_self_accounting(self):
        # sizeof/offsetof independently compiled from the local macOS SDK header.
        self.assertEqual(ctypes.sizeof(resources.RusageInfoV4), 296)
        self.assertEqual(resources.RusageInfoV4.ri_phys_footprint.offset, 72)
        self.assertEqual(resources.RusageInfoV4.ri_resident_size.offset, 64)
        self.assertEqual(resources.RusageInfoV4.ri_lifetime_max_phys_footprint.offset, 240)
        current = resources.pid_usage(os.getpid())
        self.assertGreater(current["phys_footprint_bytes"], 0)
        self.assertGreaterEqual(current["peak_phys_footprint_bytes"], current["phys_footprint_bytes"])
        self.assertFalse(current["exited"])

    def test_group_cleanup_kills_descendant_but_leaves_other_test_group(self):
        grandchild = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
        script = ("import signal,subprocess,sys,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                  f"child=subprocess.Popen([sys.executable,'-c',{grandchild!r}]); print(child.pid,flush=True); time.sleep(60)")
        owned = resources.launch_owned([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
        separate = resources.launch_owned([sys.executable, "-c", "import time;time.sleep(60)"], stdout=subprocess.DEVNULL)
        try:
            child_pid = int(owned.process.stdout.readline())
            usage = resources.owned_usage(owned)
            self.assertTrue({owned.process.pid, child_pid}.issubset({item["pid"] for item in usage["members"]}))
            with mock.patch.object(resources, "_native_group_pids", side_effect=RuntimeError("native enumeration unavailable")):
                self.assertIn(child_pid, resources.group_pids(owned.pgid))
            cleanup = resources.terminate_owned(owned, grace_seconds=.1)
            self.assertIn("SIGKILL", cleanup["signals"])
            self.assertEqual(cleanup["live_descendants"], 0)
            self.assertIsNone(separate.process.poll())
        finally:
            resources.terminate_owned(owned, grace_seconds=.05)
            resources.terminate_owned(separate, grace_seconds=.05)
            owned.process.stdout.close()

    def test_monitor_limit_logs_failure_and_reaps_only_its_child(self):
        owned = resources.launch_owned([sys.executable, "-c", "import time;time.sleep(60)"], stdout=subprocess.DEVNULL)
        state = {"status": "running", "steps": {"fixture": {"status": "running"}}}
        events = []
        actual_usage = resources.owned_usage
        def dangerous_usage(process):
            value = actual_usage(process)
            if value["members"]:
                value["group_phys_footprint_bytes"] = resources.FOOTPRINT_LIMIT + 1
            return value
        try:
            with mock.patch.object(runner, "owned_usage", side_effect=dangerous_usage), \
                    mock.patch.object(runner, "event", side_effect=lambda state, kind, **data: events.append((kind, data))):
                reasons = runner.monitor_step(owned, state, "fixture", time.monotonic(),
                                              {"free_percent": 73, "swap_used_mb": 23 * 1024})
            self.assertTrue(reasons)
            self.assertEqual(state["status"], "failed")
            self.assertIn("resource_limit", [kind for kind, _ in events])
            self.assertIn("owned_group_terminated", [kind for kind, _ in events])
            self.assertIsNotNone(owned.process.poll())
            self.assertEqual(resources.owned_usage(owned)["members"], [])
        finally:
            resources.terminate_owned(owned, grace_seconds=.05)


if __name__ == "__main__":
    unittest.main()
