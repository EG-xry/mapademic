"""Final-validation orchestration fixtures; no production graph or validator is executed."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_final_validation as final
import run_min5 as runner
import run_resources as resources


def report(**changes):
    return {"network":"min5","status":"passed","quick":False,"reconstruction_requested":True,
            "manuscript_required":True,"delivery_fingerprint":{"sha256":"f"*64},
            "checks":[{"name":name,"status":"pass"} for name in sorted(final.REQUIRED_FINAL_CHECKS)],**changes}


def fixture(root, script=None):
    run=root/"paper/runs/fixture"
    run.mkdir(parents=True)
    (root/"paper/results").mkdir()
    analysis=root/"paper/analysis"
    analysis.mkdir()
    for name in final.CODE_FILES:
        (analysis/name).write_text("# fixture\n")
    if script is None:
        script=("import json,sys\nfrom pathlib import Path\n"
                "assert sys.argv[1:] == ['--require-manuscript']\n"
                f"Path('../results/validation_min5.json').write_text(json.dumps({report()!r}))\n")
    (analysis/"validate_min5.py").write_text(script)
    state={"network":"min5","status":"analyses and figures complete; manuscript review pending","current":None,
           "steps":{step["name"]:{"status":"done"} for step in runner.STEPS}}
    (run/"state.json").write_text(json.dumps(state))
    (run/"events.jsonl").write_text("parent-event-sentinel\n")
    (run/"PROGRESS.md").write_text("parent-progress-sentinel\n")
    (run/"runner.lock").write_text("prior-run-owner\n")
    return run


def parent_bytes(run):
    return {name:(run/name).read_bytes() for name in ("state.json","events.jsonl","PROGRESS.md","runner.lock")}


class FinalValidationReportTests(unittest.TestCase):
    def test_full_report_requires_fresh_all_passing_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"validation.json"
            started=time.time_ns()-1_000_000_000
            path.write_text(json.dumps(report()))
            raw,receipt=final.fresh_full_report(path,started)
            self.assertEqual(receipt["passed_checks"],len(final.REQUIRED_FINAL_CHECKS))
            self.assertEqual(json.loads(raw)["status"],"passed")
            for change in ({"quick":True},{"reconstruction_requested":False},{"manuscript_required":False},
                           {"network":"min15"},{"status":"failed"},{"checks":[]},{"delivery_fingerprint":None}):
                with self.subTest(change=change):
                    path.write_text(json.dumps(report(**change)))
                    with self.assertRaises(ValueError):final.fresh_full_report(path,started)
            path.write_text(json.dumps(report()))
            os.utime(path,ns=(1,1))
            with self.assertRaisesRegex(ValueError,"predates"):
                final.fresh_full_report(path,started)

    def test_lock_content_and_parent_state_survive_refused_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=fixture(root);before=parent_bytes(run)
            with (run/"runner.lock").open("r+") as live:
                fcntl.flock(live,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with mock.patch.object(final,"launch_owned",side_effect=AssertionError("must not launch")):
                    with self.assertRaisesRegex(RuntimeError,"holds runner.lock"):
                        final.run_validation(root,run)
                self.assertEqual(parent_bytes(run),before)
                fcntl.flock(live,fcntl.LOCK_UN)
            self.assertFalse((run/"final_validation").exists())

    def test_incomplete_parent_records_failure_without_launch_or_parent_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=fixture(root)
            state=json.loads((run/"state.json").read_text());state["steps"][runner.STEPS[-1]["name"]]["status"]="pending"
            (run/"state.json").write_text(json.dumps(state));before=parent_bytes(run)
            with mock.patch.object(final,"launch_owned",side_effect=AssertionError("must not launch")):
                with self.assertRaisesRegex(ValueError,"Finish the main"):
                    final.run_validation(root,run)
            self.assertEqual(parent_bytes(run),before)
            receipts=list((run/"final_validation").glob("*/receipt.json"))
            self.assertEqual(len(receipts),1)
            self.assertEqual(json.loads(receipts[0].read_text())["status"],"failed")


@unittest.skipUnless(sys.platform=="darwin","owned-process watchdog uses macOS libproc")
class FinalValidationOwnedTests(unittest.TestCase):
    baseline={"free_percent":80,"swap_used_mb":20*1024,"disk_free_gb":50}

    def run_fixture(self,root,run):
        with mock.patch.object(runner,"memory",return_value=self.baseline), \
                mock.patch.object(runner,"event",side_effect=AssertionError("parent event sink used")), \
                mock.patch.object(runner,"POLL_SECONDS",.01):
            return final.run_validation(root,run)

    def test_passed_receipt_custom_sink_and_archived_report_leave_parent_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=fixture(root);before=parent_bytes(run)
            attempt=self.run_fixture(root,run)
            receipt=json.loads((attempt/"receipt.json").read_text())
            self.assertEqual(receipt["status"],"passed")
            info=receipt["steps"][final.STEP]
            self.assertEqual(info["exit_code"],0)
            self.assertEqual(info["resource_baseline"],self.baseline)
            self.assertIn("group_phys_footprint_bytes",info["resource_peaks"])
            self.assertEqual(parent_bytes(run),before)
            self.assertEqual((attempt/"validation_min5.json").read_bytes(),(root/"paper/results/validation_min5.json").read_bytes())
            command=json.loads((attempt/"command.json").read_text())["command"]
            self.assertEqual(command[-2:],["validate_min5.py","--require-manuscript"])
            with final.runner_lock(run):pass  # success releases the lock

    def test_zero_exit_with_stale_report_and_nonzero_exit_are_not_success(self):
        scripts=["# zero exit without replacing the old report\n", "raise SystemExit(3)\n"]
        for script in scripts:
            with self.subTest(script=script),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);run=fixture(root,script);before=parent_bytes(run)
                old=root/"paper/results/validation_min5.json";old.write_text(json.dumps(report()));os.utime(old,ns=(1,1))
                with self.assertRaises((ValueError,RuntimeError)):
                    self.run_fixture(root,run)
                attempt=next((run/"final_validation").iterdir())
                receipt=json.loads((attempt/"receipt.json").read_text())
                self.assertEqual(receipt["status"],"failed")
                self.assertEqual(receipt["steps"][final.STEP]["cleanup"]["live_descendants"],0)
                self.assertFalse((attempt/"validation_min5.json").exists())
                self.assertEqual(parent_bytes(run),before)

    def test_interruption_drains_only_newly_owned_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=fixture(root,"import time\ntime.sleep(60)\n");before=parent_bytes(run)
            unrelated=resources.launch_owned([sys.executable,"-c","import time;time.sleep(60)"],stdout=subprocess.DEVNULL)
            launched=[]
            real_launch=final.launch_owned
            def capture(*args,**kwargs):
                owned=real_launch(*args,**kwargs);launched.append(owned);return owned
            try:
                with mock.patch.object(final,"launch_owned",side_effect=capture), \
                        mock.patch.object(runner,"monitor_step",side_effect=InterruptedError("fixture interrupt")):
                    with self.assertRaisesRegex(InterruptedError,"fixture interrupt"):
                        self.run_fixture(root,run)
                self.assertEqual(len(launched),1)
                self.assertEqual(resources.owned_usage(launched[0])["members"],[])
                self.assertIsNone(unrelated.process.poll())
                self.assertEqual(parent_bytes(run),before)
                with final.runner_lock(run):pass
            finally:
                resources.terminate_owned(unrelated,grace_seconds=.05)

    def test_limit_uses_custom_sink_and_leaves_parent_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=fixture(root,"import time\ntime.sleep(60)\n");before=parent_bytes(run)
            usage=resources.owned_usage
            def over_limit(owned):
                current=usage(owned)
                if current["members"]:current["group_phys_footprint_bytes"]=resources.FOOTPRINT_LIMIT+1
                return current
            with mock.patch.object(runner,"owned_usage",side_effect=over_limit):
                with self.assertRaisesRegex(RuntimeError,"Full validation failed"):
                    self.run_fixture(root,run)
            attempt=next((run/"final_validation").iterdir())
            events=[json.loads(line)["event"] for line in (attempt/"events.jsonl").read_text().splitlines()]
            self.assertIn("resource_limit",events)
            self.assertIn("owned_group_terminated",events)
            self.assertEqual(parent_bytes(run),before)
            self.assertEqual(json.loads((attempt/"receipt.json").read_text())["status"],"failed")


if __name__=="__main__":
    unittest.main()
