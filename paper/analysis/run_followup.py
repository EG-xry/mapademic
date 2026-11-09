"""Follow-up analyses after Eric's review of the five-work results (3 November 2026), run serially under the
same owned-process watchdog as run_min5.py, with their own state, events and logs in
paper/runs/followup-2026-11-03/. The completed five-work run's records are not modified.

  fu_leiden15          leiden.py on the 15-work network (threshold versus implementation)
  fu_leiden15_compare  its communities against the map's GPU partition and the five-work partition
  fu_b2c               weak ties by number of shared works and team size
  fu_a9d               distances between the 13 arts-and-humanities subfields and to immunology
  fu_name_records      name-record merges among the 50,000 highest-degree profiles (live API)
  fu_repair_mask       the stricter repair-link rule (repair_link_mask.npy)
  fu_a7 ... fu_a9b     the repair experiments rerun with the revised link and same-name rules
  fu_export ... fu_threshold_comparison   deposit, checks, figures and comparison tables

Usage: python paper/analysis/run_followup.py [--status] [--through STEP]
"""
import run_min5 as R

R.RUN = R.ROOT / "paper/runs/followup-2026-11-03"
R.EARLIER = {}
step = R.step
R.STEPS = [
    step("fu_leiden15", "leiden_min15.py", extra=["paper/results/leiden_min15_cpu.json", "paper/cache_min15/leiden_cpu.npy"]),
    step("fu_leiden15_compare", "leiden_min15_compare.py", "leiden_min15_compare"),
    step("fu_b2c", "b2c_repeat_ties.py", "b2c_repeat_ties"),
    step("fu_a9d", "a9d_arts_subfields.py", "a9d_arts_subfields"),
    step("fu_name_records", "a6_name_records.py", "a6_name_records", extra=["paper/cache/a6_name_records.csv"]),
    step("fu_repair_mask", "repair_mask.py", "repair_mask", extra=["paper/cache/repair_link_mask.npy"]),
    step("fu_a7", "a7_remedy.py", "a7_remedy_peak25_strict", ["25", "strict"], ["paper/cache/a7_sets_peak25_strict.npz"]),
    step("fu_a7b", "a7b_paired.py", "a7b_paired_peak25_strict", ["peak25_strict"]),
    step("fu_a8", "a8_repair.py", "a8_repair", extra=["paper/cache/repair_rep.npy"]),
    step("fu_a8b", "a8b_communities.py", "a8b_communities"),
    step("fu_a8c", "a8c_center.py", "a8c_center"),
    step("fu_a9b", "a9b_repaired.py", "a9b_repaired"),
    step("fu_export", "export_deposit.py", extra=["data/2026-09-23/min5/deposit/SHA256SUMS"]),
    step("fu_a10", "a10_checks.py", "a10_checks"),
    step("fu_figures", "figures.py", extra=[f"paper/figures/fig_{s}.pdf" for s in ("distances", "national_heatmap")]),
    step("fu_threshold_comparison", "threshold_comparison.py", "threshold_comparison", ["--final"]),
]

if __name__ == "__main__":
    R.main()
