"""Analyses for the 8 November 2026 review comments, run serially under the same owned-process watchdog as
run_min5.py, with their own state, events and logs in paper/runs/comments-2026-11-08/. Earlier runs' records are not
modified.

  cm_a9                 A9 again, now with standard errors and the per-source sums, and the hub-removal detail
                        (surviving sources, their unmodified baseline, reach, four more random draws)
  cm_a9c                A9c again, with the fit's residual matrix and summary
  cm_a9e                which field and country differences exceed sampling error, and the arts-immunology test
  cm_leiden_benchmark   leiden.py against igraph's community_leiden on planted-community networks
  cm_leiden_passes      on the hardest kind of those networks, our modularity after 1-40 passes against igraph
  cm_b3b                betweenness rankings: the pooled estimate against the smaller of the two
  cm_a3c                the links at the far edge: shared works and flags
  cm_a5c                who is in the mixed Chinese/Japanese-script community

Usage: python paper/analysis/run_comments.py [--status] [--through STEP]
"""
import run_min5 as R

R.RUN = R.ROOT / "paper/runs/comments-2026-11-08"
R.EARLIER = {}
step = R.step
R.STEPS = [
    step("cm_a9", "a9_structure.py", "a9_structure", extra=["paper/cache/a9_per_source.npz"]),
    step("cm_a9c", "a9c_field_order.py", "a9c_field_order"),
    step("cm_a9e", "a9e_field_tests.py", "a9e_field_tests"),
    step("cm_leiden_benchmark", "leiden_benchmark.py", "leiden_benchmark"),
    step("cm_leiden_passes", "leiden_passes.py", "leiden_passes"),
    step("cm_b3b", "b3b_betweenness_pooled.py", "b3b_betweenness_pooled"),
    step("cm_a3c", "a3c_far_edge_links.py", "a3c_far_edge_links"),
    step("cm_a5c", "a5c_mixed_script.py", "a5c_mixed_script"),
]

if __name__ == "__main__":
    R.main()
