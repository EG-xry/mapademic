"""Analyses for the 8 November 2026 Scientometrics panel, run serially under the same owned-process watchdog as
run_min5.py, with their own state, events and logs in paper/runs/review-2026-11-08/. Earlier runs' records are not
modified.

  rv_ties            is B2c's overlap gradient built into the projection? overlap for two-author links and the
                     residual overlap without the shared works' own coauthors (same two million links as B2c)
  rv_remoteness      each field's remoteness at matched degree; a degree-only model of the field distances; each
                     field's cohesion residual d_ii - 2 a_i
  rv_bipartite_null  transitivity and local clustering of a random bipartite graph with the same works and team
                     sizes, checked against exact projections of shuffled 1% samples
  rv_rich_club       the share of the rich club's links that rest only on works of more than 10 authors, and the rich
                     club of the subgraph of links with a shared work of at most 10 authors
  rv_hub_links       removing as many distinct links at random as the top 1% and 5% touch, from A9's sources

Usage: python paper/analysis/run_review.py [--status] [--through STEP]
"""
import run_min5 as R

R.RUN = R.ROOT / "paper/runs/review-2026-11-08"
R.EARLIER = {}
step = R.step
R.STEPS = [
    step("rv_ties", "rv_ties.py", "rv_ties"),
    step("rv_remoteness", "rv_remoteness.py", "rv_remoteness"),
    step("rv_bipartite_null", "rv_bipartite_null.py", "rv_bipartite_null"),
    step("rv_rich_club", "rv_rich_club.py", "rv_rich_club"),
    step("rv_hub_links", "rv_hub_links.py", "rv_hub_links"),
]

if __name__ == "__main__":
    R.main()
