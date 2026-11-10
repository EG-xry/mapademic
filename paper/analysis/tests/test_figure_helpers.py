"""Small-input checks for the figure helpers added with the November 2026 redesign: interval estimates over sampled
sources (Fig. 2a), community names and groups (Figs. 1 and 8), the shortest-path corridor layout (Fig. 3a) and map
marker placement (Figs. 1 and 8). No graph data or map is read."""
import sys
from pathlib import Path

import numpy as np

ANALYSIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS))
import figures  # noqa: E402


def test_binned_shares_pools_sources_and_clusters_their_errors():
    # three sources; column 0 is the source itself; distances 1..3 and a tail bin at top=3 (3 or more)
    hist = np.array([[1, 2, 6, 2, 0],
                     [1, 0, 4, 4, 2],
                     [1, 0, 0, 0, 0]])                 # reaches no one: left out
    share, se = figures.binned_shares(hist, top=3)
    pooled = np.array([2, 10, 8]) / 20                    # distance 1, 2 and 3+ (2 + 0 + 4 + 2) over 20 targets
    assert np.allclose(share, pooled)
    # cluster (ratio-estimator) variance by hand for two equal-size clusters
    counts = np.array([[2, 6, 2], [0, 4, 6]], float)
    resid = counts - pooled[None, :] * 10
    assert np.allclose(se, np.sqrt(2 / 1 * (resid ** 2).sum(0)) / 20)
    assert np.isclose(share.sum(), 1)


def test_community_names_split_fields_countries_and_groups():
    assert figures.community_name("Medicine (Spain, Mexico, Argentina)") == ("Medicine", ["Spain", "Mexico", "Argentina"])
    assert figures.community_name("Physics & Astronomy") == ("Physics & Astronomy", [])
    assert figures.joined(["Spain", "Mexico", "Argentina"]) == "Spain, Mexico & Argentina"
    assert figures.joined(["Iran"]) == "Iran"
    assert [figures.field_group(n) for n in ("Multidisciplinary (China)", "Medicine (France)",
                                             "Medicine & Engineering (Japan)")] == [0, 1, 2]
    assert figures.members_text(2_263_224) == "2.26M" and figures.members_text(22_328) == "22k"


def test_corridor_layout_centres_steps_and_uncrosses_links():
    # a chain a-b, then b fans out to c1, c2, which lead to d2 and d1 respectively (crossed if left in input order)
    step = {"a": 0, "b": 1, "c1": 2, "c2": 2, "d1": 3, "d2": 3, "e": 4}
    coauthors = {"a": 1, "b": 3, "c1": 2, "c2": 2, "d1": 2, "d2": 2, "e": 2}
    links = [("a", "b"), ("b", "c1"), ("b", "c2"), ("c1", "d2"), ("c2", "d1"), ("d1", "e"), ("d2", "e")]
    y = figures.corridor_layout(step, coauthors, links)
    assert y["a"] == y["b"] == y["e"] == 0                # single researchers sit on the centre line
    assert sorted((y["c1"], y["c2"])) == [-0.5, 0.5] and sorted((y["d1"], y["d2"])) == [-0.5, 0.5]
    assert np.sign(y["c1"]) == np.sign(y["d2"]) and np.sign(y["c2"]) == np.sign(y["d1"])   # no crossing


def test_marker_spot_prefers_dense_dominant_cells_and_keeps_clear():
    own = np.zeros((10, 10))
    total = np.ones((10, 10))
    own[2, 3], total[2, 3] = 5, 50                        # dense but shared with others
    own[7, 6], total[7, 6] = 4, 4                         # slightly less dense, all its own
    x, y = figures.marker_spot(own, total, taken=[], radius=1, bin_px=1)
    assert (x, y) == (6.5, 7.5)
    x, y = figures.marker_spot(own, total, taken=[(6.5, 7.5)], radius=2, bin_px=1)
    assert (x, y) == (3.5, 2.5)                           # the best spot is taken
    x, y = figures.marker_spot(own, total, taken=[], radius=1, margin=3, bin_px=1)
    assert 3 <= x <= 7 and 3 <= y <= 7                    # the best spots lie in the margin
    box = (5.5, 6.5, 7.5, 8.5)                            # a label over the best spot
    assert figures.marker_spot(own, total, taken=[], radius=1, boxes=[box], bin_px=1) == (3.5, 2.5)


def test_field_groups_cover_the_26_fields_once():
    fields = np.array(["Medicine", "Arts and Humanities", None, "Physics and Astronomy", "Veterinary",
                       "Agricultural and Biological Sciences", "Decision Sciences"], dtype=object)
    names = list(figures.FIELD_GROUPS)
    got = [names[i] if i >= 0 else None for i in figures.field_group_index(fields)]
    assert got == ["Medicine & health", "Arts & humanities", None, "Physics & mathematics", "Medicine & health",
                   "Life sciences", "Social sciences"]


def test_script_circle_centres_on_the_dense_core():
    rng = np.random.default_rng(1)
    x = np.r_[rng.normal(200, 3, 900), rng.uniform(0, 400, 100)]   # a packed community and a thin halo
    y = np.r_[rng.normal(100, 3, 900), rng.uniform(0, 400, 100)]
    cx, cy, r = figures.script_circle(x, y, (400, 400), min_r=5, max_r=60)
    assert abs(cx - 200) <= 4 and abs(cy - 100) <= 4 and 5 <= r <= 12


def test_beside_puts_the_label_on_a_dark_clear_side():
    img = np.zeros((200, 400, 3), np.uint8)
    img[:, 250:] = 200                                    # the map is bright to the right of the circle
    tx, ty, ha, va, box = figures.beside(img, 200, 100, 10, "label\nline two", 7, px_per_mm=10)
    assert ha == "right" and box[2] <= 200                # to the left, over the dark part
    left = (0, 60, 195, 117)                               # something already to the left, above the circle's foot
    tx, ty, ha, va, box = figures.beside(img, 200, 100, 10, "label\nline two", 7, px_per_mm=10, avoid=[left])
    assert (ha, va) == ("right", "top") and box[1] > 100 and box[2] <= 210   # then below it, still on the dark side


def test_collisions_reports_overlaps_crossed_labels_and_text_off_the_figure():
    plt = figures.plt
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(1, 8, "alpha")
    ax.text(1.3, 8, "beta")                               # on top of alpha
    ax.plot([5, 9], [3, 3])
    ax.text(7, 3, "crossed", ha="center", va="center")    # the line runs through it
    ax.text(1, 1, "clear")
    fig.text(1.05, 0.5, "outside")                        # past the right edge
    fig.canvas.draw()
    found = figures.collisions(fig)
    plt.close(fig)
    assert "texts overlap: 'alpha' / 'beta'" in found
    assert "line through text: 'crossed'" in found
    assert "leaves the figure: 'outside'" in found
    assert not any("clear" in f for f in found)


def test_collisions_skips_tick_labels_that_are_not_drawn():
    plt = figures.plt
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.set_xticks([0.5], ["tick"])
    ax.text(0.5, -0.04, "under", transform=ax.transAxes, ha="center", va="top")   # where the tick label sits
    fig.canvas.draw()
    assert "texts overlap: 'tick' / 'under'" in figures.collisions(fig) or \
        "texts overlap: 'under' / 'tick'" in figures.collisions(fig)
    ax.axis("off")                                        # a map axis: its tick labels exist but are not drawn
    fig.canvas.draw()
    assert figures.collisions(fig) == []
    ax.axis("on")
    ax.set_xlim(0.6, 1)                                   # the tick now lies outside the view
    fig.canvas.draw()
    assert figures.collisions(fig) == []
    plt.close(fig)


def test_collisions_checks_insets_and_link_lines_but_not_masked_or_linked_text():
    from matplotlib.collections import LineCollection
    plt = figures.plt
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    inset = ax.inset_axes([0.4, 0.55, 0.58, 0.4])           # not in fig.axes
    inset.set_xlim(0, 1)
    inset.set_ylim(0, 1)
    inset.axis("off")
    inset.axvline(0.75, zorder=1)
    inset.text(0.75, 0.3, "in the inset", ha="center", fontsize=7)        # the inset's line runs through it
    inset.axvline(0.2, zorder=1)
    inset.text(0.2, 0.7, "masked", ha="center", fontsize=7, zorder=3,      # an opaque box above the line hides it
               bbox=dict(facecolor="white", edgecolor="none"))
    ax.add_collection(LineCollection([[(1, 1), (4, 4)], [(1, 5), (2, 5)]]))
    ax.text(2.5, 2.5, "crossed", ha="center", va="center")                 # the first link passes through
    ax.text(2, 5, "node", ha="center", va="center")                        # the second ends on the node it numbers
    ax.annotate("label", (9, 1), xytext=(6, 1), arrowprops=dict(arrowstyle="-"))
    ax.plot([8.5, 8.5], [0, 2])                                            # crosses the arrow, not the label
    fig.canvas.draw()
    found = figures.collisions(fig)
    plt.close(fig)
    assert "line through text: 'in the inset'" in found
    assert "line through text: 'crossed'" in found
    assert not any("masked" in f or "node" in f or "'label'" in f for f in found)


def test_collisions_reports_run_on_text_across_axes_and_gridlines_along_text():
    plt = figures.plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(4, 2))
    a.axis("off")
    b.axis("off")
    gap = 1 / 25.4 / 4                                    # 1 mm in figure width
    a.text(0.5, 0.6, "left words", ha="right", va="center", transform=fig.transFigure)
    fig.text(0.5 + gap, 0.6, "right words", ha="left", va="center")       # another owner: reads as one phrase
    a.text(0.5, 0.2, "same axes", ha="right", va="center", transform=fig.transFigure)
    a.text(0.5 + gap, 0.2, "no report", ha="left", va="center", transform=fig.transFigure)
    fig.canvas.draw()
    found = figures.collisions(fig)
    plt.close(fig)
    assert {"texts run together: 'left words' / 'right words'",
            "texts run together: 'right words' / 'left words'"} & set(found)
    assert not any("no report" in f for f in found)

    fig, ax = plt.subplots(figsize=(3, 2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.grid(axis="x", visible=False)
    ax.text(0.1, 0.5, "struck", va="center")              # the 0.5 gridline runs along its middle
    ax.text(0.6, 0.5, "boxed", va="center", bbox=dict(facecolor="white", edgecolor="none"))
    ax.text(0.1, 0.25, "between", va="center")
    fig.canvas.draw()
    found = figures.collisions(fig)
    plt.close(fig)
    assert "gridline through text: 'struck'" in found
    assert not any("boxed" in f or "between" in f for f in found)


def test_far_edge_labels_replace_only_low_cited_names_by_run():
    cited = [0, 1, 2, 1402, 45705, 88, 0, 2, 1]
    names = ["r0", "r1", "r2", "Hugh Roberts", "Catriona McLean", "Rehab A. Mohamed", "r6", "r7", "r8"]
    labels = figures.far_edge_labels(cited, names)
    assert labels == ["A1", "A2", "A3", "Hugh Roberts", "Catriona McLean", "Rehab A. Mohamed", "B1", "B2", "B3"]
