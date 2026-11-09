"""Figures for the paper, from results/*.json and cache/*. Writes figures/<name>.pdf and .png, and the files for
submission (figures/submission/Fig<n>.eps and .tif).

Drawn at Springer's print widths (84, 129 and 174 mm) in 8 pt Helvetica, so nothing is scaled at print. Colours: the
dataviz reference palette (slots 1-2 validated all-pairs against white; gray for a baseline series; one blue ramp for
magnitude; blue-gray-red for affinity). Hairline solid grid, white gaps between cells, direct labels where they help.
"""
import json
import sys
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib import patheffects
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator, PercentFormatter
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from common import CACHE, D, FIELD_GROUPS, FIGURES, MIN_WORKS, NET, RESULTS, ROOT, SCRIPT_COMMUNITIES

sys.path.insert(0, str(ROOT))
from pipeline.palette import community_rgb  # noqa: E402  (the colours the map tiles were drawn with)

# the map: on the map's own network its tiles and labels; on the paper's, the same layout recoloured by its
# communities (map_recolour.py, map_labels.py)
MAP = D if NET == "min15" else CACHE / "map"

MM = 1 / 25.4
W1, W15, W2 = 84 * MM, 129 * MM, 174 * MM             # Springer: single, 1.5 and double column
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2 = "#2a78d6", "#eb6834"
GOLD = "#ffd27a"                                       # annotations on the dark map
BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                                                    "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#9ec5f4", "#f0efec", "#f2a29f", "#e34948"])
# The comparison figures (Figs. 2-4, 9, 10, A1, A2) use Eric's palette of 8 November 2026: blue against teal or brown,
# and two shades of one hue for two similar groups. C holds the roles they draw with: the main and contrast hues and a
# second shade of each. Option A, navy against teal, is the default; colour_options.py redraws these figures under
# every option. Checked with the dataviz palette validator: blue #2171b5 or #4292c6 against teal is hard to tell
# apart even with full colour vision (Delta E 8.9 and 12.0), so the teal options pair teal with navy and pale blue.
NAVY, BLUE, BLUE_MID, BLUE_PALE = "#08306b", "#2171b5", "#4292c6", "#9ecae1"
TEAL, TEAL_LIGHT, BROWN, BROWN_LIGHT = "#0b7a85", "#3ba0a8", "#9e7a2b", "#c0993f"
OPTIONS = {"A": ("navy and teal", NAVY, BLUE_PALE, TEAL, TEAL_LIGHT),
           "B": ("blue and brown", BLUE, NAVY, BROWN, BROWN_LIGHT),
           "C": ("navy and brown", NAVY, BLUE_MID, BROWN, BROWN_LIGHT),
           "D": ("blue and teal", BLUE, NAVY, TEAL, TEAL_LIGHT)}
# Eric's choices of 8 November 2026, applied by the manuscript run below: blue and brown for Figs. 3, 9 and A1; navy
# and teal for Figs. 2, 4 and A2, and for Fig. 10, which he did not name (the default)
CHOSEN = {"far_edge": "B", "ties": "B", "affiliations": "B"}
C = {}


def use_colours(option="A"):
    _, C["main"], C["main2"], C["contrast"], C["contrast2"] = OPTIONS[option]


use_colours()
RAMP = LinearSegmentedColormap.from_list("ramp", ["#f7fbff", BLUE_PALE, BLUE_MID, BLUE, NAVY])
MAGNITUDE = (BLUE_PALE, BLUE_MID, NAVY)                    # three steps of one hue, light to dark


def tint(colour, alpha):
    """The colour at this opacity over white, as an opaque colour (EPS has no transparency)."""
    rgb = np.array(matplotlib.colors.to_rgb(colour))
    return tuple(1 - alpha * (1 - rgb))
NUMBER = {"fig_map": "1", "fig_distances": "2", "fig_far_edge": "3", "fig_field_distances": "4", "fig_fields": "5",
          "fig_national_heatmap": "6", "fig_countries": "7", "fig_hub_definitions": "8", "fig_ties": "9",
          "fig_uncited": "10", "fig_affiliations": "A1", "fig_twins": "A2"}
SHORT = {"Biochemistry, Genetics and Molecular Biology": "Biochemistry & Genetics",
         "Pharmacology, Toxicology and Pharmaceutics": "Pharmacology", "Economics, Econometrics and Finance": "Economics",
         "Business, Management and Accounting": "Business & Management",
         "Agricultural and Biological Sciences": "Agricultural & Biological Sci."}
matplotlib.rcParams.update({
    "font.size": 8, "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "mathtext.fontset": "custom", "mathtext.rm": "Helvetica", "mathtext.bf": "Helvetica:bold",
    "mathtext.it": "Helvetica:italic", "mathtext.sf": "Helvetica", "mathtext.cal": "Helvetica", "mathtext.tt": "Helvetica",
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.5, "ytick.minor.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.5, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "legend.frameon": False, "legend.handlelength": 1.6, "legend.borderaxespad": 0.3,
    "figure.constrained_layout.h_pad": 0.03, "figure.constrained_layout.w_pad": 0.03,
    "savefig.dpi": 300, "figure.dpi": 110, "pdf.fonttype": 42, "ps.fonttype": 42})


def short(field):
    return SHORT.get(field, field.replace(" and ", " & "))


# The map: every coloured community with at least MARKED members. Communities named for fields only are named in it;
# those the map's naming rule names for countries are numbered (key order: their main fields, then size), and marked
# regional when no country holds half of their members with a known country (Section 4's rule for national).
MARKED = 20_000
FIELD_NAMES = {"Medicine": "Clinical medicine", "Social Sciences & Humanities": "Social sciences\n& humanities",
               "Environmental Science, Biology, Agriculture & Earth Sciences": "Environmental, agricultural\n& earth sciences",
               "Engineering & Materials Science": "Engineering &\nmaterials science",
               "Computer Science & Engineering": "Computer science\n& engineering",
               "Physics & Astronomy": "Physics &\nastronomy"}
MARKER_SIZE, MARKER_FILL = 100, "#111111"                 # points squared: a 10 pt disc holds two 7 pt digits
KEY_PITCH = 3.9                                           # mm between key rows, clear of the 3.5 mm discs
# the metadata-poor script communities (Section 4) the map draws as dust, circled in Fig. 1
SCRIPT_MARKS = (("Japanese script", "Japanese-script community"), ("Chinese characters", "Chinese-character community"),
                ("Mixed CJK script", "Surname-first Japanese community"))


def community_name(name):
    """A map label split into its fields and countries: 'Medicine (Spain, Mexico, Argentina)' -> ('Medicine',
    ['Spain', 'Mexico', 'Argentina']); a label without countries has none."""
    if "(" not in name:
        return name, []
    field, _, rest = name.partition(" (")
    return field, [c.strip() for c in rest.rstrip(")").split(",")]


def field_group(name):
    """A country-named community's main fields: 0 many (labelled Multidisciplinary), 1 medicine, 2 others."""
    field = community_name(name)[0]
    return 0 if field == "Multidisciplinary" else 1 if field == "Medicine" else 2


def joined(words):
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " & " + words[-1]


def members_text(n):
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.0f}k"


def map_density(x, y, shape, bin_px=4, sigma=15.0):
    """Points per bin_px-pixel cell of an image of this shape, smoothed (sigma in cells: about 3 mm in Fig. 1)."""
    gy, gx = shape[0] // bin_px + 1, shape[1] // bin_px + 1
    ok = (x >= 0) & (x < shape[1]) & (y >= 0) & (y < shape[0])
    counts, _, _ = np.histogram2d(y[ok] / bin_px, x[ok] / bin_px, bins=(gy, gx), range=((0, gy), (0, gx)))
    return gaussian_filter(counts, sigma)


def marker_spot(own, total, taken, radius, boxes=(), margin=0.0, bin_px=4):
    """Where a community is both dense and dominant on the map (its density times its share of all density), in
    pixels: at least margin from the edges, radius from the spots taken and radius / 2 from the boxes (x0, y0, x1,
    y1)."""
    score = own * own / np.maximum(total, 1e-9)
    gy, gx = score.shape
    yy, xx = np.mgrid[0:gy, 0:gx]
    xx, yy = xx * bin_px + bin_px / 2, yy * bin_px + bin_px / 2
    score[(xx < margin) | (yy < margin) | (xx > gx * bin_px - margin) | (yy > gy * bin_px - margin)] = -np.inf
    for x0, y0, x1, y1 in boxes:
        score[(xx > x0 - radius / 2) & (xx < x1 + radius / 2) & (yy > y0 - radius / 2) & (yy < y1 + radius / 2)] = -np.inf
    for x, y in taken:
        score[(xx - x) ** 2 + (yy - y) ** 2 < radius ** 2] = -np.inf
    iy, ix = np.unravel_index(score.argmax(), score.shape)
    return float(xx[iy, ix]), float(yy[iy, ix])


def text_box(x, y, text, size_pt, px_per_pt, em=0.62):
    """An estimated box around centred text (Helvetica: about 0.62 em a character in bold, 0.54 regular), in
    pixels."""
    lines = text.split("\n")
    half_w = max(len(s) for s in lines) * em * size_pt * px_per_pt / 2
    half_h = len(lines) * 1.15 * size_pt * px_per_pt / 2
    return x - half_w, y - half_h, x + half_w, y + half_h


def dust_layer(shape, at, peak=0.6):
    """The records the tiles leave out (palette.sparse_communities and communities under 1,000 members; the outer ring
    lies outside the crop), as the viewer's dust layer: grey, brighter where denser on a log scale, up to `peak`. An RGB
    image in [0, 1]."""
    c = pq.read_table(MAP / "coords_web.parquet", columns=["xw", "yw", "is_ring", "is_dust"]).to_pandas()
    c = c[c.is_dust & ~c.is_ring]
    x, y = (np.asarray(v) for v in at(c.xw.to_numpy(), c.yw.to_numpy()))
    h, w = shape[:2]
    xi, yi = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    cnt = gaussian_filter(np.bincount(yi[ok] * w + xi[ok], minlength=h * w).reshape(h, w).astype(float), 0.7)
    lit = cnt > 0.02
    v = np.where(lit, 0.22 + 0.78 * (np.log1p(cnt) / np.log1p(cnt.max())) ** 0.7, 0.0) * peak
    return v[..., None] * np.array([0.93, 0.94, 1.0])


def script_circle(x, y, shape, min_r, max_r, share=0.8):
    """A circle around a community's map members: centred where they are densest, with the radius that holds `share`
    of the members within max_r of that spot (pixels; at least min_r)."""
    d = map_density(x, y, shape, bin_px=4, sigma=1.5)
    iy, ix = np.unravel_index(d.argmax(), d.shape)
    cx, cy = ix * 4 + 2.0, iy * 4 + 2.0
    r = np.hypot(x - cx, y - cy)
    near = r[r <= max_r]
    return cx, cy, float(np.clip(np.quantile(near, share), min_r, max_r)) if len(near) else min_r


def beside(img, cx, cy, radius, text, size_pt, px_per_mm, avoid=()):
    """Where a label fits beside a circle: the side (right, left, below or above) whose box lies inside the image, clear
    of the boxes to avoid, over the darkest part of the map. Returns the text position and its alignment."""
    px_per_pt = px_per_mm * 25.4 / 72
    x0, y0, x1, y1 = text_box(0, 0, text, size_pt, px_per_pt, em=0.54)
    tw, th, gap = x1 - x0, y1 - y0, radius + 0.8 * px_per_mm
    h, w = img.shape[:2]
    lum = img.max(axis=2).astype(float)
    best = None
    for tx, ty, ha, va, box in (
            (cx + gap, cy, "left", "center", (cx + gap, cy - th / 2, cx + gap + tw, cy + th / 2)),
            (cx - gap, cy, "right", "center", (cx - gap - tw, cy - th / 2, cx - gap, cy + th / 2)),
            (cx + radius, cy + gap, "right", "top", (cx + radius - tw, cy + gap, cx + radius, cy + gap + th)),
            (cx - radius, cy + gap, "left", "top", (cx - radius, cy + gap, cx - radius + tw, cy + gap + th)),
            (cx, cy - gap, "center", "bottom", (cx - tw / 2, cy - gap - th, cx + tw / 2, cy - gap))):
        bx0, by0, bx1, by1 = box
        if bx0 < 0 or by0 < 0 or bx1 > w or by1 > h:
            continue
        hit = any(bx0 < a1 and a0 < bx1 and by0 < b1 and b0 < by1 for a0, b0, a1, b1 in avoid)
        score = lum[int(by0):int(by1), int(bx0):int(bx1)].mean() + (1e3 if hit else 0)
        if best is None or score < best[0]:
            best = (score, tx, ty, ha, va, box)
    assert best is not None and best[0] < 1e3, f"no clear side for {text!r}"
    return best[1:]


_MAP = {}


def map_communities(z=4):
    """The map's coloured communities with at least MARKED members: those named for fields only (named at their
    median position) and those named for countries in key order, each with its tile colour, whether one country holds
    half of its members with a known country (and a quarter have one: national in Section 4's sense), and a marker
    spot where it is dense and dominant, clear of the field names and of the other markers (pixels of the z-4 map, at
    Fig. 1's print scale). Computed once."""
    if z in _MAP:
        return _MAP[z]
    img, at, _ = stitched_map(z, room_below=False)
    img = tight(img)
    h, w = img.shape[:2]
    dust = dust_layer(img.shape, at)
    img = np.maximum(img, np.rint(dust * 255).astype(np.uint8))
    px_per_mm = w / (W2 / MM)
    regions = sorted(json.loads((MAP / "regions.json").read_text()), key=lambda r: -r["members"])
    regions = [r for r in regions if r.get("zmin", 9) <= 2 and r["members"] >= MARKED]
    fields = [r for r in regions if not community_name(r["name"])[1]]
    named = sorted((r for r in regions if community_name(r["name"])[1]), key=lambda r: (field_group(r["name"]), -r["members"]))
    shares = pd.read_csv(CACHE / "a5b_communities.csv").set_index("community")
    majority = {r["community"]: bool(shares.top_country_share[r["community"]] >= 0.5
                                     and shares.share_no_country[r["community"]] <= 0.75) for r in named}
    c = pq.read_table(MAP / "coords_web.parquet", columns=["xw", "yw", "community", "is_ring"]).to_pandas()
    stats = c.groupby("community").agg(n=("xw", "size"), cx=("xw", "mean"), cy=("yw", "mean"))
    colour = {int(k): community_rgb(int(k), int(s.n), float(s.cx), float(s.cy)) for k, s in stats.iterrows()
              if int(k) in {r["community"] for r in regions}}
    c = c[~c.is_ring]
    x, y = (np.asarray(v) for v in at(c.xw.to_numpy(), c.yw.to_numpy()))
    total = map_density(x, y, (h, w))
    boxes = [text_box(*at(r["xw"], r["yw"]), FIELD_NAMES.get(r["name"], r["name"]), 7.5, px_per_mm * 25.4 / 72)
             for r in fields]
    spots, taken = {}, []
    community = c.community.to_numpy()
    for r in sorted(named, key=lambda r: -r["members"]):          # the largest communities choose first
        mine = community == r["community"]
        spots[r["community"]] = marker_spot(map_density(x[mine], y[mine], (h, w)), total, taken, 4.6 * px_per_mm,
                                            boxes, margin=2.8 * px_per_mm)
        taken.append(spots[r["community"]])
    # the script communities: circled where their members are, labelled beside the circle, clear of the names and
    # numbers
    size = pq.read_table(CACHE / "node_attrs.parquet", columns=["community"]).column("community").to_pandas().value_counts()
    half = 1.9 * px_per_mm
    avoid = boxes + [(sx - half, sy - half, sx + half, sy + half) for sx, sy in taken]
    scripts = []
    for key, label in SCRIPT_MARKS:
        comm = SCRIPT_COMMUNITIES[key]
        mine = community == comm
        cx, cy, radius = script_circle(x[mine], y[mine], (h, w), 1.2 * px_per_mm, 3.0 * px_per_mm)
        text = f"{label}\n{round(int(size[comm]), -3):,} researchers"
        tx, ty, ha, va, box = beside(img, cx, cy, radius, text, 7, px_per_mm, avoid)
        avoid.append(box)
        scripts.append(dict(key=key, community=comm, members=int(size[comm]), centre=(cx, cy), radius=radius,
                            text=text, at=(tx, ty), ha=ha, va=va, inside=int(mine.sum())))
    _MAP[z] = dict(img=img, at=at, fields=fields, named=named, majority=majority, colour=colour, spots=spots,
                   px_per_mm=px_per_mm, scripts=scripts, dust=dust)
    return _MAP[z]


def tight(img, pad=36, threshold=25):
    """Crop the empty right and bottom edges of the stitched map (the origin is kept, so pixel positions hold)."""
    lit = img.max(axis=2) > threshold
    rows, cols = np.flatnonzero(lit.sum(1) > 3), np.flatnonzero(lit.sum(0) > 3)
    return img[:min(img.shape[0], rows.max() + pad), :min(img.shape[1], cols.max() + pad)]


def number_marker(ax, x, y, number, colour, size=MARKER_SIZE, z=4):
    ax.scatter([x], [y], s=size, c=MARKER_FILL, edgecolors=[colour], linewidths=1.0, zorder=z)
    ax.text(x, y, str(number), color="white", fontsize=7, fontweight="bold", ha="center", va="center_baseline",
            zorder=z + 1)


def title(ax, letter, text):
    ax.set_title(rf"$\mathbf{{{letter}}}$   {text}", loc="left", pad=6)


def collisions(fig):
    """Texts that overlap one another; texts from different axes less than 2 mm apart on one line, which read as one
    phrase; texts a plotted line runs through, unless an opaque text box above the line hides it or the line segment
    ends inside the text (a link to the node the text numbers); gridlines along a text, through its middle; and texts
    that leave the figure. At print size, after fig.canvas.draw(); inset axes included. Reported, not refused: a
    review finds which ones matter."""
    renderer = fig.canvas.get_renderer()
    axes = fig.findobj(matplotlib.axes.Axes)                # with inset axes, which fig.axes leaves out
    undrawn = set()                                         # tick labels of hidden axes or of ticks outside the view
    for ax in axes:
        box = ax.get_window_extent(renderer)
        for axis, centre, lo, hi in ((ax.xaxis, lambda e: (e.x0 + e.x1) / 2, box.x0, box.x1),
                                     (ax.yaxis, lambda e: (e.y0 + e.y1) / 2, box.y0, box.y1)):
            for t in axis.get_ticklabels(which="both"):
                if not (ax.axison and axis.get_visible()) or not lo - 1 <= centre(t.get_window_extent(renderer)) <= hi + 1:
                    undrawn.add(id(t))
    texts = [t for t in fig.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text().strip() and id(t) not in undrawn]
    # the text alone: an annotation's own extent also covers its arrow
    boxes = [(t, matplotlib.text.Text.get_window_extent(t, renderer)) for t in texts]
    boxes = [(t, b) for t, b in boxes if b.width > 0 and b.height > 0]
    found = []
    for t, b in boxes:
        if b.x0 < fig.bbox.x0 - 0.5 or b.x1 > fig.bbox.x1 + 0.5 or b.y0 < fig.bbox.y0 - 0.5 or b.y1 > fig.bbox.y1 + 0.5:
            found.append(f"leaves the figure: {t.get_text()!r}")
    owner = {}                                              # innermost axes: findobj lists insets before their parents
    for ax in axes:
        for t in ax.findobj(matplotlib.text.Text):
            owner.setdefault(id(t), id(ax))
    near = 2 * fig.dpi / 25.4
    for i, (t, b) in enumerate(boxes):
        for u, c in boxes[i + 1:]:
            same_line = min(b.y1, c.y1) - max(b.y0, c.y0) > min(b.height, c.height) / 2
            if owner.get(id(t)) != owner.get(id(u)) and same_line and 0 < max(c.x0 - b.x1, b.x0 - c.x1) < near:
                found.append(f"texts run together: {t.get_text()!r} / {u.get_text()!r}")

    def hidden(t, artist):                                  # an opaque text box drawn above the artist
        patch = t.get_bbox_patch()
        return patch is not None and patch.get_facecolor()[3] > 0 and t.get_zorder() > artist.get_zorder()

    padded = [(t, b.padded(0.5)) for t, b in boxes]         # touching counts as overlapping
    for i, (t, b) in enumerate(padded):
        found += [f"texts overlap: {t.get_text()!r} / {u.get_text()!r}" for u, c in padded[i + 1:] if b.overlaps(c)]
    for ax in axes:
        for line in ax.get_lines():
            if line.get_visible() and line.get_linestyle() not in ("None", "none", "", " "):
                path = line.get_transform().transform_path(line.get_path())
                found += [f"line through text: {t.get_text()!r}" for t, b in padded
                          if not hidden(t, line) and path.intersects_bbox(b, filled=False)]
        for coll in ax.collections:
            if isinstance(coll, LineCollection) and coll.get_visible():
                for path in coll.get_paths():
                    path = coll.get_transform().transform_path(path)
                    found += [f"line through text: {t.get_text()!r}" for t, b in padded
                              if not hidden(t, coll) and path.intersects_bbox(b, filled=False)
                              and not any(b.contains(*v) for v in path.vertices[[0, -1]])]
        if not ax.axison:
            continue
        box = ax.get_window_extent(renderer)
        for line in ax.xaxis.get_gridlines() + ax.yaxis.get_gridlines():
            (x0, y0), (x1, y1) = line.get_transform().transform_path(line.get_path()).vertices[[0, -1]]
            if not line.get_visible() or not (box.x0 - 1 <= min(x0, x1) and max(x0, x1) <= box.x1 + 1 and
                                              box.y0 - 1 <= min(y0, y1) and max(y0, y1) <= box.y1 + 1):
                continue                                    # not drawn, or the tick lies outside the view
            for t, b in boxes:
                if hidden(t, line):
                    continue
                if round(t.get_rotation()) % 180 == 0:
                    along = abs(y1 - y0) < 0.5 and b.y0 + b.height / 4 < y0 < b.y1 - b.height / 4 and \
                        min(x0, x1) < b.x1 and max(x0, x1) > b.x0
                else:
                    along = abs(x1 - x0) < 0.5 and b.x0 + b.width / 4 < x0 < b.x1 - b.width / 4 and \
                        min(y0, y1) < b.y1 and max(y0, y1) > b.y0
                if along:
                    found.append(f"gridline through text: {t.get_text()!r}")
    return found


def save(fig, name, dpi=300, eps=True):
    """PDF (vector) and PNG preview; for submission, named by figure number, a 600-dpi TIFF and (for figures that are
    not mostly raster) an EPS. Refuses text under 7 pt at print size and reports text collisions."""
    fig.canvas.draw()                                       # tick labels get their text at draw time
    sizes = [t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
    print(f"{name}: {len(sizes)} texts, {min(sizes):g}-{max(sizes):g} pt")
    assert min(sizes) >= 7, f"{name}: text under 7 pt"
    for problem in collisions(fig):
        print(f"{name}: {problem}")
    sub = FIGURES / "submission"
    sub.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", dpi=dpi)
    fig.savefig(FIGURES / f"{name}.png", dpi=300)
    if eps:
        fig.savefig(sub / f"Fig{NUMBER[name]}.eps", dpi=dpi)
    fig.savefig(sub / f"Fig{NUMBER[name]}.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def percent_axis(axis, decimals=0):
    axis.set_major_formatter(PercentFormatter(1, decimals=decimals))


def national_country_groups(pivot, threshold=0.005):
    """Partition displayed countries by measured share; retain all-missing columns."""
    some, low, missing = [], [], []
    for country in pivot.columns:
        values = pivot[country].dropna()
        if values.empty:
            missing.append(country)
        elif values.max() < threshold:
            low.append(country)
        else:
            some.append(country)
    return some, low, missing


def country_region_blocks(names):
    """Keep the familiar regional order and include new countries in a final block."""
    region_order = [["US", "CA", "MX", "BR"],
                    ["GB", "NL", "BE", "FR", "DE", "AT", "CH", "SE", "IT", "ES", "PL", "RU", "UA"],
                    ["TR", "IR"], ["IN", "ID"], ["AU"], ["CN", "TW", "JP", "KR"]]
    blocks = [[country for country in block if country in names] for block in region_order]
    included = {country for block in blocks for country in block}
    blocks.append([country for country in names if country not in included])
    return [block for block in blocks if block]


def stitched_map(z=4, dim=1.0, room_below=True):
    """The map's raster tiles at zoom z, stitched and cropped to the drawn area (the ring of isolated researchers left
    out), with room below for the Japanese-script marker. Returns the image and a map from layout coordinates to its
    pixels."""
    from PIL import Image
    tiles = MAP / "tiles" / str(z)
    xs = sorted(int(p.name) for p in tiles.iterdir() if p.is_dir())
    ys = sorted({int(f.stem) for x in xs for f in (tiles / str(x)).glob("*.png")})
    x0, y0 = xs[0], ys[0]
    canvas = Image.new("RGB", ((xs[-1] - x0 + 1) * 256, (ys[-1] - y0 + 1) * 256), (0, 0, 0))
    for x in xs:
        for f in (tiles / str(x)).glob("*.png"):
            canvas.paste(Image.open(f).convert("RGB"), ((x - x0) * 256, (int(f.stem) - y0) * 256))
    img = np.asarray(canvas)
    world = 256 * 2 ** z
    c = pq.read_table(MAP / "coords_web.parquet", columns=["xw", "yw", "is_ring"]).to_pandas()
    core = c[~c.is_ring]

    def px(xw, yw):
        return xw * world - x0 * 256, (1 - yw) * world - y0 * 256

    (ax0, ay1), (ax1, ay0) = px(core.xw.quantile(0.001), core.yw.quantile(0.001)), px(core.xw.quantile(0.999), core.yw.quantile(0.999))
    pad = 40 * 2 ** (z - 4)
    cx0, cy0 = max(0, int(ax0) - pad), max(0, int(ay0) - pad)
    img = img[cy0:int(ay1) + pad, cx0:int(ax1) + pad]
    if room_below:                                        # the tiles end at the Japanese-script marker
        img = np.pad(img, ((0, pad), (0, 0), (0, 0)))
    if dim != 1.0:
        img = (img.astype(np.float32) * dim).astype(np.uint8)

    def at(xw, yw):
        x, y = px(xw, yw)
        return x - cx0, y - cy0

    return img, at, 2 ** (z - 4)


def map_overview(z=4):
    """Fig. 1: the map; its field communities named, the communities named for countries numbered and listed below
    by their main fields (the regional ones marked), the two script communities circled."""
    m = map_communities(z)
    img, at, px_per_mm = m["img"], m["at"], m["px_per_mm"]
    h, w = img.shape[:2]
    rows = max(sum(field_group(r["name"]) == g for r in m["named"]) for g in range(3))
    key_h, map_h = (10.6 + rows * KEY_PITCH) * MM, W2 * h / w
    fig = plt.figure(figsize=(W2, map_h + key_h))
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0, key_h / (map_h + key_h), 1, map_h / (map_h + key_h)])
    ax.imshow(img, interpolation="antialiased")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_axis_off()
    halo = [patheffects.withStroke(linewidth=2.2, foreground="black")]
    for r in m["fields"]:
        ax.text(*at(r["xw"], r["yw"]), FIELD_NAMES.get(r["name"], r["name"]), color="white", fontsize=7.5,
                fontweight="bold", ha="center", va="center", linespacing=1.05, path_effects=halo, zorder=6)
    for number, r in enumerate(m["named"], start=1):
        number_marker(ax, *m["spots"][r["community"]], number, m["colour"][r["community"]])
    for s in m["scripts"]:                                   # drawn as dust: circled where they are, named beside
        ax.add_patch(plt.Circle(s["centre"], s["radius"], fill=False, color=GOLD, lw=0.8, zorder=5))
        ax.text(*s["at"], s["text"], color=GOLD, fontsize=7, ha=s["ha"], va=s["va"], linespacing=1.15,
                path_effects=halo, zorder=6)
    map_key(fig.add_axes([0, 0, 1, key_h / (map_h + key_h)]), m, m["named"], key_h / MM, note=True)
    save(fig, "fig_map", dpi=600, eps=False)


def map_key(kx, m, shown, height_mm, note=False, columns=((4, 56), (62, 56), (118, 56)), title=True, sizes=True,
            footnote=None):
    """The key to the numbered communities, on black: three columns by their main fields (bold headings), each row a
    number disc, the countries (and the fields where they are not many or medicine), the regional mark and the size."""
    kx.set_facecolor("black")                               # an axes background, not the figure's
    kx.set_xticks([])
    kx.set_yticks([])
    kx.grid(False)
    for spine in kx.spines.values():
        spine.set_visible(False)
    width = columns[-1][0] + columns[-1][1]
    kx.set_xlim(0, width)
    kx.set_ylim(height_mm, 0)
    numbers = {r["community"]: n for n, r in enumerate(m["named"], start=1)}
    top = 0.0
    if title:
        kx.text(columns[0][0], 2.6, "Communities named for countries (numbered in the map), by their main fields",
                color="#bdbcb5", fontsize=7, fontweight="bold", va="center")
        top = 3.9
    for g, ((x0, span), heading) in enumerate(zip(columns, ("Many fields", "Mainly medicine", "Other fields"))):
        kx.text(x0, top + 2.7, heading, color="white", fontsize=7.5, fontweight="bold", va="center")
        group = [r for r in shown if field_group(r["name"]) == g]
        for j, r in enumerate(group):
            y = top + 7.0 + j * KEY_PITCH
            number_marker(kx, x0 + 1.8, y, numbers[r["community"]], m["colour"][r["community"]])
            field, countries = community_name(r["name"])
            text = joined(countries) + ("" if g < 2 else f"  ({field.lower()})")
            kx.text(x0 + 4.6, y, text + ("" if m["majority"][r["community"]] else " \u2020"), color="white",
                    fontsize=7, va="center_baseline")
            if sizes:
                kx.text(x0 + span - 2, y, members_text(r["members"]), color="#8f8e88", fontsize=7, ha="right",
                        va="center_baseline")
        if footnote and g == 2:
            kx.text(x0, top + 7.0 + (len(group) - 0.45) * KEY_PITCH, footnote, color="#8f8e88", fontsize=7, va="top",
                    linespacing=1.3)
        if note and g == 2:
            kx.text(x0, top + 7.0 + (len(group) - 0.45) * KEY_PITCH, "\u2020 Regional: no country holds half of the "
                    "members\nwith a known country. The others, and clinical\nmedicine, are national (Section 4).\n"
                    "Named in the map: field communities, in which\nno country is strongly over-represented.",
                    color="#8f8e88", fontsize=7, va="top", linespacing=1.3)


# Fig. 5: the field groups (common.FIELD_GROUPS) in amber over the map's grey
AMBER = LinearSegmentedColormap.from_list("amber", ["#5a5a5e", "#8a7a5a", "#d9a441", "#ffd27a", "#fff1cf"])
SHARE_TOP = 0.6                                           # the ramp's end: a group holding 60% or more of a spot


def field_group_index(fields):
    """Each researcher's field group (index into FIELD_GROUPS), or -1 without a field (no topic, or the fallback
    topic)."""
    index = {f: i for i, group in enumerate(FIELD_GROUPS.values()) for f in group}
    assert len(index) == sum(len(g) for g in FIELD_GROUPS.values()) == 26, "each field in exactly one group"
    out = np.full(len(fields), -1, np.int8)
    for f, i in index.items():
        out[np.asarray(fields == f)] = i
    unknown = set(pd.unique(fields[out < 0])) - {None}
    assert not {u for u in unknown if isinstance(u, str)}, f"fields in no group: {unknown}"
    return out


def tile_brightness(count):
    """pipeline/tiles.py's brightness: the rank of log(1 + count) among occupied cells, gamma 1.25, floor 0.05."""
    v = np.zeros(count.shape)
    lit = count > 0
    logs = np.log1p(count[lit])
    v[lit] = 0.05 + 0.95 * (np.searchsorted(np.sort(logs), logs, side="right") / len(logs)) ** 1.25
    return v


def fields_on_map(bin_px=4, sigma=2.0):
    """Fig. 5: where each field group's researchers are on the map. In each panel, the map's researchers in grey
    (brighter where denser, as the tiles), coloured by the group's share of all researchers around each spot
    (first OpenAlex topic; records with the fallback topic or no topic count in the total and in no group)."""
    m = map_communities(4)
    h, w = m["img"].shape[:2]
    c = pq.read_table(MAP / "coords_web.parquet", columns=["id", "xw", "yw", "is_ring"]).to_pandas()
    f = pq.read_table(CACHE / "map_fields.parquet", columns=["id", "field"]).to_pandas().set_index("id")
    assert f.index.is_unique and len(f) == len(c) and f.index.isin(c.id).all(), "map_fields.py is out of date"
    c = c[~c.is_ring]
    group = field_group_index(f.field.reindex(c.id).to_numpy(dtype=object))
    x, y = (np.asarray(v) for v in m["at"](c.xw.to_numpy(), c.yw.to_numpy()))
    gy, gx = h // bin_px + 1, w // bin_px + 1

    def counts(mask):
        return np.histogram2d(y[mask] / bin_px, x[mask] / bin_px, bins=(gy, gx), range=((0, gy), (0, gx)))[0]

    every = counts(np.ones(len(x), bool))
    bright, around = tile_brightness(every), gaussian_filter(every, sigma)
    cols, gap, title_h, bar_h, side = 4, 2.5, 4.6, 10.5, 0.8
    panel_w = (W2 / MM - 2 * side - (cols - 1) * gap) / cols
    panel_h = panel_w * h / w
    rows = -(-len(FIELD_GROUPS) // cols)
    height = rows * (title_h + panel_h) + (rows - 1) * gap + bar_h
    fig = plt.figure(figsize=(W2, height * MM))
    fig.patch.set_facecolor("black")
    for i, name in enumerate(FIELD_GROUPS):
        r, k = divmod(i, cols)
        x0, y0 = side + k * (panel_w + gap), r * (title_h + panel_h + gap) + title_h
        ax = fig.add_axes([x0 / (W2 / MM), 1 - (y0 + panel_h) / height, panel_w / (W2 / MM), panel_h / height])
        share = gaussian_filter(counts(group == i), sigma) / np.maximum(around, 1e-9)
        rgb = AMBER(np.clip(share / SHARE_TOP, 0, 1))[..., :3] * bright[..., None]
        ax.imshow(np.clip(rgb, 0, 1), interpolation="antialiased")
        ax.set_axis_off()
        n = int((group == i).sum())
        ax.text(0, 1.0, name, transform=ax.transAxes, color="white", fontsize=7.5, fontweight="bold",
                va="bottom", ha="left")
        ax.text(1.0, 1.0, members_text(n), transform=ax.transAxes, color="#9a9992", fontsize=7, va="bottom",
                ha="right")
    cb = fig.add_axes([0.38, 4.0 / height, 0.30, 2.0 / height])
    cb.imshow(AMBER(np.linspace(0, 1, 256))[None, :, :3], aspect="auto", extent=(0, SHARE_TOP, 0, 1))
    cb.set_yticks([])
    cb.grid(False)
    cb.set_xticks([0, 0.2, 0.4, 0.6], ["0%", "20%", "40%", "60% or more"])
    cb.tick_params(axis="x", length=0, pad=1.5, colors="#bdbcb5", labelsize=7)
    for spine in cb.spines.values():
        spine.set_visible(False)
    fig.text(0.37, 5.0 / height, "the group's share of the researchers around each spot", color="#bdbcb5",
             fontsize=7, ha="right", va="center")
    fig.text(0.70, 5.0 / height, "brighter: more researchers", color="#bdbcb5", fontsize=7, ha="left",
             va="center")
    save(fig, "fig_fields", dpi=600, eps=False)


def binned_shares(hist, top=10):
    """Per-source distance histograms (sources x steps; column 0 is the source itself) -> the pooled share of pairs
    at 1, 2, ..., top-1 and top or more steps, and standard errors that treat the sources as clusters (the ratio
    estimator's linearization). Sources that reach no one are left out."""
    h = np.asarray(hist, dtype=float)[:, 1:]
    h = np.column_stack([h[:, :top - 1], h[:, top - 1:].sum(1)])
    n = h.sum(1)
    h, n = h[n > 0], n[n > 0]
    share = h.sum(0) / n.sum()
    resid = h - share[None, :] * n[:, None]
    k = len(n)
    return share, np.sqrt(k / (k - 1) * (resid ** 2).sum(0)) / n.sum()


def tail_steps(v):
    """Sorted values -> the step curve of the share of values above each (floored at half a value, for the log axis)."""
    farther = 1 - np.arange(1, len(v) + 1) / len(v)
    last = np.r_[np.diff(v) > 0, True]
    return np.r_[v[0], v[last]], np.r_[1.0, np.maximum(farther[last], 0.5 / len(v))]


def distances():
    """Fig. 2: (a) coauthor steps between random pairs, as delivered and repaired (A2, A8), with 95% intervals over
    the 1,000 random sources as delivered (the repaired sources are these mapped through the repair, not a fresh
    sample, so the paper reports no repaired interval) and brackets for the shares within 5 and 6 steps; (b) each
    researcher's mean distance to the sampled sources, as the share farther away, as delivered and (dashed) repaired;
    (c) median local clustering by number of coauthors (A1b) against a random network with the same degrees."""
    delivered = np.load(CACHE / "a2c_hist_cap50.npz")
    repaired = np.load(CACHE / "repair_distance_summaries.npz")
    a2 = json.loads((RESULTS / "a2_distances.json").read_text())["all_links"]
    a8 = json.loads((RESULTS / "a8_repair.json").read_text())["all_sources"]["after"]
    series = [("as delivered", delivered["hist"], a2, C["main"]), ("repaired", repaired["histograms"], a8, C["contrast"])]
    fig, (a1, b, c) = plt.subplots(1, 3, figsize=(W2, 68 * MM), layout="constrained",
                                   gridspec_kw={"width_ratios": [1.08, 0.98, 0.94]})
    fig.get_layout_engine().set(w_pad=0.05)                 # about 2.5 mm between one panel's labels and the next
    w = 0.38
    for i, (lab, hist, result, col) in enumerate(series):
        share, se = binned_shares(hist)
        pmf = result["pmf"]                                 # the per-source histograms reproduce the reported shares
        assert np.allclose(share[:9], [float(pmf.get(str(k), 0)) for k in range(1, 10)], rtol=0, atol=1e-12), lab
        x = np.arange(10) + (i - 0.5) * w
        a1.bar(x, share, w, color=col, edgecolor="white", linewidth=0.4, label=f"{lab} (mean {float(result['mean']):.2f})")
        if lab == "as delivered":                           # a random sample of sources; the repaired sources are
            a1.errorbar(x, share, yerr=1.96 * se, fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.4,
                        capthick=0.6)                       # these mapped through the repair, so no interval there
            within = np.cumsum(share)
    a1.set_xticks(range(10), [str(k) for k in range(1, 10)] + ["10+"])
    a1.tick_params(axis="x", length=0)
    a1.set_xlim(-0.6, 9.6)
    a1.set_ylim(0, 0.66)                                    # room above the bars and the 50% line for the brackets
    a1.set_yticks(np.arange(0, 0.51, 0.1))
    percent_axis(a1.yaxis)
    a1.grid(axis="x", visible=False)
    a1.set_xlabel("coauthor steps")                         # the title says between whom
    a1.set_ylabel("share of pairs")

    def bracket(last, y, text):                             # from the first bar to the bars of step last + 1
        a1.plot([-0.4, -0.4, last + 0.4, last + 0.4], [y - 0.014, y, y, y - 0.014], color=INK, lw=0.7,
                solid_capstyle="butt")
        a1.text(-0.32, y - 0.019, text, ha="left", va="top", fontsize=7.5, color=INK2)   # below the end ticks

    bracket(5, 0.648, f"{within[5]:.0%} of pairs within 6 steps")
    bracket(4, 0.586, f"{within[4]:.0%} within 5 steps")
    title(a1, "a", "distance between two researchers")
    handles, labels = a1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, handlelength=1.2, columnspacing=2.2)

    closeness = np.asarray(np.load(CACHE / "closeness_1000.npy", mmap_mode="r"))
    reached = np.sort(closeness[np.isfinite(closeness)])
    counts, sums = repaired["counts"], repaired["sums"]
    full = counts == counts.max()                           # reached from every source that reaches anyone
    after = np.sort(sums[full] / counts.max())
    print("fig_distances b: delivered", len(reached), "median %.3f p99 %.3f p99.99 %.3f max %.2f" % (
        np.median(reached), np.quantile(reached, 0.99), np.quantile(reached, 0.9999), reached[-1]),
        "| repaired", len(after), "median %.3f p99 %.3f p99.99 %.3f max %.2f" % (
        np.median(after), np.quantile(after, 0.99), np.quantile(after, 0.9999), after[-1]))
    xs, ys = tail_steps(reached)
    b.fill_between(xs, ys, 1e-9, step="post", color=tint(C["main"], 0.12), linewidth=0)   # opaque for EPS
    b.step(xs, ys, where="post", color=C["main"], lw=1.3)
    b.step(*tail_steps(after), where="post", color=C["contrast"], lw=1.1, ls=(0, (3, 2)))
    b.set_yscale("log")
    b.set_ylim(3e-8, 1.6)
    b.yaxis.set_major_locator(FixedLocator([1, 1e-2, 1e-4, 1e-6]))
    b.yaxis.set_major_formatter(FixedFormatter(["100%", "1%", "0.01%", "0.0001%"]))
    b.yaxis.set_minor_locator(NullLocator())
    b.set_xlim(3, 28.5)
    b.set_xticks([3, 5, 10, 15, 20, 25])
    b.set_xlabel("mean coauthor steps to the sampled researchers")
    b.set_ylabel("share of researchers farther away")
    median, p99, p9999 = np.median(reached), np.quantile(reached, 0.99), np.quantile(reached, 0.9999)
    for xv, yv, text, dx, dy, ha, va in (
            (median, 0.5, f"half beyond {median:.2f}", 7, -3, "left", "top"),
            (p99, 0.01, f"1% ({0.01 * len(reached) / 1e3:,.0f},000) beyond {p99:.2f}", 7, 2, "left", "bottom"),
            (p9999, 1e-4, f"0.01% ({1e-4 * len(reached):,.0f})\nbeyond {p9999:.1f}", 5, 4, "left", "bottom"),
            (reached[-1], 1 / len(reached), f"farthest {reached[-1]:.1f}", -7, 0, "right", "center")):
        b.plot([xv], [yv], "o", ms=3.2, color=INK, mec="white", mew=0.6, zorder=4)
        b.annotate(text, (xv, yv), xytext=(dx, dy), textcoords="offset points", fontsize=7.5, color=INK2, ha=ha, va=va)
    title(b, "b", "mean distance to everyone")

    cl = json.loads((RESULTS / "a1b_clustering.json").read_text())
    start = np.array([r["degree_from"] for r in cl["by_degree"]], float)
    end = np.r_[start[1:] - 1, start[-1] * 2]               # the last band is open-ended
    middle = np.sqrt(start * end)
    median_c = np.array([r["median"] for r in cl["by_degree"]])
    chance = cl["all"]["random_expectation"]
    c.plot(middle, median_c, color=C["main"], marker="o", ms=3.2, mec="white", mew=0.6)
    c.axhline(chance, color=MUTED, lw=0.9, ls=(0, (3, 2)))
    c.text(2.4, chance * 1.9, "a random network with\nthe same degrees", ha="left", va="bottom", fontsize=7.5,
           color=INK2, linespacing=1.15)
    c.annotate("this network", (middle[6], median_c[6]), xytext=(5, 3), textcoords="offset points", ha="left",
               va="bottom", fontsize=7.5, color=INK2, bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    # RV bipartite null: the same works and team sizes with coauthors drawn at random, by its own coauthor counts
    nb = json.loads((RESULTS / "rv_bipartite_null.json").read_text())["null_by_degree"]
    n_start = np.array([r["degree_from"] for r in nb], float)
    assert np.array_equal(n_start, start), "null and observed clustering use different degree bands"
    null_c = np.array([r["median"] for r in nb])
    c.plot(middle, null_c, color=MUTED, ls=(0, (1, 1.5)), marker="o", ms=2.6, mec="white", mew=0.5)
    c.annotate("same works,\nrandom coauthors", (middle[9], null_c[9]), xytext=(-3, -6),
               textcoords="offset points", ha="right", va="top", fontsize=7.5, color=INK2, linespacing=1.15)
    c.set_xscale("log")
    c.set_yscale("log")
    c.set_xlim(2, 1.4e4)
    c.set_ylim(4e-5, 1.2)
    c.xaxis.set_major_locator(FixedLocator([2, 10, 100, 1000, 10000]))
    c.xaxis.set_major_formatter(FixedFormatter(["2", "10", "100", "1,000", "10,000"]))
    c.xaxis.set_minor_locator(NullLocator())
    c.tick_params(axis="x", labelsize=7)
    c.yaxis.set_major_locator(FixedLocator([1, 1e-1, 1e-2, 1e-3, 1e-4]))
    c.yaxis.set_major_formatter(FixedFormatter(["1", "0.1", "0.01", "0.001", "0.0001"]))
    c.yaxis.set_minor_locator(NullLocator())
    c.set_xlabel("coauthors")
    c.set_ylabel("median clustering coefficient")
    title(c, "c", "clustering by coauthors")
    save(fig, "fig_distances")


def script(name):
    """The script a name is recorded in, for the far-edge chain."""
    if any(0x600 <= ord(ch) <= 0x6FF for ch in name):
        return "Arabic"
    return "Cyrillic" if any(0x400 <= ord(ch) <= 0x4FF for ch in name) else "Latin"


def corridor_layout(step, coauthors, links, sweeps=8):
    """Vertical slots for the researchers on some shortest path: each step's researchers centred on one line,
    ordered by the mean slot of their linked researchers in the previous (or, sweeping back, the next) step, so that
    links rarely cross. step and coauthors map each researcher to its step and degree."""
    order = defaultdict(list)
    for u in sorted(step, key=lambda u: -coauthors[u]):
        order[step[u]].append(u)
    nb = defaultdict(set)
    for u, v in links:
        nb[u].add(v)
        nb[v].add(u)

    def slots():
        return {u: i - (len(us) - 1) / 2 for us in order.values() for i, u in enumerate(us)}

    steps = sorted(order)
    for k in range(sweeps):
        side = -1 if k % 2 == 0 else 1
        for s in (steps[1:] if side < 0 else steps[-2::-1]):
            y = slots()
            key = {u: np.mean([y[v] for v in nb[u] if step[v] == s + side] or [y[u]]) for u in order[s]}
            order[s] = sorted(order[s], key=lambda u: (key[u], -coauthors[u]))
    return slots()


LABEL_CITED = 2     # Fig. 3b names researchers with more citations; records with at most this many get a label


def far_edge_labels(cited, names):
    """Fig. 3b's text for each step of the path: the recorded name, or for records with at most LABEL_CITED citations a
    label by run of consecutive such steps (A1, A2, ... for the first run, B1, ... for the next)."""
    labels, run, index, previous = [], -1, 0, False
    for c, name in zip(cited, names):
        low = c <= LABEL_CITED
        if low and not previous:
            run, index = run + 1, 0
        if low:
            index += 1
            labels.append(f"{chr(ord('A') + run)}{index}")
        else:
            labels.append(name)
        previous = low
    return labels


def far_edge():
    """Fig. 3: the two researchers farthest apart (A3b). (a) Every shortest path between them: each dot a
    researcher, placed by its step from one end, coloured by the script of its name and sized by its citations; (b)
    one of the paths, with each researcher's citations: researchers with more than LABEL_CITED citations named in the
    scripts OpenAlex records them in, the others labelled by their run of consecutive steps (A1, A2, ...; B1, ...).
    Most of the far edge has fewer than 15 works and is not on the map, so the path is not drawn over it."""
    d = json.loads((RESULTS / "a3b_far_edge.json").read_text())
    nodes = pd.DataFrame(d["nodes"]).set_index("node")
    nodes["script"] = nodes.name.map(script)
    nxt = {}
    for u, v in d["links"]:
        nxt.setdefault(u, []).append(v)
    path = [d["pair"][0]]                       # each step: the linked record with most coauthors, never a placeholder
    while path[-1] != d["pair"][1]:
        path.append(max((v for v in nxt[path[-1]] if v not in d["placeholder"]), key=lambda v: nodes.coauthors[v]))
    p = nodes.loc[path].reset_index()
    p["label"] = far_edge_labels(p["cited"].to_numpy(), p["name"].tolist())
    width = np.array(d["width_by_step"])
    assert (nodes.groupby("step").size().to_numpy() == width).all()
    others = width - 1
    colour = {"Arabic": C["contrast"], "Latin": C["main"], "Cyrillic": C["contrast2"]}

    # runs of one script along the path, each under a heading, in two columns split between runs
    runs, first = [], 0
    for i in range(1, len(p) + 1):
        if i == len(p) or p.script[i] != p.script[first]:
            runs.append((p.script[first], list(range(first, i))))
            first = i
    size = [len(steps) + 1 for _, steps in runs]
    cut = min(range(1, len(runs)), key=lambda k: max(sum(size[:k]), sum(size[k:]))) if len(runs) > 1 else 1

    def rows(part):
        return [r for s, steps in part for r in [("head", f"{s} script")] + [("step", i) for i in steps]]

    # Leave room for glyph ink beyond its advance width at the canvas edge.
    cols = [(0.0, 0.44, rows(runs[:cut])), (0.53, 0.455, rows(runs[cut:]))]
    nrows = max(len(r) for _, _, r in cols)
    top_mm, row_mm = 58, 3.25
    fig_h = (top_mm + row_mm * (nrows + 1.9)) * MM
    fig = plt.figure(figsize=(W2, fig_h))
    mm = MM / fig_h                                         # one millimetre as a fraction of the figure's height
    fig.text(0.0, 1 - 0.5 * mm, r"$\mathbf{a}$   every shortest path between the two researchers", va="top")
    fig.text(0.0, 1 - (top_mm - 4) * mm, r"$\mathbf{b}$   one shortest path, step by step", va="top")

    a1 = fig.add_axes([0.012, 1 - 47 * mm, 0.976, 38 * mm])
    a1.set_axis_off()
    slot = corridor_layout(nodes.step.to_dict(), nodes.coauthors.to_dict(), d["links"])
    at = {u: (nodes.step[u], -slot[u]) for u in nodes.index}
    on_path = set(zip(path, path[1:]))
    a1.add_collection(LineCollection([(at[u], at[v]) for u, v in d["links"] if (u, v) not in on_path],
                                     colors=AXIS, linewidths=0.45, zorder=1))
    a1.add_collection(LineCollection([(at[u], at[v]) for u, v in zip(path, path[1:])], colors=INK, linewidths=0.9,
                                     zorder=2))

    def dot(cited):                                         # marker area, points squared, by citations
        return (2.4 + 1.1 * np.log10(np.asarray(cited) + 1)) ** 2

    for s in colour:
        us = nodes.index[nodes.script == s]
        if len(us):
            a1.scatter([at[u][0] for u in us], [at[u][1] for u in us], s=dot(nodes.cited[us]), c=colour[s],
                       edgecolors="white", linewidths=0.5, zorder=3)
    a1.scatter([at[u][0] for u in path], [at[u][1] for u in path], s=dot(nodes.cited[path]), facecolors="none",
               edgecolors=INK, linewidths=0.6, zorder=4)
    last = len(width) - 1
    a1.set_xlim(-0.8, last + 0.8)
    a1.set_ylim(-6.6, 8.4)
    for s in range(0, last + 1, 5):
        a1.text(s, -6.3, str(s), ha="center", va="center", fontsize=7, color=MUTED)
    a1.text(last / 2, -8.2, "coauthor steps from one end", ha="center", va="center", fontsize=8, color=INK2)
    # the corridor's parts, as described in Section 3
    single = [s for s in range(1, last) if width[s] == 1]
    wide = [s for s in range(1, last) if width[s] > 1]
    first_wide = min(wide)
    middle = [s for s in range(first_wide, last) if width[s] > 1 and all(width[t] > 1 for t in range(first_wide, s))]
    chain1 = [s for s in range(0, first_wide)]
    gate = max(middle) + 1
    chain2 = list(range(gate + 1, last + 1))
    assert all(p.script[s] == "Arabic" for s in chain1 + chain2) and p.script[gate] == "Latin"
    assert nodes[nodes.script == "Arabic"].cited.max() <= 2 and len(single) == 39

    def bracket(x0, x1, yb, text):
        a1.plot([x0, x0, x1, x1], [yb - 0.35, yb, yb, yb - 0.35], color=AXIS, lw=0.6, solid_capstyle="butt")
        a1.text((x0 + x1) / 2, yb + 0.55, text, ha="center", va="bottom", fontsize=7, color=INK2)

    bracket(chain1[0] - 0.3, chain1[-1] + 0.3, 1.4,
            f"{len(chain1)} Arabic-script records in a chain, none cited more than twice")
    bracket(chain2[0] - 0.3, chain2[-1] + 0.3, 1.9, f"{len(chain2)} Arabic-script records in a second chain")
    bracket(middle[0] - 0.3, middle[-1] + 0.3, 6.3,
            f"{len(middle)} steps with {width[middle].min()}-{width[middle].max()} researchers each")
    institution = nodes.institution[path[gate]]
    a1.annotate(f"one researcher at the\n{institution}", (gate, 0), xytext=(gate + 3.5, -3.6), textcoords="data",
                fontsize=7, color=INK2, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.6, shrinkA=1, shrinkB=3))
    lx, ly = 29.2, 7.7                                      # key: scripts, the path of b, dot size
    for i, (s, lab) in enumerate((("Arabic", "Arabic script"), ("Latin", "Latin script"))):
        a1.scatter([lx + i * 5.6], [ly], s=dot(100), c=colour[s], edgecolors="white", linewidths=0.5)
        a1.text(lx + i * 5.6 + 0.55, ly, lab, fontsize=7, color=INK2, va="center")
    a1.plot([lx + 11.4, lx + 12.6], [ly, ly], color=INK, lw=0.9)
    a1.text(lx + 12.9, ly, "the path in b", fontsize=7, color=INK2, va="center")
    a1.text(lx, ly - 1.3, "citations", fontsize=7, color=INK2, va="center")
    for x, c, lab in ((lx + 3.6, 0, "0"), (lx + 5.6, 100, "100"), (lx + 8.4, 10_000, "10,000")):
        a1.scatter([x], [ly - 1.3], s=dot(c), c=MUTED, edgecolors="white", linewidths=0.5)
        a1.text(x + 0.5, ly - 1.3, lab, fontsize=7, color=INK2, va="center")

    a2 = fig.add_axes([0.0, 0.0, 1.0, 1 - (top_mm - 1) * mm])
    a2.set_xlim(0, 1)
    a2.set_ylim(nrows - 0.1, -1.25)
    a2.set_axis_off()
    renderer = fig.canvas.get_renderer()
    for x0, col_w, rows_ in cols:
        a2.text(x0 + col_w, -0.95, "citations", fontsize=7, color=MUTED, ha="right", va="center")
        steps_at = [(r, v) for r, (kind, v) in enumerate(rows_) if kind == "step"]
        for (ra, _), (rb, _) in zip(steps_at, steps_at[1:]):
            a2.plot([x0 + 0.03] * 2, [ra, rb], color=AXIS, lw=1.2, zorder=1, solid_capstyle="butt")
        for r, (kind, v) in enumerate(rows_):
            if kind == "head":
                a2.scatter([x0 + 0.03], [r], s=16, color=colour[v.split()[0]], edgecolor="white", linewidth=0.6,
                           zorder=3)
                a2.text(x0 + 0.05, r, v, fontsize=7.5, fontweight="bold", color=INK, va="center")
                continue
            row = p.loc[v]
            a2.text(x0 + 0.017, r, str(v), fontsize=7, color=MUTED, ha="right", va="center")
            a2.scatter([x0 + 0.03], [r], s=10, color=colour[row.script], edgecolor="white", linewidth=0.6, zorder=3)
            named = row["label"] == row["name"]
            name = a2.text(x0 + 0.05, r, row["label"], fontsize=7, color=INK if named else INK2, va="center",
                           fontfamily="Geeza Pro" if named and row.script == "Arabic" else "sans-serif")
            if others[v]:
                end = name.get_window_extent(renderer).transformed(a2.transData.inverted()).x1
                a2.text(end + 0.01, r, f"+{others[v]}", fontsize=7, color=MUTED, va="center")
            a2.text(x0 + col_w, r, f"{row.cited:,}", fontsize=7, ha="right", va="center",
                    color=INK if row.cited else MUTED)
    save(fig, "fig_far_edge", dpi=600, eps=False)


SUBFIELD = {"Literature and Literary Theory": "Literature", "Language and Linguistics": "Languages & linguistics",
            "Archeology": "Archaeology", "Visual Arts and Performing Arts": "Visual & performing arts",
            "History and Philosophy of Science": "History & philosophy of science",
            "General Arts and Humanities": "General arts & humanities"}


def arts_subfields(ax, field_within, field_immunology):
    """Fig. 4b (A9d): for each arts-and-humanities subfield, the mean distance between its researchers and to
    immunologists (both directions averaged), as dumbbells; dashed, the whole field's values from panel a."""
    d = json.loads((RESULTS / "a9d_arts_subfields.json").read_text())
    rows = sorted(d["subfields"], key=lambda r: r["within"])
    assert all(r["within"] > r["immunology_both_directions"] for r in rows)
    y = np.arange(len(rows))
    for yi, r in zip(y, rows):
        ax.plot([r["immunology_both_directions"], r["within"]], [yi, yi], color=AXIS, lw=1.2, zorder=1,
                solid_capstyle="butt")
    ax.scatter([r["within"] for r in rows], y, s=22, color=C["contrast"], edgecolor="white", linewidth=0.6, zorder=3)
    ax.scatter([r["immunology_both_directions"] for r in rows], y, s=22, color=C["main"], edgecolor="white", linewidth=0.6,
               zorder=3)
    for v in (field_within, field_immunology):
        ax.axvline(v, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=0)
    ax.set_yticks(y, [SUBFIELD.get(r["subfield"], r["subfield"]) for r in rows], fontsize=7)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(4.5, 5.75)
    ax.set_xticks([4.5, 5.0, 5.5])
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("mean coauthor steps", fontsize=7, labelpad=2)
    top = rows[-1]
    ax.annotate("to each other", (top["within"], y[-1]), xytext=(0, 7), textcoords="offset points", ha="center",
                fontsize=7, color=INK2, zorder=4, bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ax.annotate("to an\nimmunologist", (top["immunology_both_directions"], y[-1]), xytext=(0, 7),
                textcoords="offset points", ha="center", va="bottom", fontsize=7, color=INK2, linespacing=1.0)
    ax.text(5.73, 0, "dashed: the whole field", ha="right", va="center", fontsize=7, color=MUTED, zorder=4,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ax.set_title(r"$\mathbf{b}$   arts & humanities, by subfield", loc="left", pad=16, fontsize=8)
    ax.patch.set_alpha(0)


def field_distances():
    """Fig. 4: (a) mean coauthor steps between researchers of the 26 fields (A9), as a lower triangle in cluster
    order, the fields named along the diagonal; (b) the arts and humanities by subfield (A9d), in the empty upper
    triangle."""
    from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
    from scipy.spatial.distance import squareform
    d = json.loads((RESULTS / "a9_structure.json").read_text())["field"]
    names, m = d["names"], np.array(d["mean_distance"])
    sym = (m + m.T) / 2
    off = sym.copy()
    np.fill_diagonal(off, 0)
    cond = squareform(off, checks=False)
    order = leaves_list(optimal_leaf_ordering(linkage(cond, "average"), cond))
    if sym[order[0], order[0]] > sym[order[-1], order[-1]]:
        order = order[::-1]                                 # the tightest block at the top
    S = sym[np.ix_(order, order)]
    n = len(order)
    upper = np.triu(np.ones((n, n), bool), k=1)
    fig, ax = plt.subplots(figsize=(W2, 142 * MM), layout="constrained")
    cmap = RAMP.reversed()
    lo, hi = np.floor(S.min() * 10) / 10 - 0.1, np.ceil(S.max() * 10) / 10
    mesh = ax.pcolormesh(np.ma.masked_where(upper, S), cmap=cmap, vmin=lo, vmax=hi, edgecolors="white", linewidth=0.6)
    W, H = n + 8.2, n
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    for i, k in enumerate(order):
        ax.text(i + 1.25, i + 0.5, short(names[k]), va="center", ha="left", fontsize=7.5, color=INK2)
    # selective values: the closest and farthest pairs of fields, the tightest and loosest field
    lower = np.tril(np.ones((n, n), bool), k=-1)
    marks = [np.unravel_index(np.where(lower, S, np.inf).argmin(), S.shape),
             np.unravel_index(np.where(lower, S, -np.inf).argmax(), S.shape),
             (np.diag(S).argmin(),) * 2, (np.diag(S).argmax(),) * 2]
    for i, j in marks:
        v = S[i, j]
        rgb = cmap((v - lo) / (hi - lo))[:3]
        light = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2] > 0.55
        ax.text(j + 0.5, i + 0.52, f"{v:.1f}", ha="center", va="center", fontsize=7, color=INK if light else "white")
    ax.text(0, -0.6, r"$\mathbf{a}$   the 26 fields", ha="left", va="bottom", fontsize=8, color=INK)

    def frac(x0, y_top, w, h):                              # cells (rows down from the top) -> axes fraction
        return [x0 / W, 1 - (y_top + h) / H, w / W, h / H]

    cax = ax.inset_axes(frac(22.0, 12.6, 11.4, 0.42))
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cb.set_ticks(np.arange(np.ceil(lo * 2) / 2, hi + 0.01, 0.5))
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, pad=2)
    cax.set_title("mean coauthor steps between their researchers", fontsize=7.5, color=INK2, pad=4)
    arts, immuno = names.index("Arts and Humanities"), names.index("Immunology and Microbiology")
    arts_subfields(ax.inset_axes(frac(25.6, 0.6, 8.4, 9.0)), sym[arts, arts], sym[arts, immuno])
    save(fig, "fig_field_distances")


def national_heatmap():
    """Fig. 6: national-community shares (A5), separating shares below 0.5% in every shown field."""
    cells = pd.read_csv(CACHE / "a5_field_country.csv")
    known = pq.read_table(CACHE / "node_attrs.parquet", columns=["country"]).column("country").to_pandas()
    big = known.value_counts().head(14).index                 # the 14 countries with the most researchers
    piv = cells.pivot(index="field", columns="country", values="own_national_share").reindex(columns=big)
    piv = piv[piv.notna().sum(axis=1) >= 9]
    if piv.empty:
        raise ValueError("national heatmap has no fields meeting the nine-country coverage rule")
    some, low, missing = national_country_groups(piv)
    print(f"national_heatmap: {len(piv)} fields; below 0.5% in every shown field: {low}; no data: {missing}")
    sort_by = some or low
    if sort_by:
        piv = piv.loc[piv[sort_by].mean(axis=1).sort_values(ascending=False).index]
    groups = [(label, group) for label, group in
              (("at least 0.5% in a shown field", some), ("below 0.5% throughout", low), ("no eligible cells", missing))
              if group]
    cols = []
    for _, group in groups:
        if cols:
            cols.append(None)
        cols.extend(group)
    M = np.column_stack([piv[c].to_numpy() if c else np.full(len(piv), np.nan) for c in cols])
    nr, nc = M.shape
    # below a gap, every field at once: each country's cells (shown or not) pooled by their researchers
    pooled = cells.assign(w=cells.n * cells.own_national_share).groupby("country")[["w", "n"]].sum()
    pooled = pooled.w / pooled.n
    M = np.vstack([M, np.full(nc, np.nan), [pooled.get(c, np.nan) if c else np.nan for c in cols]])
    labels = [short(f) for f in piv.index] + ["", "All fields"]
    print("national_heatmap all fields:", {c: round(float(pooled.get(c, np.nan)), 3) for c in cols if c})
    fig, ax = plt.subplots(figsize=(W2, 120 * MM), layout="constrained")
    mesh = ax.pcolormesh(np.ma.masked_invalid(M), cmap=BLUES, vmin=0, vmax=1, edgecolors="white", linewidth=0.8)
    ax.set_ylim(len(labels), 0)
    ax.set_xlim(0, nc)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.xaxis.tick_top()
    ax.set_xticks([j + 0.5 for j, c in enumerate(cols) if c], [c for c in cols if c])
    ax.set_yticks(np.arange(len(labels)) + 0.5, labels)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.tick_params(length=0, pad=3)
    for i in range(len(labels)):
        for j, c in enumerate(cols):
            v = M[i, j]
            if c in some and not np.isnan(v):
                ax.text(j + 0.5, i + 0.53, f"{v * 100:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.55 else INK2)
    for lab, group in groups:
        j0, j1 = cols.index(group[0]), cols.index(group[-1]) + 1
        ax.annotate("", xy=(j0 + 0.1, -1.35), xytext=(j1 - 0.1, -1.35), xycoords="data", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.6))
        ax.text((j0 + j1) / 2, -1.55, lab, ha="center", va="bottom", fontsize=8, color=INK2, clip_on=False)
    if low:
        j0 = cols.index(low[0])
        ax.text(j0 + len(low) / 2, nr / 2, "<0.5% in every\nshown field", ha="center", va="center",
                fontsize=8, color=MUTED)
    cb = fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.035, pad=0.02, aspect=50, shrink=0.6)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1])
    percent_axis(cb.ax.xaxis)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    cb.set_label("share of the country's researchers in the field who sit in its national community", color=INK2)
    save(fig, "fig_national_heatmap")


def affinity_log_scale(values):
    """Symmetric around one, with powers-of-two bounds enclosing every affinity."""
    positive = np.asarray(values)[np.isfinite(values) & (np.asarray(values) > 0)]
    power = max(1, int(np.ceil(np.max(np.abs(np.log2(positive)))))) if positive.size else 1
    stride = max(1, int(np.ceil(power / 3)))
    exponents = sorted(set(range(-power, power + 1, stride)) | {0, power})
    ticks = [2.0 ** p for p in exponents]
    labels = [f"1/{2 ** -p}" if p < 0 else str(2 ** p) for p in exponents]
    return 2.0 ** -power, 2.0 ** power, ticks, labels


def countries():
    """Fig. 7: collaboration abroad by country (share of link weight) and the international affinity of country pairs
    (B4)."""
    d = json.loads((RESULTS / "b4_countries.json").read_text())
    names = d["countries"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 98 * MM), layout="constrained",
                                 gridspec_kw={"width_ratios": [0.72, 1.28]})
    ab = sorted(d["share_abroad"].items(), key=lambda kv: kv[1])
    y = np.arange(len(ab))
    a1.barh(y, [v for _, v in ab], 0.62, color=S1)
    for yi, (_, v) in zip(y, ab):
        a1.text(v + 0.015, yi, f"{v:.0%}", va="center", fontsize=7, color=INK2)
    a1.set_yticks(y, [c for c, _ in ab], fontsize=7.5)
    a1.set_ylim(-0.7, len(ab) - 0.3)
    a1.set_xlim(0, 1)
    a1.set_xticks([0, 0.25, 0.5, 0.75, 1])
    percent_axis(a1.xaxis)
    a1.tick_params(axis="y", length=0)
    a1.grid(axis="y", visible=False)
    a1.set_xlabel("share of collaboration with other countries")
    title(a1, "a", "collaboration abroad")
    region_order = country_region_blocks(names)
    labels = [c for block in region_order for c in block]
    keep = [names.index(c) for c in labels]
    m = np.array(d["intl_affinity_matrix"])[np.ix_(keep, keep)].copy()
    np.fill_diagonal(m, np.nan)
    k = len(labels)
    lo, hi, ticks, ticklabels = affinity_log_scale(m)
    im = a2.pcolormesh(np.ma.masked_invalid(m), cmap=DIVERGING, norm=LogNorm(vmin=lo, vmax=hi), edgecolors="white",
                       linewidth=0.3)                       # vector cells: no transparent pixels in PDF or EPS
    a2.set_xlim(0, k)
    a2.set_ylim(k, 0)
    a2.set_aspect("equal")
    a2.grid(False)
    a2.set_xticks(np.arange(k) + 0.5, labels, rotation=90, fontsize=7)
    a2.set_yticks(np.arange(k) + 0.5, labels, fontsize=7)
    a2.tick_params(length=0, pad=2)
    for sp in a2.spines.values():
        sp.set_visible(False)
    edges = np.cumsum([len(block) for block in region_order])[:-1]
    for e in edges:
        a2.axhline(e, color="white", lw=1.6)
        a2.axvline(e, color="white", lw=1.6)
    title(a2, "b", "international affinity of country pairs (1 = random mixing)")
    cb = fig.colorbar(im, ax=a2, fraction=0.045, pad=0.02, aspect=30)
    cb.ax.yaxis.set_major_locator(FixedLocator(ticks))
    cb.ax.yaxis.set_major_formatter(FixedFormatter(ticklabels))
    cb.ax.yaxis.set_minor_locator(NullLocator())
    cb.ax.tick_params(length=0)
    cb.outline.set_visible(False)
    save(fig, "fig_countries")


def link_glow(img, at, links, chunk=4000, points=1600):
    """The links as gentle arcs, accumulated as light (length of arc per pixel), tone-mapped and screen-blended over
    the map: bundles glow where many long links run together."""
    from scipy.ndimage import gaussian_filter
    h, w = img.shape[:2]
    x1, y1 = at(links.x1.to_numpy(np.float64), links.y1.to_numpy(np.float64))
    x2, y2 = at(links.x2.to_numpy(np.float64), links.y2.to_numpy(np.float64))
    flip = x1 > x2                                          # every arc runs left to right and bows upward
    p0 = np.stack([np.where(flip, x2, x1), np.where(flip, y2, y1)], -1)
    p2 = np.stack([np.where(flip, x1, x2), np.where(flip, y1, y2)], -1)
    chord = p2 - p0
    p1 = (p0 + p2) / 2 + 0.2 * np.stack([chord[:, 1], -chord[:, 0]], -1)
    t = np.linspace(0, 1, points)[None, :, None]
    light = np.zeros(h * w)
    for k in range(0, len(p0), chunk):
        a, b, c = p0[k:k + chunk, None], p1[k:k + chunk, None], p2[k:k + chunk, None]
        pts = (1 - t) ** 2 * a + 2 * (1 - t) * t * b + t ** 2 * c
        step = np.hypot(*(c - a).squeeze(1).T) * 1.05 / points  # arc length each point stands for
        xi, yi = np.rint(pts[..., 0]).astype(np.int64), np.rint(pts[..., 1]).astype(np.int64)
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        light += np.bincount((yi * w + xi)[ok], np.broadcast_to(step[:, None], ok.shape)[ok], minlength=h * w)
    light = gaussian_filter(light.reshape(h, w), 0.6)
    lit = light[light > 0]
    v = np.log1p(light / np.percentile(lit, 50))
    v = np.clip(v / np.percentile(v[light > 0], 99.95), 0, 1) ** 1.6
    glow = np.array([1.0, 0.82, 0.55])                      # warm gold
    base = img.astype(np.float32) / 255
    return 1 - (1 - base) * (1 - v[..., None] * glow)


def ties():
    """Fig. 9: repeated ties stay inside groups (B2c): (a) the share of links crossing communities and, among links
    with known countries at both ends, borders, by the number of works the two researchers share, for all links and
    for links formed only in teams of at most five (exact counts over every link); (b) the overlap of the two
    researchers' coauthor neighbourhoods in a random two million links, with 95% intervals, and the same without the
    coauthors of their shared works (RV ties); (c) the longest links on the map (B2b), its communities named and
    numbered as in Fig. 1."""
    rep = json.loads((RESULTS / "b2c_repeat_ties.json").read_text())
    m = map_communities(4)
    img, at, _ = stitched_map(4, dim=0.5, room_below=False)
    img = img[:m["img"].shape[0], :m["img"].shape[1]]       # Fig. 1's crop, its dust dimmed alike
    img = np.maximum(img, np.rint(m["dust"] * 0.5 * 255).astype(np.uint8))
    links = pq.read_table(CACHE / "b2_long_links.parquet").to_pandas()
    lit = link_glow(img, at, links)
    h, w = img.shape[:2]
    big = 500_000                                           # communities named or numbered here
    shown = [r for r in m["named"] if r["members"] >= big]
    rows = max(sum(field_group(r["name"]) == g for r in shown) for g in range(3))
    width, map_w, key_h = W2 / MM, 110.0, 10.8 + (rows - 1) * KEY_PITCH
    map_h = map_w * h / w
    height = 5.5 + map_h + key_h
    fig = plt.figure(figsize=(W2, height * MM))

    def axes_mm(x, y, w_, h_):                              # mm from the top left
        return fig.add_axes([x / width, 1 - (y + h_) / height, w_ / width, h_ / height])

    top_h = 0.43 * height                                   # a above b, room for b's title and the x labels
    a1, a2 = axes_mm(12.5, 6.0, 47.5, top_h), axes_mm(12.5, 14.5 + top_h, 47.5, height - 27.5 - top_h)
    a3 = axes_mm(width - map_w, 5.5, map_w, map_h)
    kx = axes_mm(width - map_w, 5.2 + map_h, map_w, key_h + 0.3)         # overlapping the map: no hairline
    groups = [r["group"] for r in rep["by_shared_works"]]
    assert [r["group"] for r in rep["small_team_links_by_shared_works"]] == groups
    x = np.arange(len(groups))

    def ends(ax, v, dy, fmt):
        ax.annotate(fmt(v[0]), (0, v[0]), xytext=(-6, dy), textcoords="offset points", ha="right", va="center",
                    fontsize=7.5, color=INK2)
        ax.annotate(fmt(v[-1]), (x[-1], v[-1]), xytext=(6, dy), textcoords="offset points", ha="left", va="center",
                    fontsize=7.5, color=INK2)

    # series named beside their lines (x, and offset in data units), clear of the other lines; the dashed small-team
    # lines are named once, in the empty lower left
    dashed = (0, (3, 2))
    for rows, key, col, style, dy, lab, at_x, lab_dy in (
            (rep["by_shared_works"], "across_countries", C["contrast"], "-", 4, "crosses a border", 3.2, 0.06),
            (rep["small_team_links_by_shared_works"], "across_countries", C["contrast2"], dashed, -4, None, 0, 0),
            (rep["by_shared_works"], "across_communities", C["main"], "-", 4, "crosses communities", 3.2, 0.096),
            (rep["small_team_links_by_shared_works"], "across_communities", C["main2"], dashed, -4, None, 0, 0)):
        v = [r[key] for r in rows]
        a1.plot(x, v, color=col, ls=style, marker="o", ms=3.2, mec="white", mew=0.6)
        ends(a1, v, dy, lambda s: f"{s:.0%}")
        if lab:
            a1.text(at_x, np.interp(at_x, x, v) + lab_dy, lab, ha="center", va="center", fontsize=7.5, color=INK2)
    a1.plot([-0.2, 0.35], [0.06, 0.06], color=MUTED, ls=dashed, lw=1.0)
    a1.text(0.45, 0.06, "small teams only", ha="left", va="center", fontsize=7.5, color=INK2)
    a1.set_ylim(0, 0.5)
    a1.set_yticks(np.arange(0, 0.51, 0.1))
    percent_axis(a1.yaxis)
    a1.set_ylabel("share of links")
    title(a1, "a", "repeated ties stay inside groups")

    rv = json.loads((RESULTS / "rv_ties.json").read_text())
    residual = rv["residual_overlap_by_shared_works"]
    overlap = {**{k: (rep[f"overlap_{k}"], rep[f"overlap_se_{k}"]) for k in ("by_shared_works", "small_team_by_shared_works")},
               "residual": ({g: residual[g]["mean"] for g in groups}, {g: residual[g]["se"] for g in groups})}
    for key, col, style, dy, lab, at_x, lab_dy in (("by_shared_works", C["main"], "-", 0, "all links", 1.5, 0.022),
                                                   ("small_team_by_shared_works", C["main2"], dashed, 0,
                                                    "small teams only", 4.6, -0.032),
                                                   ("residual", C["contrast"], (0, (1, 1.5)), -2, None, 0, 0)):
        mean, se = overlap[key]
        v = np.array([mean[g] for g in groups])
        e = 1.96 * np.array([se[g] for g in groups])
        a2.errorbar(x, v, yerr=e, color=col, ls=style, marker="o", ms=3.2, mec="white", mew=0.6, elinewidth=0.7,
                    capsize=1.6, capthick=0.7)
        ends(a2, v, dy, lambda s: f"{s:.3f}")
        if lab:
            a2.text(at_x, np.interp(at_x, x, v) + lab_dy, lab, ha="center", va="center", fontsize=7.5, color=INK2)
    # the dotted residual line is named in the empty upper left, like panel a's dashed lines
    a2.plot([-1.3, -0.8], [0.135, 0.135], color=C["contrast"], ls=(0, (1, 1.5)), lw=1.0)
    a2.text(-0.7, 0.135, "without the shared\nworks' coauthors", ha="left", va="center", fontsize=7.5, color=INK2,
            linespacing=1.15)
    a2.set_ylim(0, 0.16)
    a2.set_yticks([0, 0.05, 0.1, 0.15])
    a2.set_ylabel("neighbourhood overlap")
    a2.set_xlabel("works the two researchers share")
    title(a2, "b", "and share more of their coauthors")
    for ax in (a1, a2):
        ax.set_xlim(-1.55, len(groups) - 0.1)
        ax.set_xticks(x, groups)
        ax.grid(axis="x", visible=False)
    a1.tick_params(axis="x", labelbottom=False)

    a3.imshow(lit, interpolation="antialiased")
    a3.set_xlim(0, w)
    a3.set_ylim(h, 0)
    a3.set_axis_off()
    halo = [patheffects.withStroke(linewidth=1.8, foreground="black")]
    for r in m["fields"]:
        if r["members"] >= big:
            a3.text(*at(r["xw"], r["yw"]), FIELD_NAMES.get(r["name"], r["name"]), color="white", fontsize=7,
                    fontweight="bold", ha="center", va="center", linespacing=1.0, path_effects=halo, zorder=6)
    for number, r in enumerate(m["named"], start=1):
        if r["members"] >= big:
            number_marker(a3, *m["spots"][r["community"]], number, m["colour"][r["community"]])
    title(a3, "c", f"the longest 1% of links on the map (a random {len(links):,})")
    # the key, as Fig. 1's: the numbered communities shown here, by their main fields
    map_key(kx, m, shown, key_h + 0.3, columns=((2.0, 29.5), (31.5, 36.5), (68, map_w - 68)), title=False, sizes=False,
            footnote="† regional; numbers as in Fig. 1")
    save(fig, "fig_ties", dpi=600, eps=False)


CITATION_CLASSES = [("0", 0, 1, "uncited"), ("1-9", 1, 10, "under_10"), ("10-99", 10, 100, "10_to_99"),
                    ("100-999", 100, 1000, "100_to_999"), ("1,000+", 1000, None, "1000_plus")]


def uncited():
    """Fig. 10: the uncited periphery (B1): (a) the share of profiles with no citations and with one to nine, by the
    script of the name; (b) the distribution of profiles' estimated mean distance to the 1,000 sampled sources, by
    citations, among profiles every source reaches (violins: density; bars: quartiles; dots: medians), with the share
    not reached."""
    d = json.loads((RESULTS / "b1_uncited.json").read_text())
    label = {"Japanese kanji/kana": "Japanese script", "Hangul": "Korean (Hangul)", "Latin": "Latin / other"}
    shades = [C["contrast"], C["contrast2"], *MAGNITUDE]    # the two uncited-or-nearly classes, then blues by citations
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 66 * MM), layout="constrained",
                                 gridspec_kw={"width_ratios": [1.2, 1]})
    scripts = sorted(d["by_script"].items(), key=lambda kv: kv[1]["uncited"])
    y = np.arange(len(scripts))
    zero = np.array([v["uncited"] for _, v in scripts])
    few = np.array([v["under_10"] for _, v in scripts])
    a1.barh(y, zero, 0.62, color=shades[0], edgecolor="white", linewidth=0.6, label="no citations")
    a1.barh(y, few - zero, 0.62, left=zero, color=shades[1], edgecolor="white", linewidth=0.6, label="1-9 citations")
    for yi, (_, v) in zip(y, scripts):
        z, f = v["uncited"], v["under_10"] - v["uncited"]
        a1.text(z / 2, yi, f"{z:.0%}", ha="center", va="center_baseline", fontsize=7.5, color="white")
        a1.text(z + f / 2, yi, f"{f:.0%}", ha="center", va="center_baseline", fontsize=7.5, color=INK)
        a1.text(v["under_10"] + 0.02, yi, f"n = {v['n']:,}", ha="left", va="center_baseline", fontsize=7, color=MUTED)
    a1.set_yticks(y, [label.get(k, k) for k, _ in scripts])
    a1.tick_params(axis="y", length=0)
    a1.set_xlim(0, 1.2)
    a1.set_xticks([0, 0.25, 0.5, 0.75, 1])
    percent_axis(a1.xaxis)
    a1.set_ylim(-0.6, len(scripts) - 0.1)
    a1.set_xlabel(f"share of profiles ({MIN_WORKS}+ works)")
    a1.grid(axis="y", visible=False)
    a1.legend(loc="lower right", handlelength=1.0, handleheight=0.8)
    title(a1, "a", "citations by the script of the name")

    closeness = np.asarray(np.load(CACHE / "closeness_1000.npy", mmap_mode="r"))
    cited = pq.read_table(CACHE / "node_attrs.parquet", columns=["cited"]).column("cited").to_numpy()
    lo, hi = 3.4, 7.6
    edges = np.arange(lo, hi + 1e-9, 0.02)
    mid = (edges[:-1] + edges[1:]) / 2
    # under the violins, a small table: the class, its median (the dot) and the share not reached
    rows = (("citations:", 8, INK2, -3.0), ("median:", 7.5, INK2, -14.5), ("not reached:", 7, MUTED, -25.5))

    def cell(x, text, row, ha="center"):
        name, size, colour, dy = rows[row]
        a2.annotate(text, (x, 0), xycoords=("data", "axes fraction"), xytext=(0, dy), textcoords="offset points",
                    ha=ha, va="top", fontsize=size, color=colour)

    for i, ((lab, c0, c1, key), col) in enumerate(zip(CITATION_CLASSES, shades)):
        v = closeness[(cited >= c0) if c1 is None else (cited >= c0) & (cited < c1)]
        reached = np.isfinite(v)
        v = v[reached]
        med = float(np.median(v))
        assert abs(med - d["position"][key]["median_distance_to_all"]) < 1e-9, lab
        q1, q3 = np.quantile(v, [0.25, 0.75])
        dens = gaussian_filter1d(np.histogram(v, bins=edges)[0].astype(float), 2.5)
        half = 0.42 * dens / dens.max()
        a2.fill_betweenx(mid, i - half, i + half, color=col, linewidth=0)
        a2.plot([i, i], [q1, q3], color=INK, lw=2.2, solid_capstyle="butt", zorder=3)
        a2.scatter([i], [med], s=16, color="white", edgecolor=INK, linewidth=0.8, zorder=4)
        for row, text in enumerate((lab, f"{med:.2f}", f"{1 - reached.mean():.0%}")):
            cell(i, text, row)
    for row, (name, *_) in enumerate(rows):
        cell(-0.62, name, row, ha="right")
    a2.set_xticks(range(len(CITATION_CLASSES)))
    a2.tick_params(axis="x", length=0, labelbottom=False)
    a2.set_xlim(-0.6, len(CITATION_CLASSES) - 0.4)
    a2.set_ylim(lo, hi)
    a2.set_yticks(np.arange(3.5, hi, 0.5))
    a2.grid(axis="x", visible=False)
    a2.set_ylabel("mean coauthor steps\nto the sampled researchers")
    title(a2, "b", "distance to everyone, by citations")
    save(fig, "fig_uncited")


def affiliations():
    """Fig. A1: share of profiles listing at least x institutions in their busiest year (A6)."""
    a = pd.read_csv(CACHE / "a6_affiliations.csv")
    a = a[a.alive]
    top = a[a.group == "top"].sort_values("degree", ascending=False)
    control = a[a.group == "control"]
    groups = [(f"{min(5000, len(top)):,} highest-degree resolved profiles", top.head(5000).peak_insts, C["main"]),
              (f"all {len(top):,} resolved high-degree profiles", top.peak_insts, C["main2"]),
              (f"controls (50-300 coauthors; n={len(control):,})", control.peak_insts, C["contrast"])]
    fig, ax = plt.subplots(figsize=(W15, 62 * MM), layout="constrained")
    xs = np.unique(np.round(np.logspace(0, 3, 240)))
    for lab, s, col in groups:
        v = s.clip(lower=0).to_numpy()
        ax.plot(xs, [(v >= x).mean() for x in xs], color=col, label=lab)
    ax.axvline(25, color=AXIS, lw=0.8, zorder=1)
    ax.annotate("25 in one year", (25, 1.0), xytext=(4, 0), textcoords="offset points", ha="left", va="center",
                fontsize=7.5, color=INK2, bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ax.set_xscale("log")
    ax.set_xlim(0.9, 1100)
    ax.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
    ax.xaxis.set_major_formatter(FixedFormatter(["1", "10", "100", "1,000"]))
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    percent_axis(ax.yaxis)
    ax.set_xlabel("distinct institutions listed in the busiest single year (at least)")
    ax.set_ylabel("share of profiles")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.59))   # its rows between the 50% and 75% lines
    save(fig, "fig_affiliations")


def twins():
    """Fig. A2: candidate script twins with at least t coauthor-name matches.
    Candidate and decoy rates use their respective tested-case denominators."""
    scripts = [("zh", "Chinese characters"), ("ja", "Japanese script")]
    fig, axes = plt.subplots(1, 2, figsize=(W15, 60 * MM), layout="constrained", sharey=True)
    for ax, (code, label) in zip(axes, scripts):
        d = json.loads((RESULTS / f"a6_twins_{code}.json").read_text())
        used = min(int(t) for t, r in d["pass_rate"].items() if r["decoys"] <= 0.05)
        ts = sorted(int(t) for t in d["pass_rate"])
        x = np.arange(len(ts))
        ax.axvline(ts.index(used), color=AXIS, lw=0.8, zorder=1)
        ax.annotate(f"candidate threshold {used}+", (ts.index(used), 1.0), xytext=(4, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=7, color=INK2, bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
        ax.plot(x, [d["pass_rate"][str(t)]["twins"] for t in ts], color=C["main"], marker="o", ms=3.6, mec="white", mew=0.7,
                label="namesake (same full name)")
        ax.plot(x, [d["pass_rate"][str(t)]["decoys"] for t in ts], color=C["contrast"], marker="o", ms=3.6, mec="white",
                mew=0.7, label="decoy (same surname only)")
        ax.set_xticks(x, ts)
        ax.set_xlabel("coauthor-name matches, at least")
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        percent_axis(ax.yaxis)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{label} ({d['tested']:,} tested)", loc="left", pad=6)
    axes[0].set_ylabel("share of tested cases")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2)
    save(fig, "fig_twins")


if __name__ == "__main__":
    from fig_concepts import hub_definitions
    for f in (map_overview, distances, far_edge, field_distances, fields_on_map, national_heatmap, countries,
              hub_definitions, ties, uncited, affiliations, twins):
        use_colours(CHOSEN.get(f.__name__, "A"))
        f()
    print("figures:", sorted(p.name for p in FIGURES.glob("*.png")))
