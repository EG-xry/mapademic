"""Fig. 8: five definitions of a hub, drawn on one toy coauthorship network (Section 6).

The toy network is built as the real one is, from works: a work with n authors adds 1/(n-1) to the weight of each link
among them, so a researcher's strength is the number of works they share. Its 27 researchers echo Section 6: a
consortium around its leader (most coauthors), a trio of compilers who co-sign many short data entries (most shared
works), a peripheral community that reaches everyone else only through its gatekeeper (betweenness), a researcher who
works with four groups (closest to everyone) and, in one of two small groups, the author of a landmark method (most
cited). Every measure is computed with networkx; the figure asserts that each panel highlights the unique top by its
definition, that the five are different researchers, and that no text collides or clips.
"""
import itertools
import math

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from matplotlib.text import Text
from matplotlib.transforms import ScaledTranslation

from figures import AXIS, BLUES, INK, INK2, MM, S1, W2, save, title


def ring(cx, cy, r, names, start, step):
    """Points on a circle around (cx, cy): the first at angle start, the others step degrees on."""
    return {n: (cx + r * math.cos(math.radians(start + k * step)), cy + r * math.sin(math.radians(start + k * step)))
            for k, n in enumerate(names)}


MEMBERS = [f"c{i}" for i in range(1, 8)]
WORKS = (                                                  # (authors, number of such works)
    [(("H", a, b), 1) for a, b in zip(MEMBERS, MEMBERS[1:])]          # a consortium: H writes with its member groups
    + [(("b1", "b2", "b3"), 1), (("b2", "b4"), 1), (("b3", "b4"), 1),  # two small groups; K is the most cited
       (("d1", "K", "d3"), 1), (("K", "d4"), 1), (("d3", "d4"), 1)]
    + [(("S", "s1"), 15), (("S", "s2"), 8), (("S", "s1", "s2"), 6)]   # compilers: 29 short entries, two partners
    + [((a, b), 1) for a, b in (("G", "p1"), ("p1", "p2"), ("p2", "p4"), ("p4", "p6"), ("p6", "p5"), ("p5", "p3"),
                                ("p3", "p1"))]                         # a peripheral community behind its gatekeeper G
    + [((a, b), 1) for a, b in (("X", "H"), ("X", "b1"), ("X", "d1"), ("X", "s2"),   # X works with four groups
                                ("c6", "b2"), ("c2", "s1"), ("s2", "K"), ("G", "b3"), ("G", "d3"))])
CITATIONS = {"H": 1400, "c1": 260, "c2": 410, "c3": 180, "c4": 520, "c5": 230, "c6": 340, "c7": 150, "X": 900,
             "b1": 380, "b2": 290, "b3": 160, "b4": 95, "K": 5200, "d1": 450, "d3": 200, "d4": 120,
             "S": 310, "s1": 190, "s2": 220, "G": 60, "p1": 20, "p2": 35, "p3": 8, "p4": 0, "p5": 12, "p6": 3}
POS = {"H": (4.5, 10.0), **ring(4.5, 10.0, 1.75, MEMBERS, -45, 45),   # spokes 45 degrees apart, the eighth to X
       "X": (4.5, 6.6), "s1": (7.8, 9.25), "S": (8.6, 7.85), "s2": (7.05, 7.2),
       "b1": (3.0, 5.15), "b2": (1.45, 6.6), "b3": (2.5, 3.35), "b4": (1.0, 4.55),
       "d1": (6.0, 5.15), "K": (7.55, 5.75), "d3": (6.5, 3.35), "d4": (8.1, 4.35),
       "G": (4.5, 2.45), "p1": (4.5, 1.3), "p2": (5.58, 0.8), "p4": (5.58, -0.2), "p6": (4.5, -0.7),
       "p5": (3.42, -0.2), "p3": (3.42, 0.8)}
FOCUS = {"degree": "H", "strength": "S", "betweenness": "G", "closeness": "X", "citations": "K"}
XLIM, YLIM = (0.0, 9.3), (-1.2, 13.2)
FOOT = -10                                               # footer baseline, points below the network

NODE = "#b4b3aa"                                         # everyone else recedes: warm gray, white ring
TINT = BLUES(0.3)                                        # the top researcher's coauthors
FAR = HULL = "#f0efec"                                   # 4+ steps away (d); the peripheral community (c)
D_NODE, D_FOCUS, D_BADGE = 5.2, 8.0, 8.2                 # node diameters, points
LW, LW_FOCUS = 0.6, 1.1                                  # link widths, points


def toy():
    g = nx.Graph()
    for authors, k in WORKS:
        for u, v in itertools.combinations(authors, 2):
            g.add_edge(u, v, weight=g.get_edge_data(u, v, {"weight": 0})["weight"] + k / (len(authors) - 1))
    nx.set_node_attributes(g, CITATIONS, "citations")
    return g


def measures(g):
    """Each definition: values per researcher, and whether higher ranks first."""
    mean_steps = {u: sum(nx.single_source_shortest_path_length(g, u).values()) / (len(g) - 1) for u in g}
    return {"degree": (dict(g.degree()), True), "strength": (dict(g.degree(weight="weight")), True),
            "betweenness": (nx.betweenness_centrality(g), True), "closeness": (mean_steps, False),
            "citations": (dict(g.nodes(data="citations")), True)}


class Panel:
    """One drawing of the network; remembers its nodes and texts for the collision check."""

    def __init__(self, ax, g):
        self.ax, self.g, self.discs, self.texts = ax, g, [], []

    def links(self, pairs, color, width, z):
        pairs = list(pairs)
        widths = [width(u, v) if callable(width) else width for u, v in pairs]
        self.ax.add_collection(LineCollection([(POS[u], POS[v]) for u, v in pairs], colors=color, linewidths=widths,
                                              capstyle="round", zorder=z))

    def nodes(self, names, color, size, z, ring_width=0.6, edgecolor="white"):
        names = list(names)
        sizes = [size(u) if callable(size) else size for u in names]
        self.ax.scatter([POS[u][0] for u in names], [POS[u][1] for u in names], s=[d * d for d in sizes],
                        c=[color(u) if callable(color) else color for u in names], edgecolors=edgecolor,
                        linewidths=ring_width, zorder=z)
        self.discs += [(self.ax.transData, POS[u], d + ring_width, u) for u, d in zip(names, sizes)]

    def network(self, focus, tint=(), width=LW):
        """Hairline links and gray researchers; the focus and its links in blue, the tinted researchers light blue."""
        g = self.g
        self.links([uv for uv in g.edges() if focus not in uv], AXIS, width, 1)
        self.links([uv for uv in g.edges() if focus in uv], S1,
                   lambda u, v: max(LW_FOCUS, width(u, v) if callable(width) else width), 2)
        self.nodes([u for u in g if u != focus], lambda u: TINT if u in tint else NODE, D_NODE, 3)
        self.nodes([focus], S1, D_FOCUS, 4, ring_width=0.8)

    def ground(self, names, pad=0.5):
        """A soft ground under a group: the polygon through its members (in angular order), stroked pad units wide
        with round joins."""
        cx = sum(POS[u][0] for u in names) / len(names)
        cy = sum(POS[u][1] for u in names) / len(names)
        pts = sorted((POS[u] for u in names), key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
        width = 2 * pad * (self.ax.transData.transform((1, 0))[0] - self.ax.transData.transform((0, 0))[0])
        self.ax.add_patch(Polygon(pts, closed=True, facecolor=HULL, edgecolor=HULL, joinstyle="round",
                                  linewidth=width * 72 / self.ax.figure.dpi, zorder=0))

    def heading(self, letter, name, definition, value):
        """Title in two lines (letter and name, then the definition) and, under the network, the top researcher's
        value beside a blue dot."""
        ax, fig = self.ax, self.ax.figure
        title(ax, letter, name)
        self.texts += [t for t in ax.findobj(Text) if t.get_text() == ax.get_title(loc="left")]
        self.texts.append(ax.annotate(definition, (0, 1), xycoords="axes fraction", xytext=(0, -3.5),
                                      textcoords="offset points", ha="left", va="baseline", fontsize=7.5, color=INK2))
        dot = ax.transAxes + ScaledTranslation(2.75 / 72, (FOOT + 2.6) / 72, fig.dpi_scale_trans)
        ax.scatter([0], [0], s=5.5 ** 2, c=S1, linewidths=0, transform=dot, clip_on=False, zorder=5)
        self.discs.append((dot, (0, 0), 5.5, None))
        self.texts.append(ax.annotate(value, (0, 0), xycoords="axes fraction", xytext=(8, FOOT),
                                      textcoords="offset points", ha="left", va="baseline", fontsize=7.5, color=INK))


def assert_clear(fig, panels, clearance=1.5, between=8):
    """Nothing collides or clips: every text keeps clearance (points) from every other text and from every node it
    does not label, texts of neighbouring panels stay apart by `between` (points) so that they never read as one
    line, and every text and node lies inside the figure (a node inside its own panel)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    px = fig.dpi / 72
    boxes = [(t, t.get_window_extent(r)) for p in panels for t in p.texts]
    owner = {id(t): i for i, p in enumerate(panels) for t in p.texts}
    for (t1, b1), (t2, b2) in itertools.combinations(boxes, 2):
        pad = clearance if owner[id(t1)] == owner[id(t2)] else between
        assert not b1.padded(pad * px).overlaps(b2), f"{t1.get_text()!r} is too close to {t2.get_text()!r}"
    for t, b in boxes:
        assert fig.bbox.contains(b.x0, b.y0) and fig.bbox.contains(b.x1, b.y1), f"{t.get_text()!r} is clipped"
    for p in panels:
        for transform, xy, d, u in p.discs:
            x, y = transform.transform(xy)
            rad = d / 2 * px
            frame = fig.bbox if u is None else p.ax.bbox             # nodes are clipped at their panel's edge
            assert frame.contains(x - rad, y - rad) and frame.contains(x + rad, y + rad), f"{u} is clipped"
            for t, b in boxes:
                if u is not None and t in p.texts and getattr(t, "node", None) == u:
                    continue                                # a badge sits on its own node
                gap = math.hypot(max(b.x0 - x, 0, x - b.x1), max(b.y0 - y, 0, y - b.y1)) - rad
                assert gap > clearance * px, f"{t.get_text()!r} is too close to {u or 'a footer dot'}"


def hub_definitions():
    """Fig. 8: the toy network five times, each panel highlighting the top researcher by one definition."""
    g = toy()
    tops = {}
    for name, (values, higher) in measures(g).items():
        best = FOCUS[name]
        runner = (max if higher else min)(v for u, v in values.items() if u != best)
        assert (values[best] > runner) if higher else (values[best] < runner), f"{name}: {best} is not the unique top"
        tops[name] = (best, values[best], runner)
    assert len({best for best, _, _ in tops.values()}) == 5, "the five definitions must pick five researchers"
    for name, (best, v, runner) in tops.items():
        print(f"{name:12s} {best:2s} {v:9.3f}   runner-up {runner:9.3f}")

    pw, gap, side, foot, head = 32.8 * MM, 2.0 * MM, 1.0 * MM, 5.4 * MM, 5.4 * MM
    ph = pw * (YLIM[1] - YLIM[0]) / (XLIM[1] - XLIM[0])
    fh = foot + ph + head
    fig = plt.figure(figsize=(W2, fh))
    panels = []
    for i in range(5):
        ax = fig.add_axes([(side + i * (pw + gap)) / W2, foot / fh, pw / W2, ph / fh])
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_aspect("equal")
        ax.set_axis_off()
        panels.append(Panel(ax, g))
    a, b, c, d, e = panels

    # a. degree: the leader's links to its coauthors
    leader = FOCUS["degree"]
    a.network(leader, tint=set(g[leader]))
    a.heading("a", "degree", "most coauthors", f"{tops['degree'][1]:.0f} coauthors")

    # b. strength: link width grows with weight (every link, so a weight-3 link is wider than a weight-1 one), the
    # shared works discounted for team size
    compiler = FOCUS["strength"]
    b.network(compiler, tint=set(g[compiler]), width=lambda u, v: 0.3 + 0.16 * g[u][v]["weight"])
    b.heading("b", "strength", "most shared works",
              f"{tops['strength'][1]:.0f} works, {g.degree(compiler)} coauthors")

    # c. betweenness: the gatekeeper's links carry every path out of the peripheral community
    gatekeeper = FOCUS["betweenness"]
    community = nx.node_connected_component(g.subgraph(set(g) - {gatekeeper}), "p1")
    assert all(set(g[u]) <= community | {gatekeeper} for u in community)      # its only way out is the gatekeeper
    c.ground(community)
    c.network(gatekeeper, tint=set(g[gatekeeper]))
    c.heading("c", "betweenness", "most often in between", f"on {tops['betweenness'][1]:.0%} of shortest paths")

    # d. closeness: coauthor steps from the most central researcher to everyone else
    central = FOCUS["closeness"]
    steps = nx.single_source_shortest_path_length(g, central)
    d.links([uv for uv in g.edges() if central not in uv], AXIS, LW, 1)
    d.links([uv for uv in g.edges() if central in uv], S1, LW_FOCUS, 2)
    shade = {1: BLUES(0.3), 2: BLUES(0.15), 3: BLUES(0.05)}
    others = [u for u in g if u != central]
    d.nodes(others, lambda u: shade.get(steps[u], FAR), D_BADGE, 3)
    for u in others:
        t = d.ax.text(*POS[u], str(steps[u]), ha="center", va="center_baseline", fontsize=7, color=INK, zorder=4)
        t.node = u
        d.texts.append(t)
    d.nodes([central], S1, D_BADGE, 4, ring_width=0.8)
    d.heading("d", "closeness", "fewest steps to everyone", f"{tops['closeness'][1]:.2f} steps on average")

    # e. citations, which are not a network measure: node area by citations (above a floor that keeps dots visible)
    cited = FOCUS["citations"]

    def area(u):
        return max(3.0, 14.0 * math.sqrt(CITATIONS[u] / CITATIONS[cited]))

    e.links(g.edges(), AXIS, LW, 1)
    e.nodes([u for u in g if u != cited], NODE, area, 3)
    e.nodes([cited], S1, area, 4, ring_width=0.8)
    e.heading("e", "citations", "not a network measure", f"{CITATIONS[cited]:,} citations")

    assert_clear(fig, panels)
    save(fig, "fig_hub_definitions")
    return tops


if __name__ == "__main__":
    hub_definitions()
