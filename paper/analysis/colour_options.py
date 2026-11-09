"""Redraw the comparison figures (Figs. 2, 3, 4, 9, 10, A1 and A2) under every colour option in figures.OPTIONS, for
choosing the final palette: paper/figures/colour_options/Fig<n>/<option>_<name>.png, with a README.md listing the
options and their palette checks. Not part of the manuscript build; figures.py draws the manuscript with Eric's
choices (figures.CHOSEN).

Usage: python paper/analysis/colour_options.py
"""
import matplotlib.pyplot as plt

import figures as F

OUT = F.FIGURES / "colour_options"
DRAW = (F.distances, F.far_edge, F.field_distances, F.ties, F.uncited, F.affiliations, F.twins)
CHECKS = {  # dataviz validate_palette.js, light mode, all pairs: worst normal-vision Delta E of the pairs drawn together
    "A": "navy-teal 22.7, navy-pale blue 49.8, pale blue-teal 28.9, pale blue-light teal 17.2; teal shades 12.1 "
         "(labelled where used together); pale blue 1.7:1 on white",
    "B": "blue-brown 24.2, blue-navy 21.6, navy-brown 35.2; brown shades 10.3 (labelled where used together); "
         "light brown 2.6:1 on white",
    "C": "navy-brown 35.2, navy-mid blue 31.3, mid blue-brown 21.3; brown shades 10.3; light brown 2.6:1 on white",
    "D": "blue-teal 8.9: below the floor of 15, hard to tell apart even with full colour vision (shown for comparison)",
}
current = {}


def preview(fig, name, dpi=300, eps=True):
    fig.canvas.draw()
    for problem in F.collisions(fig):
        print(f"{name} [{current['key']}]: {problem}")
    folder = OUT / f"Fig{F.NUMBER[name]}"
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{current['key']}_{current['slug']}.png", dpi=200)
    plt.close(fig)


F.save = preview
for key, (label, main, main2, contrast, contrast2) in F.OPTIONS.items():
    F.use_colours(key)
    current.update(key=key, slug=label.replace(" ", "-"))
    for draw in DRAW:
        draw()
F.use_colours("A")
lines = ["# Colour options for Figs. 2, 3, 4, 9, 10, A1 and A2", "",
         "Each folder holds one figure drawn four ways. Roles: the main hue, a second shade of it for a group like the main",
         "one, the contrasting hue, and a second shade of that. The manuscript uses Eric's choices of 8 November 2026",
         "(figures.CHOSEN): B for Figs. 3, 9 and A1, and A for Figs. 2, 4, 10 and A2.", "",
         "| Option | Main | Second shade | Contrast | Contrast shade | Palette check (Delta E, normal vision; 15+ passes) |",
         "|---|---|---|---|---|---|"]
for key, (label, *colours) in F.OPTIONS.items():
    lines.append(f"| {key}, {label} | " + " | ".join(f"`{c}`" for c in colours) + f" | {CHECKS[key]} |")
lines += ["", "Fig. 4a (the heatmap) uses the same blue ramp in every option: white, #9ecae1, #4292c6, #2171b5, #08306b.",
          "Fig. 9c (the map) keeps the community colours of Fig. 1.", ""]
(OUT / "README.md").write_text("\n".join(lines))
print("colour options:", sorted(str(p.relative_to(OUT)) for p in OUT.glob("*/*.png")))
