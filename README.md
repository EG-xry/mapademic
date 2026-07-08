# mapademic

A zoomable, colorful map of the academic world: every sufficiently active
researcher is a dot, positioned by force-directed layout over the OpenAlex
coauthorship graph, colored by community. Companion project to
[academic-degree-of-separation](https://github.com/riptideiv/academic-degree-of-separation).

Design spec: `docs/superpowers/specs/2026-07-05-mapademic-design.md`

## Pipeline

    download -> extract -> filter -> edges -> [Anvil: layout -> communities] -> tiles -> index -> web/

Run stages via `python -m pipeline <stage>`. Data checkpoints live under
`$MAPADEMIC_DATA` (default `./data`, git-ignored).
