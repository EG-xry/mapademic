# Mapademic - Design Spec

Date: 2026-07-05
Status: Approved by Eric (brainstorming session, sections approved incrementally)

## What this is

A zoomable, colorful map of the academic world: every sufficiently active
researcher is a dot, positioned by force-directed layout over the coauthorship
graph, colored by detected community (which in practice means field). Inspired
by the "Map of Wikipedia" video (python-igraph, DRL layout, Leiden communities,
prerendered deep-zoom map); built on OpenAlex data instead of Wikipedia dumps.

Standalone project and repo (`mapademic`), separate from
`academic-degree-of-separation` (where Eric is second contributor, not owner).
The two cross-link in their READMEs. A shared GitHub organization is a possible
later step but requires the other repo's owner (riptideiv) to transfer their
repo; nothing in this design depends on that happening.

## Decisions (locked during brainstorming)

- Scale: top ~5-10M authors (works_count threshold, tuned to land in range)
- Edges: coauthorship only for v1; citations are a possible v2
- Rendering: prerendered raster tile pyramid (deep-zoom web map), not a live
  WebGL graph
- Interactivity v1: pan/zoom + name search + node info card + path overlay
  that reuses the existing degree-of-separation backend
- Pipeline split: M4 Max for all data and rendering stages; Purdue Anvil A100
  (user `x-egao2`) for layout + community detection only
- Name: mapademic

## Architecture overview

Batch pipeline of idempotent stages, each reading/writing Parquet checkpoints
under `DATA_DIR` (git-ignored; may live on an external drive - the heavy pass
is sequential IO so external SSD is fine, spinning HDD merely slower). Any
stage can re-run without redoing upstream work.

```
download -> filter -> edges ->  [Anvil: layout -> communities]  -> tiles -> index -> web/
   M4         M4       M4              A100 (Slurm)                 M4       M4     static
```

Repo layout:

- `pipeline/` - Python package; one CLI subcommand per stage
  (`download`, `filter`, `edges`, `layout`, `communities`, `tiles`, `index`)
- `slurm/` - Anvil job scripts (layout + Leiden stage only)
- `web/` - static MapLibre site
- `docs/` - specs, plans, run logs
- `data/` - checkpoints, git-ignored (often a symlink to external drive)

## Stage details

### 1. Data acquisition (M4 Max)

- Source: OpenAlex S3 snapshot (free, no egress; ~330GB compressed JSONL).
  Entities needed: `works` (authorship lists) and `authors`.
- Disk plan: snapshot lives on an external drive; only final small artifacts
  (nodes, coords, tiles - a few GB) need internal disk.
- Documented fallback if no drive is available: the OpenAlex BigQuery public
  dataset (the degree-of-separation repo already scaffolds BigQuery access);
  costs a few dollars of query, skips the download.

### 2. Filter + edge build (M4 Max, DuckDB/Polars)

- Node selection: authors with `works_count >= ~5`; tune threshold until node
  count lands in the 5-10M window. Keep: id, display name, works_count,
  cited_by_count, last known institution, primary field/topic.
- Edges: explode each work's authorship list into author pairs restricted to
  selected authors. Edge weight = sum over shared works of `1/(n_authors - 1)`.
- Hyperauthorship rule: works with more than 50 authors are EXCLUDED from edge
  building (they still count toward node stats). Otherwise a single 3,000-author
  CERN paper contributes ~4.5M clique edges and collapses physics into a black
  hole at the map center. This is the analogue of the video excluding
  "See also"/footnote links: connections that do not represent a real
  relationship.
- Output: `nodes.parquet` (5-10M rows), `edges.parquet` (est. 50-200M rows,
  a few GB). These two files are all that goes to Anvil.

### 3. Layout + communities (Anvil, the only GPU stage)

- Transfer the two Parquet files to Anvil scratch (scp or Globus,
  `x-egao2@anvil.rcac.purdue.edu`).
- One Slurm job, 1x A100 (Anvil GPUs are 40GB - fits this edge list), RAPIDS
  via Anvil's conda/module system.
- Primary layout: cuGraph ForceAtlas2, ~1000 iterations, degree-weighted
  repulsion; expect 2-3 tuning runs of tens of minutes each. GPU-hour budget
  is comfortable.
- Communities: GPU Leiden in the same job -> community id per node. Top ~40
  communities get distinct hues; the long tail gets muted greys.
- Fallback layout (behind a flag in the same job script): cuGraph node2vec ->
  cuML UMAP to 2D. Render both cheaply, keep the prettier one.
- Job checkpoints coordinates every N iterations, so wall-time kills lose
  nothing.
- Return trip: `coords.parquet` (~300MB) back to the Mac. Anvil is done.

### 4. Tile rendering (M4 Max)

- Standard XYZ web-map pyramid: 256px PNG tiles, zoom 0 through 9 (zoom 10
  only if the pyramid stays within hosting budget), rendered with datashader +
  multiprocessing; empty tiles skipped. Estimated total 2-10GB.
- Node dots: size proportional to log(citations), color by Leiden community,
  alpha accumulation so dense regions glow.
- Edges: faint low-alpha strokes from mid-zoom onward; omitted at far-out
  zooms where they would read as fog.
- Labels are NOT baked into tiles (deliberate deviation from the video):
  the most-cited author names per tile per zoom level (label budget per tile
  is a render parameter, tuned during visual QA; starting point ~10) are
  emitted as small JSON "label tiles" rendered client-side as crisp text. Buys retina-sharp labels, hoverable
  names, and restyling without re-rendering the pyramid.
- QA gate: after each layout run, render only zooms 0-5 for visual review
  before committing to the full pyramid.

### 5. Index build (M4 Max)

- Search index: normalized name -> (id, x, y), sharded into a few thousand
  small JSON files by name prefix, fetched per keystroke. Zero backend.
- Hit tiles: tiny per-tile JSON of node ids + positions at high zooms, used to
  resolve clicks to nodes.

### 6. Web frontend (`web/`, static)

- MapLibre GL JS: raster tile layer over an abstract plane (plain XYZ, no
  geographic projection), inertial pan/zoom, vector overlays on top.
- Search box: prefix-shard lookup, fly-to on select.
- Click -> info card: hit-tile lookup resolves the node; card details
  (institution, works, links) fetched live from the OpenAlex API - single
  cheap calls, no quota concern.
- Path overlay: pick two researchers -> call the existing
  degree-of-separation backend on Render (author ids are OpenAlex ids in both
  projects, directly compatible) -> draw the path as a glowing polyline
  through mapped node coords and fly the camera along it. If a hop is not
  among the mapped authors it appears in the path card list and is drawn as a
  dashed skip segment. Required change to the other repo: a CORS allowlist
  entry for the map origin (a 2-line PR).

### 7. Hosting

- Cloudflare Pages (site) + Cloudflare R2 (tile pyramid + index shards).
  R2 free tier is 10GB with free egress, which fits the estimate and matters
  because map tiles get hammered.
- GitHub Pages ruled out (1GB ceiling vs multi-GB pyramid).
- Running cost ~$0/month at hobby traffic; path overlay rides the existing
  Render deployment.

## Testing & failure handling

- Unit tests on edge building with tiny fixtures: hyperauthorship cutoff,
  weight formula, pair-explosion correctness. Silent bugs here poison
  everything downstream, so this stage gets the densest tests.
- 100k-node sample pipeline runs end-to-end on the Mac alone (CPU igraph
  layout stands in for cuGraph) - the fast full-system test before any Anvil
  submission.
- Stages are idempotent over Parquet checkpoints; a crashed stage never
  corrupts upstream data.
- Visual QA before publishing: do fields form continents; does a spot-checked
  researcher (e.g., Hinton) sit in a sensible neighborhood.

## Non-goals for v1

- Citation edges (v2 candidate; pipeline design does not preclude them)
- Live WebGL rendering of the full graph
- Any server-side component beyond the existing degree-of-separation API
- Time sliders, per-year filtering, or animated growth

## Known risks

- RAPIDS environment setup friction on Anvil (mitigation: conda env spec
  committed to `slurm/`, tested with a tiny graph before the real run).
- FA2 hairball risk at this scale (mitigation: node2vec+UMAP fallback in the
  same job).
- Tile pyramid exceeding the 10GB R2 free tier if zoom 10 is kept
  (mitigation: cap max zoom or accept R2's modest paid storage rate).
- OpenAlex author disambiguation is imperfect; some nodes are merged or split
  people. Accepted as a data-source limitation, noted in the site's About.
