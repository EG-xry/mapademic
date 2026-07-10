# Mapademic Visual Phase - Design Spec (Plans 3 & 4)

Date: 2026-07-10
Status: Approved by Eric (brainstorming session; parts 1 and 2 approved)
Supersedes/refines: stages 4-6 of `2026-07-05-mapademic-design.md`

## Input (fixed, produced by Plan 2)

`coords.parquet` - 8,587,906 rows, verified on Eric's Mac (`data/`), external
drive, and Anvil scratch. ACTUAL schema (authoritative, supersedes older docs):
`id VARCHAR, display_name VARCHAR, x FLOAT, y FLOAT, community INTEGER,
works_count INTEGER, cited_by_count INTEGER, institution VARCHAR, field VARCHAR`.
Row order is arbitrary - always join by `id`. Layout: run-1 FA2; dense core
within roughly ±100k units, sparse halo to ±700k. 216,620 Leiden communities;
19 OpenAlex fields. Authors snapshot (topics per author) available locally
(external drive) and on Anvil for the region-naming job.

## Decisions (locked during brainstorming)

- Aesthetic: dark cosmic - black space, glowing communities, luminous density
- Color: hybrid - each of the 19 fields owns a hue family; community id gives
  a deterministic shade/saturation jitter within the family
- Halo: asinh radial compression (linear core, log tail) - all 8.59M stay on-map
- Labels: auto-named regions (topic-derived) at low/mid zoom + most-cited
  researcher names at deep zoom; nothing baked into raster tiles
- Zoom: raster pyramid 0-9, measure size, optionally add partial zoom 10 for
  the densest central tiles
- Renderer: custom Python raster pipeline (approach A) - aggregate once at max
  zoom, pyramid-reduce upward; NOT vector tiles (fights the glow aesthetic)

## Architecture

Four new pipeline stages (Mac, no cluster needed) + a static site + deploy:

```
coords.parquet -> webcoords -> tiles -> [PNG pyramid 0-9]
                     |-> regions  -> regions.json
                     |-> index    -> label tiles + hit tiles + search shards
web/ (MapLibre)  <- consumes all of the above ->  R2 + Pages (deploy script)
```

Plan 3 = webcoords, tiles, regions, index (ends: Eric approves rendered tiles).
Plan 4 = web/ + hosting + the 2-line CORS PR to degree-of-separation (ends:
public URL).

## Stage details

### 1. `webcoords` (transform)

- Centroid of all points, then radial transform `r' = asinh(r / s)` with
  `s = median radius`; rescale to the unit square [0,1]^2 with a small margin.
- Output `coords_web.parquet`: id, display_name, xw DOUBLE, yw DOUBLE,
  community, works_count, cited_by_count, institution, field, plus
  `tx, ty` (zoom-9 tile indices) for cheap bucketing.
- Every downstream consumer reads ONLY web coords - one source of truth.
- Tests: all points in [0,1]^2; radius monotonic (order preserved);
  core proportions approximately linear (small-r round-trip error bound).

### 2. `tiles` (aggregate -> pyramid -> styled PNGs)

- One DuckDB pass buckets points into zoom-9 pixels (131,072^2 virtual grid,
  sparse; <=8.6M occupied): per-pixel count + dominant (field, community).
- Zoom N-1 = 2x2 box-sum of zoom N (count-weighted dominant color); nine
  reductions down to zoom 0. Counts conserved at every level (tested).
- Styling per zoom: brightness = histogram-equalized log-count (per-zoom CDF);
  hue = dominant field's family; shade = deterministic jitter from community
  id; subtle 1-2px bloom at zooms 8-9 so isolated researchers read as stars.
- Output: 256px PNG XYZ tiles, zooms 0-9, empty tiles skipped; parallel per
  tile-block (multiprocessing). Expected 2-5GB vs R2 free tier 10GB.
- QA gate: render zooms 0-5 only, Eric eyeballs before the full burn.
- Golden test: tiny synthetic fixture -> stable tile bytes (styling
  regressions show as pixel diffs).
- The 19 field-family hue anchors live in one palette module, hand-tunable.

### 3. `regions` (community naming)

- For the ~200-300 largest communities: members' primary topics from the
  authors snapshot, name = most distinctive topic (high share in community,
  low share overall - TF-IDF-like); centroid + spread in web coords.
- Output `regions.json` (~120 curated labels with zoom bands by size rank:
  continent-scale z2-3, country-scale z4-6). Hand-editable data file - name
  fixes never require re-rendering.
- Tests on fixtures: distinctive-topic extraction picks the intended topic;
  output schema stable.

### 4. `index` (labels, hits, search)

- Label/hit tiles (zooms 6-9): per-tile JSON with researchers ranked by
  cited_by_count (name, id, xw, yw). The client draws text for the top ~10;
  the SAME file carries a deeper list (top ~50 at z6-7, ~200 at z8-9) used
  only for click hit-testing (nearest dot within a pixel radius), so clicking
  an unlabeled dot still resolves. Clicks with no entry in radius show a
  "zoom in to select individuals" hint rather than failing silently.
- Search: normalized-name prefix shards (first 2-3 chars -> a few thousand
  JSON files) covering all 8.6M; client fetches one shard per query prefix,
  matches locally, fly-to on select. Zero backend.

### 5. `web/` (static MapLibre site)

- MapLibre GL: raster tile source + overlay layers (region labels, researcher
  labels, selection markers); inertial pan/zoom on the abstract plane.
- Click -> info card: live OpenAlex API call for details (institution, works,
  links); include OPENALEX_KEY-less anonymous calls (cheap, per-click).
- Path overlay: two researchers -> existing degree-of-separation backend on
  Render -> glowing polyline through web coords; unmapped hops listed in the
  card and drawn as dashed skips. Needs the CORS allowlist PR (2 lines).
- About panel: snapshot date (June 2026), >=15-works threshold, OpenAlex
  disambiguation caveat, link to both repos.

### 6. Hosting + deploy

- R2 (tiles + JSON artifacts; immutable cache headers) + Cloudflare Pages
  (site). A `deploy` script wraps wrangler for idempotent uploads.
- Budget: $0/month at hobby traffic; if zoom 10 pushes past 10GB, either cap
  it or accept R2's ~$0.015/GB-month.

## Error handling & operational notes

- All stages are idempotent over checkpointed outputs like the data phase;
  tile rendering resumes by skipping existing tiles.
- The pipeline runs entirely locally; source parquets also exist on the
  external drive and Anvil if the Mac copy is lost.
- Region auto-names WILL have a quality tail - the curation loop (edit
  regions.json, redeploy JSON only) is the mitigation, not re-rendering.

## Testing summary

- Unit: transform bounds/monotonicity, pyramid count conservation, palette
  determinism, region-name extraction, search-shard lookup.
- Golden-image: synthetic fixture tiles byte-stable.
- Human gates: zooms 0-5 QA render (before full pyramid); local site preview
  (before publish).

## Non-goals (v1)

- Vector-tile rendering, WebGL point clouds
- Time sliders / animated growth
- Theme toggle (dark only)
- Server-side anything beyond the existing degree-of-separation API
