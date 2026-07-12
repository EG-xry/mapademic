# Search, Path Overlay, z10 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eric's round-3 asks: fix region-name display, author search with fly-to, degree-of-separation path overlay (borrowing the academic-degree-of-separation backend), fix "zero coauthorship" areas, and native z10 resolution so max zoom is neither dense nor blurry.

**Architecture:** Pipeline: MAXZ 9 -> 10 (native z10 tiles, ~503k tiles), edgepx regenerated in z10 px space plus an optional weight-1 short-edge set drawn only at z10, index gains id-bucket shards (for path coord lookup) and 3-char splitting of hot search shards. Viewer: region-label collision culling + rank-continuous zoom windows, search box over the prefix shards, path overlay consuming the degree-of-separation `/api/path` SSE endpoint (CORS already `*`; base URL constant, local default `http://localhost:8000`).

**Tech Stack:** unchanged (DuckDB/numpy/PIL/pytest; vanilla JS + MapLibre).

## Global Constraints

- Repo `/Users/eric/Downloads/mapademic`, branch `data-pipeline`. Never touch `master`.
- `.venv/bin/pytest`; stage-module pattern; deterministic outputs; atomic writes (.tmp + os.replace); no `data/` commits.
- No em dashes in docs/commits. No Claude co-author line.
- Do NOT run stages against real `data/` in tasks (controller does that; tests use tmp_path).
- Do NOT re-run the `regions` stage (it clobbers name curation). The 7 unnamed/suffixed region names are Eric's to hand-edit; they are listed for him in the QA artifact.
- Diagnosed facts: at z10, 8,580,406 of 8.59M authors occupy distinct pixels across 502,942 tiles; every >=100k community has >=50k drawable edges (the "zero coauthorship" look = dust + weight-1-only areas); label-tile ids are FULL OpenAlex URLs; label tile FILES are XYZ y-down keyed.
- `/api/path?from=<id>&to=<id>&edges=coauthor` on the degree-of-separation backend streams SSE; the implementer of Task 6 must read `/Users/eric/Downloads/academic-degree-of-separation-master/backend/graph_backend.py` (find_path events) and `frontend/` consumption to pin the exact `path` event payload before writing the overlay.

## File Structure

- Modify `pipeline/tiles.py` - MAXZ=10, splat radii, w1-edge layer (Tasks 1, 3)
- Modify `pipeline/edge_px.py` - z10 px space + weight-1 mode (Task 2)
- Modify `pipeline/build_index.py` - id shards + 3-char hot-shard split (Task 4)
- Modify `web/dev.html` - region culling, search UI, path overlay (Tasks 5-7)
- Tests: `tests/test_tiles.py`, `tests/test_edge_px.py`, `tests/test_build_index.py`

---

### Task 1: MAXZ 10

**Files:** Modify `pipeline/tiles.py`; Test `tests/test_tiles.py`

**Interfaces:** `MAXZ = 10`, `PIX = (2**MAXZ) * TILE = 262144`. `SPLAT_RADIUS = {9: 1, 10: 2}`. Bloom applies at `z >= MAXZ - 1`. `EDGE_MINZ` stays 8. Aggregate cache file renamed `pixels_z10.parquet` (constant `PIXELS_NAME = f"pixels_z{MAXZ}.parquet"`; run() uses it — the old pixels_z9.parquet becomes dead, staleness logic unchanged otherwise). `aggregate_z9` renamed `aggregate_maxz` (update callers/tests).

Steps: update every test in `tests/test_tiles.py` that hardcodes z=9/PIX=131072/511 tile paths to derive from MAXZ/PIX (the existing bloom/splat/edge tests); run to see failures; implement; full suite green; commit `feat(tiles): native z10 pyramid (MAXZ=10)`.

Note: `render_zoom` and `reduce_level` are already MAXZ-parametric; the work is constants + rename + tests. Verify `_brightness` and bucketing make no z9 assumptions (they don't; `_bucket_edges` shifts by `MAXZ - z`).

---

### Task 2: edgepx in z10 space + weight-1 mode

**Files:** Modify `pipeline/edge_px.py`; Test `tests/test_edge_px.py`

**Interfaces:** `build_edge_px(edges_path, web_path, out_path, min_weight=2, max_weight=None, max_len_px=1536)` — px math uses `PIX` from tiles (now 262144), so `max_len_px` default DOUBLES to 1536 (same world distance as before). New `max_weight` (exclusive upper bound, None = no bound) enables a weight-1-only extraction: `min_weight=1, max_weight=2`. CLI gains `--max-weight` and keeps the rest.

Tests: update pixel-coord expectations for PIX=262144 (0.5 -> 131072); add a max_weight test (weight-3 row excluded when max_weight=2, weight-1 row kept with min_weight=1). Commit `feat(edgepx): z10 pixel space + weight-band selection`.

---

### Task 3: tiles draws the w1 layer at z10 only

**Files:** Modify `pipeline/tiles.py`; Test `tests/test_tiles.py`

**Interfaces:** `render_zoom(..., edges=None, edges_w1=None)`; `edges_w1` drawn ONLY when `z == MAXZ`, additive with `W1_ALPHA = 0.05` and the same `EDGE_RGB`, drawn before `edges` (both before nodes). CLI: `--edges-w1 [PATH]` mirroring `--edges` (default `data/edges_px_w1.parquet`), same skip-existing footgun note in help. Reuse `_bucket_edges` for both sets.

Tests: w1 edge renders at z10, does NOT render at z9 even when passed; regular edges still render at z8-10. Commit `feat(tiles): weight-1 short edges at max zoom only`.

---

### Task 4: index — id shards + hot search-shard split

**Files:** Modify `pipeline/build_index.py`; Test `tests/test_build_index.py`

**Interfaces:**
- `build_id_shards(web, out_dir) -> int`: writes `out_dir/ids/<bucket>.json` where `bucket = int-digits of the OpenAlex id (strip non-digits) % 1000`, content `{"<full id>": [xw6, yw6]}` (round 6). Non-ring AND ring both included (path nodes may be dust). Called from `run()` after search shards.
- Hot-shard split in `build_search_shards`: after building 2-char buckets, any bucket whose serialized size would exceed `SHARD_SPLIT_BYTES = 4_000_000` is split into 3-char sub-buckets (`abc.json`); entries whose normalized name is exactly 2 chars stay in the 2-char shard, which is always written (possibly small) so the viewer fallback order is: 3-char shard -> 2-char shard -> `_`. Document that order in a module docstring.

Tests: id shard bucket math + content shape; a synthetic oversized bucket splits into 3-char shards while the 2-char shard keeps exact-2-char names. Commit `feat(index): id-bucket coord shards + 3-char hot search shards`.

---

### Task 5: viewer — region label display fix

**Files:** Modify `web/dev.html`

Replace `drawRegions` zmin/zmax gating with rank-continuous windows + collision culling:
- Visibility: region visible when `zoom >= zminByRank && zoom <= 8` where `zminByRank = 1.5 + 1.2 * log10(rank)` (rank 1 appears ~z1.5, rank 100 ~z3.9); regions never all vanish mid-range (fixes "cluster that's never named" — names were hidden by the old zmax windows).
- Style: font size `13 + 6 * min(1, members / 1_000_000)` px, keep the uppercase letter-spaced look.
- Collision: same greedy screen-rect culling as author labels (extract the rect-overlap check into a shared helper `collides(placed, x, y, w, h)`), regions culled by rank order (biggest first). Region labels and author labels cull against SEPARATE placed-lists (author labels may sit under a region name; that is fine).
- Redraw on the same moveend/zoomend hooks with the drawLabels generation-token pattern if any await is introduced (there is none today; keep it synchronous).

Verify in browser (labels de-overlap at the z4-7 range that was cluttered; regions visible continuously as you zoom). Commit `fix(web): region labels - collision culling + rank-continuous visibility`.

---

### Task 6: viewer — author search

**Files:** Modify `web/dev.html`

- Top-left search box (`#search`, input + dropdown). `normalize()` in JS replicating build_index.normalize (NFKD, strip combining, lowercase, non-alnum -> space, collapse).
- Shard fetch: `key3 = norm.replace(/ /g,"").slice(0,3)`, try `/data/index/search/${key3}.json`; on 404 try 2-char, then `_`. Cache fetched shards (Map, cap ~20).
- Rank: entries whose `norm` starts with the query first, then substring matches; sort by cited desc; top 12 in dropdown (name + formatted citations).
- Select: `map.flyTo({center: toLngLat(xw, yw), zoom: 10.5})`, drop a pulsing highlight marker (CSS animation, accent ring) that persists until the next search or Escape; show the tooltip content for it.
- Keyboard: arrows + Enter, Escape clears.

Verify live against the real index. Commit `feat(web): author search with fly-to highlight`.

---

### Task 7: viewer — degree-of-separation path overlay

**Files:** Modify `web/dev.html`

- FIRST read `/Users/eric/Downloads/academic-degree-of-separation-master/backend/app.py` `/api/path`, `backend/graph_backend.py` find_path event shapes, and how `frontend/` consumes the `path` event; pin the exact payload (node id list with names/types) before coding.
- UI: a "Connect" toggle next to search swaps in two inputs (A, B) reusing the search component; Go button opens `new EventSource(API_BASE + "/api/path?from=..&to=..&edges=coauthor")` with `const API_BASE = "http://localhost:8000"` defined at the top next to HIT_ZOOM (comment: Render URL goes here when deployed).
- Progress: show streamed progress events' text in a small status pill; `app_error` -> show message.
- On `path` event: resolve each path author id via `/data/index/ids/<bucket>.json` (same bucket math as Task 4; cache). Nodes missing from the map (sub-threshold authors) get a "ghost" position = midpoint of their resolved neighbors, rendered hollow.
- Draw: one GeoJSON source + two layers — `line` (accent `#7fae8e`, width 2.5, `line-blur` 1.5, opacity 0.9) and per-node markers (filled = on-map, hollow = ghost) with name labels always visible (skip collision culling for path labels); dim the base raster to `raster-opacity: 0.45` while a path is shown; fitBounds to the path. Status pill shows hop count ("3 hops"). Clear button restores opacity and removes source/markers and closes the EventSource.
- Path mode suspends normal label redraw culling only if it visually fights the path labels (implementer judgment, note the choice).

Verify live with the backend running (`.venv/bin/uvicorn backend.app:app --port 8000` in the degree-of-separation repo) using the known-good pair Hinton A5108093963 <-> Chomsky A5072532913. Commit `feat(web): degree-of-separation path overlay`.

---

### Task 8: production batch + QA (controller-run)

- edgepx (weight>=2, z10 space) + edgepx w1 (min 1, max 2, `--max-len-px 512`; run the count first and raise/lower the length cap to keep the w1 set under ~50M).
- `tiles --zooms 0-10 --out data/tiles_v3 --force-aggregate --edges --edges-w1`, then swap; `index` re-run (id shards + split search shards; labels unchanged z6-9).
- Composites + a z10 crop; live checks: search flies to an author; path overlay draws Hinton-Chomsky with the local backend; region labels culled; z10 sharpness.
- Lavish round 3 update + ledger + memory. Note for Eric: total hosting size (expect ~7-8GB with z10; R2 free tier is 10GB) and the 7 region names awaiting his hand-edit.

## Self-Review

- Eric's five asks map: region display (T5), search (T6), path overlay (T7), zero-coauthor areas (T2/T3 w1 edges + dust explanation in QA report), density+resolution (T1 native z10). Coverage complete.
- Interfaces consistent: PIX flows from tiles into edge_px and build_index binning; id-shard bucket math defined once and mirrored in the viewer; MAXZ-derived constants replace all z9 literals.
- No placeholders; viewer tasks carry exact behavioral specs and verification steps; pipeline tasks carry exact signatures/defaults.
