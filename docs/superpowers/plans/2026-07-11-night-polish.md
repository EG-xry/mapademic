# Night Polish Implementation Plan (R4)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Eric is asleep; all design calls are the controller's. Everything must be live-verified in the browser, not just node --check.

**Goal:** User-facing polish for the map site: display tuning (contrast/brightness/saturation/label density), shareable URLs, author info cards, help overlay, path hop list.

**Architecture:** All viewer-side (`web/dev.html`), zero pipeline changes, zero hosting cost. Display tuning uses MapLibre raster paint properties (GPU, free). URL state via a custom location.hash codec. Info cards call api.openalex.org directly (public CORS).

## Global Constraints

- Repo `/Users/eric/Downloads/mapademic`, branch `data-pipeline`; only `web/dev.html` changes in tasks 1-4.
- No em dashes; no Claude co-author; node --check + live chrome-devtools verification per task (controller runs the live checks if the implementer lacks browser tools; implementer must at minimum node --check and curl-serve).
- Keep the file's established idioms: generation counters for async redraws, Map caches, textContent (no innerHTML injection of data), separate marker arrays, dark minimal styling consistent with the legend panel (system font, no gradients).
- Do not degrade existing features (search, connect/path, labels, regions, hover, legend).

### Task 1: Display settings panel

Gear button (bottom-right, above MapLibre attribution). Panel (collapsible, styled like #legend) with sliders:
- Brightness: `raster-brightness-max` 0.6..1.0 (default 1.0)
- Contrast: `raster-contrast` -0.4..0.4 (default 0)
- Saturation: `raster-saturation` -0.6..0.6 (default 0)
- Label density: multiplier 0.5x/1x/2x applied to drawLabels' cap (60/150 base) - radio row
- Region names: on/off checkbox (drawRegions early-return when off)
- Reset button restoring defaults.
All persisted to localStorage key `mapademic.display` (JSON), applied on load. Path-mode dimming must compose: when a path is active, raster-opacity stays 0.45 regardless of sliders; sliders touch only brightness/contrast/saturation. Live-verify each slider visibly changes the canvas and persists across reload.

### Task 2: Shareable URL state

Custom hash codec (do NOT enable MapLibre's hash option): `#<zoom>/<lat>/<lng>` maintained on moveend (replaceState, no history spam); optional `&a=<bare id>` when a search highlight is active; `&p=<bareA>,<bareB>` when a path is drawn. On load: parse hash -> jumpTo view; `a=` -> resolve coords via id shards, place highlight + tooltip; `p=` -> prefill Connect inputs (names resolved from id shard? ids-only is fine: fetch author names lazily from api.openalex.org/people/<id> fields=display_name) and auto-run Go ONCE (guard against loops; only if API_BASE reachable - probe /health with a 2s timeout first, else show pill "path backend offline"). A "Share" button (next to Connect) copies the current URL to clipboard and flashes "copied". Live-verify: set view+highlight, copy, open in a new tab, same state restores.

### Task 3: Author info card on click

At zoom >= HIT_ZOOM, clicking a hit now opens an info card (fixed panel, bottom-left, dismissible via X or Escape) instead of window.open:
- Instant content from the hit entry: name, citations.
- Then fetch `https://api.openalex.org/people/<bare id>?select=display_name,works_count,cited_by_count,summary_stats,last_known_institutions,topics` (has public CORS; handle failure with a "couldn't load details" row): institution name, works count, h-index, top 3 topic display_names.
- Footer buttons: "Open on OpenAlex" (window.open, noopener), "Connect from here" (opens Connect panel, fills input A with this author), "Copy link" (share URL with a=<id>).
- Cache responses (Map, cap 50). Only one card at a time; opening a new card replaces the old.
Live-verify with a real click on a real dot (z>=10 over current tiles) and confirm real API data renders.

### Task 4: Help overlay + path hop list + final polish

- "?" button (top-right next to legend). Overlay (modal, dark, Escape/click-out closes): what the map is (8,587,906 OpenAlex authors with >= 15 works, positioned by coauthorship ForceAtlas2, colored by community), what colors/dots/edges/splats mean, controls (search, connect, hover/click at deep zoom, share, display settings), data provenance + degree-of-separation cross-link, note that one-off collaborations only show at max zoom.
- Path hop list: when a path renders, the connect panel shows the ordered hop list (name + on-map/ghost badge); clicking a hop flies to it. Clear removes it.
- Keyboard: "/" focuses search (unless typing in an input), "?" toggles help, Escape closes topmost (card > help > dropdowns > highlight).
- Sweep: consistent focus styles, aria-labels on icon buttons, all new panels respond to Escape, no console errors on load.

### Task 5 (controller): production swap + full QA + wrap-up

Swap tiles_v3 when render completes; composites incl. z10 crop; Lavish round-3/final update with all new features + screenshots; final whole-branch review 6c2462c..HEAD; ledger + memory; leave preview + backend running for Eric.
