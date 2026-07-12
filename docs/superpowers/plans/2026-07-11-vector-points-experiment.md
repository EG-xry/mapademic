# Vector Points Experiment (R7)

> Experiment, Eric-requested: can the deepest zoom be "infinitely sharp" via vector points instead of raster upscaling? Prototype behind a toggle, judged via before/after screenshots in Lavish.

**Insight:** the existing hit/label tiles (data/index/labels/9/{x}/{y}.json, entries [norm-ignored, name, id? — actual shape [name, id, xw, yw, cited]], cap 4000/tile) are already per-tile point data with coords + citations, cached by the viewer for hover. A MapLibre GeoJSON circle layer fed from those tiles renders resolution-independent dots — crisp at any zoom — with zero new hosting data.

**v0 scope (this round):**
- Display-settings toggle "Vector points (beta)", default OFF, persisted.
- When ON and zoom >= 10.25: build/update a GeoJSON source from the visible z9 label tiles (reuse the existing tile cache + visibleTiles logic) on throttled moveend; circle layer on top of the raster: radius interpolated by zoom and by sqrt(cited) (ties into the graduated-splat idea), color uniform #cfd6d2 at 0.85 opacity with a subtle darker stroke; raster stays beneath for context (community colors still visible around/under points).
- Known caps documented: label tiles cap 4000/z9-tile (dense cores drop the tail — acceptable at z>=10 where a tile covers >=512px screen); dust/ring authors excluded (they are excluded from label tiles).
- No pipeline changes. Escape/keyboard, hover, rays, path overlays unaffected (circle layer is pointer-events-free; hitTest unchanged).

**v1 candidates (pending Eric's judgment):** community-colored circles (append community id to label-tile entries + legend.json color map, needs one index rebuild); raise maxZoom beyond 11 for vector mode only; move hover/hitTest onto queryRenderedFeatures; replace raster z10 with vector-only deepest zoom (kills 4.7GB tiles -> ~1.3GB labels, the radical hosting win).

**Verification:** live toggle at max zoom over a dense area and a sparse area; screenshot raster-only vs vector-on at identical viewport; no console errors; label/hover/path features still work with toggle on.
