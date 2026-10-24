#!/bin/bash
# Build every map artifact the viewer serves, from a release's coords.parquet
# (run after the GPU layout):
#
#   MAPADEMIC_DATA=data/2026-09-23 bash scripts/build_map.sh
#
# Writes <release>/{subcommunities.parquet, coords_web.parquet,
# regions.json, tiles/, pathgraph/, paths/, index/,
# vector_experiment/}.
# subcommunities (about 15 min on a laptop) reruns only when the layout is
# newer than its output. regions runs before tiles because the tiles stage
# names the legend from regions.json, and needs the OpenAlex institutions and
# topics tables (python -m pipeline download --entities institutions topics).
# The tiles stage skips tiles that already exist, so the old tiles/ is
# removed first: every stage above it changes them.
set -euo pipefail
cd "$(dirname "$0")/.."
D=${MAPADEMIC_DATA:?set MAPADEMIC_DATA to the release directory}
PY=${PY:-.venv/bin/python}
[ -f "$D/coords.parquet" ] || { echo "no $D/coords.parquet: run the layout first" >&2; exit 1; }

step() { echo "== $* ($(date +%H:%M:%S))"; "$@"; }
if [ ! -f "$D/subcommunities.parquet" ] || [ "$D/coords.parquet" -nt "$D/subcommunities.parquet" ]; then
  step "$PY" -m pipeline subcommunities
fi
step "$PY" -m pipeline webcoords
step "$PY" -m pipeline regions
rm -rf "$D/tiles"
step "$PY" -m pipeline tiles --zooms 0-10
[ -f "$D/pathgraph/neighbors.npy" ] && [ "$D/pathgraph/neighbors.npy" -nt "$D/graph/edges_int32.parquet" ] \
  || step "$PY" -m pipeline pathgraph
step "$PY" -m scripts.export_paths_bin          # paths/: Connect's graph for the browser (~9 min)
step "$PY" -m pipeline index                    # id shards carry each author's paths/ number
step "$PY" -m scripts.export_points_bin
step "$PY" -m scripts.export_ids_bin
step "$PY" -m scripts.export_edges_bin
echo "== done ($(date +%H:%M:%S)) -> $D"
