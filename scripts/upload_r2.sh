#!/bin/bash
# Upload one release's site data to the R2 bucket behind data.mapademic.app,
# staying inside Cloudflare R2's free tier (10GB stored, 1M writes and 10M
# reads a month):
#
#   MAPADEMIC_DATA=data/2026-09-23 bash scripts/upload_r2.sh
#
# Needs an rclone remote named "r2" (create it yourself; keys never go in
# chat):   rclone config create r2 s3 provider=Cloudflare \
#            access_key_id=... secret_access_key=... \
#            endpoint=https://4de7bc56bd00677fac379de0cc51717d.r2.cloudflarestorage.com
#
# Writes under mapademic/<release>/data/ -- the same /data/... paths the
# viewer requests, behind the release prefix, so DATA_BASE in
# scripts/build_site.py becomes https://data.mapademic.app/<release>.
#
# Free-tier choices:
#  - JSON (index/, regions.json; 3.2GB raw) is stored gzip-compressed with
#    Content-Encoding: gzip; browsers decompress it transparently (~0.9GB).
#  - The viewer never loads the full and 3M connection-line tiers; skipped.
#  - Refuses to upload if the total would pass FREE_BUDGET_GB.
#  - --size-only compares sizes from the listing (no per-file HEAD reads),
#    so re-running after an interruption only sends what is missing.
#  - Every object gets Cache-Control: public, max-age=604800, so Cloudflare
#    serves repeat requests from its cache instead of reading R2; Connect's
#    paths/manifest.json alone is cached for five minutes (it names the
#    content-hashed graph files, see scripts/export_paths_bin.py).
# After the site points at the new release, delete the previous one
# (deletes are free): rclone purge r2:mapademic/<old prefix>
set -euo pipefail
cd "$(dirname "$0")/.."
D=${MAPADEMIC_DATA:?set MAPADEMIC_DATA to the release directory}
REL=$(basename "$D")
BUCKET=${BUCKET:-r2:mapademic}
DEST="$BUCKET/$REL/data"
FREE_BUDGET_GB=${FREE_BUDGET_GB:-9.5}
STAGE=.tmp/r2-stage/$REL
CACHE=(--header-upload "Cache-Control: public, max-age=604800")
COMMON=(--size-only --transfers "${TRANSFERS:-128}" --checkers 64 --stats 60s --stats-one-line --stats-log-level NOTICE)
SKIP=(--exclude "edges_pos.bin" --exclude "edges_attr.bin" --exclude "*_3m.bin")

rclone listremotes | grep -qx "${BUCKET%%:*}:" || { echo "no rclone remote '${BUCKET%%:*}' - see the header of this script" >&2; exit 1; }
for d in tiles index paths vector_experiment; do
  [ -d "$D/$d" ] || { echo "missing $D/$d - run scripts/build_map.sh first" >&2; exit 1; }
done

echo "== staging gzipped JSON in $STAGE"
.venv/bin/python - "$D" "$STAGE" <<'EOF'
import gzip, os, sys
from concurrent.futures import ThreadPoolExecutor
src, stage = sys.argv[1], sys.argv[2]
jobs = [os.path.join(r, f) for r, _, fs in os.walk(os.path.join(src, "index")) for f in fs if f.endswith(".json")]
jobs.append(os.path.join(src, "regions.json"))
def one(path):
    out = os.path.join(stage, os.path.relpath(path, src))
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(path):
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(path, "rb") as f:
        data = gzip.compress(f.read(), 9, mtime=0)
    with open(out + ".tmp", "wb") as f:
        f.write(data)
    os.replace(out + ".tmp", out)
with ThreadPoolExecutor(8) as pool:
    list(pool.map(one, jobs))
print(f"{len(jobs):,} JSON files staged")
EOF

total=$(.venv/bin/python - "$D" "$STAGE" <<'EOF'
import os, sys
src, stage = sys.argv[1], sys.argv[2]
skip = {"edges_pos.bin", "edges_attr.bin", "edges_pos_3m.bin", "edges_attr_3m.bin"}
def size(p, keep=lambda f: True):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs if keep(f))
t = size(os.path.join(src, "tiles")) + size(os.path.join(src, "paths")) + size(stage)
t += size(os.path.join(src, "vector_experiment"), lambda f: f not in skip)
print(f"{t / 1e9:.2f}")
EOF
)
echo "== upload size ${total} GB (free tier 10 GB, budget ${FREE_BUDGET_GB} GB)"
awk -v t="$total" -v b="$FREE_BUDGET_GB" 'BEGIN { exit !(t <= b) }' || { echo "over the free budget - not uploading" >&2; exit 1; }

echo "== JSON (gzip)"
rclone copy "$STAGE" "$DEST" "${COMMON[@]}" "${CACHE[@]}" \
  --header-upload "Content-Encoding: gzip" --header-upload "Content-Type: application/json"
echo "== tiles"
rclone copy "$D/tiles" "$DEST/tiles" "${COMMON[@]}" "${CACHE[@]}" --header-upload "Content-Type: image/png"
echo "== point, line and dust binaries"
rclone copy "$D/vector_experiment" "$DEST/vector_experiment" "${COMMON[@]}" "${CACHE[@]}" "${SKIP[@]}"
echo "== Connect's coauthor graph"
# data files are content-hashed (a re-export gets new names), so sync: new
# files go up, files no longer listed are deleted (free). The manifest goes
# last, with a five-minute cache, so it never names a file not yet uploaded.
rclone sync "$D/paths" "$DEST/paths" "${COMMON[@]}" "${CACHE[@]}" --exclude manifest.json
rclone copyto "$D/paths/manifest.json" "$DEST/paths/manifest.json" \
  --header-upload "Cache-Control: public, max-age=300" --header-upload "Content-Type: application/json"
echo "== bucket now holds:"
rclone size "$BUCKET"
echo "done. Next: DATA_BASE = https://data.mapademic.app/$REL in scripts/build_site.py, rebuild docs/,"
echo "check the site, publish, then delete the previous release's objects (deletes are free)."
