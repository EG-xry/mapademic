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
#  - JSON (index/, identities/, optional regions.json) is gzip-compressed with
#    Content-Encoding: gzip; browsers decompress it transparently (~0.9GB).
#  - The viewer never loads the full and 3M connection-line tiers; skipped.
#  - Refuses to upload if the total would pass FREE_BUDGET_GB.
#  - --size-only compares sizes from the listing (no per-file HEAD reads),
#    so re-running after an interruption only sends what is missing.
#  - Most objects are cached for a week. The graph manifest is cached for
#    five minutes and uploaded last. It points to a content-addressed
#    identity directory, so cached manifests can still read their metadata.
# Existing release prefixes and previously uploaded graph files are retained.
set -euo pipefail
cd "$(dirname "$0")/.."
D=${MAPADEMIC_DATA:?set MAPADEMIC_DATA to the release directory}
REL=$(basename "$D")
BUCKET=${BUCKET:-r2:mapademic}
DEST="$BUCKET/$REL/data"
FREE_BUDGET_GB=${FREE_BUDGET_GB:-9.5}
STAGE=.tmp/r2-stage/$REL
CACHE=(--header-upload "Cache-Control: public, max-age=604800")
SHORT_CACHE=(--header-upload "Cache-Control: public, max-age=300")
COMMON=(--transfers "${TRANSFERS:-128}" --checkers 64 --stats 60s --stats-one-line --stats-log-level NOTICE)
GZIP_JSON=(--header-upload "Content-Encoding: gzip" --header-upload "Content-Type: application/json")
SKIP=(--exclude "edges_pos.bin" --exclude "edges_attr.bin" --exclude "edges_fine.bin" --exclude "*_3m.bin")

rclone listremotes | grep -qx "${BUCKET%%:*}:" || { echo "no rclone remote '${BUCKET%%:*}' - see the header of this script" >&2; exit 1; }
for d in tiles index paths vector_experiment; do
  [ -d "$D/$d" ] || { echo "missing $D/$d - run scripts/build_map.sh first" >&2; exit 1; }
done

# A tagged graph must never be published without its matching identity data.
.venv/bin/python - "$D" <<'EOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
try:
    paths = json.loads((root / "paths/manifest.json").read_text())
    identities = root / "identities"
    if paths.get("version") == 3 or identities.exists():
        manifest = json.loads((identities / "manifest.json").read_text())
        flags = json.loads((identities / "flags.json").read_text())
        ids = paths["files"]["ids"]
        if (manifest.get("version") != 1 or not manifest.get("graph_repaired")
                or flags.get("version") != 1
                or manifest.get("paths_ids_file") != ids or flags.get("paths_ids_file") != ids
                or not isinstance(ids, str) or Path(ids).name != ids
                or not (root / "paths" / ids).is_file()):
            raise ValueError("identity metadata does not match the graph IDs")
except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
    raise SystemExit(f"invalid release identity/graph metadata: {error}")
EOF

echo "== staging gzipped JSON in $STAGE"
identity_generation=$(.venv/bin/python - "$D" "$STAGE" <<'EOF'
import gzip, hashlib, os, sys
from concurrent.futures import ThreadPoolExecutor
src, stage = sys.argv[1], sys.argv[2]
jobs = [os.path.join(r, f) for directory in ("index", "identities")
        for r, _, fs in os.walk(os.path.join(src, directory)) for f in fs if f.endswith(".json")]
if os.path.isfile(os.path.join(src, "regions.json")):
    jobs.append(os.path.join(src, "regions.json"))
# Retain reusable gzip checkpoints, but do not upload removed profiles/shards
# left over from an earlier staging pass under the same local release name.
keep = {os.path.join(stage, os.path.relpath(path, src)) for path in jobs}
for root, _, files in os.walk(stage, topdown=False):
    for filename in files:
        path = os.path.join(root, filename)
        if path not in keep:
            os.unlink(path)
    if root != stage and not os.listdir(root):
        os.rmdir(root)
def is_identity(path):
    return os.path.relpath(path, src).split(os.sep, 1)[0] == "identities"

def one(path):
    out = os.path.join(stage, os.path.relpath(path, src))
    if not is_identity(path) and os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(path):
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(path, "rb") as f:
        data = gzip.compress(f.read(), 9, mtime=0)
    with open(out + ".tmp", "wb") as f:
        f.write(data)
    os.replace(out + ".tmp", out)
with ThreadPoolExecutor(8) as pool:
    list(pool.map(one, jobs))
print(f"{len(jobs):,} JSON files staged", file=sys.stderr)
identity_files = sorted(os.path.relpath(p, src) for p in jobs if is_identity(p))
if identity_files:
    generation = hashlib.sha256()
    for path in identity_files:
        generation.update(os.fsencode(path) + b"\0")
        with open(os.path.join(stage, path), "rb") as f:
            generation.update(hashlib.file_digest(f, "sha256").digest())
    print(generation.hexdigest())
EOF
)

total=$(.venv/bin/python - "$D" "$STAGE" <<'EOF'
import os, sys
src, stage = sys.argv[1], sys.argv[2]
skip = {"edges_pos.bin", "edges_attr.bin", "edges_fine.bin", "edges_pos_3m.bin", "edges_attr_3m.bin", "edges_fine_3m.bin"}
def size(p, keep=lambda f: True):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs if keep(f))
t = size(os.path.join(src, "tiles")) + size(os.path.join(src, "paths")) + size(stage)
t += size(os.path.join(src, "vector_experiment"), lambda f: f not in skip)
print(f"{t / 1e9:.9f}")
EOF
)
echo "== upload size ${total} GB (free tier 10 GB, budget ${FREE_BUDGET_GB} GB)"
awk -v t="$total" -v b="$FREE_BUDGET_GB" 'BEGIN { exit !(t <= b) }' || { echo "over the free budget - not uploading" >&2; exit 1; }

# This upper bound may count objects that will be replaced or skipped.
# Warn without deleting previous releases or blocking a resumable upload.
stored=$(rclone size "$BUCKET" --json | .venv/bin/python -c 'import json,sys; print(json.load(sys.stdin)["bytes"] / 1e9)')
if awk -v t="$total" -v s="$stored" -v b="$FREE_BUDGET_GB" 'BEGIN { exit !(t + s > b) }'; then
  echo "warning: existing bucket data ($stored GB) plus this upload may exceed the $FREE_BUDGET_GB GB budget" >&2
fi

echo "== JSON (gzip)"
rclone copy "$STAGE" "$DEST" --size-only "${COMMON[@]}" "${CACHE[@]}" "${GZIP_JSON[@]}" --exclude "identities/**"
echo "== tiles"
rclone copy "$D/tiles" "$DEST/tiles" --size-only "${COMMON[@]}" "${CACHE[@]}" --header-upload "Content-Type: image/png"
echo "== point, line and dust binaries"
rclone copy "$D/vector_experiment" "$DEST/vector_experiment" --size-only "${COMMON[@]}" "${CACHE[@]}" "${SKIP[@]}"
echo "== Connect's coauthor graph"
# Retain older content-hashed files so in-flight readers of the previous
# manifest can finish. Copy the new manifest only after all uploads succeed.
rclone copy "$D/paths" "$DEST/paths" --size-only "${COMMON[@]}" "${CACHE[@]}" --exclude manifest.json
graph_manifest="$D/paths/manifest.json"
if [ -n "$identity_generation" ]; then
  echo "== repaired identity JSON (gzip)"
  identity_dest="$DEST/identities/$identity_generation"
  rclone copy "$STAGE/identities" "$identity_dest" --checksum "${COMMON[@]}" "${CACHE[@]}" "${GZIP_JSON[@]}" \
    --exclude /flags.json --exclude /manifest.json
  for name in flags.json manifest.json; do
    rclone copyto "$STAGE/identities/$name" "$identity_dest/$name" --checksum "${CACHE[@]}" "${GZIP_JSON[@]}"
  done
  graph_manifest=$(mktemp "$STAGE/../$REL-paths.XXXXXXXX")
  trap 'rm -f "$graph_manifest"' EXIT
  .venv/bin/python - "$D/paths/manifest.json" "$graph_manifest" "$identity_generation" <<'EOF'
import json, sys
from pathlib import Path
source, target, generation = sys.argv[1:]
meta = json.loads(Path(source).read_text())
meta["identity_generation"] = generation
Path(target).write_text(json.dumps(meta, indent=1))
EOF
fi
rclone copyto "$graph_manifest" "$DEST/paths/manifest.json" --checksum \
  "${SHORT_CACHE[@]}" --header-upload "Content-Type: application/json"
echo "== bucket now holds:"
rclone size "$BUCKET"
echo "done. Next: DATA_BASE = https://data.mapademic.app/$REL in scripts/build_site.py, rebuild docs/,"
echo "check the site and publish. Previous release objects have been retained."
