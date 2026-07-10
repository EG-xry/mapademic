#!/bin/bash
# Local verification server: open http://localhost:8123/web/dev.html
cd "$(dirname "$0")/.."
exec python3 -m http.server 8123
