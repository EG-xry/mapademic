"""Regenerate docs/index.html from web/dev.html (R15 review cleanup).

docs/ is the public GitHub Pages deploy; web/dev.html is the local dev page
(same markup/JS, plus in-progress experiments not meant for public eyes yet).
The only difference the deployed page needs is a window.MAPADEMIC_CONFIG
block, injected as a <script> tag right before dev.html's first bare
<script> tag (dev.html's own scripts read window.MAPADEMIC_CONFIG.dataBase/
apiBase and fall back to same-origin/localhost defaults when it's absent --
see the "Deploy config" comment in web/dev.html).

This script is NOT run automatically on every dev.html change -- docs/
index.html is regenerated deliberately at deploy time (a human decides the
local experiment-of-the-moment is ready to go public, then runs this without
--check). Use --check as a pre-deploy gate: it exits 1 if docs/index.html is
stale relative to a fresh build, without writing anything.
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEV_HTML = REPO_ROOT / "web" / "dev.html"
SITE_HTML = REPO_ROOT / "docs" / "index.html"

# Exact config currently deployed (read out of the last docs/index.html
# generation). apiBase is deliberately left unset -- see the config
# block's own comment below.
DATA_BASE = "https://pub-3627024db5924826912107ea4f16e739.r2.dev"

BARE_SCRIPT_TAG = "<script>"


def build_config_block(data_base: str) -> str:
    return (
        "<script>\n"
        "window.MAPADEMIC_CONFIG = {\n"
        f'  dataBase: "{data_base}"\n'
        "  // apiBase intentionally unset: path/ray features probe localhost:8000 and\n"
        "  // stay gracefully disabled for public visitors without a running backend.\n"
        "};\n"
        "</script>\n"
    )


def inject_config(html: str, data_base: str = DATA_BASE) -> str:
    """Insert the MAPADEMIC_CONFIG script tag before the first bare <script>
    tag (a <script> tag with no attributes -- dev.html's own inline code,
    as opposed to the maplibre-gl CDN <script src="..."> above it).

    Pure string transform, no I/O -- kept separate from build() so it's easy
    to unit test.
    """
    idx = html.index(BARE_SCRIPT_TAG)
    return html[:idx] + build_config_block(data_base) + html[idx:]


def build(dev_html_path: Path = DEV_HTML, data_base: str = DATA_BASE) -> str:
    return inject_config(dev_html_path.read_text(), data_base)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if docs/index.html is stale relative to a fresh build; write nothing",
    )
    args = parser.parse_args()

    fresh = build()

    if args.check:
        current = SITE_HTML.read_text() if SITE_HTML.exists() else None
        if current != fresh:
            print(f"{SITE_HTML} is stale relative to {DEV_HTML}; run scripts/build_site.py to regenerate", file=sys.stderr)
            return 1
        print(f"{SITE_HTML} is up to date")
        return 0

    SITE_HTML.write_text(fresh)
    print(f"wrote {SITE_HTML}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
