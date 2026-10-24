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
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEV_HTML = REPO_ROOT / "web" / "dev.html"
SITE_HTML = REPO_ROOT / "docs" / "index.html"

# The R2 bucket behind its custom domain (connected 2026-10-10). The bucket's
# r2.dev development URL is HTTP/1.1-only with no edge cache: every zoom step
# queued ~40 tile requests six at a time. Through the zone, requests go over
# HTTP/2 and repeat tiles are Cloudflare cache hits; in an A/B run of the same
# page the worst zoom step went from 1,566 ms to 638 ms.
DATA_BASE = "https://data.mapademic.app/2026-09-23"   # R35c: one prefix per release in the bucket (scripts/upload_r2.sh)
# R31: the public degree-of-separation backend for Connect/coauthor-ray
# lookups -- the companion project's own Render deploy, the same instance
# powering xingruic.net/tools/researchers. Verified 2026-07-24: /health,
# /api/path and /api/graph/expand all answer, and CORS is
# access-control-allow-origin: * so a GitHub Pages origin is fine. Visitors
# need no key of any kind; the backend's OpenAlex key is server-side and
# optional. Set to None to omit apiBase entirely, which sends dev.html back to
# its localhost:8000 default (Connect off for public visitors). No trailing
# slash: dev.html builds URLs as `${API_BASE}/api/path`.
# R35c (2026-10-11): None. Connect now searches the release's own coauthor
# graph in the visitor's browser (data/paths/ on the data host, see
# scripts/export_paths_bin.py), and coauthor rays read OpenAlex directly, so
# the public site needs no backend -- and no host to pay for or keep awake.
API_BASE = None
# R41 (Eric, 2026-10-24): Cloudflare Web Analytics -- free, no cookies; it
# counts visits, referrers (reddit.com, linkedin.com, ...), countries and
# devices. The site token comes from the dashboard (Analytics & Logs > Web
# Analytics > mapademic.app) and is public by design: every visitor's browser
# sends it. Only docs/index.html gets the beacon; the local-data review build
# (inject_config alone) has none. None = no beacon.
ANALYTICS_TOKEN = "999d99d8c9dc4bef80e1ab3121a6b632"

BARE_SCRIPT_TAG = "<script>"


def build_config_block(data_base: str, api_base: str | None = None) -> str:
    if api_base:
        return (
            "<script>\n"
            "window.MAPADEMIC_CONFIG = {\n"
            f'  dataBase: "{data_base}",\n'
            f'  apiBase: "{api_base}"\n'
            "};\n"
            "</script>\n"
        )
    return (
        "<script>\n"
        "window.MAPADEMIC_CONFIG = {\n"
        f'  dataBase: "{data_base}"\n'
        "  // apiBase intentionally unset: Connect searches data/paths in the browser\n"
        "  // and coauthor rays read OpenAlex directly, so no backend is needed.\n"
        "};\n"
        "</script>\n"
    )


def inject_config(html: str, data_base: str = DATA_BASE, api_base: str | None = API_BASE) -> str:
    """Insert the MAPADEMIC_CONFIG script tag before the first bare <script>
    tag (a <script> tag with no attributes -- dev.html's own inline code,
    as opposed to the maplibre-gl CDN <script src="..."> above it).

    Pure string transform, no I/O -- kept separate from build() so it's easy
    to unit test.
    """
    idx = html.index(BARE_SCRIPT_TAG)
    return html[:idx] + build_config_block(data_base, api_base) + html[idx:]


def inject_analytics(html: str, token: str | None = ANALYTICS_TOKEN) -> str:
    """Add the Cloudflare Web Analytics beacon just before </head>, or
    nothing when token is None."""
    if not token:
        return html
    if not re.fullmatch(r"[0-9a-f]{32}", token):
        raise ValueError(f"not a Cloudflare Web Analytics token: {token!r}")
    tag = ('<script type="module" src="https://static.cloudflareinsights.com/beacon.min.js" '
           f'data-cf-beacon=\'{{"token": "{token}"}}\'></script>\n')
    idx = html.index("</head>")
    return html[:idx] + tag + html[idx:]


def build(
    dev_html_path: Path = DEV_HTML,
    data_base: str = DATA_BASE,
    api_base: str | None = API_BASE,
    analytics_token: str | None = ANALYTICS_TOKEN,
) -> str:
    return inject_analytics(inject_config(dev_html_path.read_text(), data_base, api_base), analytics_token)


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
