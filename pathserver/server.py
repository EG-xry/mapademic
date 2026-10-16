"""Coauthor-path server for the map's Connect tool and coauthor rays.

    python -m pathserver.server --graph data/2026-09-23/pathgraph --port 8000

Speaks the API web/dev.html already uses (so it can replace the hosted
backend without viewer changes): GET /health, GET /api/path (server-sent
events: progress, then result or app_error) and GET /api/graph/expand
(events: expansion, done). Paths are exact shortest paths over the release's
full coauthor graph (pathserver/search.py on pipeline/pathgraph.py's arrays,
memory-mapped, so it starts in seconds and needs ~3.5GB of disk and modest
RAM). The paper shown for each hop is looked up live from OpenAlex.
Standard library + numpy (and certifi, when installed, for HTTPS).
"""
import argparse
import json
import re
import ssl
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from pathserver.search import shortest_path

OPENALEX = "https://api.openalex.org"
LABEL_TIMEOUT = 4.0
try:   # python.org builds on macOS ship without root certificates
    import certifi
    SSL = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL = ssl.create_default_context()


class Graph:
    def __init__(self, d: Path):
        self.offsets = np.load(d / "offsets.npy", mmap_mode="r")
        self.neighbors = np.load(d / "neighbors.npy", mmap_mode="r")
        self.ids = np.load(d / "ids.npy")
        self.order = np.argsort(self.ids, kind="stable")
        self.sorted_ids = self.ids[self.order]
        self.name_offsets = np.load(d / "names_offsets.npy")
        self.names = (d / "names.bin").read_bytes()
        self.n = len(self.offsets) - 1

    def node(self, author_id: str) -> int | None:
        m = re.fullmatch(r"(?:https://openalex\.org/)?A(\d+)", author_id or "")
        if not m:
            return None
        key = np.uint64(int(m.group(1)))
        i = int(np.searchsorted(self.sorted_ids, key))
        return int(self.order[i]) if i < self.n and self.sorted_ids[i] == key else None

    def author(self, node: int) -> tuple[str, str]:
        name = self.names[self.name_offsets[node]:self.name_offsets[node + 1]].decode("utf-8")
        return f"A{int(self.ids[node])}", name

    def coauthors(self, node: int, k: int) -> list[int]:
        lo = int(self.offsets[node])
        return [int(x) for x in self.neighbors[lo:min(int(self.offsets[node + 1]), lo + k)]]


def shared_paper(a: str, b: str) -> str | None:
    """Title of the most-cited work two authors wrote together, or None."""
    q = urllib.parse.urlencode({"filter": f"authorships.author.id:{a},authorships.author.id:{b}",
                                "per_page": 1, "select": "title", "sort": "cited_by_count:desc"})
    try:
        with urllib.request.urlopen(f"{OPENALEX}/works?{q}", timeout=LABEL_TIMEOUT, context=SSL) as r:
            results = json.load(r).get("results") or []
        return results[0].get("title") if results else None
    except Exception:
        return None


class Handler(BaseHTTPRequestHandler):
    graph: Graph = None
    labels = True
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # one line per request, no noise
        print(f"{self.address_string()} {fmt % args}", flush=True)

    def _head(self, status: int, ctype: str, extra: dict | None = None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)

    def _json(self, status: int, obj):
        body = json.dumps(obj).encode()
        self._head(status, "application/json", {"Content-Length": str(len(body))})
        self.end_headers()
        self.wfile.write(body)

    def _events(self):
        self._head(200, "text/event-stream", {"Connection": "close", "X-Accel-Buffering": "no"})
        self.end_headers()
        self.close_connection = True

        def send(event: str, data: dict):
            self.wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode())
            self.wfile.flush()
        return send

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
        if url.path == "/health":
            return self._json(200, {"status": "ok", "store": "csr", "authors": self.graph.n})
        if url.path == "/api/path":
            return self._path(q.get("from", ""), q.get("to", ""))
        if url.path == "/api/graph/expand":
            return self._expand(q.get("new_id", ""), int(q.get("top_k", 25)))
        return self._json(404, {"error": "not found"})

    def _path(self, src: str, dst: str):
        send = self._events()
        g = self.graph
        a, b = g.node(src), g.node(dst)
        if a is None or b is None:
            missing = src if a is None else dst
            return send("app_error", {"code": "unknown_author",
                                      "message": f"{missing} is not on the map (no coauthor links)."})
        t0 = time.time()

        def progress(side, depth, found):
            send("progress", {"type": "progress", "message":
                              f"searching {depth} hop{'s' if depth > 1 else ''} out from "
                              f"{'the first' if side == 0 else 'the second'} author ({found:,} new)…"})
        nodes = shortest_path(g.offsets, g.neighbors, a, b, progress=progress)
        if nodes is None:
            return send("result", {"found": False, "reason": "no coauthor path between these authors"})
        authors = [g.author(x) for x in nodes]
        labels = [None] * len(authors)
        if self.labels and len(authors) > 1:
            send("progress", {"type": "progress", "message": "finding the papers that link each step…"})
            with ThreadPoolExecutor(max_workers=6) as pool:
                labels[:-1] = pool.map(lambda i: shared_paper(authors[i][0], authors[i + 1][0]),
                                       range(len(authors) - 1))
        send("result", {"found": True, "hops": len(nodes) - 1, "seconds": round(time.time() - t0, 2),
                        "path": [{"author_id": aid, "author_name": name, "label": lab}
                                 for (aid, name), lab in zip(authors, labels)]})

    def _expand(self, author_id: str, k: int):
        send = self._events()
        g = self.graph
        a = g.node(author_id)
        if a is None:
            return send("app_error", {"code": "unknown_author", "message": f"{author_id} is not on the map."})
        nodes = []
        for x in g.coauthors(a, max(1, min(k, 200))):
            aid, name = g.author(x)
            nodes.append({"id": aid, "name": name, "type": "expansion"})
        send("expansion", {"nodes": nodes})
        send("done", {})


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--graph", required=True, help="pipeline/pathgraph.py output directory")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-labels", action="store_true", help="skip the OpenAlex paper lookup per hop")
    args = p.parse_args()
    t0 = time.time()
    Handler.graph = Graph(Path(args.graph))
    Handler.labels = not args.no_labels
    print(f"{Handler.graph.n:,} authors loaded in {time.time() - t0:.1f}s; "
          f"listening on {args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
