import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipeline.pathgraph import build_pathgraph
from pathserver.server import Graph, Handler

# local requests must not go through any HTTP(S)_PROXY in the environment
OPEN = urllib.request.build_opener(urllib.request.ProxyHandler({})).open


@pytest.fixture
def server(tmp_path):
    # chain A10 - A11 - A12 - A13, plus an isolated A14 (no links)
    pq.write_table(pa.table({"node_idx": pa.array(range(5), pa.int32()),
                             "id": [f"https://openalex.org/A{i}" for i in range(10, 15)],
                             "display_name": ["Ann", "Bo", "Cy", "Di", "Ed"]}), tmp_path / "nodes.parquet")
    pq.write_table(pa.table({"src": pa.array([0, 1, 2], pa.int32()), "dst": pa.array([1, 2, 3], pa.int32()),
                             "weight": pa.array([1.0, 1.0, 1.0], pa.float32())}), tmp_path / "edges.parquet")
    build_pathgraph(str(tmp_path / "nodes.parquet"), str(tmp_path / "edges.parquet"), tmp_path / "pg")
    Handler.graph = Graph(tmp_path / "pg")
    Handler.labels = False
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def events(url):
    with OPEN(url, timeout=10) as r:
        assert r.headers["Access-Control-Allow-Origin"] == "*"
        text = r.read().decode()
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_health(server):
    with OPEN(server + "/health") as r:
        assert json.load(r) == {"status": "ok", "store": "csr", "authors": 5}


def test_path_streams_progress_then_the_shortest_path(server):
    ev = events(server + "/api/path?from=A10&to=A13&edges=coauthor")
    assert ev[0][0] == "progress" and ev[-1][0] == "result"
    result = ev[-1][1]
    assert result["found"] and result["hops"] == 3
    assert [s["author_name"] for s in result["path"]] == ["Ann", "Bo", "Cy", "Di"]
    assert result["path"][0]["author_id"] == "A10"


def test_unreachable_and_unknown_authors(server):
    assert events(server + "/api/path?from=A10&to=A14")[-1] == ("result", {"found": False,
        "reason": "no coauthor path between these authors"})
    kind, data = events(server + "/api/path?from=A10&to=A999")[-1]
    assert kind == "app_error" and data["code"] == "unknown_author"


def test_expand_lists_coauthors(server):
    ev = events(server + "/api/graph/expand?new_id=A11&depth=1&top_k=25&edges=coauthor")
    assert ev[0][0] == "expansion" and ev[-1][0] == "done"
    assert sorted(n["name"] for n in ev[0][1]["nodes"]) == ["Ann", "Cy"]
