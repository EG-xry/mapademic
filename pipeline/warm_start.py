"""Seed the GPU layout with the previous release's positions (graph/init_pos.parquet)."""
from pathlib import Path

import duckdb

from pipeline.config import apply_resource_limits, data_dir, spill_dir

# Neighbour-averaging rounds for authors new in this release: round 1 places
# those with a carried-over coauthor, round 2 those one hop further out.
HOPS = 2
# Jitter, as a fraction of the previous median radius, so new authors sharing
# the same coauthors do not start on one point (zero-distance repulsion).
JITTER = 1e-4


def build_init_positions(nodes_path: str, edges_path: str, prev_coords: str,
                         out_path: str, hops: int = HOPS) -> dict:
    """Write (vertex, x, y) for every node of the new graph; returns counts.

    Carried-over authors keep their previous raw layout position. A new
    author starts at the weight-averaged position of its already-placed
    coauthors; one with none after `hops` rounds is scattered (hash-seeded,
    so reruns agree) inside the previous median radius. The FA2 run then only
    has to settle what changed, and the map keeps its familiar arrangement.
    """
    con = duckdb.connect()
    apply_resource_limits(con)
    con.execute(f"SET temp_directory='{spill_dir(Path(out_path).parent)}'")
    con.execute(f"CREATE TEMP TABLE nodes AS SELECT node_idx, id FROM read_parquet('{nodes_path}')")
    con.execute(
        f"""CREATE TEMP TABLE pos AS
            SELECT n.node_idx AS vertex, CAST(c.x AS DOUBLE) AS x, CAST(c.y AS DOUBLE) AS y
            FROM nodes n JOIN read_parquet('{prev_coords}') c ON c.id = n.id
            WHERE c.x IS NOT NULL AND c.y IS NOT NULL"""
    )
    carried = con.execute("SELECT count(*) FROM pos").fetchone()[0]
    cx, cy, radius = con.execute(
        f"""WITH p AS (SELECT CAST(x AS DOUBLE) x, CAST(y AS DOUBLE) y
                       FROM read_parquet('{prev_coords}') WHERE x IS NOT NULL)
            SELECT median(x), median(y),
                   greatest(median(sqrt((x - (SELECT median(x) FROM p)) ** 2
                                      + (y - (SELECT median(y) FROM p)) ** 2)), 1e-9)
            FROM p"""
    ).fetchone()
    jitter = max(JITTER * radius, 1e-3)   # floor: a degenerate previous layout has radius 0
    from_neighbours = 0
    for _ in range(hops):
        # both directions of every edge touching a still-unplaced node
        added = con.execute(
            f"""INSERT INTO pos
                WITH open AS (SELECT node_idx AS v FROM nodes
                              WHERE node_idx NOT IN (SELECT vertex FROM pos)),
                     touch AS (
                         SELECT dst AS v, src AS u, weight FROM read_parquet('{edges_path}')
                         WHERE dst IN (SELECT v FROM open)
                         UNION ALL
                         SELECT src AS v, dst AS u, weight FROM read_parquet('{edges_path}')
                         WHERE src IN (SELECT v FROM open))
                SELECT t.v,
                       sum(t.weight * p.x) / sum(t.weight)
                         + {jitter} * ((hash(t.v) % 2001) / 1000.0 - 1.0),
                       sum(t.weight * p.y) / sum(t.weight)
                         + {jitter} * ((hash(t.v * 7 + 3) % 2001) / 1000.0 - 1.0)
                FROM touch t JOIN pos p ON p.vertex = t.u
                GROUP BY t.v"""
        ).fetchone()[0]
        from_neighbours += added
        if not added:
            break
    scattered = con.execute(
        f"""INSERT INTO pos
            SELECT node_idx,
                   {cx} + {radius} * sqrt(u1) * cos(2 * pi() * u2),
                   {cy} + {radius} * sqrt(u1) * sin(2 * pi() * u2)
            FROM (SELECT node_idx,
                         (hash(node_idx) % 1000000) / 1000000.0 AS u1,
                         (hash(node_idx * 31 + 17) % 1000000) / 1000000.0 AS u2
                  FROM nodes WHERE node_idx NOT IN (SELECT vertex FROM pos))"""
    ).fetchone()[0]
    con.execute(
        f"""COPY (SELECT CAST(vertex AS INTEGER) AS vertex, CAST(x AS FLOAT) AS x,
                         CAST(y AS FLOAT) AS y
                  FROM pos ORDER BY vertex) TO '{out_path}' (FORMAT PARQUET)"""
    )
    return {"carried": carried, "from_neighbours": from_neighbours, "scattered": scattered}


def add_parser(parser) -> None:
    parser.add_argument("--prev", required=True,
                        help="previous release's coords.parquet (raw layout x, y by id)")
    parser.add_argument("--graph", default=None, help="graph dir (default DATA_DIR/graph)")
    parser.add_argument("--hops", type=int, default=HOPS)


def run(args) -> int:
    graph = Path(args.graph) if args.graph else data_dir() / "graph"
    out = graph / "init_pos.parquet"
    stats = build_init_positions(str(graph / "nodes_int32.parquet"),
                                 str(graph / "edges_int32.parquet"),
                                 args.prev, str(out), hops=args.hops)
    print(f"{stats['carried']:,} carried over, {stats['from_neighbours']:,} placed by "
          f"coauthors, {stats['scattered']:,} scattered -> {out}")
    print("pass --init-pos to slurm/run_layout.py (remote_layout.sh does it when present)")
    return 0
