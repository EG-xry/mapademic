"""Split each large community into sub-groups (Leiden on its own coauthor links).

The layout's communities are coarse: one holds most of the world's clinical
medicine (1.4M authors in 2026-09-23), and national research networks sit
beside international fields. Clustering each large community on its internal
links gives the map's second level. In the 2026-10-11 prototype, community 0
split into US specialties (oncology, cardiology, neurology, ...) and national
networks (UK, Germany/Austria/Switzerland, Canada, Australia, the Nordics).
Those sub-groups overlap in the layout itself (well under half of a small area
belongs to its most common sub-group), so webcoords gives each its own area,
palette shades it from its parent's colour and regions names it.

Writes subcommunities.parquet (id, community, sub) for every member of a
split community, where sub 0 is its largest sub-group. Leiden's many small
sub-groups are folded into the major one they share the most link weight
with, so every member lands in a named, shaded area. Also writes
subcommunity_links.parquet (community, a, b, weight): link weight between
major sub-groups, which webcoords uses to place linked sub-groups side by side.
"""
import json
import os
import random
import time
from pathlib import Path

import duckdb
import igraph as ig
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.config import apply_resource_limits, data_dir
from pipeline.palette import sparse_communities

SPLIT_MIN = 20_000        # communities at least this big are split
MAJOR_SHARE = 0.01        # a sub-group needs this share of its community ...
MAJOR_MIN = 1_500         # ... and this many members to keep its own area
MAX_SUBS = 24             # at most this many areas per community (shades stay distinct)
RESOLUTION = 1.0          # Leiden modularity resolution
SEED = 42


def leiden(n: int, src: np.ndarray, dst: np.ndarray, weight: np.ndarray,
           resolution: float = RESOLUTION) -> np.ndarray:
    """Membership array of a modularity Leiden run, deterministic for a given
    graph (python-igraph draws from the `random` module, seeded here)."""
    g = ig.Graph(n=n, edges=np.stack([src, dst], axis=1), directed=False)
    g.es["weight"] = weight.astype(np.float64)
    random.seed(SEED)
    part = g.community_leiden(objective_function="modularity", weights="weight",
                              resolution=resolution, n_iterations=2)
    return np.asarray(part.membership, dtype=np.int64)


def fold_minor(membership: np.ndarray, src: np.ndarray, dst: np.ndarray,
               weight: np.ndarray, major_share: float = MAJOR_SHARE,
               major_min: int = MAJOR_MIN, max_subs: int = MAX_SUBS):
    """-> (sub per node, 0 = largest; links {(a, b): weight} between subs).
    Returns (None, {}) when fewer than two sub-groups are big enough to split."""
    n = len(membership)
    sizes = np.bincount(membership)
    order = np.lexsort((np.arange(len(sizes)), -sizes))           # size desc, label asc
    need = max(major_share * n, major_min)
    major = [int(c) for c in order[:max_subs] if sizes[c] >= need]
    if len(major) < 2:
        return None, {}
    rank = np.full(len(sizes), -1, dtype=np.int64)
    rank[major] = np.arange(len(major))
    # each minor sub-group joins the major one it shares the most weight with
    ra, rb = rank[membership[src]], rank[membership[dst]]
    target = rank.copy()
    minor_ids = np.flatnonzero(rank < 0)
    if len(minor_ids):
        k = len(major)
        votes = np.zeros((len(sizes), k))
        for mine, other, theirs in ((membership[src], rb, ra), (membership[dst], ra, rb)):
            sel = (theirs < 0) & (other >= 0)
            np.add.at(votes, (mine[sel], other[sel]), weight[sel])
        best = votes[minor_ids].argmax(axis=1)
        best[votes[minor_ids].max(axis=1) <= 0] = 0                # unlinked -> largest
        target[minor_ids] = best
    sub = target[membership]
    # re-rank by final size so sub 0 is the largest area
    final = np.bincount(sub, minlength=len(major))
    order2 = np.lexsort((np.arange(len(major)), -final))
    remap = np.empty(len(major), dtype=np.int64)
    remap[order2] = np.arange(len(major))
    sub = remap[sub]
    a, b = sub[src], sub[dst]
    cross = a != b
    lo, hi = np.minimum(a[cross], b[cross]), np.maximum(a[cross], b[cross])
    keys = lo * len(major) + hi
    totals = np.bincount(keys, weights=weight[cross], minlength=len(major) ** 2)
    links = {(int(k // len(major)), int(k % len(major))): float(w)
             for k, w in enumerate(totals) if w > 0}
    return sub, links


def build_subcommunities(coords_path: str, nodes_path: str, edges_path: str,
                         out_path: str, links_path: str, split_min: int = SPLIT_MIN,
                         major_share: float = MAJOR_SHARE, major_min: int = MAJOR_MIN,
                         sparse_min: int = 1000) -> dict:
    con = duckdb.connect()
    apply_resource_limits(con)
    faded = sparse_communities(con, coords_path, min_members=sparse_min)
    sizes = con.execute(
        f"""SELECT community, count(*) n FROM read_parquet('{coords_path}')
            WHERE community IS NOT NULL GROUP BY 1 HAVING count(*) >= {int(split_min)}
            ORDER BY n DESC, community"""
    ).fetchall()
    todo = [(int(c), int(n)) for c, n in sizes if int(c) not in faded]
    con.execute(
        f"""CREATE TEMP TABLE node_comm AS
            SELECT n.node_idx, n.id, c.community
            FROM read_parquet('{nodes_path}') n
            JOIN read_parquet('{coords_path}') c ON c.id = n.id
            WHERE c.community IN ({','.join(str(c) for c, _ in todo) or 'NULL'})"""
    )
    ids_out, comm_out, sub_out, link_rows, report = [], [], [], [], {}
    for comm, n in todo:
        t0 = time.time()
        con.execute(f"""CREATE OR REPLACE TEMP TABLE mem AS
                        SELECT node_idx, id FROM node_comm WHERE community = {comm}
                        ORDER BY node_idx""")
        mem = con.execute("SELECT node_idx, id FROM mem").fetchnumpy()
        idx = mem["node_idx"].astype(np.int64)
        e = con.execute(
            f"""SELECT e.src, e.dst, e.weight FROM read_parquet('{edges_path}') e
                SEMI JOIN mem a ON a.node_idx = e.src
                SEMI JOIN mem b ON b.node_idx = e.dst"""
        ).fetchnumpy()
        src = np.searchsorted(idx, e["src"].astype(np.int64))
        dst = np.searchsorted(idx, e["dst"].astype(np.int64))
        weight = e["weight"].astype(np.float64)
        del e
        membership = leiden(len(idx), src, dst, weight)
        sub, links = fold_minor(membership, src, dst, weight, major_share, major_min)
        if sub is None:
            report[comm] = {"members": n, "subs": 0}
            print(f"community {comm}: {n:,} members, not split", flush=True)
            continue
        ids_out.append(mem["id"])
        comm_out.append(np.full(len(idx), comm, dtype=np.int32))
        sub_out.append(sub.astype(np.int16))
        link_rows += [(comm, a, b, w) for (a, b), w in links.items()]
        k = int(sub.max()) + 1
        report[comm] = {"members": n, "subs": k, "leiden_groups": int(membership.max()) + 1,
                        "edges": int(len(src)), "seconds": round(time.time() - t0, 1)}
        print(f"community {comm}: {n:,} members, {len(src):,} internal links -> "
              f"{k} sub-groups ({time.time() - t0:.0f}s)", flush=True)
    table = pa.table({
        "id": pa.array(np.concatenate(ids_out) if ids_out else np.array([], dtype=object), pa.string()),
        "community": pa.array(np.concatenate(comm_out) if comm_out else np.array([], np.int32), pa.int32()),
        "sub": pa.array(np.concatenate(sub_out) if sub_out else np.array([], np.int16), pa.int16()),
    })
    for path, tab in ((out_path, table), (links_path, pa.table({
            "community": pa.array([r[0] for r in link_rows], pa.int32()),
            "a": pa.array([r[1] for r in link_rows], pa.int16()),
            "b": pa.array([r[2] for r in link_rows], pa.int16()),
            "weight": pa.array([r[3] for r in link_rows], pa.float64())}))):
        tmp = str(path) + ".tmp"
        pq.write_table(tab, tmp)
        os.replace(tmp, path)
    meta = {"split_min": split_min, "major_share": major_share, "major_min": major_min,
            "max_subs": MAX_SUBS, "resolution": RESOLUTION, "faded": sorted(faded),
            "communities": {str(c): r for c, r in report.items()}}
    Path(str(out_path) + ".meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def add_parser(parser) -> None:
    parser.add_argument("--coords", default=None, help="layout output (default coords.parquet)")
    parser.add_argument("--split-min", type=int, default=SPLIT_MIN,
                        help="split communities with at least this many members")


def run(args) -> int:
    d = data_dir()
    meta = build_subcommunities(
        args.coords or str(d / "coords.parquet"), str(d / "graph" / "nodes_int32.parquet"),
        str(d / "graph" / "edges_int32.parquet"), str(d / "subcommunities.parquet"),
        str(d / "subcommunity_links.parquet"), split_min=args.split_min)
    split = {c: r for c, r in meta["communities"].items() if r["subs"]}
    print(f"{len(split)} communities split into {sum(r['subs'] for r in split.values())} "
          f"sub-groups -> {d / 'subcommunities.parquet'}")
    return 0
