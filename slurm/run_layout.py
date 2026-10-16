"""Chunked ForceAtlas2 + Leiden on the prepped int32 graph. Resumable via checkpoints."""
import argparse
import json
import os
import re
import time
from pathlib import Path


def latest_checkpoint(ckpt_dir: Path) -> tuple[int, Path | None]:
    best = (0, None)
    for f in ckpt_dir.glob("pos_*.parquet"):
        m = re.fullmatch(r"pos_(\d+)\.parquet", f.name)
        if m and int(m.group(1)) > best[0]:
            best = (int(m.group(1)), f)
    return best


GRAPH_FILES = ("nodes_int32.parquet", "edges_int32.parquet")
COORDS_COLUMNS = ["id", "display_name", "x", "y", "community",
                  "works_count", "cited_by_count", "institution", "field"]


def write_host(frame, path: Path) -> None:
    """Write a cudf frame through host memory: one bulk device->host copy
    (to_arrow) then pyarrow's CPU writer, atomically via a .tmp rename.

    libcudf's own GPU parquet writer crawled at ~4MB/min under RMM managed
    memory on an Expanse V100 (2026-10-10): a 100MB checkpoint would have
    taken ~25 minutes, the final coords far longer.
    """
    import pyarrow.parquet as pq
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(frame.to_arrow(), tmp)
    os.replace(tmp, path)   # atomic: a kill mid-write never yields a half file


def read_host(path):
    """Read a parquet file to the GPU through host memory (see write_host)."""
    import cudf
    import pyarrow.parquet as pq
    return cudf.DataFrame.from_arrow(pq.read_table(path))


def assemble_coords(nodes, pos, parts):
    """Join node metadata with FA2 positions and Leiden communities (pyarrow
    tables) on node index. Edgeless nodes outside the graph keep null x/y/community."""
    out = nodes.join(pos.select(["vertex", "x", "y"]), keys="node_idx",
                     right_keys="vertex", join_type="left outer")
    out = out.join(parts.select(["vertex", "partition"]), keys="node_idx",
                   right_keys="vertex", join_type="left outer")
    out = out.rename_columns(["community" if c == "partition" else c for c in out.column_names])
    return out.select(COORDS_COLUMNS)


def graph_fingerprint(graph_dir: Path) -> dict:
    """Byte sizes of the prepped graph: differ for any other release or filter."""
    return {name: (graph_dir / name).stat().st_size for name in GRAPH_FILES}


def check_checkpoints(ckpt_dir: Path, fingerprint: dict, adopt: bool = False) -> None:
    """Checkpoints belong to one graph; refuse to resume another graph's.

    Positions are indexed by node_idx, which `prep` renumbers per graph. A
    pos_1000 left in a reused DATA dir would otherwise be taken as a finished
    layout for the new graph and silently scramble every coordinate.
    """
    stamp = ckpt_dir / "graph.json"
    if stamp.exists():
        held = json.loads(stamp.read_text())
        if held != fingerprint:
            raise SystemExit(
                f"{ckpt_dir} holds checkpoints for a different graph "
                f"({held} vs {fingerprint}); move layout_ckpt/ aside or use a "
                "fresh --data dir")
        return
    if any(ckpt_dir.glob("pos_*.parquet")) and not adopt:
        raise SystemExit(
            f"{ckpt_dir} holds checkpoints with no graph.json, so they cannot be "
            "matched to this graph. If they were computed for exactly this graph, "
            "pass --adopt-checkpoints; otherwise move layout_ckpt/ aside.")
    stamp.write_text(json.dumps(fingerprint))


def main() -> None:
    import cudf
    import cugraph

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with graph/ subdir")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--scaling-ratio", type=float, default=2.0)
    ap.add_argument("--gravity", type=float, default=1.0)
    ap.add_argument("--strong-gravity", action="store_true")
    ap.add_argument("--lin-log", action="store_true")
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--managed-memory", action="store_true",
                    help="RMM managed memory: oversubscribe GPU into host RAM")
    ap.add_argument("--init-pos", default=None,
                    help="starting positions (vertex, x, y) for a cold start, e.g. "
                         "graph/init_pos.parquet from `pipeline warmstart`; ignored "
                         "once checkpoints exist")
    ap.add_argument("--adopt-checkpoints", action="store_true",
                    help="stamp pre-existing unstamped checkpoints as this graph's")
    args = ap.parse_args()
    if args.chunk <= 0 or args.iters < 0:
        raise SystemExit("--chunk must be > 0 and --iters >= 0")

    if args.managed_memory:
        import rmm
        rmm.reinitialize(managed_memory=True)
        print("rmm managed memory enabled", flush=True)

    data = Path(args.data)
    ckpt_dir = data / "layout_ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    check_checkpoints(ckpt_dir, graph_fingerprint(data / "graph"),
                      adopt=args.adopt_checkpoints)

    edges = cudf.read_parquet(data / "graph" / "edges_int32.parquet")
    print(f"edges loaded: {len(edges):,} rows from "
          f"{data / 'graph' / 'edges_int32.parquet'}", flush=True)
    G = cugraph.Graph(directed=False)
    G.from_cudf_edgelist(
        edges, source="src", destination="dst", weight="weight", renumber=False
    )
    del edges
    print(f"graph loaded: {G.number_of_nodes():,} nodes", flush=True)

    done, ckpt = latest_checkpoint(ckpt_dir)
    pos = read_host(ckpt) if ckpt else None
    if done:
        print(f"resuming from {ckpt} ({done} iters done)", flush=True)
    elif args.init_pos:
        pos = read_host(args.init_pos)[["vertex", "x", "y"]]
        n = G.number_of_nodes()
        # renumber=False sizes the graph as max connected vertex + 1; edgeless
        # vertices above that have no place in FA2 (webcoords scatters them)
        pos = pos[pos["vertex"] < n]
        if len(pos) != n:
            raise SystemExit(f"--init-pos has {len(pos):,} rows for a {n:,}-node graph")
        print(f"warm start from {args.init_pos}", flush=True)
    if done >= args.iters:
        print(f"checkpoint at {done} iters >= --iters {args.iters}; "
              "skipping FA2, reusing positions", flush=True)

    while done < args.iters:
        step = min(args.chunk, args.iters - done)
        t0 = time.time()
        pos = cugraph.force_atlas2(
            G,
            max_iter=step,
            pos_list=pos,
            outbound_attraction_distribution=True,
            barnes_hut_optimize=True,
            barnes_hut_theta=0.5,
            scaling_ratio=args.scaling_ratio,
            gravity=args.gravity,
            strong_gravity_mode=args.strong_gravity,
            lin_log_mode=args.lin_log,
        )
        done += step
        write_host(pos, ckpt_dir / f"pos_{done}.parquet")
        print(f"iters {done}/{args.iters} (+{step} in {time.time()-t0:.0f}s)", flush=True)

    if pos is None:
        raise SystemExit(
            "no positions to write: no checkpoint found and --iters is 0"
        )

    parts, modularity = cugraph.leiden(G, resolution=args.resolution)
    print(f"leiden: modularity={modularity:.4f}, "
          f"communities={parts.partition.nunique():,}", flush=True)

    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    out = assemble_coords(pq.read_table(data / "graph" / "nodes_int32.parquet"),
                          pos.to_arrow(), parts.to_arrow())
    tmp = data / "coords.parquet.tmp"
    pq.write_table(out, tmp)
    os.replace(tmp, data / "coords.parquet")
    isolated = int(pc.sum(pc.is_null(out["x"])).as_py() or 0)
    print(f"coords.parquet written ({out.num_rows:,} rows; "
          f"{isolated:,} isolated nodes with null coords)", flush=True)

if __name__ == "__main__":
    main()
