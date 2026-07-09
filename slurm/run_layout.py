"""Chunked ForceAtlas2 + Leiden on the prepped int32 graph. Resumable via checkpoints."""
import argparse
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
    args = ap.parse_args()

    data = Path(args.data)
    ckpt_dir = data / "layout_ckpt"
    ckpt_dir.mkdir(exist_ok=True)

    edges = cudf.read_parquet(data / "graph" / "edges_int32.parquet")
    G = cugraph.Graph(directed=False)
    G.from_cudf_edgelist(
        edges, source="src", destination="dst", weight="weight", renumber=False
    )
    del edges
    print(f"graph loaded: {G.number_of_nodes():,} nodes", flush=True)

    done, ckpt = latest_checkpoint(ckpt_dir)
    pos = cudf.read_parquet(ckpt) if ckpt else None
    if done:
        print(f"resuming from {ckpt} ({done} iters done)", flush=True)

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
        pos.to_parquet(ckpt_dir / f"pos_{done}.parquet")
        print(f"iters {done}/{args.iters} (+{step} in {time.time()-t0:.0f}s)", flush=True)

    parts, modularity = cugraph.leiden(G, resolution=args.resolution)
    print(f"leiden: modularity={modularity:.4f}, "
          f"communities={parts.partition.nunique():,}", flush=True)

    nodes = cudf.read_parquet(data / "graph" / "nodes_int32.parquet")
    out = (
        nodes.merge(pos, left_on="node_idx", right_on="vertex", how="left")
        .merge(parts, left_on="node_idx", right_on="vertex", how="left")
    )
    out = out.rename(columns={"partition": "community"})
    out[
        ["id", "display_name", "x", "y", "community",
         "works_count", "cited_by_count", "institution", "field"]
    ].to_parquet(data / "coords.parquet")
    isolated = int(out.x.isnull().sum())
    print(f"coords.parquet written ({len(out):,} rows; "
          f"{isolated:,} isolated nodes with null coords)", flush=True)


if __name__ == "__main__":
    main()
