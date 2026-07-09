"""Tiny FA2 + Leiden on a synthetic graph - proves the exact API Task 3 relies on."""
import cudf
import cugraph
import numpy as np

print("cudf", cudf.__version__, "| cugraph", cugraph.__version__)

rng = np.random.default_rng(0)
n, m = 10_000, 100_000
edges = cudf.DataFrame(
    {
        "src": rng.integers(0, n, m, dtype=np.int32),
        "dst": rng.integers(0, n, m, dtype=np.int32),
        "weight": rng.random(m, dtype=np.float32),
    }
)
edges = edges[edges.src != edges.dst]

G = cugraph.Graph(directed=False)
G.from_cudf_edgelist(
    edges, source="src", destination="dst", weight="weight", renumber=False
)

pos = cugraph.force_atlas2(
    G,
    max_iter=50,
    outbound_attraction_distribution=True,
    barnes_hut_optimize=True,
    barnes_hut_theta=0.5,
    scaling_ratio=2.0,
    gravity=1.0,
)
assert {"vertex", "x", "y"} <= set(pos.columns), pos.columns
assert not pos.x.isnull().any() and not pos.y.isnull().any()

# warm-restart contract used for checkpointing in Task 3
pos2 = cugraph.force_atlas2(G, max_iter=10, pos_list=pos)
assert len(pos2) == len(pos)

parts, modularity = cugraph.leiden(G, resolution=1.0)
assert {"vertex", "partition"} <= set(parts.columns), parts.columns
print("leiden modularity:", modularity, "| communities:", parts.partition.nunique())
print("SMOKE OK")
