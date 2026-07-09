# Mapademic Layout + Communities (Plan 2 of 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `nodes.parquet` (8,587,906 authors) + `edges.parquet` (379,431,851 weighted coauthor edges) on Anvil scratch into `coords.parquet`: one row per author with `x, y` map position (cuGraph ForceAtlas2) and `community` (GPU Leiden).

**Architecture:** One new CPU pipeline stage (`prep`) renumbers author ids to int32 and emits GPU-ready edge/node files (testable locally, runs free on Anvil login). The GPU work is a single Slurm job on 1× A100: chunked ForceAtlas2 with parquet checkpoints every chunk (wall-time kills lose nothing), then Leiden on the same in-GPU graph, then a merge back to author ids. QA is a cheap scatter render reviewed before spending more SUs on tuning re-runs.

**Tech Stack:** DuckDB (prep), RAPIDS pip wheels (`cudf-cu12`, `cugraph-cu12` from pypi.nvidia.com) in a dedicated `.venv-gpu` on Anvil, Slurm (`gpu-debug` for smoke, `gpu` for real runs), matplotlib locally for QA renders.

**Spec:** `docs/superpowers/specs/2026-07-05-mapademic-design.md` stage 3. **One planned deviation:** the spec's node2vec→UMAP fallback layout is NOT built in this plan - cuGraph's node2vec emits random walks that still need CPU word2vec training, a large build for something we may never use. We build FA2 with tuning knobs first; if visual QA fails after 2-3 tuning runs, a follow-up task adds the fallback. (YAGNI; flagged to Eric in the plan summary.)

## Global Constraints

- Anvil account `bio260224-gpu` has **~194 GPU SUs** (1 SU = 1 GPU-hour). Hard budget for this plan: **≤ 40 SUs**; expected ≈ 10-20. Login-node CPU work is free. Never submit to CPU partitions (no CPU allocation).
- Anvil paths: repo `~/mapademic`, data `$SCRATCH/mapademic` (`/anvil/scratch/x-egao2/mapademic`), login node with tmux sessions: `login06`. CPU venv `~/mapademic/.venv` (python 3.11 via `module load anaconda/2024.02-py311`); GPU venv `~/mapademic/.venv-gpu` (created in Task 2).
- All login-node DuckDB work runs with `MAPADEMIC_THREADS=8 MAPADEMIC_MEMORY_LIMIT=24GB` and `taskset -c 0-7` (login cgroup OOM-kills uncapped DuckDB - learned the hard way).
- int32 renumbering: `node_idx` is `0..N-1 INTEGER`, deterministic (`ORDER BY id`); edge weights cast to `FLOAT`. GPU memory fallback if FA2 OOMs on 379M edges: `prep --min-weight 0.34` (drops pairs whose only link is one ≥4-author paper) - do NOT invent other pruning schemes.
- ForceAtlas2 defaults for the first real run (from the spec: ~1000 iterations, degree-weighted repulsion): `max_iter=1000` in chunks of 100, `barnes_hut_optimize=True`, `barnes_hut_theta=0.5`, `outbound_attraction_distribution=True`, `scaling_ratio=2.0`, `gravity=1.0`, `strong_gravity_mode=False`, `lin_log_mode=False`. Tuning runs change ONLY these named knobs.
- Leiden: `cugraph.leiden(G, resolution=1.0)`; record modularity in the run log.
- `coords.parquet` output columns (Plan 3 consumes exactly this): `id VARCHAR, display_name VARCHAR, x DOUBLE, y DOUBLE, community BIGINT, works_count, cited_by_count, institution VARCHAR, field VARCHAR`.
- Mac-side test commands use `.venv/bin/pytest`; commits conventional-commit style; NEVER Claude as co-author (CLAUDE.md rule 6).
- GPU scripts (`slurm/`) cannot be unit-tested locally (no CUDA on the Mac). Their gate is the Task 2/3 smoke runs on `gpu-debug` plus code review - keep them small and boring.

---

### Task 1: `prep` stage - int32 graph files (Mac, TDD)

**Files:**
- Create: `pipeline/prep_graph.py`
- Modify: `pipeline/__main__.py` (register stage as `prep`)
- Test: `tests/test_prep_graph.py`

**Interfaces:**
- Consumes: `nodes.parquet` (`id, display_name, works_count, cited_by_count, institution, field`), `edges.parquet` (`src, dst, weight`)
- Produces: `pipeline.prep_graph.prep_graph(nodes_path: str, edges_path: str, out_dir: Path, min_weight: float = 0.0) -> tuple[int, int]` returning `(n_nodes, n_edges)`; writes `<out_dir>/nodes_int32.parquet` (`node_idx INTEGER` + all node columns) and `<out_dir>/edges_int32.parquet` (`src INTEGER, dst INTEGER, weight FLOAT`). CLI: `python -m pipeline prep [--min-weight W] [--nodes P] [--edges P] [--out DIR]`, default out `<DATA_DIR>/graph`.

- [ ] **Step 1: Write the failing tests**

`tests/test_prep_graph.py`:

```python
import duckdb
import pytest

from pipeline.prep_graph import prep_graph

A, B, C = "https://openalex.org/A1", "https://openalex.org/A2", "https://openalex.org/A3"


@pytest.fixture
def graph_inputs(tmp_path):
    con = duckdb.connect()
    con.execute(
        f"""
        COPY (SELECT * FROM (VALUES
            ('{B}', 'Bee', 20, 200, 'UCSD', 'Neuroscience'),
            ('{A}', 'Aye', 30, 300, NULL, NULL),
            ('{C}', 'Sea', 15, 50, NULL, 'Biology')
        ) t(id, display_name, works_count, cited_by_count, institution, field))
        TO '{tmp_path / "nodes.parquet"}' (FORMAT PARQUET)
        """
    )
    con.execute(
        f"""
        COPY (SELECT * FROM (VALUES
            ('{A}', '{B}', 2.0),
            ('{B}', '{C}', 0.25)
        ) t(src, dst, weight))
        TO '{tmp_path / "edges.parquet"}' (FORMAT PARQUET)
        """
    )
    return tmp_path


def test_renumbers_nodes_deterministically_by_id(graph_inputs):
    out = graph_inputs / "graph"
    n_nodes, n_edges = prep_graph(
        str(graph_inputs / "nodes.parquet"),
        str(graph_inputs / "edges.parquet"),
        out,
    )
    assert (n_nodes, n_edges) == (3, 2)
    rows = duckdb.sql(
        f"SELECT node_idx, id, display_name FROM '{out / 'nodes_int32.parquet'}' ORDER BY node_idx"
    ).fetchall()
    assert rows == [(0, A, "Aye"), (1, B, "Bee"), (2, C, "Sea")]  # ORDER BY id


def test_edges_remapped_to_int32_with_float_weight(graph_inputs):
    out = graph_inputs / "graph"
    prep_graph(
        str(graph_inputs / "nodes.parquet"),
        str(graph_inputs / "edges.parquet"),
        out,
    )
    schema = {
        name: typ
        for name, typ, *_ in duckdb.sql(
            f"DESCRIBE SELECT * FROM '{out / 'edges_int32.parquet'}'"
        ).fetchall()
    }
    assert schema == {"src": "INTEGER", "dst": "INTEGER", "weight": "FLOAT"}
    rows = duckdb.sql(
        f"SELECT src, dst, weight FROM '{out / 'edges_int32.parquet'}' ORDER BY src"
    ).fetchall()
    assert rows == [(0, 1, 2.0), (1, 2, 0.25)]


def test_min_weight_prunes_edges_not_nodes(graph_inputs):
    out = graph_inputs / "graph"
    n_nodes, n_edges = prep_graph(
        str(graph_inputs / "nodes.parquet"),
        str(graph_inputs / "edges.parquet"),
        out,
        min_weight=0.34,
    )
    assert (n_nodes, n_edges) == (3, 1)  # weak B-C edge dropped, node C kept
    rows = duckdb.sql(
        f"SELECT src, dst FROM '{out / 'edges_int32.parquet'}'"
    ).fetchall()
    assert rows == [(0, 1)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_prep_graph.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.prep_graph'`

- [ ] **Step 3: Write the implementation**

`pipeline/prep_graph.py`:

```python
"""Renumber author ids to int32 and emit GPU-ready graph files for layout."""
from pathlib import Path

import duckdb

from pipeline.config import apply_resource_limits, data_dir


def prep_graph(
    nodes_path: str, edges_path: str, out_dir: Path, min_weight: float = 0.0
) -> tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    apply_resource_limits(con)
    con.execute(f"SET temp_directory='{out_dir / '.duckdb_tmp'}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute(
        f"""
        CREATE VIEW numbered AS
        SELECT CAST(row_number() OVER (ORDER BY id) - 1 AS INTEGER) AS node_idx, *
        FROM read_parquet('{nodes_path}')
        """
    )
    n_nodes = con.execute(
        f"""
        COPY (SELECT * FROM numbered ORDER BY node_idx)
        TO '{out_dir / "nodes_int32.parquet"}' (FORMAT PARQUET)
        """
    ).fetchone()[0]
    n_edges = con.execute(
        f"""
        COPY (
            SELECT na.node_idx AS src, nb.node_idx AS dst,
                   CAST(e.weight AS FLOAT) AS weight
            FROM read_parquet('{edges_path}') e
            JOIN numbered na ON na.id = e.src
            JOIN numbered nb ON nb.id = e.dst
            WHERE e.weight >= {float(min_weight)}
        ) TO '{out_dir / "edges_int32.parquet"}' (FORMAT PARQUET)
        """
    ).fetchone()[0]
    return n_nodes, n_edges


def add_parser(parser) -> None:
    parser.add_argument("--min-weight", type=float, default=0.0)
    parser.add_argument("--nodes", default=None, help="override nodes.parquet path")
    parser.add_argument("--edges", default=None, help="override edges.parquet path")
    parser.add_argument("--out", default=None, help="override output dir")


def run(args) -> int:
    nodes = args.nodes or str(data_dir() / "nodes.parquet")
    edges = args.edges or str(data_dir() / "edges.parquet")
    out = Path(args.out) if args.out else data_dir() / "graph"
    n_nodes, n_edges = prep_graph(nodes, edges, out, min_weight=args.min_weight)
    print(f"{n_nodes:,} nodes, {n_edges:,} edges (min_weight={args.min_weight}) -> {out}")
    return 0
```

In `pipeline/__main__.py`, grow the registry:

```python
from pipeline import build_edges, download, extract_works, filter_authors, prep_graph

STAGES: dict = {
    "download": download,
    "extract": extract_works,
    "filter": filter_authors,
    "edges": build_edges,
    "prep": prep_graph,
}
```

Note: the `ORDER BY node_idx` inside the nodes COPY is required - `preserve_insertion_order=false` would otherwise let window-function output land in arbitrary row order, and Plan 3 treats row order as meaningful for compact per-tile indexes.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest -v`
Expected: all PASS (3 new + 24 existing = 27)

- [ ] **Step 5: Commit**

```bash
git add pipeline/prep_graph.py pipeline/__main__.py tests/test_prep_graph.py
git commit -m "feat: prep stage - int32 renumbered graph files for GPU layout"
```

---

### Task 2: GPU environment + smoke test on gpu-debug (Anvil)

**Files:**
- Create: `slurm/setup_gpu_env.sh`
- Create: `slurm/smoke.sbatch`
- Create: `slurm/smoke_test.py`

**Interfaces:**
- Consumes: nothing from the pipeline (synthetic graph)
- Produces: `~/mapademic/.venv-gpu` on Anvil with working `cudf`/`cugraph`; proof (job log) that FA2 + Leiden run on an A100 with the exact API calls Task 3 uses. Cost: ≤ 0.5 SU.

- [ ] **Step 1: Write the three files**

`slurm/setup_gpu_env.sh`:

```bash
#!/bin/bash
# Run ON ANVIL (login node OK - pip install only, no GPU needed):
#   bash slurm/setup_gpu_env.sh
set -euo pipefail
module load anaconda/2024.02-py311
cd "$HOME/mapademic"
python3 -m venv .venv-gpu
.venv-gpu/bin/pip install --quiet --upgrade pip
# RAPIDS pip wheels (CUDA 12 bundled; node driver must be >= 525 - smoke job verifies)
.venv-gpu/bin/pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12>=24.10" "cugraph-cu12>=24.10" pyarrow
echo "GPU venv ready: $HOME/mapademic/.venv-gpu"
```

`slurm/smoke_test.py`:

```python
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
```

`slurm/smoke.sbatch`:

```bash
#!/bin/bash
#SBATCH -A bio260224-gpu
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:15:00
#SBATCH -J mapademic-smoke
#SBATCH -o %x-%j.log
set -euo pipefail
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
cd "$HOME/mapademic"
.venv-gpu/bin/python slurm/smoke_test.py
```

- [ ] **Step 2: Sync to Anvil, build the env, submit the smoke job**

```bash
rsync -az --exclude .venv --exclude .git --exclude data --exclude .superpowers \
  --exclude .pytest_cache --exclude __pycache__ --exclude "*.egg-info" \
  /Users/eric/Downloads/mapademic/ x-egao2@anvil.rcac.purdue.edu:mapademic/
ssh x-egao2@anvil.rcac.purdue.edu 'bash mapademic/slurm/setup_gpu_env.sh'
ssh x-egao2@anvil.rcac.purdue.edu 'cd mapademic && sbatch slurm/smoke.sbatch'
```

- [ ] **Step 3: Verify the smoke log**

```bash
ssh x-egao2@anvil.rcac.purdue.edu 'cd mapademic && tail -20 mapademic-smoke-*.log'
```

Expected: A100 line from nvidia-smi (driver ≥ 525), version prints, `leiden modularity: ...`, `SMOKE OK`. If the pip wheels fail on the node (driver too old / import error), fall back to Apptainer: `apptainer pull docker://rapidsai/base:24.10-cuda12.0-py3.11` into `$SCRATCH` and swap the sbatch python line for `apptainer exec --nv` - record whichever path worked in the run log.

- [ ] **Step 4: Commit (from the Mac)**

```bash
git add slurm/setup_gpu_env.sh slurm/smoke.sbatch slurm/smoke_test.py
git commit -m "feat: Anvil GPU env setup and FA2/Leiden smoke test"
```

---

### Task 3: Layout + Leiden job with checkpointed chunks (Anvil GPU)

**Files:**
- Create: `slurm/run_layout.py`
- Create: `slurm/layout.sbatch`

**Interfaces:**
- Consumes: `$SCRATCH/mapademic/graph/{nodes_int32,edges_int32}.parquet` (Task 1 output, produced on Anvil)
- Produces: `$SCRATCH/mapademic/coords.parquet` (schema in Global Constraints); checkpoints `$SCRATCH/mapademic/layout_ckpt/pos_<iters_done>.parquet`. Resume = resubmit the same job; it loads the newest checkpoint.

- [ ] **Step 1: Write the layout runner**

`slurm/run_layout.py`:

```python
"""Chunked ForceAtlas2 + Leiden on the prepped int32 graph. Resumable via checkpoints."""
import argparse
import re
import time
from pathlib import Path

import cudf
import cugraph


def latest_checkpoint(ckpt_dir: Path) -> tuple[int, Path | None]:
    best = (0, None)
    for f in ckpt_dir.glob("pos_*.parquet"):
        m = re.fullmatch(r"pos_(\d+)\.parquet", f.name)
        if m and int(m.group(1)) > best[0]:
            best = (int(m.group(1)), f)
    return best


def main() -> None:
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
```

Note: nodes with zero surviving edges never enter the cugraph graph, so their `x/y/community` are null after the left merges. They are kept in `coords.parquet` (Plan 3 decides whether to scatter them in a ring or drop them); the count is printed so the decision is data-informed.

`slurm/layout.sbatch`:

```bash
#!/bin/bash
#SBATCH -A bio260224-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -t 06:00:00
#SBATCH -J mapademic-layout
#SBATCH -o %x-%j.log
set -euo pipefail
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
cd "$HOME/mapademic"
.venv-gpu/bin/python slurm/run_layout.py \
    --data "${DATA:-$SCRATCH/mapademic}" \
    --iters "${ITERS:-1000}" --chunk "${CHUNK:-100}" \
    ${LAYOUT_ARGS:-}
```

(`DATA`/`ITERS`/`CHUNK`/`LAYOUT_ARGS` come via `sbatch --export=ALL,ITERS=...,LAYOUT_ARGS="--lin-log"` so tuning re-runs and the Task 3 minitest don't edit the script.)

- [ ] **Step 2: Mini end-to-end validation on gpu-debug**

Run Task 1's `prep` on Anvil against a small synthetic nodes/edges pair to produce a real `graph/` dir, then run the layout job on it (few thousand nodes, `--iters 50`):

```bash
ssh x-egao2@anvil.rcac.purdue.edu 'cd mapademic && module load anaconda/2024.02-py311 && \
  MAPADEMIC_THREADS=4 MAPADEMIC_MEMORY_LIMIT=8GB .venv/bin/python - <<EOF
import duckdb, os
S = os.environ["SCRATCH"] + "/mapademic/minitest"
os.makedirs(S, exist_ok=True)
con = duckdb.connect()
con.execute(f"""COPY (
  SELECT 'https://openalex.org/A' || i AS id, 'Author ' || i AS display_name,
         20 AS works_count, 100 AS cited_by_count,
         CAST(NULL AS VARCHAR) AS institution, CAST(NULL AS VARCHAR) AS field
  FROM range(5000) t(i)) TO '{S}/nodes.parquet' (FORMAT PARQUET)""")
con.execute(f"""COPY (
  SELECT 'https://openalex.org/A' || (random()*4999)::INT AS src,
         'https://openalex.org/A' || (random()*4999)::INT AS dst,
         random()::DOUBLE AS weight
  FROM range(50000)) TO '{S}/edges.parquet' (FORMAT PARQUET)""")
EOF
MAPADEMIC_DATA=$SCRATCH/mapademic/minitest MAPADEMIC_THREADS=4 MAPADEMIC_MEMORY_LIMIT=8GB \
  .venv/bin/python -m pipeline prep && \
  sbatch -p gpu-debug -t 00:15:00 \
    --export=ALL,ITERS=50,CHUNK=25,DATA=$SCRATCH/mapademic/minitest \
    --job-name mapademic-minitest \
    -o minitest-%j.log slurm/layout.sbatch'
```

(The submit uses `--export=ALL,ITERS=50,CHUNK=25,DATA=$SCRATCH/mapademic/minitest` - the sbatch's `DATA` override targets the minitest dir.)

Wait for the job, then verify: log shows a resume-free run, two checkpoints (`pos_25`, `pos_50`), a Leiden modularity line, and `$SCRATCH/mapademic/minitest/coords.parquet` with 5,000 rows and non-null x/y for connected nodes. **Also verify resume:** resubmit the same job with `ITERS=75` and confirm the log prints `resuming from ... (50 iters done)`.

- [ ] **Step 3: Commit (from the Mac)**

```bash
git add slurm/run_layout.py slurm/layout.sbatch
git commit -m "feat: checkpointed FA2 + Leiden layout job for Anvil A100"
```

---

### Task 4: Real prep + full layout run + QA render

**Files:**
- Create: `scripts/qa_render.py` (Mac)
- No pipeline code changes - this is the production run.

**Interfaces:**
- Consumes: everything above
- Produces: `$SCRATCH/mapademic/coords.parquet` (real), QA PNGs on the Mac, and a go/no-go decision on layout quality. Expected cost: 2-6 SUs for the first full run.

- [ ] **Step 1: Run `prep` on Anvil against the real graph** (login node, free; ~20-40 min)

```bash
ssh x-egao2@anvil.rcac.purdue.edu 'cd mapademic && module load anaconda/2024.02-py311 && \
  tmux new-session -d -s prep "MAPADEMIC_DATA=$SCRATCH/mapademic MAPADEMIC_THREADS=8 \
  MAPADEMIC_MEMORY_LIMIT=24GB taskset -c 0-7 .venv/bin/python -m pipeline prep \
  2>&1 | tee -a $SCRATCH/mapademic/prep.log"'
```

Verify after: `graph/nodes_int32.parquet` has 8,587,906 rows, `graph/edges_int32.parquet` ~379M rows, `node_idx` max = 8,587,905.

- [ ] **Step 2: Submit the full layout job**

```bash
ssh x-egao2@anvil.rcac.purdue.edu 'cd mapademic && sbatch slurm/layout.sbatch'
```

Monitor via `squeue -u x-egao2` and the job log's per-chunk timings. If the first chunk OOMs on the A100 (40GB): rerun prep with `--min-weight 0.34` (Global Constraints fallback) and resubmit - do not tune other memory knobs first.

- [ ] **Step 3: Write the QA renderer (Mac)**

`scripts/qa_render.py`:

```python
"""Quick-look renders of coords.parquet - one full view + one zoom, colored by community."""
import argparse
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(coords: str, out_dir: Path, sample: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = duckdb.sql(
        f"""
        SELECT x, y, community % 40 AS hue
        FROM '{coords}'
        WHERE x IS NOT NULL
        USING SAMPLE {sample} ROWS (reservoir, 42)
        """
    ).df()
    for name, (w, h) in {"overview": (16, 16), "zoom": (16, 16)}.items():
        fig, ax = plt.subplots(figsize=(w, h), facecolor="black")
        d = df
        if name == "zoom":
            cx, cy = df.x.median(), df.y.median()
            sx, sy = df.x.std() * 0.15, df.y.std() * 0.15
            d = df[(df.x.between(cx - sx, cx + sx)) & (df.y.between(cy - sy, cy + sy))]
        ax.scatter(d.x, d.y, s=0.05, c=d.hue, cmap="tab20", alpha=0.5, linewidths=0)
        ax.set_axis_off()
        fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight",
                    facecolor="black")
        plt.close(fig)
        print(f"wrote {out_dir / f'{name}.png'} ({len(d):,} points)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("coords")
    ap.add_argument("--out", default="data/qa")
    ap.add_argument("--sample", type=int, default=2_000_000)
    args = ap.parse_args()
    render(args.coords, Path(args.out), args.sample)
```

Install matplotlib into the Mac venv for this: `.venv/bin/pip install matplotlib` (dev-only tool; do not add to pyproject deps).

- [ ] **Step 4: Pull coords, render, review with Eric**

```bash
rsync -avP x-egao2@anvil.rcac.purdue.edu:/anvil/scratch/x-egao2/mapademic/coords.parquet \
  /Users/eric/Downloads/mapademic/data/
.venv/bin/python scripts/qa_render.py data/coords.parquet
```

QA checklist (spec): do fields form continents (colors cluster spatially)? Does zoom show local structure, not uniform fuzz? Spot-check Hinton's neighborhood. **Show Eric the PNGs and get an explicit verdict before any tuning re-runs** - each re-run costs SUs. Budget: at most 2-3 tuning runs (`LAYOUT_ARGS` knobs only), then either accept or escalate to the node2vec fallback discussion.

- [ ] **Step 5: Commit QA tooling + record the run**

```bash
git add scripts/qa_render.py
git commit -m "feat: QA render script for layout coordinate review"
```

Append run facts (SUs spent, chunk timings, modularity, community count, verdict) to `docs/superpowers/plans/2026-07-09-layout-communities.md` as a "Run log" section and commit.

---

## Run log (2026-07-09/10)

- **Run 1 = FINAL** (Anvil job 19007814, A100, full 379M edges, default knobs,
  1000 iters, ~15 min): modularity 0.8034, 216,620 communities, 8,587,906
  rows, 0 isolated. Macro field-lobes real (confirmed by density-equalized
  renders); soft cluster edges; large sparse halo (extent ±700k, core ±~100k).
- Run 2 (job 19013701, `--lin-log --strong-gravity`): over-compressed uniform
  disk, structure destroyed. Rejected.
- Run 3 (job 19018133, `--lin-log`, 2000 iters): halo exploded to ±56M,
  single-blob core, colors don't separate. Rejected. Lesson: lin-log variants
  hurt at this scale/weighting; default FA2 balance wins.
- Expanse lane (V100 32GB): both attempts OOM'd at graph build even with
  edges pruned to 95.9M (min-weight 0.34) - the allocator asks for the same
  ~6.07GB regardless, memory bound is structural. A100 40GB required for the
  full graph. Expanse env is built and staged as standby.
- Cost: ~3 SUs Anvil (3 layout runs + smoke + minitest), ~0.4 SUs Expanse.
- **ACTUAL coords.parquet schema (Plan 3: code against THIS, not the plan text
  above):** `id VARCHAR, display_name VARCHAR, x FLOAT, y FLOAT,
  community INTEGER, works_count INTEGER, cited_by_count INTEGER,
  institution VARCHAR, field VARCHAR`. x/y are float32 (cuGraph output;
  ~0.06-unit resolution at the ±700k halo edge - fine for tiles, stated
  deliberately). The plan's earlier `DOUBLE`/`BIGINT` wording is superseded.
- **coords.parquet row order is ARBITRARY** (cudf hash merges; verified
  ~3k inversions per 100k rows). Plan 3 must join/sort by `id` (or re-derive
  node order from `nodes_int32.parquet`, whose physical order IS pinned).
  Any future layout re-run should add `.sort_values("node_idx")` before
  `to_parquet` if aligned order is wanted.
- **Checkpoint discipline:** `layout_ckpt/` is shared per DATA dir and keyed
  only by iteration count, NOT by knobs. Runs 2 and 3 were cold-started (the
  controller archived/wiped `layout_ckpt` between runs - so the run
  comparisons above are clean), but nothing enforces this: wipe or redirect
  the checkpoint dir before any re-run with different LAYOUT_ARGS.
- Run 1 executed with `sbatch -p gpu -t 02:00:00` (not the sbatch file's 6h
  default): account `bio260224-gpu` has no Slurm association for the `ai`
  partition, and the short walltime was chosen for backfill (it worked -
  started ~2 days before its scheduler estimate).
- `--managed-memory` flag: added during Expanse OOM debugging, placement
  follows the documented RAPIDS pattern, but it has never run on a successful
  job - treat as unproven, not as an established OOM escape hatch.
- Expanse standby env was built ad hoc (py3.13 venv + pinned 26.6 wheels +
  `LD_LIBRARY_PATH` over the lib package dirs, per the sbatch files); there is
  no committed Expanse setup script - write one before relying on that lane.
- Artifacts: coords.parquet (=run1) canonical on Anvil scratch, Eric's Mac
  (`data/coords.parquet`), and external drive (`artifacts/`), all verified by
  row count 8,587,906.

## Out of scope for this plan

- node2vec→UMAP fallback layout (built only if FA2 QA fails; see deviation note)
- Tile rendering, label tiles, search index (Plan 3)
- Web frontend, hosting (Plan 4)
