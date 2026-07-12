# mapademic

A zoomable map of 8,587,906 researchers: OpenAlex authors with 15 or more
works, positioned by coauthorship and colored by detected community. The map
is served as a static site: raster tiles plus JSON indexes, rendered in the
browser by a vanilla-JS MapLibre viewer (`web/dev.html`). No backend is
required for panning, zooming, or search; live path-finding between two
researchers is provided by a companion project,
[academic-degree-of-separation](https://github.com/riptideiv/academic-degree-of-separation).

## Features

- Continuous zoom z0-11, with native pre-rendered tiles through z10.
- Author search across all 8.59M names, served from prefix-sharded JSON
  indexed by first and last name token.
- Hover and click for author details, enriched live from the OpenAlex API.
- Coauthor rays for a searched author, and a shortest-path overlay between
  two authors via the companion backend.
- Shareable URLs that encode view, active search, and active path.
- Display settings: brightness, contrast, saturation, label density, author
  label modes (normal / faint / hover-only), point rendering modes (raster,
  vector, or both), and citation-size contrast.
- Community legend with an isolate mode, and named regions overlaid on the
  map.

## Methodology

### Data

An OpenAlex snapshot is the source. Authors are filtered to those with 15 or
more works: 8.59 million of roughly 100 million author records are kept.
Coauthor edges are built from works with 50 or fewer authors (a single
paper with thousands of authors would otherwise contribute millions of
clique edges), weighted by repeat collaboration with fractional credit per
work. 379,431,851 edges are produced this way.

### Layout

Positions are computed with ForceAtlas2 on cuGraph, run on an A100 GPU for
1000 iterations with Barnes-Hut approximation (theta 0.5). The run is
chunked with checkpoints so it can resume after interruption
(`slurm/run_layout.py`). Communities are detected with Leiden in the same
job: modularity 0.8034, roughly 216,000 communities, of which 37 have 1000
or more members and together hold 97.2% of authors.

### Post-processing

Raw layout coordinates are median-centered (mean is sensitive to outlier
halos) and radially compressed with an asinh transform into the unit
square. Roughly 207,000 authors in tiny disconnected components are
re-scattered as background dust rather than left at their natural position,
which is a ring at the layout's gravity/repulsion equilibrium. The centroid
of each community with 1000 or more members is expanded outward (spread
factor 1.6), opening separation between adjacent fields.

### Color

Hue is assigned from each community's centroid angle around the map center,
so spatially neighboring communities receive neighboring hues, with a small
per-community jitter in lightness and saturation. Communities below the
1000-member threshold are rendered grey.

### Rendering

Dominant community and log-density brightness are computed per pixel at
zoom 10 (a 262144 x 262144 virtual grid), then reduced by pyramid
downsampling through zoom 0 and written as PNG tiles. Coauthor edges with
weight 2 or higher and bounded length are baked into tiles at zoom 8-9.
Citation "splats" (bright markers for highly-cited authors) are baked at
zoom 9-10 in three size tiers.

### Validation

Across all 379M edges, compared against a null of 2M random author pairs:
frequent-collaborator pairs (weight 5 or higher) have a median map distance
of 0.0145, versus 0.266 for random pairs, roughly 18x closer, and share a
detected community 92% of the time versus 7% expected by chance. In a
300,000-author sample, 94% have at least one coauthor within 2% of the map
width.

## Reproducing the pipeline

### Prerequisites

- Python 3.11 or newer.
- A virtual environment with the project installed:

  ```
  python3 -m venv .venv
  .venv/bin/pip install -e '.[dev]'
  ```

  `duckdb`, `numpy`, and `pillow` (declared in `pyproject.toml`) are
  installed as dependencies this way; the `dev` extra additionally provides
  `pytest` for the test suite.
- Roughly 30GB of free disk for pipeline artifacts, written under
  `MAPADEMIC_DATA` (default `./data`, git-ignored).
- For the layout stage only: a machine with CUDA and RAPIDS cuGraph
  (`cudf`, `cugraph`) available. This is run through the Slurm scripts in
  `slurm/` or on any A100 machine directly; `slurm/setup_gpu_env.sh`
  documents the environment setup, and `slurm/layout.sbatch` /
  `slurm/layout-expanse.sbatch` are cluster-specific submission wrappers
  around `slurm/run_layout.py`.

Two environment variables tune resource use on shared hosts, read by every
DuckDB-backed stage: `MAPADEMIC_THREADS` caps the thread count, and
`MAPADEMIC_MEMORY_LIMIT` caps memory (e.g. `24GB`).

### Stages

Each stage is a subcommand of `python -m pipeline`; output is checkpointed
under `MAPADEMIC_DATA` (default `./data`), so a stage can be re-run without
redoing upstream work. The order below is required; several stages depend
on files written by earlier ones. All paths below assume the default;
`MAPADEMIC_DATA` can be exported before the first stage to relocate
everything, in which case the same directory should also be passed as
`--data` to `slurm/run_layout.py`, which has its own required flag and does
not read the environment variable.

```
python -m pipeline download                          # data/snapshot/authors/  (~53GB, S3 sync)
python -m pipeline extract                            # data/works_authorships/*.parquet
python -m pipeline filter --min-works 15               # data/nodes.parquet     (8.59M authors)
python -m pipeline edges                               # data/edges.parquet     (379M edges, ~7.5GB)
python -m pipeline prep                                # data/graph/*_int32.parquet (GPU-ready)
```

The layout stage runs separately, on GPU hardware, and writes
`data/coords.parquet`:

```
python slurm/run_layout.py --data data --iters 1000
```

With `data/coords.parquet` in place, the remaining stages run on CPU again.
`edgepx` must run after `webcoords` and before `tiles --edges`, since it
reads both `coords_web.parquet` and `edges.parquet`:

```
python -m pipeline webcoords --spread 1.6                          # data/coords_web.parquet
python -m pipeline edgepx                                          # data/edges_px.parquet
python -m pipeline tiles --zooms 0-10 --edges data/edges_px.parquet  # data/tiles/  (~4.7GB)
python -m pipeline regions --authors "data/snapshot/authors/*/*.parquet"  # data/regions.json
python -m pipeline index                                           # data/index/  (~3.5GB: label tiles, search shards, legend, community rosters)
```

`regions` needs the authors snapshot from `download` to name communities by
topic; its `--authors` default is a path specific to the original run and
should be overridden as shown, pointing at wherever `download` wrote the
snapshot.

### Preview

```
bash scripts/preview.sh
```

The viewer is then available at `http://localhost:8123/web/dev.html`.

Search, hover, click, labels, and the community legend work without any
other service running. Coauthor rays and the shortest-path overlay require
the companion backend on `localhost:8000`, run from within the
`academic-degree-of-separation` repository:

```
uvicorn backend.app:app --port 8000
```

The map degrades gracefully when that backend is unreachable: it is probed
before path features are enabled, and those features are otherwise left
off.

## Data

Everything under `data/` is a pipeline checkpoint or output and is
git-ignored; nothing large is committed to this repository.

## License and attribution

Author and work data originates from [OpenAlex](https://openalex.org),
released under CC0. The viewer is built on
[MapLibre GL JS](https://maplibre.org).
