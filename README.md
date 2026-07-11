# mapademic

A zoomable, colorful map of the academic world: every sufficiently active
researcher is a dot, positioned by force-directed layout over the OpenAlex
coauthorship graph, colored by community. Companion project to
[academic-degree-of-separation](https://github.com/riptideiv/academic-degree-of-separation).

Design spec: `docs/superpowers/specs/2026-07-05-mapademic-design.md`

## Pipeline

    download -> extract -> filter -> edges -> prep -> webcoords -> tiles -> regions -> index -> edgepx -> web/

Run stages via `python -m pipeline <stage>`. Data checkpoints live under
`$MAPADEMIC_DATA` (default `./data`, git-ignored).

## Running the real pipeline (M4 Max)

    # 0. Everything targets the external drive
    export MAPADEMIC_DATA=/Volumes/Untitled/mapademic

    # 1. Authors snapshot, ~53GB (aws s3 sync; resumable, safe to re-run)
    .venv/bin/python -m pipeline download

    # 2. Works authorships streamed from S3, ~195GB transfer / ~30-50GB stored.
    #    Checkpoints per partition; re-run after any interruption to resume.
    .venv/bin/python -m pipeline extract

    # 3. Tune --min-works until the kept count lands in 5-10M
    .venv/bin/python -m pipeline filter --min-works 15  # (production run: 15 -> 8.59M authors)

    # 4. Build edges (prints edge count; expect 50-200M rows)
    .venv/bin/python -m pipeline edges

    # 5. Ship to Anvil (Plan 2): scp nodes.parquet edges.parquet \
    #      x-egao2@anvil.rcac.purdue.edu:$SCRATCH/mapademic/

No external drive? Fallback: skip `download`/`extract` and run the
filter/edge queries against the OpenAlex BigQuery public dataset instead
(the degree-of-separation repo already scaffolds BigQuery access); costs a
few dollars of query and produces the same two Parquet files.
