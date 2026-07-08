# Mapademic Data Pipeline (Plan 1 of 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Mac-side data stages of the mapademic pipeline - CLI scaffold, authors snapshot download, works authorship streaming extract, author filtering, and weighted coauthorship edge building - producing the `nodes.parquet` + `edges.parquet` pair that ships to Anvil.

**Architecture:** A Python package `pipeline/` with one CLI subcommand per stage (`python -m pipeline <stage>`). Each stage is an idempotent function reading/writing Parquet checkpoints under `DATA_DIR`; DuckDB does the heavy lifting. Works are NEVER stored locally: DuckDB's httpfs reads the S3 Parquet snapshot with column projection and writes compact per-partition authorship extracts. Stage modules expose a pure core function (unit-testable, no CLI/S3) plus `add_parser`/`run` glue.

**Tech Stack:** Python 3.11+, DuckDB with httpfs (only runtime dependency), pytest, AWS CLI (subprocess, `--no-sign-request`) for the authors sync.

**Spec:** `docs/superpowers/specs/2026-07-05-mapademic-design.md`. This plan covers spec stages 1-2. Plan 2 = layout/communities (Anvil), Plan 3 = tiles + index, Plan 4 = web frontend + hosting.

> **Revision 2026-07-06 (supersedes the first version of this plan):** the live
> snapshot measured much larger than the spec's ~330GB estimate - jsonl
> authors+works is **740GB**, parquet is **778GB**; the external drive has 429GB
> free, so "download everything" is impossible. Measured way out (DuckDB over
> S3, `EXPLAIN ANALYZE` HTTP counts on a representative part file):
> - Parquet works file ~880MB: needed columns (`id`, `authors_count`,
>   `authorships`) are ~27% of bytes; DuckDB projection pushdown reads only
>   those (struct-LEAF pruning does NOT happen - `author.id` alone would be
>   ~4% - but column-level pruning is confirmed empirically: id-only scan 8.2s
>   vs authorships scan 62.9s on the same file).
> - So: sync only `authors` locally (53GB parquet), stream-extract works
>   authorships from S3 (~195GB transferred, ~30-50GB stored), never store raw
>   works. Total disk ≈ 100GB. Fits with room to spare.

## Global Constraints

- Python `>=3.11`; runtime dependency is `duckdb` only (dev extra adds `pytest`)
- `DATA_DIR` resolves from env `MAPADEMIC_DATA`, default `./data`; `data/` is git-ignored. Real runs set `MAPADEMIC_DATA=/Volumes/Untitled/mapademic`
- Snapshot source is the **Parquet** variant: `s3://openalex/data/parquet/{authors,works}/updated_date=*/part_*.parquet`, anonymous access
- Node selection threshold: `works_count >= 5` by default, tunable via `--min-works` (spec: tune until node count lands in 5-10M)
- Hyperauthorship rule: works with more than **50** authors are excluded from edge building (spec-locked value; tunable via `--max-authors`)
- Edge weight: sum over shared works of `1 / (n_authors - 1)`, where `n_authors` is the work's FULL author count (snapshot's `authors_count`, before filtering to selected authors)
- Stages must be idempotent: re-running overwrites the stage's own output, never touches upstream checkpoints; the works extract checkpoints per partition and skips completed ones
- S3 DuckDB settings (VPN'd connection drops long reads): `http_timeout=600000`, `http_retries=5` - empirically required, default timeout failed mid-file
- Commits: conventional-commit style (`feat:`, `test:`, `docs:`); NEVER add Claude as co-author (CLAUDE.md rule 6)

---

### Task 1: Repo scaffold + CLI skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `pipeline/__init__.py`
- Create: `pipeline/config.py`
- Create: `pipeline/__main__.py`
- Test: `tests/test_config.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: nothing (first task)
- Produces: `pipeline.config.data_dir() -> pathlib.Path`; `pipeline.__main__.main(argv: list[str] | None = None) -> int`; `pipeline.__main__.STAGES: dict[str, module]` registry that Tasks 2-5 append to. Each stage module must define `add_parser(parser: argparse.ArgumentParser) -> None` and `run(args: argparse.Namespace) -> int`.

- [ ] **Step 1: Create project files**

`pyproject.toml`:

```toml
[project]
name = "mapademic"
version = "0.1.0"
description = "A zoomable map of the academic world (OpenAlex coauthorship graph)"
requires-python = ">=3.11"
dependencies = ["duckdb>=1.0"]

[project.optional-dependencies]
dev = ["pytest>=8"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["pipeline"]
```

`.gitignore`:

```
data/
data
__pycache__/
*.egg-info/
.venv/
.DS_Store
```

`README.md`:

```markdown
# mapademic

A zoomable, colorful map of the academic world: every sufficiently active
researcher is a dot, positioned by force-directed layout over the OpenAlex
coauthorship graph, colored by community. Companion project to
[academic-degree-of-separation](https://github.com/riptideiv/academic-degree-of-separation).

Design spec: `docs/superpowers/specs/2026-07-05-mapademic-design.md`

## Pipeline

    download -> extract -> filter -> edges -> [Anvil: layout -> communities] -> tiles -> index -> web/

Run stages via `python -m pipeline <stage>`. Data checkpoints live under
`$MAPADEMIC_DATA` (default `./data`, git-ignored).
```

`pipeline/__init__.py`: empty file.

`pipeline/config.py`:

```python
"""Shared paths for pipeline stages."""
import os
from pathlib import Path


def data_dir() -> Path:
    """Checkpoint root; override with MAPADEMIC_DATA (e.g. an external drive)."""
    return Path(os.environ.get("MAPADEMIC_DATA", "data")).expanduser()
```

`pipeline/__main__.py`:

```python
"""mapademic pipeline CLI: python -m pipeline <stage> [options]."""
import argparse

STAGES: dict = {}  # name -> stage module; stages register in Tasks 2-5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pipeline", description="mapademic batch pipeline"
    )
    sub = parser.add_subparsers(dest="stage", required=True)
    for name, mod in STAGES.items():
        mod.add_parser(sub.add_parser(name, help=(mod.__doc__ or "").strip()))
    args = parser.parse_args(argv)
    return STAGES[args.stage].run(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Write the failing tests**

`tests/test_config.py`:

```python
from pathlib import Path

from pipeline import config


def test_data_dir_defaults_to_local_data(monkeypatch):
    monkeypatch.delenv("MAPADEMIC_DATA", raising=False)
    assert config.data_dir() == Path("data")


def test_data_dir_env_override(monkeypatch):
    monkeypatch.setenv("MAPADEMIC_DATA", "/Volumes/Untitled/mapademic")
    assert config.data_dir() == Path("/Volumes/Untitled/mapademic")
```

`tests/test_cli.py`:

```python
import subprocess
import sys

import pytest

from pipeline.__main__ import main


def test_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_module_is_runnable():
    proc = subprocess.run(
        [sys.executable, "-m", "pipeline", "--help"], capture_output=True
    )
    assert proc.returncode == 0
```

- [ ] **Step 3: Set up venv and run tests to verify they pass**

```bash
cd /Users/eric/Downloads/mapademic
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -v
```

Expected: 4 tests PASS (scaffold and tests land together in this bootstrap task; TDD's fail-first starts in Task 2 once there is a package to test against).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml .gitignore README.md pipeline/ tests/
git commit -m "feat: pipeline CLI scaffold with stage registry"
```

---

### Task 2: Download stage (authors parquet sync)

Only `authors` (53GB) is downloaded; `works` is handled by Task 3's streaming extract because the full works snapshot (666-725GB) does not fit the 429GB-free drive.

**Files:**
- Create: `pipeline/download.py`
- Modify: `pipeline/__main__.py` (register stage)
- Test: `tests/test_download.py`

**Interfaces:**
- Consumes: `pipeline.config.data_dir()`
- Produces: `pipeline.download.sync_command(dest: Path) -> list[str]`; snapshot files land at `<DATA_DIR>/snapshot/authors/updated_date=*/part_*.parquet`. Task 4 reads that glob.

- [ ] **Step 1: Write the failing test**

`tests/test_download.py`:

```python
from pathlib import Path

from pipeline import download


def test_sync_command_is_anonymous_authors_parquet_sync():
    cmd = download.sync_command(Path("/data"))
    assert cmd[:3] == ["aws", "s3", "sync"]
    assert "s3://openalex/data/parquet/authors/" in cmd
    assert "--no-sign-request" in cmd
    assert str(Path("/data/snapshot/authors")) in cmd
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_download.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.download'` (or ImportError)

- [ ] **Step 3: Write the implementation**

`pipeline/download.py`:

```python
"""Sync the OpenAlex authors parquet snapshot (~53GB) from S3."""
import subprocess
from pathlib import Path

from pipeline.config import data_dir


def sync_command(dest: Path) -> list[str]:
    return [
        "aws", "s3", "sync",
        "s3://openalex/data/parquet/authors/",
        str(dest / "snapshot" / "authors"),
        "--no-sign-request",
        "--no-progress",
    ]


def add_parser(parser) -> None:
    parser.add_argument(
        "--dest", default=None,
        help="download root (default: DATA_DIR; put this on the external drive)",
    )


def run(args) -> int:
    dest = Path(args.dest) if args.dest else data_dir()
    cmd = sync_command(dest)
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return 0
```

In `pipeline/__main__.py`, replace the registry lines:

```python
from pipeline import download

STAGES: dict = {
    "download": download,
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest -v`
Expected: all PASS (including Task 1's CLI tests, which now list the `download` subcommand)

- [ ] **Step 5: Commit**

```bash
git add pipeline/download.py pipeline/__main__.py tests/test_download.py
git commit -m "feat: download stage - anonymous S3 sync of authors parquet"
```

---

### Task 3: Extract stage (works authorships streamed from S3)

Streams `id`, `authors_count`, `authorships -> author ids` from the S3 works parquet (DuckDB httpfs, column projection: ~27% of bytes actually transferred, ~195GB total). Checkpoints one output file per `updated_date=*` partition so an interrupted run (VPN drop, sleep, quota) resumes for free.

**Files:**
- Create: `pipeline/extract_works.py`
- Modify: `pipeline/__main__.py` (register stage as `extract`)
- Test: `tests/test_extract_works.py`, `tests/conftest.py`

**Interfaces:**
- Consumes: S3 works parquet (or any local directory with the same `<root>/updated_date=*/part_*.parquet` shape - that is what tests use)
- Produces:
  - `pipeline.extract_works.connect(src_root: str) -> duckdb.DuckDBPyConnection` (loads httpfs + timeout settings only for `s3://` roots)
  - `pipeline.extract_works.partitions(con, src_root: str) -> list[str]` (sorted `updated_date=...` directory names)
  - `pipeline.extract_works.extract_partition(con, src_root: str, partition: str, out_dir: Path) -> bool` (False = checkpoint already existed)
  - Output files `<DATA_DIR>/works_authorships/<partition>.parquet` with columns `work_id VARCHAR, n_authors INTEGER, author_ids VARCHAR[]`. Task 5 consumes exactly this schema.

- [ ] **Step 1: Write the shared works fixture helper and failing tests**

`tests/conftest.py`:

```python
"""Fixture helpers: build tiny parquet files shaped like the OpenAlex snapshot."""
import duckdb


def authorships_sql(author_ids):
    if not author_ids:
        return "CAST([] AS STRUCT(author STRUCT(id VARCHAR))[])"
    inner = ", ".join(f"{{'author': {{'id': '{a}'}}}}" for a in author_ids)
    return f"[{inner}]"


def write_works_partition(part_dir, works):
    """works: list of (work_id, authors_count, [author_ids]).

    Writes <part_dir>/part_0000.parquet shaped like the snapshot's works files
    (only the columns the extract stage touches).
    """
    part_dir.mkdir(parents=True, exist_ok=True)
    rows = ", ".join(
        f"('{wid}', {count}, {authorships_sql(aids)})"
        for wid, count, aids in works
    )
    duckdb.sql(
        f"COPY (SELECT * FROM (VALUES {rows}) t(id, authors_count, authorships)) "
        f"TO '{part_dir / 'part_0000.parquet'}' (FORMAT PARQUET)"
    )


def write_extract_output(path, rows):
    """rows: list of (work_id, n_authors, [author_ids]) in the EXTRACT OUTPUT schema."""
    vals = ", ".join(f"('{w}', {n}, {ids})" for w, n, ids in rows)
    duckdb.sql(
        f"COPY (SELECT * FROM (VALUES {vals}) t(work_id, n_authors, author_ids)) "
        f"TO '{path}' (FORMAT PARQUET)"
    )
```

`tests/test_extract_works.py`:

```python
import duckdb

from pipeline.extract_works import connect, extract_partition, partitions
from tests.conftest import write_works_partition


def make_src(tmp_path):
    src = tmp_path / "works"
    write_works_partition(
        src / "updated_date=2026-01-01",
        [("W1", 2, ["A1", "A2"]), ("W2", 1, ["A1"])],
    )
    write_works_partition(
        src / "updated_date=2026-02-01",
        [("W3", 3, ["A1", "A2", "A3"])],
    )
    return str(src)


def test_partitions_sorted_from_glob(tmp_path):
    src = make_src(tmp_path)
    con = connect(src)
    assert partitions(con, src) == [
        "updated_date=2026-01-01", "updated_date=2026-02-01"
    ]


def test_extract_partition_writes_authorship_columns(tmp_path):
    src = make_src(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    con = connect(src)
    assert extract_partition(con, src, "updated_date=2026-01-01", out_dir) is True
    rows = duckdb.sql(
        f"SELECT work_id, n_authors, author_ids "
        f"FROM '{out_dir / 'updated_date=2026-01-01.parquet'}' ORDER BY work_id"
    ).fetchall()
    assert rows == [("W1", 2, ["A1", "A2"]), ("W2", 1, ["A1"])]


def test_extract_partition_skips_existing_checkpoint(tmp_path):
    src = make_src(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    con = connect(src)
    assert extract_partition(con, src, "updated_date=2026-01-01", out_dir) is True
    assert extract_partition(con, src, "updated_date=2026-01-01", out_dir) is False


def test_no_leftover_tmp_files(tmp_path):
    src = make_src(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    con = connect(src)
    for part in partitions(con, src):
        extract_partition(con, src, part, out_dir)
    assert not list(out_dir.glob("*.tmp"))
    assert len(list(out_dir.glob("*.parquet"))) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_extract_works.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.extract_works'`

- [ ] **Step 3: Write the implementation**

`pipeline/extract_works.py`:

```python
"""Stream work authorships from the S3 parquet snapshot; raw works never touch disk."""
from pathlib import Path

import duckdb

from pipeline.config import data_dir

DEFAULT_SRC = "s3://openalex/data/parquet/works"


def connect(src_root: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    if src_root.startswith("s3://"):
        con.execute("INSTALL httpfs; LOAD httpfs;")
        # long timeout + retries: VPN'd reads of ~200MB column chunks time out
        # on DuckDB's defaults
        con.execute(
            "SET s3_region='us-east-1';"
            "SET http_timeout=600000;"
            "SET http_retries=5;"
        )
    return con


def partitions(con, src_root: str) -> list[str]:
    rows = con.execute(
        "SELECT file FROM glob(?)", [f"{src_root}/*/*.parquet"]
    ).fetchall()
    return sorted({row[0].rsplit("/", 2)[1] for row in rows})


def extract_partition(con, src_root: str, partition: str, out_dir: Path) -> bool:
    """Extract one updated_date partition. Returns False if already done."""
    out = out_dir / f"{partition}.parquet"
    if out.exists():
        return False
    tmp = out.with_name(out.name + ".tmp")
    tmp.unlink(missing_ok=True)
    con.execute(
        f"""
        COPY (
            SELECT
                id AS work_id,
                authors_count AS n_authors,
                list_transform(authorships, a -> a.author.id) AS author_ids
            FROM read_parquet('{src_root}/{partition}/*.parquet')
        ) TO '{tmp}' (FORMAT PARQUET)
        """
    )
    tmp.rename(out)  # atomic: a crash mid-COPY never yields a half checkpoint
    return True


def add_parser(parser) -> None:
    parser.add_argument("--src", default=DEFAULT_SRC, help="works parquet root")
    parser.add_argument("--out", default=None, help="override output dir")


def run(args) -> int:
    out_dir = Path(args.out) if args.out else data_dir() / "works_authorships"
    out_dir.mkdir(parents=True, exist_ok=True)
    con = connect(args.src)
    parts = partitions(con, args.src)
    done = 0
    for i, part in enumerate(parts, 1):
        fresh = extract_partition(con, args.src, part, out_dir)
        done += fresh
        state = "extracted" if fresh else "skip (checkpoint)"
        print(f"[{i}/{len(parts)}] {part}: {state}", flush=True)
    print(f"{done} extracted, {len(parts) - done} skipped -> {out_dir}")
    return 0
```

In `pipeline/__main__.py`, grow the registry:

```python
from pipeline import download, extract_works

STAGES: dict = {
    "download": download,
    "extract": extract_works,
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/extract_works.py pipeline/__main__.py tests/conftest.py tests/test_extract_works.py
git commit -m "feat: extract stage - stream works authorships from S3 with partition checkpoints"
```

---

### Task 4: Filter stage (authors -> nodes.parquet)

**Files:**
- Create: `pipeline/filter_authors.py`
- Modify: `pipeline/__main__.py` (register stage as `filter`)
- Test: `tests/test_filter_authors.py`

**Interfaces:**
- Consumes: authors parquet at `<DATA_DIR>/snapshot/authors/*/*.parquet` (Task 2's sync)
- Produces: `pipeline.filter_authors.filter_authors(src_glob: str, out_path: str, min_works: int = 5) -> int` (returns rows kept); `nodes.parquet` with columns `id VARCHAR, display_name VARCHAR, works_count BIGINT, cited_by_count BIGINT, institution VARCHAR, field VARCHAR`. Task 5 joins on `id`; Plan 2 (layout) and Plan 3 (index) consume the full row.

Schema note (verified against the live snapshot): `last_known_institutions` is `STRUCT(...)[]` and `topics` is `STRUCT(..., field STRUCT(display_name VARCHAR, ...), ...)[]`; both `last_known_institutions[1].display_name` and `topics[1].field.display_name` type-check as VARCHAR and return NULL on empty lists.

- [ ] **Step 1: Write the failing tests**

`tests/test_filter_authors.py`:

```python
import duckdb
import pytest

from pipeline.filter_authors import filter_authors

EMPTY_INST = "CAST([] AS STRUCT(display_name VARCHAR)[])"
EMPTY_TOPICS = "CAST([] AS STRUCT(display_name VARCHAR, field STRUCT(display_name VARCHAR))[])"


@pytest.fixture
def authors_glob(tmp_path):
    part = tmp_path / "authors" / "updated_date=2026-01-01"
    part.mkdir(parents=True)
    duckdb.sql(
        f"""
        COPY (
            SELECT * FROM (VALUES
                ('https://openalex.org/A1', 'Low Works', 2, 20,
                 {EMPTY_INST}, {EMPTY_TOPICS}),
                ('https://openalex.org/A2', 'Busy Bee', 5, 50,
                 [{{'display_name': 'UCSD'}}],
                 [{{'display_name': 'ULM',
                    'field': {{'display_name': 'Neuroscience'}}}}]),
                ('https://openalex.org/A3', 'Prolific', 100, 1000,
                 {EMPTY_INST}, {EMPTY_TOPICS})
            ) t(id, display_name, works_count, cited_by_count,
                last_known_institutions, topics)
        ) TO '{part / "part_0000.parquet"}' (FORMAT PARQUET)
        """
    )
    return str(tmp_path / "authors" / "*" / "*.parquet")


def test_threshold_drops_low_works_count_authors(authors_glob, tmp_path):
    out = tmp_path / "nodes.parquet"
    kept = filter_authors(authors_glob, str(out), min_works=5)
    ids = {r[0] for r in duckdb.sql(f"SELECT id FROM '{out}'").fetchall()}
    assert kept == 2
    assert ids == {"https://openalex.org/A2", "https://openalex.org/A3"}


def test_extracts_first_institution_and_field(authors_glob, tmp_path):
    out = tmp_path / "nodes.parquet"
    filter_authors(authors_glob, str(out), min_works=5)
    row = duckdb.sql(
        f"SELECT institution, field FROM '{out}' WHERE id LIKE '%A2'"
    ).fetchone()
    assert row == ("UCSD", "Neuroscience")


def test_missing_institution_and_topics_yield_nulls(authors_glob, tmp_path):
    out = tmp_path / "nodes.parquet"
    filter_authors(authors_glob, str(out), min_works=5)
    row = duckdb.sql(
        f"SELECT institution, field FROM '{out}' WHERE id LIKE '%A3'"
    ).fetchone()
    assert row == (None, None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_filter_authors.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.filter_authors'`

- [ ] **Step 3: Write the implementation**

`pipeline/filter_authors.py`:

```python
"""Select authors with works_count >= threshold into nodes.parquet."""
import duckdb

from pipeline.config import data_dir


def filter_authors(src_glob: str, out_path: str, min_works: int = 5) -> int:
    con = duckdb.connect()
    con.execute(
        f"""
        CREATE TABLE nodes AS
        SELECT
            id,
            display_name,
            works_count,
            cited_by_count,
            last_known_institutions[1].display_name AS institution,
            topics[1].field.display_name AS field
        FROM read_parquet('{src_glob}')
        WHERE works_count >= {int(min_works)}
        """
    )
    con.execute(f"COPY nodes TO '{out_path}' (FORMAT PARQUET)")
    return con.execute("SELECT count(*) FROM nodes").fetchone()[0]


def add_parser(parser) -> None:
    parser.add_argument("--min-works", type=int, default=5)
    parser.add_argument("--src", default=None, help="override authors glob")
    parser.add_argument("--out", default=None, help="override output path")


def run(args) -> int:
    src = args.src or str(data_dir() / "snapshot" / "authors" / "*" / "*.parquet")
    out = args.out or str(data_dir() / "nodes.parquet")
    kept = filter_authors(src, out, min_works=args.min_works)
    print(f"kept {kept:,} authors (min_works={args.min_works}) -> {out}")
    print("target window: 5-10M; re-run with a different --min-works to tune")
    return 0
```

In `pipeline/__main__.py`, grow the registry:

```python
from pipeline import download, extract_works, filter_authors

STAGES: dict = {
    "download": download,
    "extract": extract_works,
    "filter": filter_authors,
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/filter_authors.py pipeline/__main__.py tests/test_filter_authors.py
git commit -m "feat: filter stage - works_count-thresholded nodes.parquet"
```

---

### Task 5: Edges stage (authorship extracts -> weighted edges.parquet)

This is the stage the spec calls out for the densest tests - a silent bug here poisons layout, communities, and the map.

**Files:**
- Create: `pipeline/build_edges.py`
- Modify: `pipeline/__main__.py` (register stage as `edges`)
- Test: `tests/test_build_edges.py`

**Interfaces:**
- Consumes: `<DATA_DIR>/works_authorships/*.parquet` from Task 3 (`work_id, n_authors, author_ids`); `nodes.parquet` from Task 4 (join on `id`)
- Produces: `pipeline.build_edges.build_edges(works_glob: str, nodes_path: str, out_path: str, max_authors: int = 50) -> int` (returns edge count); `edges.parquet` with columns `src VARCHAR, dst VARCHAR, weight DOUBLE`, one row per unordered author pair (`src < dst` lexicographically). Plan 2's layout consumes exactly this.

- [ ] **Step 1: Write the failing tests**

`tests/test_build_edges.py`:

```python
import duckdb
import pytest

from pipeline.build_edges import build_edges
from tests.conftest import write_extract_output

A, B, C = "https://openalex.org/A1", "https://openalex.org/A2", "https://openalex.org/A3"


def write_nodes(tmp_path, ids):
    path = tmp_path / "nodes.parquet"
    con = duckdb.connect()
    con.execute("CREATE TABLE n (id VARCHAR)")
    con.executemany("INSERT INTO n VALUES (?)", [(i,) for i in ids])
    con.execute(f"COPY n TO '{path}' (FORMAT PARQUET)")
    return str(path)


def write_works(tmp_path, rows):
    path = tmp_path / "works_authorships.parquet"
    write_extract_output(path, rows)
    return str(path)


def edges_of(out_path):
    rows = duckdb.sql(
        f"SELECT src, dst, weight FROM '{out_path}' ORDER BY src, dst"
    ).fetchall()
    return {(r[0], r[1]): pytest.approx(r[2]) for r in rows}


def test_pair_explosion_and_weight(tmp_path):
    works = write_works(tmp_path, [("W1", 3, [A, B, C])])
    nodes = write_nodes(tmp_path, [A, B, C])
    out = tmp_path / "edges.parquet"
    n = build_edges(works, nodes, str(out))
    assert n == 3
    assert edges_of(out) == {(A, B): 0.5, (A, C): 0.5, (B, C): 0.5}


def test_weights_sum_across_shared_works(tmp_path):
    works = write_works(tmp_path, [("W1", 2, [A, B]), ("W2", 2, [A, B])])
    nodes = write_nodes(tmp_path, [A, B])
    out = tmp_path / "edges.parquet"
    build_edges(works, nodes, str(out))
    assert edges_of(out) == {(A, B): 2.0}


def test_hyperauthorship_cutoff_excludes_work(tmp_path):
    big = [f"https://openalex.org/AX{i:02d}" for i in range(51)]
    works = write_works(tmp_path, [("W1", 51, big)])
    nodes = write_nodes(tmp_path, big)
    out = tmp_path / "edges.parquet"
    assert build_edges(works, nodes, str(out)) == 0


def test_cutoff_boundary_50_authors_included(tmp_path):
    big = [f"https://openalex.org/AX{i:02d}" for i in range(50)]
    works = write_works(tmp_path, [("W1", 50, big)])
    nodes = write_nodes(tmp_path, big)
    out = tmp_path / "edges.parquet"
    assert build_edges(works, nodes, str(out)) == 50 * 49 // 2


def test_unselected_authors_drop_from_pairs_but_count_in_weight(tmp_path):
    # C is not in nodes: no edges touch C, but n_authors=3 so weight is 1/2
    works = write_works(tmp_path, [("W1", 3, [A, B, C])])
    nodes = write_nodes(tmp_path, [A, B])
    out = tmp_path / "edges.parquet"
    build_edges(works, nodes, str(out))
    assert edges_of(out) == {(A, B): 0.5}


def test_single_author_work_yields_no_edges(tmp_path):
    works = write_works(tmp_path, [("W1", 1, [A])])
    nodes = write_nodes(tmp_path, [A])
    out = tmp_path / "edges.parquet"
    assert build_edges(works, nodes, str(out)) == 0


def test_duplicate_author_in_list_no_self_edge_or_double_count(tmp_path):
    # OpenAlex data noise: same author listed twice on one work
    works = write_works(tmp_path, [("W1", 3, [A, A, B])])
    nodes = write_nodes(tmp_path, [A, B])
    out = tmp_path / "edges.parquet"
    build_edges(works, nodes, str(out))
    edges = edges_of(out)
    assert set(edges) == {(A, B)}
    assert edges[(A, B)] == pytest.approx(0.5)  # n_authors=3 per snapshot count
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_build_edges.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.build_edges'`

- [ ] **Step 3: Write the implementation**

`pipeline/build_edges.py`:

```python
"""Build weighted coauthorship edges from authorship extracts and selected nodes."""
import duckdb

from pipeline.config import data_dir

# Spec-locked: works with more than 50 authors are excluded from edge building
# (a 3,000-author CERN paper would otherwise contribute ~4.5M clique edges).
MAX_AUTHORS = 50


def build_edges(
    works_glob: str, nodes_path: str, out_path: str, max_authors: int = MAX_AUTHORS
) -> int:
    con = duckdb.connect()
    con.execute(
        f"""
        CREATE TABLE edges AS
        WITH exploded AS (
            SELECT work_id, unnest(author_ids) AS author_id, n_authors
            FROM read_parquet('{works_glob}')
            WHERE n_authors BETWEEN 2 AND {int(max_authors)}
        ),
        -- DISTINCT guards against the same author appearing twice on one work
        kept AS (
            SELECT DISTINCT e.work_id, e.author_id, e.n_authors
            FROM exploded e
            JOIN read_parquet('{nodes_path}') n ON n.id = e.author_id
        ),
        pairs AS (
            SELECT
                a.author_id AS src,
                b.author_id AS dst,
                1.0 / (a.n_authors - 1) AS w
            FROM kept a
            JOIN kept b
              ON a.work_id = b.work_id AND a.author_id < b.author_id
        )
        SELECT src, dst, sum(w) AS weight
        FROM pairs
        GROUP BY src, dst
        """
    )
    con.execute(f"COPY edges TO '{out_path}' (FORMAT PARQUET)")
    return con.execute("SELECT count(*) FROM edges").fetchone()[0]


def add_parser(parser) -> None:
    parser.add_argument("--max-authors", type=int, default=MAX_AUTHORS)
    parser.add_argument("--works", default=None, help="override extracts glob")
    parser.add_argument("--nodes", default=None, help="override nodes.parquet path")
    parser.add_argument("--out", default=None, help="override output path")


def run(args) -> int:
    works = args.works or str(data_dir() / "works_authorships" / "*.parquet")
    nodes = args.nodes or str(data_dir() / "nodes.parquet")
    out = args.out or str(data_dir() / "edges.parquet")
    n = build_edges(works, nodes, out, max_authors=args.max_authors)
    print(f"built {n:,} edges (max_authors={args.max_authors}) -> {out}")
    return 0
```

In `pipeline/__main__.py`, grow the registry:

```python
from pipeline import build_edges, download, extract_works, filter_authors

STAGES: dict = {
    "download": download,
    "extract": extract_works,
    "filter": filter_authors,
    "edges": build_edges,
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest -v`
Expected: all PASS (7 new edge tests + everything prior)

- [ ] **Step 5: Commit**

```bash
git add pipeline/build_edges.py pipeline/__main__.py tests/test_build_edges.py
git commit -m "feat: edges stage - weighted coauthor pairs with hyperauthorship cutoff"
```

---

### Task 6: End-to-end fixture test + pipeline docs

**Files:**
- Test: `tests/test_end_to_end.py`
- Modify: `README.md` (usage section)

**Interfaces:**
- Consumes: `extract_partition(...)`, `filter_authors(...)`, `build_edges(...)` exactly as defined in Tasks 3-5
- Produces: confidence that the three stages compose over one shared fixture; README documents the real-run sequence Plan 2 picks up from

- [ ] **Step 1: Write the failing test**

`tests/test_end_to_end.py`:

```python
"""extract -> filter -> edges over one coherent fixture, as the real run composes them."""
import duckdb

from pipeline.build_edges import build_edges
from pipeline.extract_works import connect, extract_partition, partitions
from pipeline.filter_authors import filter_authors
from tests.conftest import write_works_partition

HINTON = "https://openalex.org/A_hinton"
COLLAB = "https://openalex.org/A_collab"
ONEHIT = "https://openalex.org/A_onehit"

EMPTY_INST = "CAST([] AS STRUCT(display_name VARCHAR)[])"
EMPTY_TOPICS = "CAST([] AS STRUCT(display_name VARCHAR, field STRUCT(display_name VARCHAR))[])"


def test_extract_filter_edges(tmp_path):
    # authors snapshot fixture
    authors_part = tmp_path / "authors" / "updated_date=2026-01-01"
    authors_part.mkdir(parents=True)
    duckdb.sql(
        f"""
        COPY (
            SELECT * FROM (VALUES
                ('{HINTON}', 'G. Hinton', 400, 900000, {EMPTY_INST}, {EMPTY_TOPICS}),
                ('{COLLAB}', 'Co Author', 12, 300, {EMPTY_INST}, {EMPTY_TOPICS}),
                ('{ONEHIT}', 'One Hit', 1, 5, {EMPTY_INST}, {EMPTY_TOPICS})
            ) t(id, display_name, works_count, cited_by_count,
                last_known_institutions, topics)
        ) TO '{authors_part / "part_0000.parquet"}' (FORMAT PARQUET)
        """
    )
    # works snapshot fixture
    works_src = tmp_path / "works"
    write_works_partition(
        works_src / "updated_date=2026-01-01",
        [("W1", 3, [HINTON, COLLAB, ONEHIT])],
    )

    # extract
    extracts = tmp_path / "works_authorships"
    extracts.mkdir()
    con = connect(str(works_src))
    for part in partitions(con, str(works_src)):
        extract_partition(con, str(works_src), part, extracts)

    # filter
    nodes_out = tmp_path / "nodes.parquet"
    kept = filter_authors(
        str(tmp_path / "authors" / "*" / "*.parquet"), str(nodes_out), min_works=5
    )
    assert kept == 2

    # edges
    edges_out = tmp_path / "edges.parquet"
    n = build_edges(str(extracts / "*.parquet"), str(nodes_out), str(edges_out))
    assert n == 1

    src, dst, weight = duckdb.sql(f"SELECT * FROM '{edges_out}'").fetchone()
    assert {src, dst} == {HINTON, COLLAB}   # ONEHIT filtered out upstream
    assert weight == 0.5                     # n_authors=3 counts ONEHIT
```

- [ ] **Step 2: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_end_to_end.py -v`
Expected: PASS immediately (it exercises code from Tasks 3-5; if it fails, the stages do not compose - fix before continuing)

- [ ] **Step 3: Document the real-run sequence in README.md**

Append to `README.md`:

```markdown
## Running the real pipeline (M4 Max)

    # 0. Everything targets the external drive
    export MAPADEMIC_DATA=/Volumes/Untitled/mapademic

    # 1. Authors snapshot, ~53GB (aws s3 sync; resumable, safe to re-run)
    .venv/bin/python -m pipeline download

    # 2. Works authorships streamed from S3, ~195GB transfer / ~30-50GB stored.
    #    Checkpoints per partition; re-run after any interruption to resume.
    .venv/bin/python -m pipeline extract

    # 3. Tune --min-works until the kept count lands in 5-10M
    .venv/bin/python -m pipeline filter --min-works 5

    # 4. Build edges (prints edge count; expect 50-200M rows)
    .venv/bin/python -m pipeline edges

    # 5. Ship to Anvil (Plan 2): scp nodes.parquet edges.parquet \
    #      x-egao2@anvil.rcac.purdue.edu:$SCRATCH/mapademic/

No external drive? Fallback: skip `download`/`extract` and run the
filter/edge queries against the OpenAlex BigQuery public dataset instead
(the degree-of-separation repo already scaffolds BigQuery access); costs a
few dollars of query and produces the same two Parquet files.
```

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/pytest -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_end_to_end.py README.md
git commit -m "test: end-to-end extract->filter->edges fixture; docs: real-run sequence"
```

---

## Out of scope for this plan

- Layout, communities, Slurm scripts (Plan 2 - written after this plan lands, so it can target the actual Parquet schemas produced here)
- Tile rendering + search/hit index (Plan 3)
- Web frontend, hosting, CORS PR to the degree-of-separation repo (Plan 4)
