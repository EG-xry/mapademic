# Mapademic Data Pipeline (Plan 1 of 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Mac-side data stages of the mapademic pipeline - CLI scaffold, OpenAlex snapshot download, author filtering, and weighted coauthorship edge building - producing the `nodes.parquet` + `edges.parquet` pair that ships to Anvil.

**Architecture:** A Python package `pipeline/` with one CLI subcommand per stage (`python -m pipeline <stage>`). Each stage is an idempotent function reading/writing Parquet checkpoints under `DATA_DIR`; DuckDB does all heavy lifting via streaming SQL over the snapshot's gzipped JSONL. Stage modules expose a pure core function (unit-testable, no CLI) plus `add_parser`/`run` glue.

**Tech Stack:** Python 3.11+, DuckDB (only runtime dependency), pytest, AWS CLI (invoked as a subprocess for the S3 sync, `--no-sign-request`).

**Spec:** `docs/superpowers/specs/2026-07-05-mapademic-design.md`. This plan covers spec stages 1-2. Plan 2 = layout/communities (Anvil), Plan 3 = tiles + index, Plan 4 = web frontend + hosting.

## Global Constraints

- Python `>=3.11`; runtime dependency is `duckdb` only (dev extra adds `pytest`)
- `DATA_DIR` resolves from env `MAPADEMIC_DATA`, default `./data`; `data/` is git-ignored (often a symlink to an external drive)
- Node selection threshold: `works_count >= 5` by default, tunable via `--min-works` (spec: tune until node count lands in 5-10M)
- Hyperauthorship rule: works with more than **50** authors are excluded from edge building (spec-locked value; tunable via `--max-authors`)
- Edge weight: sum over shared works of `1 / (n_authors - 1)`, where `n_authors` is the work's FULL author count (before filtering to selected authors)
- Stages must be idempotent: re-running overwrites the stage's own output, never touches upstream checkpoints
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
- Produces: `pipeline.config.data_dir() -> pathlib.Path`; `pipeline.__main__.main(argv: list[str] | None = None) -> int`; `pipeline.__main__.STAGES: dict[str, module]` registry that Tasks 2-4 append to. Each stage module must define `add_parser(parser: argparse.ArgumentParser) -> None` and `run(args: argparse.Namespace) -> int`.

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

    download -> filter -> edges -> [Anvil: layout -> communities] -> tiles -> index -> web/

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

STAGES: dict = {}  # name -> stage module; stages register in Tasks 2-4


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
    monkeypatch.setenv("MAPADEMIC_DATA", "/Volumes/ext/mapademic")
    assert config.data_dir() == Path("/Volumes/ext/mapademic")
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

Expected: 3 tests PASS (scaffold and tests land together in this bootstrap task; TDD's fail-first starts in Task 2 once there is a package to test against).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml .gitignore README.md pipeline/ tests/
git commit -m "feat: pipeline CLI scaffold with stage registry"
```

---

### Task 2: Download stage (OpenAlex S3 snapshot sync)

**Files:**
- Create: `pipeline/download.py`
- Modify: `pipeline/__main__.py` (register stage)
- Test: `tests/test_download.py`

**Interfaces:**
- Consumes: `pipeline.config.data_dir()`
- Produces: `pipeline.download.sync_commands(dest: Path) -> list[list[str]]`; snapshot files land at `<DATA_DIR>/snapshot/{authors,works}/updated_date=*/part_*.gz`. Tasks 3-4 read those globs.

- [ ] **Step 1: Write the failing test**

`tests/test_download.py`:

```python
from pathlib import Path

from pipeline import download


def test_sync_commands_cover_authors_and_works():
    cmds = download.sync_commands(Path("/data"))
    assert len(cmds) == 2
    joined = [" ".join(c) for c in cmds]
    assert any("s3://openalex/data/authors/" in c for c in joined)
    assert any("s3://openalex/data/works/" in c for c in joined)


def test_sync_commands_are_anonymous_s3_syncs():
    for cmd in download.sync_commands(Path("/data")):
        assert cmd[:3] == ["aws", "s3", "sync"]
        assert "--no-sign-request" in cmd
        assert "/data/snapshot/" in " ".join(cmd)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_download.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.download'` (or ImportError)

- [ ] **Step 3: Write the implementation**

`pipeline/download.py`:

```python
"""Sync the OpenAlex snapshot (authors + works, ~330GB compressed) from S3."""
import subprocess
from pathlib import Path

from pipeline.config import data_dir

ENTITIES = ("authors", "works")


def sync_commands(dest: Path) -> list[list[str]]:
    return [
        [
            "aws", "s3", "sync",
            f"s3://openalex/data/{entity}/",
            str(dest / "snapshot" / entity),
            "--no-sign-request",
        ]
        for entity in ENTITIES
    ]


def add_parser(parser) -> None:
    parser.add_argument(
        "--dest", default=None,
        help="download root (default: DATA_DIR; put this on the external drive)",
    )


def run(args) -> int:
    dest = Path(args.dest) if args.dest else data_dir()
    for cmd in sync_commands(dest):
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

(Keep the `# name -> stage module` comment off; the import block grows in Tasks 3-4.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest -v`
Expected: all PASS (including Task 1's CLI tests, which now list the `download` subcommand)

- [ ] **Step 5: Commit**

```bash
git add pipeline/download.py pipeline/__main__.py tests/test_download.py
git commit -m "feat: download stage - anonymous S3 sync of OpenAlex authors+works"
```

---

### Task 3: Filter stage (authors -> nodes.parquet)

**Files:**
- Create: `pipeline/filter_authors.py`
- Modify: `pipeline/__main__.py` (register stage as `filter`)
- Test: `tests/test_filter_authors.py`

**Interfaces:**
- Consumes: snapshot JSONL.gz at `<DATA_DIR>/snapshot/authors/*/*.gz`
- Produces: `pipeline.filter_authors.filter_authors(src_glob: str, out_path: str, min_works: int = 5) -> int` (returns rows kept); `nodes.parquet` with columns `id VARCHAR, display_name VARCHAR, works_count BIGINT, cited_by_count BIGINT, institution VARCHAR, field VARCHAR`. Task 4 joins on `id`; Plan 2 (layout) and Plan 3 (index) consume the full row.

- [ ] **Step 1: Write the failing tests**

`tests/test_filter_authors.py`:

```python
import gzip
import json

import duckdb
import pytest

from pipeline.filter_authors import filter_authors


def author(id_, works_count, name="A. Author", institutions=None, topics=None):
    return {
        "id": id_,
        "display_name": name,
        "works_count": works_count,
        "cited_by_count": works_count * 10,
        "last_known_institutions": institutions if institutions is not None else [],
        "topics": topics if topics is not None else [],
    }


@pytest.fixture
def authors_file(tmp_path):
    rows = [
        author("https://openalex.org/A1", 2),
        author(
            "https://openalex.org/A2", 5, name="Busy Bee",
            institutions=[{"display_name": "UCSD"}],
            topics=[{"display_name": "ULM", "field": {"display_name": "Neuroscience"}}],
        ),
        author("https://openalex.org/A3", 100),
    ]
    path = tmp_path / "part_000.jsonl.gz"
    with gzip.open(path, "wt") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def test_threshold_drops_low_works_count_authors(authors_file, tmp_path):
    out = tmp_path / "nodes.parquet"
    kept = filter_authors(str(authors_file), str(out), min_works=5)
    ids = {r[0] for r in duckdb.sql(f"SELECT id FROM '{out}'").fetchall()}
    assert kept == 2
    assert ids == {"https://openalex.org/A2", "https://openalex.org/A3"}


def test_extracts_first_institution_and_field(authors_file, tmp_path):
    out = tmp_path / "nodes.parquet"
    filter_authors(str(authors_file), str(out), min_works=5)
    row = duckdb.sql(
        f"SELECT institution, field FROM '{out}' WHERE id LIKE '%A2'"
    ).fetchone()
    assert row == ("UCSD", "Neuroscience")


def test_missing_institution_and_topics_yield_nulls(authors_file, tmp_path):
    out = tmp_path / "nodes.parquet"
    filter_authors(str(authors_file), str(out), min_works=5)
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

# Explicit schema: read_json then only parses what we keep, and missing keys
# become NULL instead of erroring.
AUTHOR_COLUMNS = {
    "id": "VARCHAR",
    "display_name": "VARCHAR",
    "works_count": "BIGINT",
    "cited_by_count": "BIGINT",
    "last_known_institutions": "STRUCT(display_name VARCHAR)[]",
    "topics": "STRUCT(display_name VARCHAR, field STRUCT(display_name VARCHAR))[]",
}


def _columns_sql(columns: dict[str, str]) -> str:
    return "{" + ", ".join(f"'{k}': '{v}'" for k, v in columns.items()) + "}"


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
        FROM read_json(
            '{src_glob}',
            format='newline_delimited',
            compression='auto',
            columns={_columns_sql(AUTHOR_COLUMNS)}
        )
        WHERE works_count >= {int(min_works)}
        """
    )
    con.execute(f"COPY nodes TO '{out_path}' (FORMAT PARQUET)")
    return con.execute("SELECT count(*) FROM nodes").fetchone()[0]


def add_parser(parser) -> None:
    parser.add_argument("--min-works", type=int, default=5)
    parser.add_argument("--src", default=None, help="override snapshot glob")
    parser.add_argument("--out", default=None, help="override output path")


def run(args) -> int:
    src = args.src or str(data_dir() / "snapshot" / "authors" / "*" / "*.gz")
    out = args.out or str(data_dir() / "nodes.parquet")
    kept = filter_authors(src, out, min_works=args.min_works)
    print(f"kept {kept:,} authors (min_works={args.min_works}) -> {out}")
    print("target window: 5-10M; re-run with a different --min-works to tune")
    return 0
```

In `pipeline/__main__.py`, grow the registry:

```python
from pipeline import download, filter_authors

STAGES: dict = {
    "download": download,
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

### Task 4: Edges stage (works -> weighted coauthorship edges.parquet)

This is the stage the spec calls out for the densest tests - a silent bug here poisons layout, communities, and the map.

**Files:**
- Create: `pipeline/build_edges.py`
- Modify: `pipeline/__main__.py` (register stage as `edges`)
- Test: `tests/test_build_edges.py`

**Interfaces:**
- Consumes: works JSONL at `<DATA_DIR>/snapshot/works/*/*.gz`; `nodes.parquet` from Task 3 (join on `id`)
- Produces: `pipeline.build_edges.build_edges(works_glob: str, nodes_path: str, out_path: str, max_authors: int = 50) -> int` (returns edge count); `edges.parquet` with columns `src VARCHAR, dst VARCHAR, weight DOUBLE`, one row per unordered author pair (`src < dst` lexicographically). Plan 2's layout consumes exactly this.

- [ ] **Step 1: Write the failing tests**

`tests/test_build_edges.py`:

```python
import gzip
import json

import duckdb
import pytest

from pipeline.build_edges import build_edges

A, B, C = "https://openalex.org/A1", "https://openalex.org/A2", "https://openalex.org/A3"


def work(id_, author_ids):
    return {
        "id": id_,
        "authorships": [{"author": {"id": a}} for a in author_ids],
    }


def write_works(tmp_path, works):
    path = tmp_path / "works_000.jsonl.gz"
    with gzip.open(path, "wt") as f:
        for w in works:
            f.write(json.dumps(w) + "\n")
    return str(path)


def write_nodes(tmp_path, ids):
    path = tmp_path / "nodes.parquet"
    con = duckdb.connect()
    con.execute("CREATE TABLE n (id VARCHAR)")
    con.executemany("INSERT INTO n VALUES (?)", [(i,) for i in ids])
    con.execute(f"COPY n TO '{path}' (FORMAT PARQUET)")
    return str(path)


def edges_of(out_path):
    rows = duckdb.sql(f"SELECT src, dst, weight FROM '{out_path}' ORDER BY src, dst").fetchall()
    return {(r[0], r[1]): pytest.approx(r[2]) for r in rows}


def test_pair_explosion_and_weight(tmp_path):
    works_glob = write_works(tmp_path, [work("W1", [A, B, C])])
    nodes = write_nodes(tmp_path, [A, B, C])
    out = tmp_path / "edges.parquet"
    n = build_edges(works_glob, nodes, str(out))
    assert n == 3
    assert edges_of(out) == {(A, B): 0.5, (A, C): 0.5, (B, C): 0.5}


def test_weights_sum_across_shared_works(tmp_path):
    works_glob = write_works(tmp_path, [work("W1", [A, B]), work("W2", [A, B])])
    nodes = write_nodes(tmp_path, [A, B])
    out = tmp_path / "edges.parquet"
    build_edges(works_glob, nodes, str(out))
    assert edges_of(out) == {(A, B): 2.0}


def test_hyperauthorship_cutoff_excludes_work(tmp_path):
    big = [f"https://openalex.org/AX{i}" for i in range(51)]
    works_glob = write_works(tmp_path, [work("W1", big)])
    nodes = write_nodes(tmp_path, big)
    out = tmp_path / "edges.parquet"
    assert build_edges(works_glob, nodes, str(out)) == 0


def test_cutoff_boundary_50_authors_included(tmp_path):
    big = [f"https://openalex.org/AX{i:02d}" for i in range(50)]
    works_glob = write_works(tmp_path, [work("W1", big)])
    nodes = write_nodes(tmp_path, big)
    out = tmp_path / "edges.parquet"
    assert build_edges(works_glob, nodes, str(out)) == 50 * 49 // 2


def test_unselected_authors_drop_from_pairs_but_count_in_weight(tmp_path):
    # C is not in nodes: no edges touch C, but n_authors=3 so weight is 1/2
    works_glob = write_works(tmp_path, [work("W1", [A, B, C])])
    nodes = write_nodes(tmp_path, [A, B])
    out = tmp_path / "edges.parquet"
    build_edges(works_glob, nodes, str(out))
    assert edges_of(out) == {(A, B): 0.5}


def test_single_author_work_yields_no_edges(tmp_path):
    works_glob = write_works(tmp_path, [work("W1", [A])])
    nodes = write_nodes(tmp_path, [A])
    out = tmp_path / "edges.parquet"
    assert build_edges(works_glob, nodes, str(out)) == 0


def test_duplicate_author_in_authorships_no_self_edge_or_double_count(tmp_path):
    # OpenAlex data noise: same author listed twice on one work
    works_glob = write_works(tmp_path, [work("W1", [A, A, B])])
    nodes = write_nodes(tmp_path, [A, B])
    out = tmp_path / "edges.parquet"
    build_edges(works_glob, nodes, str(out))
    edges = edges_of(out)
    assert set(edges) == {(A, B)}
    assert edges[(A, B)] == pytest.approx(0.5)  # n_authors=3 (raw list length)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_build_edges.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.build_edges'`

- [ ] **Step 3: Write the implementation**

`pipeline/build_edges.py`:

```python
"""Build weighted coauthorship edges from works, restricted to selected nodes."""
import duckdb

from pipeline.config import data_dir
from pipeline.filter_authors import _columns_sql

# Spec-locked: works with more than 50 authors are excluded from edge building
# (a 3,000-author CERN paper would otherwise contribute ~4.5M clique edges).
MAX_AUTHORS = 50

WORK_COLUMNS = {
    "id": "VARCHAR",
    "authorships": "STRUCT(author STRUCT(id VARCHAR))[]",
}


def build_edges(
    works_glob: str, nodes_path: str, out_path: str, max_authors: int = MAX_AUTHORS
) -> int:
    con = duckdb.connect()
    con.execute(
        f"""
        CREATE VIEW works AS
        SELECT
            id AS work_id,
            list_transform(authorships, a -> a.author.id) AS author_ids,
            len(authorships) AS n_authors
        FROM read_json(
            '{works_glob}',
            format='newline_delimited',
            compression='auto',
            columns={_columns_sql(WORK_COLUMNS)}
        )
        """
    )
    con.execute(f"CREATE VIEW nodes AS SELECT id FROM read_parquet('{nodes_path}')")
    con.execute(
        f"""
        CREATE TABLE edges AS
        WITH exploded AS (
            SELECT work_id, unnest(author_ids) AS author_id, n_authors
            FROM works
            WHERE n_authors BETWEEN 2 AND {int(max_authors)}
        ),
        -- DISTINCT guards against the same author appearing twice on one work
        kept AS (
            SELECT DISTINCT e.work_id, e.author_id, e.n_authors
            FROM exploded e
            JOIN nodes n ON n.id = e.author_id
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
    parser.add_argument("--works", default=None, help="override works glob")
    parser.add_argument("--nodes", default=None, help="override nodes.parquet path")
    parser.add_argument("--out", default=None, help="override output path")


def run(args) -> int:
    works = args.works or str(data_dir() / "snapshot" / "works" / "*" / "*.gz")
    nodes = args.nodes or str(data_dir() / "nodes.parquet")
    out = args.out or str(data_dir() / "edges.parquet")
    n = build_edges(works, nodes, out, max_authors=args.max_authors)
    print(f"built {n:,} edges (max_authors={args.max_authors}) -> {out}")
    return 0
```

In `pipeline/__main__.py`, grow the registry:

```python
from pipeline import build_edges, download, filter_authors

STAGES: dict = {
    "download": download,
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

### Task 5: End-to-end fixture test + pipeline docs

**Files:**
- Test: `tests/test_end_to_end.py`
- Modify: `README.md` (usage section)

**Interfaces:**
- Consumes: `filter_authors(...)` and `build_edges(...)` exactly as defined in Tasks 3-4
- Produces: confidence that the two stages compose over one shared fixture; README documents the real-run sequence Plan 2 picks up from

- [ ] **Step 1: Write the failing test**

`tests/test_end_to_end.py`:

```python
"""Filter -> edges over one coherent fixture, as the real run composes them."""
import gzip
import json

import duckdb

from pipeline.build_edges import build_edges
from pipeline.filter_authors import filter_authors

HINTON = "https://openalex.org/A_hinton"
COLLAB = "https://openalex.org/A_collab"
ONEHIT = "https://openalex.org/A_onehit"


def gz_jsonl(path, rows):
    with gzip.open(path, "wt") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return str(path)


def test_filter_then_edges(tmp_path):
    authors = gz_jsonl(
        tmp_path / "authors.jsonl.gz",
        [
            {"id": HINTON, "display_name": "G. Hinton", "works_count": 400,
             "cited_by_count": 900000, "last_known_institutions": [], "topics": []},
            {"id": COLLAB, "display_name": "Co Author", "works_count": 12,
             "cited_by_count": 300, "last_known_institutions": [], "topics": []},
            {"id": ONEHIT, "display_name": "One Hit", "works_count": 1,
             "cited_by_count": 5, "last_known_institutions": [], "topics": []},
        ],
    )
    works = gz_jsonl(
        tmp_path / "works.jsonl.gz",
        [
            {"id": "W1", "authorships": [
                {"author": {"id": HINTON}},
                {"author": {"id": COLLAB}},
                {"author": {"id": ONEHIT}},
            ]},
        ],
    )
    nodes_out = tmp_path / "nodes.parquet"
    edges_out = tmp_path / "edges.parquet"

    assert filter_authors(authors, str(nodes_out), min_works=5) == 2
    assert build_edges(works, str(nodes_out), str(edges_out)) == 1

    src, dst, weight = duckdb.sql(f"SELECT * FROM '{edges_out}'").fetchone()
    assert {src, dst} == {HINTON, COLLAB}   # ONEHIT filtered out upstream
    assert weight == 0.5                     # n_authors=3 counts ONEHIT
```

- [ ] **Step 2: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_end_to_end.py -v`
Expected: PASS immediately (it exercises code from Tasks 3-4; if it fails, the stages do not compose - fix before continuing)

- [ ] **Step 3: Document the real-run sequence in README.md**

Append to `README.md`:

```markdown
## Running the real pipeline (M4 Max)

    # 1. ~330GB compressed; point DATA_DIR at the external drive first
    export MAPADEMIC_DATA=/Volumes/<drive>/mapademic
    .venv/bin/python -m pipeline download

    # 2. Tune --min-works until the kept count lands in 5-10M
    .venv/bin/python -m pipeline filter --min-works 5

    # 3. Build edges (prints edge count; expect 50-200M rows)
    .venv/bin/python -m pipeline edges

    # 4. Ship to Anvil (Plan 2): scp nodes.parquet edges.parquet \
    #      x-egao2@anvil.rcac.purdue.edu:$SCRATCH/mapademic/

No external drive? Fallback: skip `download` and run the filter/edge queries
against the OpenAlex BigQuery public dataset instead (the
degree-of-separation repo already scaffolds BigQuery access); costs a few
dollars of query and produces the same two Parquet files.
```

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/pytest -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_end_to_end.py README.md
git commit -m "test: end-to-end filter->edges fixture; docs: real-run sequence"
```

---

## Out of scope for this plan

- Layout, communities, Slurm scripts (Plan 2 - written after this plan lands, so it can target the actual Parquet schemas produced here)
- Tile rendering + search/hit index (Plan 3)
- Web frontend, hosting, CORS PR to the degree-of-separation repo (Plan 4)
