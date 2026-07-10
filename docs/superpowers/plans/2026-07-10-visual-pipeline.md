# Mapademic Visual Pipeline (Plan 3 of 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `data/coords.parquet` (8,587,906 researchers) into a locally-verifiable map: web coordinates, a styled PNG tile pyramid (zooms 0-9), auto-named regions, label/hit tiles, and search shards - all on the Mac, verified through a local preview page. Hosting is Plan 4.

**Architecture:** Four new pipeline stages (`webcoords`, `tiles`, `regions`, `index`) following the existing stage-module pattern (`add_parser`/`run` + pure core functions). One shared palette module. Aggregation happens once at zoom 9; every shallower zoom is a 2x2 reduction. A minimal static preview page (MapLibre from CDN) serves as the local verification harness.

**Tech Stack:** Python 3.11+, DuckDB, numpy, pillow (new runtime deps), pytest. No network calls anywhere in this plan except the CDN script tag in the dev preview page.

**Spec:** `docs/superpowers/specs/2026-07-10-visual-phase-design.md`. Input schema (authoritative): `id VARCHAR, display_name VARCHAR, x FLOAT, y FLOAT, community INTEGER, works_count INTEGER, cited_by_count INTEGER, institution VARCHAR, field VARCHAR`; row order arbitrary - join by `id`.

## Global Constraints

- `MAPADEMIC_DATA` defaults to `./data` on the Mac for this plan (coords.parquet lives at `data/coords.parquet` in the repo checkout; `data/` is git-ignored)
- World coordinates: `xw, yw` in [0,1] with a 2% margin (all points within [0.02, 0.98]); yw is "up" in data space; XYZ tile row is `ty = floor((1 - yw) * 2^z)` (y flips ONLY at tile addressing, in one helper used everywhere)
- Max raster zoom **9** (2^9 = 512 tiles/side; 131,072 virtual pixels/side); zoom 10 is a POST-MEASUREMENT decision, not built in this plan
- Radial transform: `r' = asinh(r/s)` with `s = median radius`; center = **median(x), median(y)** (mean is dragged by the sparse halo - caught during Task 1 implementation). DuckDB has no asinh: use `ln(v + sqrt(v*v + 1))`
- Palette: **26** fixed field hue anchors + grey (NULL/unknown field); community shade jitter deterministic via splitmix64 - same (field, community) always yields the same RGB, test-pinned. (The real coords.parquet has 26 distinct non-null fields, not the 19 the spec sketched - discovered in Task 2; no aliasing, each real field is a direct key.)
- Tile PNGs: 256x256 RGB, black background; empty tiles never written; per-zoom histogram-equalized brightness from log1p(count); 3x3 bloom kernel at zooms 8-9 only
- All stages idempotent; `tiles` resumes by skipping existing tile files
- Tests never touch `data/` - fixtures in tmp_path only; golden tests compare decoded pixel arrays, never PNG bytes (codec drift)
- New runtime deps `numpy`, `pillow` go in pyproject `dependencies`; matplotlib stays uninstalled-by-default (dev tool)
- Commits: conventional style; NEVER Claude as co-author (CLAUDE.md rule 6)

---

### Task 1: `webcoords` stage (asinh radial compression)

**Files:**
- Create: `pipeline/webcoords.py`
- Modify: `pipeline/__main__.py` (register stage as `webcoords`)
- Modify: `pyproject.toml` (add `numpy`, `pillow` to dependencies)
- Test: `tests/test_webcoords.py`

**Interfaces:**
- Consumes: `<DATA>/coords.parquet` (schema above)
- Produces: `pipeline.webcoords.build_webcoords(coords_path: str, out_path: str) -> dict` returning stats `{"n": int, "cx": float, "cy": float, "s": float, "r2max": float}`; writes `coords_web.parquet` with columns `id, display_name, xw DOUBLE, yw DOUBLE, community, works_count, cited_by_count, institution, field` where all xw/yw lie in [0.02, 0.98]. Also writes `<out_path>.meta.json` with the stats. CLI: `python -m pipeline webcoords [--coords P] [--out P]`.

- [ ] **Step 1: Add deps and write the failing tests**

In `pyproject.toml` change the dependencies line to:

```toml
dependencies = ["duckdb>=1.0", "numpy>=1.26", "pillow>=10"]
```

`tests/test_webcoords.py`:

```python
import json
import math

import duckdb
import pytest

from pipeline.webcoords import build_webcoords


def write_coords(path, rows):
    """rows: list of (id, x, y). Other columns filled with constants."""
    vals = ", ".join(
        f"('{i}', 'N {i}', CAST({x} AS FLOAT), CAST({y} AS FLOAT), 1, 20, 100,"
        f" 'Inst', 'Biology')" for i, x, y in rows
    )
    duckdb.sql(
        f"COPY (SELECT * FROM (VALUES {vals}) t(id, display_name, x, y,"
        f" community, works_count, cited_by_count, institution, field))"
        f" TO '{path}' (FORMAT PARQUET)"
    )


@pytest.fixture
def coords_file(tmp_path):
    # a center-heavy cross plus two far halo points
    rows = [("A0", 0, 0), ("A1", 10, 0), ("A2", -10, 0), ("A3", 0, 10),
            ("A4", 0, -10), ("H1", 1000, 0), ("H2", 0, -1000)]
    p = tmp_path / "coords.parquet"
    write_coords(p, rows)
    return str(p)


def test_all_points_inside_margin(coords_file, tmp_path):
    out = str(tmp_path / "coords_web.parquet")
    stats = build_webcoords(coords_file, out)
    assert stats["n"] == 7
    lo, hi = duckdb.sql(
        f"SELECT least(min(xw), min(yw)), greatest(max(xw), max(yw)) FROM '{out}'"
    ).fetchone()
    assert lo >= 0.02 - 1e-9 and hi <= 0.98 + 1e-9


def test_radius_order_preserved_and_angles_kept(coords_file, tmp_path):
    out = str(tmp_path / "coords_web.parquet")
    build_webcoords(coords_file, out)
    r = {
        i: math.hypot(xw - 0.5, yw - 0.5)
        for i, xw, yw in duckdb.sql(f"SELECT id, xw, yw FROM '{out}'").fetchall()
    }
    assert r["A0"] < r["A1"] < r["H1"]          # monotonic in original radius
    assert r["A1"] == pytest.approx(r["A2"])     # symmetric points equal radius
    # halo compressed: H1 is 100x A1's radius in data, far less on the map
    assert r["H1"] / r["A1"] < 10
    # angle preserved: A1 lies due +x of center, H2 due -y
    x1, y1 = duckdb.sql(f"SELECT xw, yw FROM '{out}' WHERE id='A1'").fetchone()
    assert y1 == pytest.approx(0.5, abs=1e-6) and x1 > 0.5
    xh, yh = duckdb.sql(f"SELECT xw, yw FROM '{out}' WHERE id='H2'").fetchone()
    assert xh == pytest.approx(0.5, abs=1e-6) and yh < 0.5


def test_meta_json_written(coords_file, tmp_path):
    out = str(tmp_path / "coords_web.parquet")
    stats = build_webcoords(coords_file, out)
    meta = json.loads((tmp_path / "coords_web.parquet.meta.json").read_text())
    assert meta == stats
    assert set(stats) == {"n", "cx", "cy", "s", "r2max"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pip install -e '.[dev]' -q && .venv/bin/pytest tests/test_webcoords.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.webcoords'`

- [ ] **Step 3: Write the implementation**

`pipeline/webcoords.py`:

```python
"""asinh radial compression: layout coords -> unit-square web coords."""
import json
from pathlib import Path

import duckdb

from pipeline.config import apply_resource_limits, data_dir

MARGIN = 0.02  # points land in [MARGIN, 1-MARGIN]


def build_webcoords(coords_path: str, out_path: str) -> dict:
    con = duckdb.connect()
    apply_resource_limits(con)
    cx, cy = con.execute(
        f"SELECT median(x), median(y) FROM read_parquet('{coords_path}')"
    ).fetchone()
    # median radius as the asinh scale; guard degenerate all-at-center inputs
    s = con.execute(
        f"""
        SELECT greatest(median(sqrt((x - {cx})*(x - {cx}) + (y - {cy})*(y - {cy}))), 1e-9)
        FROM read_parquet('{coords_path}')
        """
    ).fetchone()[0]
    # asinh(v) = ln(v + sqrt(v*v + 1)); DuckDB lacks asinh
    r2max = con.execute(
        f"""
        SELECT max(ln(r/{s} + sqrt((r/{s})*(r/{s}) + 1))) FROM (
            SELECT sqrt((x - {cx})*(x - {cx}) + (y - {cy})*(y - {cy})) AS r
            FROM read_parquet('{coords_path}')
        )
        """
    ).fetchone()[0] or 1e-9
    half = 0.5 - MARGIN
    n = con.execute(
        f"""
        COPY (
            WITH polar AS (
                SELECT *,
                       sqrt((x - {cx})*(x - {cx}) + (y - {cy})*(y - {cy})) AS r,
                       atan2(y - {cy}, x - {cx}) AS theta
                FROM read_parquet('{coords_path}')
            ),
            compressed AS (
                SELECT *,
                       CASE WHEN r = 0 THEN 0.0
                            ELSE ln(r/{s} + sqrt((r/{s})*(r/{s}) + 1)) / {r2max}
                       END AS ru
                FROM polar
            )
            SELECT id, display_name,
                   0.5 + {half} * ru * cos(theta) AS xw,
                   0.5 + {half} * ru * sin(theta) AS yw,
                   community, works_count, cited_by_count, institution, field
            FROM compressed
        ) TO '{out_path}' (FORMAT PARQUET)
        """
    ).fetchone()[0]
    stats = {"n": n, "cx": cx, "cy": cy, "s": s, "r2max": r2max}
    Path(out_path + ".meta.json").write_text(json.dumps(stats))
    return stats


def add_parser(parser) -> None:
    parser.add_argument("--coords", default=None)
    parser.add_argument("--out", default=None)


def run(args) -> int:
    coords = args.coords or str(data_dir() / "coords.parquet")
    out = args.out or str(data_dir() / "coords_web.parquet")
    stats = build_webcoords(coords, out)
    print(f"{stats['n']:,} points -> {out} (s={stats['s']:.1f}, r2max={stats['r2max']:.3f})")
    return 0
```

In `pipeline/__main__.py`, grow the registry (imports + dict entry, same pattern as existing stages): add `webcoords` -> `pipeline.webcoords`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest -v`
Expected: all PASS (3 new + 30 existing)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml pipeline/webcoords.py pipeline/__main__.py tests/test_webcoords.py
git commit -m "feat: webcoords stage - asinh radial compression to unit square"
```

---

### Task 2: Palette module (field hues + community shades)

**Files:**
- Create: `pipeline/palette.py`
- Test: `tests/test_palette.py`

**Interfaces:**
- Consumes: nothing (pure module)
- Produces: `pipeline.palette.FIELD_HUES: dict[str, float]` (19 field names -> hue degrees; grey handled separately); `pipeline.palette.field_community_rgb(field: str | None, community: int) -> tuple[float, float, float]` returning base RGB in [0,1] at full brightness (callers multiply by their brightness); deterministic. `pipeline.palette.splitmix64(x: int) -> int`.

- [ ] **Step 1: Write the failing tests**

`tests/test_palette.py`:

```python
from pipeline.palette import FIELD_HUES, field_community_rgb


def test_nineteen_fields_covered():
    assert len(FIELD_HUES) == 19
    assert "Computer Science" in FIELD_HUES and "Biology" in FIELD_HUES


def test_deterministic_and_community_varies():
    a = field_community_rgb("Biology", 42)
    assert a == field_community_rgb("Biology", 42)  # stable across calls
    b = field_community_rgb("Biology", 43)
    assert a != b                                    # neighbors distinguishable
    # same family: hues near the Biology anchor -> green channel dominates both
    assert a[1] == max(a) and b[1] == max(b)


def test_null_field_is_grey():
    r, g, b = field_community_rgb(None, 7)
    assert abs(r - g) < 0.05 and abs(g - b) < 0.05   # near-neutral
    assert field_community_rgb(None, 7) == field_community_rgb(None, 7)


def test_rgb_in_unit_range():
    for field in list(FIELD_HUES) + [None]:
        for comm in (0, 1, 999999):
            assert all(0.0 <= c <= 1.0 for c in field_community_rgb(field, comm))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_palette.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write the implementation**

`pipeline/palette.py`:

```python
"""Dark-cosmic palette: 19 field hue families, community shade jitter."""
import colorsys

# Hue anchors (degrees) spread for perceptual separation on black.
# The 19 OpenAlex fields (topics[1].field.display_name vocabulary).
FIELD_HUES: dict[str, float] = {
    "Agricultural and Biological Sciences": 95.0,
    "Arts and Humanities": 30.0,
    "Biochemistry, Genetics and Molecular Biology": 130.0,
    "Business, Management and Accounting": 45.0,
    "Chemical Engineering": 285.0,
    "Chemistry": 300.0,
    "Computer Science": 210.0,
    "Decision Sciences": 55.0,
    "Earth and Planetary Sciences": 170.0,
    "Economics, Econometrics and Finance": 40.0,
    "Energy": 15.0,
    "Engineering": 250.0,
    "Environmental Science": 150.0,
    "Health Professions": 340.0,
    "Immunology and Microbiology": 110.0,
    "Materials Science": 265.0,
    "Mathematics": 225.0,
    "Medicine": 0.0,
    "Neuroscience": 320.0,
}
# Aliases: coords.parquet 'field' values that differ from anchor keys map here.
FIELD_ALIASES: dict[str, str] = {
    "Biology": "Agricultural and Biological Sciences",
    "Physics": "Materials Science",
    "Physics and Astronomy": "Materials Science",
    "Psychology": "Neuroscience",
    "Social Sciences": "Arts and Humanities",
    "Nursing": "Health Professions",
    "Dentistry": "Health Professions",
    "Veterinary": "Agricultural and Biological Sciences",
    "Pharmacology, Toxicology and Pharmaceutics": "Chemistry",
}


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def field_community_rgb(field: str | None, community: int) -> tuple[float, float, float]:
    """Base RGB in [0,1] at full brightness; deterministic."""
    h = splitmix64(int(community))
    j1 = ((h & 0xFFFF) / 0xFFFF) * 2 - 1          # [-1, 1]
    j2 = (((h >> 16) & 0xFFFF) / 0xFFFF) * 2 - 1  # [-1, 1]
    key = FIELD_ALIASES.get(field, field) if field else None
    if key is None or key not in FIELD_HUES:
        light = 0.62 + 0.18 * j1                   # neutral grey family
        return colorsys.hls_to_rgb(0.0, light, 0.03)
    hue = (FIELD_HUES[key] + 9.0 * j1) % 360.0     # small hue wobble in-family
    sat = min(1.0, max(0.35, 0.75 + 0.15 * j2))
    light = min(0.78, max(0.45, 0.60 + 0.12 * j2))
    return colorsys.hls_to_rgb(hue / 360.0, light, sat)


def add_parser(parser) -> None:  # not a CLI stage; keeps import-shape uniform
    raise SystemExit("palette is a library module, not a stage")


def run(args) -> int:
    raise SystemExit("palette is a library module, not a stage")
```

Note: do NOT register palette in `STAGES`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/palette.py tests/test_palette.py
git commit -m "feat: field-hue + community-shade palette module"
```

---

### Task 3: `tiles` stage - aggregation, pyramid, styled PNGs

The heart of the plan. Split into pure functions so each is testable: aggregate (DuckDB) -> reduce (numpy) -> style (numpy+palette) -> write (pillow).

**Files:**
- Create: `pipeline/tiles.py`
- Modify: `pipeline/__main__.py` (register stage as `tiles`)
- Test: `tests/test_tiles.py`

**Interfaces:**
- Consumes: `coords_web.parquet` (Task 1)
- Produces:
  - `aggregate_z9(webcoords_path: str, con) -> duckdb relation persisted as <DATA>/pixels_z9.parquet` with columns `px INT, py INT, cnt BIGINT, field VARCHAR, community INT` (dominant field/community per pixel, y-UP pixel space `py = floor(yw * 131072)`)
  - `load_level9(pixels_path) -> dict[str, np.ndarray]` with keys `px, py, cnt, rgb` (rgb precomputed via palette, shape (n,3) float32)
  - `reduce_level(level: dict) -> dict` producing the next-shallower level (2x2 sum of cnt; rgb of the max-cnt child - count-weighted dominant approximation)
  - `render_zoom(level: dict, z: int, out_dir: Path, bloom: bool) -> int` writing `<out_dir>/{z}/{x}/{y}.png` (XYZ addressing, y flipped HERE only: `ty = (2^z - 1) - (py_tile)`) and returning tiles written; skips tiles whose file exists
  - CLI: `python -m pipeline tiles [--zooms 0-9] [--web P] [--out DIR]` default out `<DATA>/tiles`
- Brightness: per zoom, `bright = 0.15 + 0.85 * rank(log1p(cnt))/n` (empirical CDF, same as the QA renderer)

- [ ] **Step 1: Write the failing tests**

`tests/test_tiles.py`:

```python
import duckdb
import numpy as np
import pytest
from PIL import Image

from pipeline.tiles import (MAXZ, PIX, aggregate_z9, load_level9,
                            reduce_level, render_zoom)

# MAXZ == 9, PIX == 2**9 * 256 == 131072


def write_web(path, rows):
    """rows: (id, xw, yw, community, field)"""
    vals = ", ".join(
        f"('{i}', 'N', {xw}, {yw}, {c}, 20, 100, 'I', "
        + (f"'{f}'" if f else "NULL") + ")"
        for i, xw, yw, c, f in rows
    )
    duckdb.sql(
        f"COPY (SELECT * FROM (VALUES {vals}) t(id, display_name, xw, yw,"
        f" community, works_count, cited_by_count, institution, field))"
        f" TO '{path}' (FORMAT PARQUET)"
    )


@pytest.fixture
def tiny_web(tmp_path):
    p = tmp_path / "coords_web.parquet"
    # two points in the same z9 pixel (dominant test), one lone point far away
    eps = 0.2 / PIX
    write_web(p, [
        ("a", 0.25, 0.25, 1, "Biology"),
        ("b", 0.25 + eps, 0.25 + eps, 1, "Biology"),
        ("c", 0.25, 0.25, 2, "Chemistry"),   # same pixel, minority
        ("d", 0.75, 0.75, 3, "Computer Science"),
    ])
    return str(p)


def test_aggregate_dominant_and_counts(tiny_web, tmp_path):
    out = str(tmp_path / "pixels_z9.parquet")
    con = duckdb.connect()
    aggregate_z9(tiny_web, out, con)
    rows = duckdb.sql(
        f"SELECT px, py, cnt, field, community FROM '{out}' ORDER BY px"
    ).fetchall()
    assert len(rows) == 2                      # two occupied pixels
    dense = rows[0]
    assert dense[2] == 3                       # a+b+c share one pixel
    assert dense[3] == "Biology" and dense[4] == 1   # dominant wins
    assert rows[1][2] == 1


def test_pyramid_counts_conserved(tiny_web, tmp_path):
    out = str(tmp_path / "pixels_z9.parquet")
    aggregate_z9(tiny_web, out, duckdb.connect())
    level = load_level9(out)
    total = level["cnt"].sum()
    for _ in range(MAXZ):                      # 9 reductions -> zoom 0
        level = reduce_level(level)
        assert level["cnt"].sum() == total
    assert len(level["cnt"]) == 2              # 2 distinct pixels (128px apart at z0), same tile


def test_render_writes_expected_tiles_and_is_idempotent(tiny_web, tmp_path):
    pixels = str(tmp_path / "pixels_z9.parquet")
    aggregate_z9(tiny_web, pixels, duckdb.connect())
    level = load_level9(pixels)
    for _ in range(MAXZ - 1):                  # reduce to zoom 1 (2x2 tiles)
        level = reduce_level(level)
    out = tmp_path / "tiles"
    n = render_zoom(level, 1, out, bloom=False)
    assert n == 2                              # points at (.25,.25) and (.75,.75)
    # XYZ y-flip: yw=0.75 (upper area) -> tile row 0; yw=0.25 -> row 1
    assert (out / "1/0/1.png").exists() and (out / "1/1/0.png").exists()
    img = np.asarray(Image.open(out / "1/1/0.png"))
    assert img.shape == (256, 256, 3) and img.max() > 0
    assert render_zoom(level, 1, out, bloom=False) == 0   # resume: all skipped
    # GOLDEN pixel: the dense pixel (cnt=3, dominant Biology/community 1) has
    # rank 2/2 -> brightness 1.0, so its RGB equals the raw palette color.
    from pipeline.palette import field_community_rgb
    expected = tuple(int(c * 255) for c in field_community_rgb("Biology", 1))  # trunc, matching astype(uint8)
    dense = np.asarray(Image.open(out / "1/0/1.png"))
    ys, xs = np.nonzero(dense.sum(axis=2))
    assert tuple(dense[ys[0], xs[0]]) == expected


def test_styled_pixels_deterministic(tiny_web, tmp_path):
    pixels = str(tmp_path / "pixels_z9.parquet")
    aggregate_z9(tiny_web, pixels, duckdb.connect())
    level = load_level9(pixels)
    for _ in range(MAXZ - 1):
        level = reduce_level(level)
    a, b = tmp_path / "ta", tmp_path / "tb"
    render_zoom(level, 1, a, bloom=False)
    render_zoom(level, 1, b, bloom=False)
    ia = np.asarray(Image.open(a / "1/1/0.png"))
    ib = np.asarray(Image.open(b / "1/1/0.png"))
    assert np.array_equal(ia, ib)              # golden-by-self: bytes-level stability
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_tiles.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write the implementation**

`pipeline/tiles.py`:

```python
"""Aggregate web coords at zoom 9, pyramid-reduce, write styled PNG tiles."""
from pathlib import Path

import duckdb
import numpy as np
from PIL import Image

from pipeline.config import apply_resource_limits, data_dir
from pipeline.palette import field_community_rgb

MAXZ = 9
TILE = 256
PIX = (2 ** MAXZ) * TILE  # 131072 virtual pixels per side at zoom 9


def aggregate_z9(webcoords_path: str, out_path: str, con) -> int:
    """Per-pixel count + dominant (field, community). Returns occupied pixels."""
    apply_resource_limits(con)
    return con.execute(
        f"""
        COPY (
            WITH binned AS (
                -- cast to DOUBLE first: high-precision fixture literals are DECIMAL,
                -- which overflows when multiplied by PIX (no-op on real DOUBLE data)
                SELECT least({PIX - 1}, CAST(floor(CAST(xw AS DOUBLE) * {PIX}) AS INT)) AS px,
                       least({PIX - 1}, CAST(floor(CAST(yw AS DOUBLE) * {PIX}) AS INT)) AS py,
                       field, community
                FROM read_parquet('{webcoords_path}')
            ),
            grouped AS (
                SELECT px, py, field, community, count(*) AS c
                FROM binned GROUP BY px, py, field, community
            ),
            ranked AS (
                SELECT px, py, field, community, c,
                       sum(c) OVER (PARTITION BY px, py) AS cnt,
                       row_number() OVER (PARTITION BY px, py
                                          ORDER BY c DESC, community) AS rn
                FROM grouped
            )
            SELECT px, py, CAST(cnt AS BIGINT) AS cnt, field, community
            FROM ranked WHERE rn = 1
        ) TO '{out_path}' (FORMAT PARQUET)
        """
    ).fetchone()[0]


def load_level9(pixels_path: str) -> dict:
    con = duckdb.connect()
    # palette per DISTINCT (field, community) pair (~216k), then broadcast -
    # avoids a python loop over all 8.6M pixels
    pairs = con.execute(
        f"""SELECT field, community, row_number() OVER () - 1 AS pi
            FROM (SELECT DISTINCT field, community
                  FROM read_parquet('{pixels_path}'))"""
    ).fetchall()
    pair_rgb = np.empty((len(pairs), 3), dtype=np.float32)
    for field, community, pi in pairs:
        pair_rgb[pi] = field_community_rgb(
            None if field is None else str(field), int(community)
        )
    con.execute("CREATE TEMP TABLE pal (field VARCHAR, community INT, pi INT)")
    con.executemany("INSERT INTO pal VALUES (?, ?, ?)", pairs)
    t = con.execute(
        f"""SELECT p.px, p.py, p.cnt, pal.pi
            FROM read_parquet('{pixels_path}') p
            JOIN pal ON pal.community = p.community
                    AND pal.field IS NOT DISTINCT FROM p.field"""
    ).fetchnumpy()
    return {
        "px": t["px"].astype(np.int64),
        "py": t["py"].astype(np.int64),
        "cnt": t["cnt"].astype(np.int64),
        "rgb": pair_rgb[t["pi"].astype(np.int64)],
    }


def reduce_level(level: dict) -> dict:
    """2x2 -> 1: counts summed, rgb of the heaviest child (dominant approx)."""
    px, py = level["px"] >> 1, level["py"] >> 1
    key = px * 2**31 + py  # unique combined key (px, py < 2**17)
    order = np.argsort(key, kind="stable")
    key_s, cnt_s, rgb_s = key[order], level["cnt"][order], level["rgb"][order]
    px_s, py_s = px[order], py[order]
    boundaries = np.flatnonzero(np.diff(key_s)) + 1
    groups = np.split(np.arange(len(key_s)), boundaries)
    n = len(groups)
    out = {
        "px": np.empty(n, np.int64), "py": np.empty(n, np.int64),
        "cnt": np.empty(n, np.int64), "rgb": np.empty((n, 3), np.float32),
    }
    for gi, idx in enumerate(groups):
        out["px"][gi] = px_s[idx[0]]
        out["py"][gi] = py_s[idx[0]]
        c = cnt_s[idx]
        out["cnt"][gi] = c.sum()
        out["rgb"][gi] = rgb_s[idx[np.argmax(c)]]
    return out


def _brightness(cnt: np.ndarray) -> np.ndarray:
    vals = np.log1p(cnt.astype(np.float64))
    ranks = np.searchsorted(np.sort(vals), vals, side="right") / len(vals)
    return (0.15 + 0.85 * ranks).astype(np.float32)


BLOOM = np.array([[0.06, 0.12, 0.06], [0.12, 0.0, 0.12], [0.06, 0.12, 0.06]],
                 dtype=np.float32)


def render_zoom(level: dict, z: int, out_dir: Path, bloom: bool) -> int:
    """Write XYZ PNG tiles for one zoom; skip existing. Returns tiles written."""
    ntiles = 1 << z
    px, py = level["px"], level["py"]  # already in zoom-z pixel space
    bright = _brightness(level["cnt"])
    color = level["rgb"] * bright[:, None]
    tx, ty_up = px // TILE, py // TILE
    written = 0
    for t in np.unique(tx * ntiles + ty_up):
        x, yu = int(t // ntiles), int(t % ntiles)
        ty = (ntiles - 1) - yu                       # XYZ y-flip, here only
        path = out_dir / str(z) / str(x) / f"{ty}.png"
        if path.exists():
            continue
        sel = (tx == x) & (ty_up == yu)
        img = np.zeros((TILE, TILE, 3), dtype=np.float32)
        ix = px[sel] - x * TILE
        iy_up = py[sel] - yu * TILE
        iy = (TILE - 1) - iy_up                      # flip rows inside the tile
        img[iy, ix] = color[sel]
        if bloom:
            base = img.copy()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    w = BLOOM[dy + 1, dx + 1]
                    if w:
                        img[max(0, dy):TILE + min(0, dy) or TILE,
                            max(0, dx):TILE + min(0, dx) or TILE] += \
                            w * base[max(0, -dy):TILE + min(0, -dy) or TILE,
                                     max(0, -dx):TILE + min(0, -dx) or TILE]
        arr = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(path, optimize=True)
        written += 1
    return written


def _parse_zooms(spec: str) -> list[int]:
    lo, _, hi = spec.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def add_parser(parser) -> None:
    parser.add_argument("--zooms", default="0-9", help="e.g. 0-5 for the QA gate")
    parser.add_argument("--web", default=None)
    parser.add_argument("--out", default=None)


def run(args) -> int:
    web = args.web or str(data_dir() / "coords_web.parquet")
    out = Path(args.out) if args.out else data_dir() / "tiles"
    pixels = str(data_dir() / "pixels_z9.parquet")
    if not Path(pixels).exists():
        n = aggregate_z9(web, pixels, duckdb.connect())
        print(f"aggregated {n:,} occupied z9 pixels", flush=True)
    zooms = sorted(_parse_zooms(args.zooms), reverse=True)  # deep -> shallow
    level, at = load_level9(pixels), MAXZ
    for z in range(MAXZ, -1, -1):
        if z in zooms:
            w = render_zoom(level, z, out, bloom=(z >= 8))
            print(f"zoom {z}: {w} tiles written", flush=True)
        if z > 0:
            level = reduce_level(level)
    return 0
```

Registry: add `tiles` -> `pipeline.tiles` in `pipeline/__main__.py`.

Implementation note: the `reduce_level` group loop is python-level over occupied
parent pixels (<=8.6M at level 8, halving each level). If profiling in Task 5
shows it too slow on the real data, vectorize with `np.add.reduceat` - but do
not pre-optimize; the interface is what matters.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/tiles.py pipeline/__main__.py tests/test_tiles.py
git commit -m "feat: tiles stage - z9 aggregation, pyramid reduction, styled PNGs"
```

---

### Task 4: Local preview page + real-data QA render (zooms 0-5)

**Files:**
- Create: `web/dev.html`
- Create: `scripts/preview.sh`
- No pipeline changes.

**Interfaces:**
- Consumes: `data/tiles/` (Task 3), later `data/index/` + `data/regions.json`
- Produces: a local, hosting-free verification loop: `bash scripts/preview.sh` serves the repo root; `web/dev.html` shows the tile pyramid via MapLibre (CDN) reading `/data/tiles/{z}/{x}/{y}.png`.

- [ ] **Step 1: Create the preview page**

`web/dev.html`:

```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>mapademic dev preview</title>
  <script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
  <link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet">
  <style>html, body, #map { margin: 0; height: 100%; background: #000; }</style>
</head>
<body>
<div id="map"></div>
<script>
const map = new maplibregl.Map({
  container: "map",
  style: {
    version: 8,
    sources: {
      dots: { type: "raster", tiles: [location.origin + "/data/tiles/{z}/{x}/{y}.png"],
              tileSize: 256, minzoom: 0, maxzoom: 9 }
    },
    layers: [
      { id: "bg", type: "background", paint: { "background-color": "#000000" } },
      { id: "dots", type: "raster", source: "dots",
        paint: { "raster-fade-duration": 150 } }
    ]
  },
  center: [0, 0], zoom: 1, minZoom: 0, maxZoom: 12, renderWorldCopies: false
});
map.addControl(new maplibregl.NavigationControl());
</script>
</body>
</html>
```

`scripts/preview.sh`:

```bash
#!/bin/bash
# Local verification server: open http://localhost:8123/web/dev.html
cd "$(dirname "$0")/.."
exec python3 -m http.server 8123
```

`chmod +x scripts/preview.sh`.

Note: the map treats the unit square as Web-Mercator world - fine for a dev
preview (we only need pan/zoom over our own tiles); Plan 4's real site sets
proper camera bounds.

- [ ] **Step 2: Run the real pipeline through zoom 5 (QA gate data)**

```bash
cd /Users/eric/Downloads/mapademic
MAPADEMIC_THREADS=8 MAPADEMIC_MEMORY_LIMIT=24GB .venv/bin/python -m pipeline webcoords
MAPADEMIC_THREADS=8 MAPADEMIC_MEMORY_LIMIT=24GB .venv/bin/python -m pipeline tiles --zooms 0-5
du -sh data/tiles; find data/tiles -name "*.png" | wc -l
```

Expected: webcoords prints 8,587,906 points; tiles prints per-zoom counts; total size well under 1GB for zooms 0-5.

- [ ] **Step 3: HUMAN GATE - Eric reviews the preview**

Run `bash scripts/preview.sh`, open http://localhost:8123/web/dev.html.
Checklist from the spec: does the core fill the viewport (asinh working)?
Do field hue-territories read at zooms 1-3? Does the halo form a thin shell,
not dominate? **Get Eric's explicit approval (or palette/transform tweak
requests) before Task 5 burns zooms 6-9.** Iterate here if needed - restyling
means deleting `data/tiles` and re-rendering from the cached
`pixels_z9.parquet` (minutes, not hours).

- [ ] **Step 4: Commit**

```bash
git add web/dev.html scripts/preview.sh
git commit -m "feat: local dev preview page and server script"
```

---

### Task 5: Full pyramid render (zooms 0-9) + measurement

**Files:** none new - production run of Task 3's stage.

- [ ] **Step 1: Render the remaining zooms**

```bash
MAPADEMIC_THREADS=8 MAPADEMIC_MEMORY_LIMIT=24GB .venv/bin/python -m pipeline tiles --zooms 6-9
du -sh data/tiles; find data/tiles -name "*.png" | wc -l
```

Existing zoom 0-5 tiles are skipped (idempotent). If the zoom 8-9 render is
slow (python group loop), vectorize `reduce_level` with `np.add.reduceat`
first - conserve the tests.

- [ ] **Step 2: Record the measurement + zoom-10 decision input**

Append to this plan file: total pyramid size, tile count, render wall time.
The zoom-10 decision (partial deepest level over dense cores) goes to Eric
with those numbers; it is OUT of this plan's scope either way.

- [ ] **Step 3: Spot-verify deep zoom in the preview** (Hinton's neighborhood
should resolve to individual star-dots at z9; no seams between tiles;
y-orientation consistent across zoom levels - a landmark that is upper-left
at z3 stays upper-left at z7).

---

### Task 6: `regions` stage (community naming)

**Files:**
- Create: `pipeline/regions.py`
- Modify: `pipeline/__main__.py` (register as `regions`)
- Test: `tests/test_regions.py`

**Interfaces:**
- Consumes: `coords_web.parquet`; the authors snapshot parquet glob (for topics) - default `/Volumes/Untitled/mapademic/snapshot/authors/*/*.parquet`, overridable `--authors`; fails with a clear message if the glob matches nothing ("plug in the external drive or pass --authors")
- Produces: `pipeline.regions.build_regions(webcoords_path, authors_glob, out_path, top_n=300, keep=120) -> int` (regions written); `regions.json`: array of `{"name": str, "xw": float, "yw": float, "spread": float, "members": int, "rank": int, "community": int, "zmin": int, "zmax": int}` sorted by rank; zoom bands: rank 1-30 -> zmin 2, zmax 4; rank 31+ -> zmin 4, zmax 6
- Naming: for each top-N community, member topics (authors snapshot `topics[1].display_name` joined via id); score = count_in_community * ln(total_authors_with_any_topic / count_topic_overall); name = top-scoring topic; ties by topic count then alphabetical

- [ ] **Step 1: Write the failing tests**

`tests/test_regions.py`:

```python
import json

import duckdb
import pytest

from pipeline.regions import build_regions


def make_fixture(tmp_path):
    web = tmp_path / "coords_web.parquet"
    # community 1: 3 authors clustered near (0.3, 0.3); community 2: 2 authors
    rows = [
        ("A1", 0.30, 0.30, 1), ("A2", 0.31, 0.30, 1), ("A3", 0.30, 0.31, 1),
        ("B1", 0.70, 0.70, 2), ("B2", 0.71, 0.70, 2),
    ]
    vals = ", ".join(
        f"('{i}', 'N', {x}, {y}, {c}, 20, 100, 'I', 'Biology')" for i, x, y, c in rows
    )
    duckdb.sql(f"COPY (SELECT * FROM (VALUES {vals}) t(id, display_name, xw, yw,"
               f" community, works_count, cited_by_count, institution, field))"
               f" TO '{web}' (FORMAT PARQUET)")
    authors = tmp_path / "authors" / "updated_date=2026-01-01"
    authors.mkdir(parents=True)
    # 'Genome Editing' distinctive to community 1; 'Deep Learning' to 2;
    # 'Statistics' ubiquitous (appears in both -> low distinctiveness)
    topic_rows = [
        ("A1", "Genome Editing"), ("A2", "Genome Editing"), ("A3", "Statistics"),
        ("B1", "Deep Learning"), ("B2", "Statistics"),
    ]
    tvals = ", ".join(
        f"('{i}', [{{'display_name': '{t}'}}])" for i, t in topic_rows
    )
    duckdb.sql(f"COPY (SELECT * FROM (VALUES {tvals}) t(id, topics))"
               f" TO '{authors / 'part_0000.parquet'}' (FORMAT PARQUET)")
    return str(web), str(tmp_path / "authors" / "*" / "*.parquet")


def test_distinctive_names_and_geometry(tmp_path):
    web, authors = make_fixture(tmp_path)
    out = tmp_path / "regions.json"
    n = build_regions(web, authors, str(out), top_n=10, keep=10)
    regions = json.loads(out.read_text())
    assert n == len(regions) == 2
    by_comm = {r["community"]: r for r in regions}
    assert by_comm[1]["name"] == "Genome Editing"     # not ubiquitous Statistics
    assert by_comm[2]["name"] == "Deep Learning"
    assert by_comm[1]["members"] == 3
    assert by_comm[1]["xw"] == pytest.approx(0.3033, abs=1e-3)
    assert by_comm[1]["rank"] == 1                     # biggest first
    assert by_comm[1]["zmin"] == 2 and by_comm[2]["zmin"] == 2


def test_missing_authors_glob_fails_clearly(tmp_path):
    web, _ = make_fixture(tmp_path)
    with pytest.raises(SystemExit, match="authors snapshot"):
        build_regions(web, str(tmp_path / "nope" / "*.parquet"),
                      str(tmp_path / "r.json"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_regions.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write the implementation**

`pipeline/regions.py`:

```python
"""Auto-name the largest communities from their members' dominant topics."""
import glob as globmod
import json
from pathlib import Path

import duckdb

from pipeline.config import apply_resource_limits, data_dir

DEFAULT_AUTHORS = "/Volumes/Untitled/mapademic/snapshot/authors/*/*.parquet"


def build_regions(webcoords_path: str, authors_glob: str, out_path: str,
                  top_n: int = 300, keep: int = 120) -> int:
    if not globmod.glob(authors_glob):
        raise SystemExit(
            f"authors snapshot not found at {authors_glob!r} - plug in the "
            "external drive or pass --authors <glob>"
        )
    con = duckdb.connect()
    apply_resource_limits(con)
    con.execute(
        f"""
        CREATE TEMP TABLE named AS
        WITH top_comms AS (
            SELECT community, count(*) AS members,
                   avg(xw) AS xw, avg(yw) AS yw,
                   sqrt(var_pop(xw) + var_pop(yw)) AS spread
            FROM read_parquet('{webcoords_path}')
            GROUP BY community
            ORDER BY members DESC
            LIMIT {int(top_n)}
        ),
        member_topics AS (
            SELECT w.community, a.topics[1].display_name AS topic
            FROM read_parquet('{webcoords_path}') w
            JOIN read_parquet('{authors_glob}') a ON a.id = w.id
            JOIN top_comms tc ON tc.community = w.community
            WHERE a.topics[1].display_name IS NOT NULL
        ),
        topic_global AS (
            SELECT topic, count(*) AS g FROM member_topics GROUP BY topic
        ),
        scored AS (
            SELECT mt.community, mt.topic, count(*) AS c,
                   count(*) * ln((SELECT count(*) FROM member_topics) * 1.0
                                 / tg.g) AS score
            FROM member_topics mt JOIN topic_global tg ON tg.topic = mt.topic
            GROUP BY mt.community, mt.topic, tg.g
        ),
        best AS (
            SELECT community, topic,
                   row_number() OVER (PARTITION BY community
                                      ORDER BY score DESC, c DESC, topic) AS rn
            FROM scored
        )
        SELECT tc.community, b.topic AS name, tc.members, tc.xw, tc.yw,
               tc.spread,
               row_number() OVER (ORDER BY tc.members DESC) AS rank
        FROM top_comms tc
        JOIN best b ON b.community = tc.community AND b.rn = 1
        ORDER BY rank
        LIMIT {int(keep)}
        """
    )
    rows = con.execute("SELECT * FROM named").fetchall()
    cols = [d[0] for d in con.description]
    regions = []
    for row in rows:
        r = dict(zip(cols, row))
        rank = r["rank"]
        regions.append({
            "name": r["name"], "xw": r["xw"], "yw": r["yw"],
            "spread": r["spread"], "members": r["members"], "rank": rank,
            "community": r["community"],
            "zmin": 2 if rank <= 30 else 4,
            "zmax": 4 if rank <= 30 else 6,
        })
    Path(out_path).write_text(json.dumps(regions, indent=1))
    return len(regions)


def add_parser(parser) -> None:
    parser.add_argument("--authors", default=DEFAULT_AUTHORS)
    parser.add_argument("--web", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--top-n", type=int, default=300)
    parser.add_argument("--keep", type=int, default=120)


def run(args) -> int:
    web = args.web or str(data_dir() / "coords_web.parquet")
    out = args.out or str(data_dir() / "regions.json")
    n = build_regions(web, args.authors, out, top_n=args.top_n, keep=args.keep)
    print(f"{n} regions named -> {out} (hand-edit freely; it is data)")
    return 0
```

Registry: add `regions` in `pipeline/__main__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest -v`
Expected: all PASS

- [ ] **Step 5: Run against real data (drive plugged in) and eyeball**

```bash
MAPADEMIC_THREADS=8 MAPADEMIC_MEMORY_LIMIT=24GB .venv/bin/python -m pipeline regions
python3 -c "import json; [print(r['rank'], r['name'], r['members']) for r in json.load(open('data/regions.json'))[:20]]"
```

Silly names are EXPECTED in the tail - note the worst offenders for Eric's
curation pass; do not over-engineer the scorer.

- [ ] **Step 6: Commit**

```bash
git add pipeline/regions.py pipeline/__main__.py tests/test_regions.py
git commit -m "feat: regions stage - distinctive-topic community naming"
```

---

### Task 7: `index` stage (label/hit tiles + search shards)

**Files:**
- Create: `pipeline/build_index.py`
- Modify: `pipeline/__main__.py` (register as `index`)
- Test: `tests/test_build_index.py`

**Interfaces:**
- Consumes: `coords_web.parquet`
- Produces: `pipeline.build_index.build_label_tiles(web, out_dir) -> int` and `build_search_shards(web, out_dir) -> int`; CLI `python -m pipeline index [--web P] [--out DIR]` default out `<DATA>/index`
  - Label/hit tiles: `<out>/labels/{z}/{x}/{y}.json` for z in 6..9, XYZ addressing (same y-flip helper convention as tiles). Content: `{"l": [[name, id, xw, yw, cited_by_count], ...]}` ranked by cited_by_count desc; capacity 50 per tile at z6-7, 200 at z8-9. Client draws the first 10 as text, uses all for hit-testing.
  - Search shards: `<out>/search/{shard}.json` where shard = first 2 chars of normalized name (lowercase, accents stripped via NFKD, non-alnum -> space, collapse), non-alpha bucket `_`. Content: array of `[normalized, display_name, id, xw, yw, cited_by_count]` sorted by cited desc. Every one of the 8.6M rows lands in exactly one shard.

- [ ] **Step 1: Write the failing tests**

`tests/test_build_index.py`:

```python
import json

import duckdb
import pytest

from pipeline.build_index import build_label_tiles, build_search_shards, normalize


def write_web(path, rows):
    """rows: (id, name, xw, yw, cited)"""
    vals = ", ".join(
        f"('{i}', '{n}', {x}, {y}, 1, 20, {c}, 'I', 'Biology')"
        for i, n, x, y, c in rows
    )
    duckdb.sql(f"COPY (SELECT * FROM (VALUES {vals}) t(id, display_name, xw, yw,"
               f" community, works_count, cited_by_count, institution, field))"
               f" TO '{path}' (FORMAT PARQUET)")


def test_normalize():
    assert normalize("Géraldine  O'Brien-Smith") == "geraldine o brien smith"
    assert normalize("李明") == "李明"          # non-latin kept verbatim (lowercased)


def test_label_tiles_ranked_and_capped(tmp_path):
    web = tmp_path / "w.parquet"
    rows = [(f"A{i}", f"Name{i}", 0.25, 0.25, 1000 - i) for i in range(60)]
    rows.append(("B0", "Far Away", 0.75, 0.75, 5))
    write_web(web, rows)
    out = tmp_path / "index"
    n = build_label_tiles(str(web), out)
    assert n > 0
    z6 = json.loads((out / "labels/6/16/47.json").read_text())  # 0.25 -> tile 16, y-flip of 16 at z6 = 47
    assert len(z6["l"]) == 50                       # capped at z6
    assert z6["l"][0][0] == "Name0"                 # highest cited first
    z9 = json.loads((out / "labels/9/128/383.json").read_text())
    assert len(z9["l"]) == 60                       # cap 200 not hit
    far = json.loads((out / "labels/6/48/15.json").read_text())
    assert far["l"][0][0] == "Far Away"


def test_search_shards_cover_everyone(tmp_path):
    web = tmp_path / "w.parquet"
    write_web(web, [("A1", "Alice Zhang", 0.3, 0.3, 10),
                    ("A2", "alan turing", 0.4, 0.4, 99),
                    ("A3", "李明", 0.5, 0.5, 5)])
    out = tmp_path / "index"
    n = build_search_shards(str(web), out)
    assert n == 3
    al = json.loads((out / "search/al.json").read_text())
    assert [e[1] for e in al] == ["alan turing", "Alice Zhang"]  # cited desc
    other = json.loads((out / "search/_.json").read_text())
    assert other[0][1] == "李明"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_build_index.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write the implementation**

`pipeline/build_index.py`:

```python
"""Label/hit tiles (zooms 6-9) and prefix search shards. Zero backend."""
import json
import unicodedata
from collections import defaultdict
from pathlib import Path

import duckdb

from pipeline.config import apply_resource_limits, data_dir

LABEL_ZOOMS = {6: 50, 7: 50, 8: 200, 9: 200}   # zoom -> per-tile capacity


def normalize(name: str) -> str:
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = "".join(c if (c.isalnum() or c == " ") else " " for c in s)
    return " ".join(s.split())


def build_label_tiles(web: str, out_dir: Path) -> int:
    con = duckdb.connect()
    apply_resource_limits(con)
    written = 0
    for z, cap in LABEL_ZOOMS.items():
        ntiles = 1 << z
        rows = con.execute(
            f"""
            SELECT tx, ty_up, display_name, id, xw, yw, cited_by_count FROM (
                SELECT least({ntiles - 1}, CAST(floor(xw * {ntiles}) AS INT)) AS tx,
                       least({ntiles - 1}, CAST(floor(yw * {ntiles}) AS INT)) AS ty_up,
                       display_name, id, xw, yw, cited_by_count,
                       row_number() OVER (
                           PARTITION BY least({ntiles - 1}, CAST(floor(xw * {ntiles}) AS INT)),
                                        least({ntiles - 1}, CAST(floor(yw * {ntiles}) AS INT))
                           ORDER BY cited_by_count DESC, id
                       ) AS rn
                FROM read_parquet('{web}')
            ) WHERE rn <= {cap}
            ORDER BY tx, ty_up, cited_by_count DESC, id
            """
        ).fetchall()
        tiles = defaultdict(list)
        for tx, ty_up, name, aid, xw, yw, cited in rows:
            tiles[(tx, ty_up)].append(
                [name, aid, round(xw, 6), round(yw, 6), int(cited)]
            )
        for (tx, ty_up), entries in tiles.items():
            ty = (ntiles - 1) - ty_up               # XYZ y-flip
            p = out_dir / "labels" / str(z) / str(tx) / f"{ty}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"l": entries}, ensure_ascii=False))
            written += 1
    return written


def build_search_shards(web: str, out_dir: Path) -> int:
    con = duckdb.connect()
    apply_resource_limits(con)
    shards = defaultdict(list)
    rows = con.execute(
        f"""SELECT display_name, id, xw, yw, cited_by_count
            FROM read_parquet('{web}') ORDER BY cited_by_count DESC"""
    ).fetchall()
    for name, aid, xw, yw, cited in rows:
        norm = normalize(name or "")
        head = norm.replace(" ", "")[:2]
        key = head if len(head) == 2 and head.isascii() and head.isalpha() else "_"
        shards[key].append([norm, name, aid, round(xw, 6), round(yw, 6), int(cited)])
    sdir = out_dir / "search"
    sdir.mkdir(parents=True, exist_ok=True)
    total = 0
    for key, entries in shards.items():
        (sdir / f"{key}.json").write_text(json.dumps(entries, ensure_ascii=False))
        total += len(entries)
    return total


def add_parser(parser) -> None:
    parser.add_argument("--web", default=None)
    parser.add_argument("--out", default=None)


def run(args) -> int:
    web = args.web or str(data_dir() / "coords_web.parquet")
    out = Path(args.out) if args.out else data_dir() / "index"
    t = build_label_tiles(web, out)
    s = build_search_shards(web, out)
    print(f"{t:,} label tiles, {s:,} searchable names -> {out}")
    return 0
```

Shard-key rule (the tests define the contract): first two characters of the
space-stripped normalized name if both are ascii letters, else the `_` bucket.

Memory note: `build_search_shards` materializes 8.6M rows (~1.5GB as python
lists) - acceptable on the M4 Max; if it isn't, batch by first letter via 26
WHERE-prefix queries. Registry: add `index` -> `pipeline.build_index`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest -v`
Expected: all PASS

- [ ] **Step 5: Run on real data + integrate into preview**

```bash
MAPADEMIC_THREADS=8 MAPADEMIC_MEMORY_LIMIT=24GB .venv/bin/python -m pipeline index
du -sh data/index; ls data/index/search | wc -l
```

Then add to `web/dev.html` (append inside the `<script>` before the closing
tag) a minimal label overlay to verify labels visually:

```javascript
async function drawLabels() {
  const z = Math.round(map.getZoom());
  if (z < 6) return document.querySelectorAll(".lbl").forEach(e => e.remove());
  // naive: fetch the center tile's labels and render the top 10 as DOM pins
  const n = 1 << z, c = map.getCenter();
  const merc = maplibregl.MercatorCoordinate.fromLngLat(c);
  const tx = Math.floor(merc.x * n), ty = Math.floor(merc.y * n);
  const r = await fetch(`/data/index/labels/${z}/${tx}/${ty}.json`);
  document.querySelectorAll(".lbl").forEach(e => e.remove());
  if (!r.ok) return;
  (await r.json()).l.slice(0, 10).forEach(([name, id, xw, yw]) => {
    const ll = new maplibregl.MercatorCoordinate(xw, 1 - yw, 0).toLngLat();
    const el = document.createElement("div");
    el.className = "lbl"; el.textContent = name;
    el.style.cssText = "color:#fff;font:12px sans-serif;text-shadow:0 0 4px #000";
    new maplibregl.Marker({ element: el }).setLngLat(ll).addTo(map);
  });
}
map.on("moveend", drawLabels);
```

Verify: zoom past 6 anywhere dense -> names appear; names match dots.

- [ ] **Step 6: Commit**

```bash
git add pipeline/build_index.py pipeline/__main__.py tests/test_build_index.py web/dev.html
git commit -m "feat: index stage - label/hit tiles and search shards; preview labels"
```

---

## Out of scope for this plan

- Zoom 10 (decided after Task 5's measurement)
- The real site (search UI, info cards, path overlay, region-label rendering
  beyond the dev overlay), hosting, deploy, CORS PR - all Plan 4
- Region label curation (Eric edits `regions.json` whenever; it's data)
