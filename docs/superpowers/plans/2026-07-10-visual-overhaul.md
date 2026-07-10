# Visual Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the five visual complaints from Eric's 2026-07-10 review: outer ring, per-author color chaos, cramped clusters, missing hover/click, naive labels — plus faint coauthor edges at high zoom, all CPU-only (no GPU re-layout).

**Architecture:** All changes are post-layout: `webcoords` gains ring re-scatter + per-community expansion; `palette` switches from per-author field color to per-community color (majority-field hue family); `tiles` colors by community, splats top-cited authors as bigger dots at z8-9, and optionally bakes faint edge lines at z8-9 from a new `edgepx` precompute stage; `index` raises the z9 label cap; `web/dev.html` becomes a real viewer (multi-tile labels with collision culling, hover/click at high zoom, legend).

**Tech Stack:** Python 3.11+, DuckDB, numpy, PIL, pytest (existing patterns); vanilla JS + MapLibre GL 4.x for the viewer.

## Global Constraints

- Repo: `/Users/eric/Downloads/mapademic`, branch `data-pipeline`. Never touch `master`.
- Run tests with `.venv/bin/pytest`; run stages with `.venv/bin/python -m pipeline <stage>`.
- Stage-module pattern: each stage exposes `add_parser(parser)` and `run(args)`, pure core function importable for tests.
- DuckDB has no `asinh`: use `ln(v + sqrt(v*v + 1))`. Cast to DOUBLE before multiplying by PIX-scale constants.
- Deterministic outputs only (hash-based jitter, no `random`).
- Atomic file writes: write `.tmp` then `os.replace` (existing convention).
- Do NOT commit `data/` outputs (git-ignored).
- No em dashes in docs/commits. No Claude co-author line in commits.

## Design decisions (from Eric, 2026-07-10 conversation)

- Ring nodes (207,450; community size < 100 AND at the radial shell) are RE-SCATTERED as faint background dust, not dropped: Eric cares that all 8.59M authors stay searchable/clickable.
- Color by community (37 communities >= 1000 members cover 97.2%); hue family = community's majority field so field still means something; tail communities = dim grey dust. Field remains visible as TEXT (hover tooltip, legend).
- Post-hoc community-centroid expansion with `--spread` knob (default 1.35), QA-tuned.
- Dot size by citation: top-cited authors get 2-3 px splats at z8/z9.
- Edges: faint grey lines baked into z8/z9 raster tiles only (barely visible, zoom-gated). Client-side edge rendering is out (379M edges).
- Hover/click only at browser zoom >= 10; click opens `https://openalex.org/<id>`.

Diagnostic facts (verified 2026-07-10 against `data/coords_web.parquet`):
ring = ru in [0.93, 0.98] with empty gap at [0.85, 0.93] (~190 nodes); all ring
nodes have community size < 100; 29,210 small-community nodes are embedded in
the core (must NOT be re-scattered - radius separates them). Citation
percentiles: p50=338, p99=17.5k, p99.9=63k, p99.99=146k.

## File Structure

- Modify `pipeline/webcoords.py` - ring re-scatter + expansion (Tasks 1-2)
- Modify `pipeline/palette.py` - community palette (Task 3)
- Modify `pipeline/tiles.py` - community colors, splats, edge baking, legend.json (Tasks 4, 6)
- Create `pipeline/edge_px.py` - edge precompute stage (Task 5)
- Modify `pipeline/__main__.py` - register `edgepx` (Task 5)
- Modify `pipeline/build_index.py` - z9 cap (Task 7)
- Rewrite `web/dev.html` - viewer v2 (Task 8)
- Tests: `tests/test_webcoords.py`, `tests/test_palette.py`, `tests/test_tiles.py`, `tests/test_edge_px.py`, `tests/test_build_index.py`

---

### Task 1: webcoords ring re-scatter

**Files:**
- Modify: `pipeline/webcoords.py`
- Test: `tests/test_webcoords.py`

**Interfaces:**
- Produces: `build_webcoords(coords_path, out_path, spread=1.35, ring_comm_max=100) -> dict` (spread used in Task 2; this task adds `ring_comm_max` and returns stat key `ring_n`). Output parquet gains column `is_ring BOOLEAN`.

Ring rule: community size < `ring_comm_max` AND asinh-radius > 0.9 * max
asinh-radius. Ring nodes are re-scattered deterministically into the annulus
ru in [0.86, 0.98] with hash-based angle/radius; core normalization (`r2max`)
is computed over NON-ring nodes only so the galaxy reclaims the outer range.
`ring_comm_max=0` disables (old behavior, used by tests that pin compression
math).

- [ ] **Step 1: Update existing tests that assume no ring handling**

In `tests/test_webcoords.py`, the fixture rows all share `community=1` (size 7
< 100), so H1/H2 would be re-scattered and `test_radius_order_preserved_and_angles_kept`
would fail. Pass `ring_comm_max=0` in every existing `build_webcoords` call to
pin the pure-compression behavior. Example:

```python
stats = build_webcoords(coords_file, out, ring_comm_max=0)
```

- [ ] **Step 2: Write failing tests for the ring rule**

```python
def write_coords_comm(path, rows):
    """rows: list of (id, x, y, community)."""
    vals = ", ".join(
        f"('{i}', 'N {i}', CAST({x} AS FLOAT), CAST({y} AS FLOAT), {c}, 20,"
        f" 100, 'Inst', 'Biology')" for i, x, y, c in rows
    )
    duckdb.sql(
        f"COPY (SELECT * FROM (VALUES {vals}) t(id, display_name, x, y,"
        f" community, works_count, cited_by_count, institution, field))"
        f" TO '{path}' (FORMAT PARQUET)"
    )


@pytest.fixture
def ringed_file(tmp_path):
    # community 1: 150 core members (size >= 100 -> never ring)
    rows = [(f"C{k}", math.cos(k) * (1 + k % 7), math.sin(k) * (1 + k % 7), 1)
            for k in range(150)]
    # community 2: two members parked far out (the shell)
    rows += [("R1", 5000, 0, 2), ("R2", 0, 5000, 2)]
    # community 3: small but embedded near center -> must stay put
    rows += [("E1", 1.0, 1.0, 3), ("E2", -1.0, 1.0, 3)]
    p = tmp_path / "coords.parquet"
    write_coords_comm(p, rows)
    return str(p)


def test_ring_nodes_rescattered_into_annulus(ringed_file, tmp_path):
    out = str(tmp_path / "web.parquet")
    stats = build_webcoords(ringed_file, out, spread=1.0)
    assert stats["ring_n"] == 2
    r1, r2 = duckdb.sql(
        f"SELECT sqrt((xw-0.5)**2+(yw-0.5)**2)/0.48, is_ring FROM '{out}'"
        f" WHERE id IN ('R1','R2') ORDER BY id"
    ).fetchall()
    for ru, is_ring in (r1, r2):
        assert is_ring is True
        assert 0.86 - 1e-9 <= ru <= 0.98 + 1e-9


def test_embedded_small_community_not_rescattered(ringed_file, tmp_path):
    out = str(tmp_path / "web.parquet")
    build_webcoords(ringed_file, out, spread=1.0)
    rows = duckdb.sql(
        f"SELECT is_ring FROM '{out}' WHERE id IN ('E1','E2')").fetchall()
    assert all(r[0] is False for r in rows)


def test_core_normalization_excludes_ring(ringed_file, tmp_path):
    # with the shell excluded from r2max, the widest CORE node reaches ru ~ 1.0
    out = str(tmp_path / "web.parquet")
    build_webcoords(ringed_file, out, spread=1.0)
    (rumax,) = duckdb.sql(
        f"SELECT max(sqrt((xw-0.5)**2+(yw-0.5)**2))/0.48 FROM '{out}'"
        f" WHERE NOT is_ring").fetchone()
    assert rumax > 0.99


def test_rescatter_deterministic(ringed_file, tmp_path):
    a, b = str(tmp_path / "a.parquet"), str(tmp_path / "b.parquet")
    build_webcoords(ringed_file, a, spread=1.0)
    build_webcoords(ringed_file, b, spread=1.0)
    ra = duckdb.sql(f"SELECT id, xw, yw FROM '{a}' ORDER BY id").fetchall()
    rb = duckdb.sql(f"SELECT id, xw, yw FROM '{b}' ORDER BY id").fetchall()
    assert ra == rb
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/test_webcoords.py -v`
Expected: new tests FAIL (unexpected keyword `ring_comm_max` / missing `ring_n`).

- [ ] **Step 4: Implement**

Rewrite `build_webcoords` with temp tables (single connection):

```python
def build_webcoords(coords_path: str, out_path: str, spread: float = 1.35,
                    ring_comm_max: int = 100) -> dict:
    con = duckdb.connect()
    apply_resource_limits(con)
    cx, cy = con.execute(
        f"SELECT median(x), median(y) FROM read_parquet('{coords_path}')"
    ).fetchone()
    s = con.execute(
        f"""SELECT greatest(median(sqrt((x-{cx})*(x-{cx}) + (y-{cy})*(y-{cy}))), 1e-9)
            FROM read_parquet('{coords_path}')"""
    ).fetchone()[0]
    con.execute(
        f"""CREATE TEMP TABLE polar AS
            SELECT *,
                   sqrt((x-{cx})*(x-{cx}) + (y-{cy})*(y-{cy})) / {s} AS rs,
                   atan2(y-{cy}, x-{cx}) AS theta,
                   count(*) OVER (PARTITION BY community) AS comm_n
            FROM read_parquet('{coords_path}')"""
    )
    con.execute(
        """CREATE TEMP TABLE au AS
           SELECT *, ln(rs + sqrt(rs*rs + 1)) AS a FROM polar"""
    )
    amax = con.execute("SELECT max(a) FROM au").fetchone()[0] or 1e-9
    ring_pred = (f"(comm_n < {int(ring_comm_max)} AND a > 0.9 * {amax})"
                 if ring_comm_max > 0 else "FALSE")
    a2max = con.execute(
        f"SELECT max(a) FROM au WHERE NOT {ring_pred}"
    ).fetchone()[0] or 1e-9
    half = 0.5 - MARGIN
    # hash-based uniforms in [0,1): DuckDB hash() is a stable UBIGINT
    con.execute(
        f"""CREATE TEMP TABLE placed AS
            SELECT id, display_name, community, works_count, cited_by_count,
                   institution, field, {ring_pred} AS is_ring,
                   CASE WHEN {ring_pred}
                        THEN (0.86 + 0.12 * ((hash(id) % 100000) / 100000.0))
                        ELSE least(a / {a2max}, 1.0) END AS ru,
                   CASE WHEN {ring_pred}
                        THEN 2 * pi() * ((hash(id || '/t') % 100000) / 100000.0)
                        ELSE theta END AS th
            FROM au"""
    )
    n = con.execute(
        f"""COPY (
              SELECT id, display_name,
                     0.5 + {half} * ru * cos(th) AS xw,
                     0.5 + {half} * ru * sin(th) AS yw,
                     community, works_count, cited_by_count, institution,
                     field, is_ring
              FROM placed
            ) TO '{out_path}' (FORMAT PARQUET)"""
    ).fetchone()[0]
    ring_n = con.execute("SELECT count(*) FROM placed WHERE is_ring").fetchone()[0]
    stats = {"n": n, "cx": cx, "cy": cy, "s": s, "r2max": a2max,
             "ring_n": ring_n, "spread": spread}
    Path(out_path + ".meta.json").write_text(json.dumps(stats))
    return stats
```

Note: `ru CASE WHEN r=0` guard is preserved by `ln(0 + sqrt(0+1)) = 0`, so the
explicit r=0 branch is no longer needed. `spread` is threaded but unused until
Task 2. Update `add_parser`/`run`:

```python
def add_parser(parser) -> None:
    parser.add_argument("--coords", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--spread", type=float, default=1.35)
    parser.add_argument("--ring-comm-max", type=int, default=100)


def run(args) -> int:
    coords = args.coords or str(data_dir() / "coords.parquet")
    out = args.out or str(data_dir() / "coords_web.parquet")
    stats = build_webcoords(coords, out, spread=args.spread,
                            ring_comm_max=args.ring_comm_max)
    print(f"{stats['n']:,} points -> {out} (s={stats['s']:.1f}, "
          f"r2max={stats['r2max']:.3f}, ring={stats['ring_n']:,}, "
          f"spread={stats['spread']})")
    return 0
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest tests/test_webcoords.py tests/test_cli.py tests/test_end_to_end.py -v`
Expected: PASS (end-to-end/CLI tests may need `is_ring` added to expected schemas; fix those assertions, nothing else).

- [ ] **Step 6: Commit**

```bash
git add pipeline/webcoords.py tests/test_webcoords.py
git commit -m "feat(webcoords): re-scatter disconnected shell nodes as background dust"
```

---

### Task 2: webcoords community expansion

**Files:**
- Modify: `pipeline/webcoords.py`
- Test: `tests/test_webcoords.py`

**Interfaces:**
- Consumes: Task 1's `placed` temp table (columns `ru`, `th`, `is_ring`, `comm_n` retained through `au`).
- Produces: same `build_webcoords` signature; `spread` now takes effect. Communities with >= 1000 members get centroid displacement `(spread - 1) * (centroid - center)`; everything then renormalized so the widest core node sits at ru = 1.0. Ring dust is placed after renormalization (annulus unchanged).

- [ ] **Step 1: Write failing test**

```python
@pytest.fixture
def two_blob_file(tmp_path):
    # two 1000-member communities, blobs centered at (+-10, 0), radius <= 1
    rows = []
    for k in range(1000):
        dx, dy = math.cos(k) * (k % 10) / 10, math.sin(k) * (k % 10) / 10
        rows.append((f"L{k}", -10 + dx, dy, 1))
        rows.append((f"R{k}", 10 + dx, dy, 2))
    p = tmp_path / "coords.parquet"
    write_coords_comm(p, rows)
    return str(p)


def test_spread_moves_centroids_apart(two_blob_file, tmp_path):
    a, b = str(tmp_path / "a.parquet"), str(tmp_path / "b.parquet")
    build_webcoords(two_blob_file, a, spread=1.0)
    build_webcoords(two_blob_file, b, spread=1.6)
    def gap(p):
        return duckdb.sql(
            f"""SELECT abs(avg(xw) FILTER (community=1)
                       - avg(xw) FILTER (community=2)) FROM '{p}'"""
        ).fetchone()[0]
    def blob_radius(p):
        return duckdb.sql(
            f"""SELECT max(sqrt((xw - cxw)*(xw - cxw) + (yw - cyw)*(yw - cyw)))
                FROM (SELECT *, avg(xw) OVER (PARTITION BY community) cxw,
                              avg(yw) OVER (PARTITION BY community) cyw
                      FROM '{p}') WHERE community = 1"""
        ).fetchone()[0]
    # relative separation (centroid gap / blob size) must grow with spread
    assert gap(b) / blob_radius(b) > gap(a) / blob_radius(a) * 1.2
    # and everything still inside the margin box
    lo, hi = duckdb.sql(
        f"SELECT least(min(xw),min(yw)), greatest(max(xw),max(yw)) FROM '{b}'"
    ).fetchone()
    assert lo >= 0.02 - 1e-9 and hi <= 0.98 + 1e-9
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_webcoords.py::test_spread_moves_centroids_apart -v`
Expected: FAIL (spread currently a no-op, ratios equal).

- [ ] **Step 3: Implement**

In `build_webcoords`, replace the final COPY with an expansion pass over core
nodes (`spread=1.0` must be an exact no-op):

```python
    con.execute(
        f"""CREATE TEMP TABLE core AS
            SELECT *, ru * cos(th) AS ex, ru * sin(th) AS ey
            FROM placed WHERE NOT is_ring"""
    )
    con.execute(
        f"""CREATE TEMP TABLE shifted AS
            SELECT *,
                   ex + CASE WHEN comm_n >= 1000
                        THEN ({spread} - 1.0) * avg(ex) OVER (PARTITION BY community)
                        ELSE 0.0 END AS sx,
                   ey + CASE WHEN comm_n >= 1000
                        THEN ({spread} - 1.0) * avg(ey) OVER (PARTITION BY community)
                        ELSE 0.0 END AS sy
            FROM core"""
    )
    smax = con.execute(
        "SELECT greatest(max(sqrt(sx*sx + sy*sy)), 1e-9) FROM shifted"
    ).fetchone()[0]
    n = con.execute(
        f"""COPY (
              SELECT id, display_name,
                     0.5 + {half} * sx / {smax} AS xw,
                     0.5 + {half} * sy / {smax} AS yw,
                     community, works_count, cited_by_count, institution,
                     field, is_ring
              FROM shifted
              UNION ALL
              SELECT id, display_name,
                     0.5 + {half} * ru * cos(th) AS xw,
                     0.5 + {half} * ru * sin(th) AS yw,
                     community, works_count, cited_by_count, institution,
                     field, is_ring
              FROM placed WHERE is_ring
            ) TO '{out_path}' (FORMAT PARQUET)"""
    ).fetchone()[0]
```

`comm_n` must survive into `placed` (add it to the `placed` SELECT list in the
Task 1 code). Note `smax >= 1.0` whenever spread >= 1 (some node already at
ru=1 moves outward), so this only ever shrinks back into the box; at
spread=1.0, smax = 1.0 exactly and positions are unchanged.

- [ ] **Step 4: Run all webcoords tests**

Run: `.venv/bin/pytest tests/test_webcoords.py -v`
Expected: PASS, including Task 1 tests (they pass `spread=1.0`).

- [ ] **Step 5: Commit**

```bash
git add pipeline/webcoords.py tests/test_webcoords.py
git commit -m "feat(webcoords): post-hoc community expansion via --spread"
```

---

### Task 3: community palette

**Files:**
- Modify: `pipeline/palette.py`
- Test: `tests/test_palette.py`

**Interfaces:**
- Produces: `community_rgb(community: int, majority_field: str | None, members: int) -> tuple[float, float, float]` (pure, deterministic) and `build_community_palette(con, web_path: str, min_members: int = 1000) -> dict[int, tuple]` mapping EVERY community id in the parquet to an RGB. Keep `FIELD_HUES` and `splitmix64` exported (reused). Keep `field_community_rgb` for one release (tiles stops calling it in Task 4; delete then).

Color rule: `members >= min_members` -> hue = majority-field hue + 12deg *
j1 wobble, sat 0.85-1.0, light 0.44-0.56 (j2); `members < min_members` or
field None -> dim grey (light 0.30 + 0.06*j1, sat 0.03) so tail/dust recedes.

- [ ] **Step 1: Write failing tests**

```python
from pipeline.palette import build_community_palette, community_rgb


def test_big_community_uses_majority_field_hue_family():
    import colorsys
    r, g, b = community_rgb(7, "Medicine", 500000)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    assert s >= 0.80
    hue_deg = (h * 360.0) % 360.0
    dist = min(abs(hue_deg - 0.0), 360.0 - abs(hue_deg - 0.0))
    assert dist <= 12.0 + 1e-6            # Medicine anchor is 0.0
    assert community_rgb(7, "Medicine", 500000) == (r, g, b)  # deterministic


def test_small_or_fieldless_community_is_dim_grey():
    import colorsys
    for rgb in (community_rgb(3, "Medicine", 50), community_rgb(3, None, 10**6)):
        h, l, s = colorsys.rgb_to_hls(*rgb)
        assert s <= 0.05 and l <= 0.40


def test_same_field_communities_differ():
    assert community_rgb(1, "Medicine", 10**6) != community_rgb(2, "Medicine", 10**6)


def test_build_community_palette_covers_all(tmp_path):
    import duckdb
    p = str(tmp_path / "web.parquet")
    duckdb.sql(
        "COPY (SELECT 'A' || range::VARCHAR AS id, 'n' AS display_name,"
        " 0.5 AS xw, 0.5 AS yw, (range % 3)::INT AS community, 20 AS works_count,"
        " 10 AS cited_by_count, 'i' AS institution,"
        " CASE WHEN range % 3 = 0 THEN 'Medicine' ELSE 'Chemistry' END AS field,"
        " FALSE AS is_ring FROM range(3000))"
        f" TO '{p}' (FORMAT PARQUET)")
    pal = build_community_palette(duckdb.connect(), p, min_members=1000)
    assert set(pal) == {0, 1, 2}
    assert all(len(v) == 3 for v in pal.values())
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_palette.py -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement**

```python
def community_rgb(community: int, majority_field: str | None,
                  members: int, min_members: int = 1000) -> tuple[float, float, float]:
    """Community base RGB in [0,1]; deterministic. Big communities take their
    majority field's hue family; the tail is dim grey dust."""
    h = splitmix64(int(community))
    j1 = ((h & 0xFFFF) / 0xFFFF) * 2 - 1
    j2 = (((h >> 16) & 0xFFFF) / 0xFFFF) * 2 - 1
    if members < min_members or majority_field is None \
            or majority_field not in FIELD_HUES:
        return colorsys.hls_to_rgb(0.0, 0.30 + 0.06 * j1, 0.03)
    hue = (FIELD_HUES[majority_field] + 12.0 * j1) % 360.0
    sat = min(1.0, max(0.85, 0.925 + 0.075 * j2))
    light = min(0.56, max(0.44, 0.50 + 0.06 * j2))
    return colorsys.hls_to_rgb(hue / 360.0, light, sat)


def build_community_palette(con, web_path: str,
                            min_members: int = 1000) -> dict[int, tuple]:
    rows = con.execute(
        f"""WITH fc AS (SELECT community, field, count(*) c
                        FROM read_parquet('{web_path}') GROUP BY 1, 2),
             tot AS (SELECT community, sum(c) n FROM fc GROUP BY 1),
             maj AS (SELECT community, field,
                            row_number() OVER (PARTITION BY community
                                ORDER BY c DESC, field NULLS LAST) rn
                     FROM fc)
            SELECT t.community, m.field, t.n
            FROM tot t JOIN maj m ON m.community = t.community AND m.rn = 1"""
    ).fetchall()
    return {int(c): community_rgb(int(c), f, int(n), min_members)
            for c, f, n in rows}
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_palette.py -v`
Expected: PASS (old `field_community_rgb` tests untouched and passing).

- [ ] **Step 5: Commit**

```bash
git add pipeline/palette.py tests/test_palette.py
git commit -m "feat(palette): community-majority-field palette + grey dust tail"
```

---

### Task 4: tiles color by community, citation splats, legend.json

**Files:**
- Modify: `pipeline/tiles.py`
- Test: `tests/test_tiles.py`

**Interfaces:**
- Consumes: `build_community_palette(con, web_path, min_members)` and `community_rgb` from Task 3; webcoords parquet with `is_ring` (Task 1).
- Produces: `aggregate_z9` output schema becomes `(px, py, cnt, community)` (field dropped: color no longer needs it). `load_level9(pixels_path, palette: dict[int, tuple]) -> dict` (new required arg). New `load_splats(con, web_path, palette, min_cited) -> dict` with keys `px, py, rgb` (z9 pixel coords). New `write_legend(con, web_path, out_path, regions_path, min_members)` writing `data/index/legend.json`. `render_zoom(level, z, out_dir, bloom, splats=None)` draws splat discs at z>=8. New CLI flags: `--splat-min-cited` (default 60000), `--legend/--no-legend` behavior folded into run().

- [ ] **Step 1: Update existing tiles tests for the new schema/signature**

Existing tests construct pixels parquet with a `field` column and call
`load_level9(path)`. Update them: drop `field` from synthetic pixels files,
pass an explicit palette dict, e.g.

```python
pal = {1: (1.0, 0.0, 0.0), 2: (0.0, 1.0, 0.0)}
level = load_level9(str(pixels), pal)
```

- [ ] **Step 2: Write failing tests for the new behavior**

```python
def test_aggregate_schema_has_no_field(tmp_path):
    web = tmp_path / "web.parquet"
    duckdb.sql(
        "COPY (SELECT 'A1' id, 'n' display_name, 0.25 xw, 0.25 yw,"
        " 1 community, 20 works_count, 10 cited_by_count, 'i' institution,"
        " 'Medicine' field, FALSE is_ring)"
        f" TO '{web}' (FORMAT PARQUET)")
    out = tmp_path / "px.parquet"
    aggregate_z9(str(web), str(out), duckdb.connect())
    cols = [r[0] for r in duckdb.sql(f"DESCRIBE SELECT * FROM '{out}'").fetchall()]
    assert cols == ["px", "py", "cnt", "community"]


def test_splats_drawn_at_z9_only_above_threshold(tmp_path):
    con = duckdb.connect()
    web = tmp_path / "web.parquet"
    duckdb.sql(
        "COPY (SELECT * FROM (VALUES"
        " ('BIG', 'n', 0.5, 0.5, 1, 20, 100000, 'i', 'Medicine', FALSE),"
        " ('SML', 'n', 0.25, 0.25, 1, 20, 100, 'i', 'Medicine', FALSE))"
        " t(id, display_name, xw, yw, community, works_count, cited_by_count,"
        " institution, field, is_ring))"
        f" TO '{web}' (FORMAT PARQUET)")
    pal = {1: (1.0, 0.0, 0.0)}
    s = load_splats(con, str(web), pal, min_cited=60000)
    assert len(s["px"]) == 1                      # only BIG
    assert s["px"][0] == PIX // 2 and s["py"][0] == PIX // 2


def test_render_zoom_draws_splat_disc(tmp_path):
    level = {"px": np.array([128]), "py": np.array([128]),
             "cnt": np.array([1]), "rgb": np.array([[0.1, 0.1, 0.1]], np.float32)}
    splats = {"px": np.array([128 + 3]), "py": np.array([128]),
              "rgb": np.array([[0.0, 1.0, 0.0]], np.float32)}
    render_zoom(level, 9, tmp_path, bloom=False, splats=splats)
    # z9: level px are already zoom-9 pixel coords; tile 0/0 holds px<256
    img = np.asarray(Image.open(tmp_path / "9" / "0" / "255.png"))
    ys, xs = np.nonzero(img[:, :, 1] > 200)       # bright green disc pixels
    assert len(ys) >= 5                           # radius-2 disc, not 1 pixel


def test_legend_json(tmp_path):
    con = duckdb.connect()
    web = tmp_path / "web.parquet"
    duckdb.sql(
        "COPY (SELECT 'A' || range::VARCHAR id, 'n' display_name, 0.5 xw,"
        " 0.5 yw, 1 community, 20 works_count, 10 cited_by_count,"
        " 'i' institution, 'Medicine' field, FALSE is_ring FROM range(1500))"
        f" TO '{web}' (FORMAT PARQUET)")
    regions = tmp_path / "regions.json"
    regions.write_text(json.dumps([{"community": 1, "name": "Cardiology"}]))
    out = tmp_path / "legend.json"
    write_legend(con, str(web), str(out), str(regions), min_members=1000)
    entries = json.loads(out.read_text())
    assert entries == [{"community": 1, "name": "Cardiology",
                        "field": "Medicine", "members": 1500,
                        "color": entries[0]["color"]}]
    assert entries[0]["color"].startswith("#") and len(entries[0]["color"]) == 7
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/test_tiles.py -v`
Expected: FAIL (`load_splats`/`write_legend` undefined; schema mismatch).

- [ ] **Step 4: Implement**

In `aggregate_z9`, drop `field` everywhere (`GROUP BY px, py, community`,
rank `ORDER BY c DESC, community`). In `load_level9(pixels_path, palette)`:

```python
def load_level9(pixels_path: str, palette: dict[int, tuple]) -> dict:
    con = duckdb.connect()
    t = con.execute(
        f"SELECT px, py, cnt, community FROM read_parquet('{pixels_path}')"
        " ORDER BY py, px").fetchnumpy()
    comm = t["community"].astype(np.int64)
    uniq, inv = np.unique(comm, return_inverse=True)
    lut = np.array([palette.get(int(c), (0.35, 0.35, 0.35)) for c in uniq],
                   dtype=np.float32)
    return {"px": t["px"].astype(np.int64), "py": t["py"].astype(np.int64),
            "cnt": t["cnt"].astype(np.int64), "rgb": lut[inv]}
```

Splats and legend:

```python
SPLAT_RADIUS = {8: 1, 9: 2}   # disc radius in px at each zoom


def load_splats(con, web_path: str, palette: dict, min_cited: int) -> dict:
    t = con.execute(
        f"""SELECT least({PIX - 1}, CAST(floor(CAST(xw AS DOUBLE) * {PIX}) AS INT)) px,
                   least({PIX - 1}, CAST(floor(CAST(yw AS DOUBLE) * {PIX}) AS INT)) py,
                   community
            FROM read_parquet('{web_path}')
            WHERE cited_by_count >= {int(min_cited)} AND NOT is_ring"""
    ).fetchnumpy()
    comm = t["community"].astype(np.int64)
    rgb = np.array([palette.get(int(c), (0.6, 0.6, 0.6)) for c in comm],
                   dtype=np.float32)
    return {"px": t["px"].astype(np.int64), "py": t["py"].astype(np.int64),
            "rgb": rgb}


def write_legend(con, web_path: str, out_path: str, regions_path: str | None,
                 min_members: int = 1000) -> int:
    names = {}
    if regions_path and Path(regions_path).exists():
        names = {r["community"]: r["name"]
                 for r in json.loads(Path(regions_path).read_text())
                 if "community" in r}
    rows = con.execute(
        f"""WITH fc AS (SELECT community, field, count(*) c
                        FROM read_parquet('{web_path}') GROUP BY 1, 2),
             tot AS (SELECT community, sum(c) n FROM fc GROUP BY 1),
             maj AS (SELECT community, field, row_number() OVER (
                         PARTITION BY community ORDER BY c DESC, field NULLS LAST) rn
                     FROM fc)
            SELECT t.community, m.field, t.n FROM tot t
            JOIN maj m ON m.community = t.community AND m.rn = 1
            WHERE t.n >= {int(min_members)} ORDER BY t.n DESC"""
    ).fetchall()
    entries = []
    for c, f, n in rows:
        r, g, b = community_rgb(int(c), f, int(n), min_members)
        entries.append({
            "community": int(c), "name": names.get(int(c)),
            "field": f, "members": int(n),
            "color": "#%02x%02x%02x" % (int(r*255), int(g*255), int(b*255)),
        })
    tmp = str(out_path) + ".tmp"
    Path(tmp).write_text(json.dumps(entries, ensure_ascii=False, indent=1))
    os.replace(tmp, out_path)
    return len(entries)
```

`render_zoom(level, z, out_dir, bloom, splats=None)`: after the base
`img[iy, ix] = color[sel]` and BEFORE bloom, draw discs:

```python
        if splats is not None and z in SPLAT_RADIUS:
            shift = MAXZ - z
            spx, spy = splats["px"] >> shift, splats["py"] >> shift
            ssel = (spx // TILE == x) & (spy // TILE == yu)
            rad = SPLAT_RADIUS[z]
            for sx, sy_up, srgb in zip(spx[ssel] - x * TILE,
                                       spy[ssel] - yu * TILE,
                                       splats["rgb"][ssel]):
                sy = (TILE - 1) - sy_up
                for dy in range(-rad, rad + 1):
                    for dx in range(-rad, rad + 1):
                        if dx*dx + dy*dy > rad*rad:
                            continue
                        yy, xx = sy + dy, sx + dx
                        if 0 <= yy < TILE and 0 <= xx < TILE:
                            img[yy, xx] = np.maximum(img[yy, xx], srgb)
```

In `run()`: build palette once (`pal = build_community_palette(duckdb.connect(), web)`),
pass to `load_level9` and `load_splats` (only when rendering z>=8), call
`write_legend(con, web, str(data_dir() / "index" / "legend.json"), str(data_dir() / "regions.json"))`
(mkdir index dir first). Add `--splat-min-cited` (default 60000, ~p99.9).
`pixels_z9.parquet` schema changed: bump staleness by checking columns - if the
cached file still has a `field` column, re-aggregate (add this to the stale
check).

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest tests/test_tiles.py tests/test_end_to_end.py -v`
Expected: PASS (fix end-to-end expectations for the new schema/legend as needed).

- [ ] **Step 6: Commit**

```bash
git add pipeline/tiles.py tests/test_tiles.py
git commit -m "feat(tiles): community colors, citation splats at z8-9, legend.json"
```

---

### Task 5: edgepx stage - precompute drawable edges

**Files:**
- Create: `pipeline/edge_px.py`
- Modify: `pipeline/__main__.py` (register `edgepx` in STAGES)
- Test: `tests/test_edge_px.py`

**Interfaces:**
- Consumes: `data/edges.parquet` (`src VARCHAR, dst VARCHAR, weight`), webcoords parquet.
- Produces: `data/edges_px.parquet` with schema `(x0 INT, y0 INT, x1 INT, y1 INT)` in z9 pixel space, filtered to `weight >= min_weight AND length <= max_len_px AND` both endpoints non-ring. `build_edge_px(edges_path, web_path, out_path, min_weight=2, max_len_px=768) -> int`.

- [ ] **Step 1: Write failing test**

```python
import duckdb

from pipeline.edge_px import build_edge_px
from pipeline.tiles import PIX


def _write_inputs(tmp_path):
    web = tmp_path / "web.parquet"
    duckdb.sql(
        "COPY (SELECT * FROM (VALUES"
        " ('A', 0.5, 0.5, FALSE), ('B', 0.5005, 0.5, FALSE),"   # short edge
        " ('C', 0.9, 0.9, FALSE),"                              # far away
        " ('D', 0.1, 0.1, TRUE))"                               # ring dust
        " t(id, xw, yw, is_ring)) TO '" + str(web) + "' (FORMAT PARQUET)")
    edges = tmp_path / "edges.parquet"
    duckdb.sql(
        "COPY (SELECT * FROM (VALUES"
        " ('A', 'B', 3),"    # kept
        " ('A', 'C', 3),"    # too long
        " ('A', 'B', 1),"    # weight below min  (weights are per-row here)
        " ('A', 'D', 9))"    # ring endpoint
        " t(src, dst, weight)) TO '" + str(edges) + "' (FORMAT PARQUET)")
    return str(edges), str(web)


def test_filters_and_pixel_coords(tmp_path):
    edges, web = _write_inputs(tmp_path)
    out = str(tmp_path / "edges_px.parquet")
    n = build_edge_px(edges, web, out, min_weight=2, max_len_px=768)
    rows = duckdb.sql(f"SELECT * FROM '{out}'").fetchall()
    assert n == len(rows) == 1
    x0, y0, x1, y1 = rows[0]
    assert (x0, y0) == (PIX // 2, PIX // 2)
    assert abs(x1 - x0) <= 768 and y1 == y0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_edge_px.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

```python
"""Precompute drawable coauthor edges in z9 pixel space (for tile baking)."""
from pathlib import Path

import duckdb

from pipeline.config import apply_resource_limits, data_dir
from pipeline.tiles import PIX


def build_edge_px(edges_path: str, web_path: str, out_path: str,
                  min_weight: int = 2, max_len_px: int = 768) -> int:
    con = duckdb.connect()
    apply_resource_limits(con)
    n = con.execute(
        f"""COPY (
              SELECT p0.px AS x0, p0.py AS y0, p1.px AS x1, p1.py AS y1
              FROM read_parquet('{edges_path}') e
              JOIN (SELECT id,
                           least({PIX-1}, CAST(floor(CAST(xw AS DOUBLE)*{PIX}) AS INT)) px,
                           least({PIX-1}, CAST(floor(CAST(yw AS DOUBLE)*{PIX}) AS INT)) py
                    FROM read_parquet('{web_path}') WHERE NOT is_ring) p0
                ON p0.id = e.src
              JOIN (SELECT id,
                           least({PIX-1}, CAST(floor(CAST(xw AS DOUBLE)*{PIX}) AS INT)) px,
                           least({PIX-1}, CAST(floor(CAST(yw AS DOUBLE)*{PIX}) AS INT)) py
                    FROM read_parquet('{web_path}') WHERE NOT is_ring) p1
                ON p1.id = e.dst
              WHERE e.weight >= {int(min_weight)}
                AND sqrt((p0.px - p1.px)**2 + (p0.py - p1.py)**2) <= {int(max_len_px)}
            ) TO '{out_path}' (FORMAT PARQUET)"""
    ).fetchone()[0]
    return n


def add_parser(parser) -> None:
    parser.add_argument("--edges", default=None)
    parser.add_argument("--web", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--min-weight", type=int, default=2)
    parser.add_argument("--max-len-px", type=int, default=768)


def run(args) -> int:
    edges = args.edges or str(data_dir() / "edges.parquet")
    web = args.web or str(data_dir() / "coords_web.parquet")
    out = args.out or str(data_dir() / "edges_px.parquet")
    n = build_edge_px(edges, web, out, args.min_weight, args.max_len_px)
    print(f"{n:,} drawable edges -> {out}")
    return 0
```

Register in `pipeline/__main__.py` STAGES: `"edgepx": edge_px` (import it),
after `"webcoords"`.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_edge_px.py tests/test_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Measure real volume (knob sanity, no commit gate)**

Run against real data and record counts in the commit message body:

```bash
.venv/bin/python -c "
import duckdb
con = duckdb.connect()
print(con.execute(\"\"\"SELECT min_w, count(*) FROM (
  SELECT CASE WHEN weight >= 3 THEN 3 WHEN weight >= 2 THEN 2 ELSE 1 END min_w
  FROM read_parquet('data/edges.parquet')) GROUP BY 1 ORDER BY 1\"\"\").fetchall())"
```

If `weight >= 2` still leaves > ~60M edges, raise the default `min_weight` to 3
and note it. Target: drawable set that renders in < ~2 h on the Mac.

- [ ] **Step 6: Commit**

```bash
git add pipeline/edge_px.py pipeline/__main__.py tests/test_edge_px.py
git commit -m "feat(edgepx): precompute drawable coauthor edges in z9 pixel space"
```

---

### Task 6: tiles - bake faint edges at z8-9

**Files:**
- Modify: `pipeline/tiles.py`
- Test: `tests/test_tiles.py`

**Interfaces:**
- Consumes: `data/edges_px.parquet` from Task 5.
- Produces: `render_zoom(level, z, out_dir, bloom, splats=None, edges=None)` where `edges` is a dict `{x0, y0, x1, y1}` of int64 arrays in z9 px space; drawn only when `z >= 8`, additively, `EDGE_ALPHA = 0.045`, colour `EDGE_RGB = (0.45, 0.50, 0.60)` (cool grey). `load_edges(path) -> dict`. CLI flag `--edges [PATH]` (default off; `--edges` alone uses `data/edges_px.parquet`).

Edges are drawn BEFORE node pixels and splats so points stay crisp on top.
Rasterization: sample each edge every ~1 px with `np.linspace`, accumulate
into a float buffer, clip.

- [ ] **Step 1: Write failing test**

```python
def test_render_zoom_bakes_faint_edges_at_z9(tmp_path):
    level = {"px": np.array([10]), "py": np.array([10]),
             "cnt": np.array([1]), "rgb": np.array([[1.0, 0, 0]], np.float32)}
    edges = {"x0": np.array([10]), "y0": np.array([10]),
             "x1": np.array([110]), "y1": np.array([10])}
    render_zoom(level, 9, tmp_path, bloom=False, edges=edges)
    img = np.asarray(Image.open(tmp_path / "9" / "0" / "255.png")).astype(int)
    mid = img[(TILE - 1) - 10, 60]              # a pixel along the edge
    assert 3 <= mid.max() <= 40                 # faint but present
    assert img[(TILE - 1) - 10, 10, 0] > 200    # node still bright red


def test_no_edges_below_z8(tmp_path):
    level = {"px": np.array([0]), "py": np.array([0]),
             "cnt": np.array([1]), "rgb": np.array([[1.0, 0, 0]], np.float32)}
    edges = {"x0": np.array([0]), "y0": np.array([0]),
             "x1": np.array([200 << 2]), "y1": np.array([0])}
    render_zoom(level, 7, tmp_path, bloom=False, edges=edges)
    ntiles = 1 << 7
    img = np.asarray(Image.open(tmp_path / "7" / "0" / f"{ntiles - 1}.png")).astype(int)
    assert img[TILE - 1, 30].max() == 0         # nothing drawn along the line
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_tiles.py -k edges -v`
Expected: FAIL (unexpected keyword `edges`).

- [ ] **Step 3: Implement**

```python
EDGE_ALPHA = 0.045
EDGE_RGB = np.array([0.45, 0.50, 0.60], dtype=np.float32)
EDGE_MINZ = 8


def load_edges(path: str) -> dict:
    t = duckdb.connect().execute(
        f"SELECT x0, y0, x1, y1 FROM read_parquet('{path}')").fetchnumpy()
    return {k: t[k].astype(np.int64) for k in ("x0", "y0", "x1", "y1")}
```

In `render_zoom`, before `img[iy, ix] = color[sel]`:

```python
        if edges is not None and z >= EDGE_MINZ:
            shift = MAXZ - z
            ex0, ey0 = edges["x0"] >> shift, edges["y0"] >> shift
            ex1, ey1 = edges["x1"] >> shift, edges["y1"] >> shift
            # edges whose bbox intersects this tile
            tx0, ty0 = x * TILE, yu * TILE
            esel = ((np.minimum(ex0, ex1) < tx0 + TILE)
                    & (np.maximum(ex0, ex1) >= tx0)
                    & (np.minimum(ey0, ey1) < ty0 + TILE)
                    & (np.maximum(ey0, ey1) >= ty0))
            acc = np.zeros((TILE, TILE), dtype=np.float32)
            for a0, b0, a1, b1 in zip(ex0[esel], ey0[esel], ex1[esel], ey1[esel]):
                n = max(2, int(max(abs(a1 - a0), abs(b1 - b0))) + 1)
                xs = np.linspace(a0, a1, n).round().astype(np.int64) - tx0
                ys_up = np.linspace(b0, b1, n).round().astype(np.int64) - ty0
                keep = (xs >= 0) & (xs < TILE) & (ys_up >= 0) & (ys_up < TILE)
                np.add.at(acc, ((TILE - 1) - ys_up[keep], xs[keep]), EDGE_ALPHA)
            img += np.clip(acc, 0, 0.25)[:, :, None] * EDGE_RGB
```

(Cap per-pixel accumulation at 0.25 so dense bundles glow gently instead of
whiting out.) Wire into `run()`:

```python
    parser.add_argument("--edges", nargs="?", const="__default__", default=None)
    # in run():
    edges = None
    if args.edges:
        epath = (str(data_dir() / "edges_px.parquet")
                 if args.edges == "__default__" else args.edges)
        edges = load_edges(epath)
        print(f"baking {len(edges['x0']):,} edges into z>={EDGE_MINZ} tiles", flush=True)
    # pass edges=edges to render_zoom
```

Per-tile bbox filtering over tens of millions of edges re-scans the arrays for
every tile (512*512 tiles at z9): pre-sort once by `x0 // TILE` and slice, OR
accept the O(tiles * edges) cost only for tiles that exist. Implement the
simple pre-bucketing: group edge indices by z-level tile of their midpoint
plus neighbours within `max_len_px // TILE + 1` tiles; a dict
`{(tx, ty): np.ndarray}` built once per zoom before the tile loop. Keep it
private (`_bucket_edges(edges, z)`), no need to over-engineer beyond that.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_tiles.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/tiles.py tests/test_tiles.py
git commit -m "feat(tiles): bake faint coauthor edges into z8-9 tiles"
```

---

### Task 7: index z9 hit capacity

**Files:**
- Modify: `pipeline/build_index.py`
- Test: `tests/test_build_index.py`

**Interfaces:**
- Produces: `LABEL_ZOOMS = {6: 50, 7: 50, 8: 200, 9: 1000}` and ring dust excluded from label tiles (`WHERE NOT is_ring` - dust must not win label slots; it stays in search shards so every author remains findable).

- [ ] **Step 1: Measure density first**

```bash
.venv/bin/python -c "
import duckdb
print(duckdb.sql(\"\"\"SELECT quantile_cont(c, [0.5, 0.9, 0.99, 1.0]) FROM (
 SELECT count(*) c FROM (
  SELECT CAST(floor(xw * 512) AS INT) tx, CAST(floor(yw * 512) AS INT) ty
  FROM read_parquet('data/coords_web.parquet')) GROUP BY tx, ty)\"\"\").fetchall())"
```

If p99 tile occupancy is far above 1000, note the coverage gap in the commit
body (cap stays 1000; hosting size matters).

- [ ] **Step 2: Write failing test**

```python
def test_z9_cap_and_ring_excluded_from_labels(tmp_path):
    web = tmp_path / "web.parquet"
    duckdb.sql(
        "COPY (SELECT 'A' || range::VARCHAR id, 'name' || range::VARCHAR display_name,"
        " 0.5 xw, 0.5 yw, 1 community, 20 works_count,"
        " (2000000 - range)::INT cited_by_count, 'i' institution, 'Medicine' field,"
        " (range = 0) is_ring FROM range(1200))"
        f" TO '{web}' (FORMAT PARQUET)")
    n = build_label_tiles(str(web), tmp_path / "index")
    z9 = json.loads((tmp_path / "index" / "labels" / "9" / "256" / "255.json").read_text())
    assert len(z9["l"]) == 1000
    names = {e[0] for e in z9["l"]}
    assert "name0" not in names          # the ring node (highest cited) is excluded
```

(Adjust the expected tile path to where xw=yw=0.5 lands: tx=256, ty_up=256 ->
ty = 511 - 256 = 255.)

- [ ] **Step 3: Run to verify failure, implement, re-run**

Change `LABEL_ZOOMS` z9 value to 1000 and add `WHERE NOT is_ring` to the
`build_label_tiles` inner SELECT (`FROM read_parquet('{web}') WHERE NOT is_ring`).
Search shards: NO filter (coverage promise). Existing tests write webcoords
fixtures without `is_ring`; add the column to their fixture writers
(`FALSE AS is_ring`).

Run: `.venv/bin/pytest tests/test_build_index.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add pipeline/build_index.py tests/test_build_index.py
git commit -m "feat(index): z9 hit capacity 1000, ring dust excluded from labels"
```

---

### Task 8: viewer v2

**Files:**
- Rewrite: `web/dev.html`

**Interfaces:**
- Consumes: raster tiles `data/tiles/{z}/{x}/{y}.png`, label tiles `data/index/labels/{z}/{x}/{y}.json` (entries `[name, id, xw, yw, cited]`), `data/index/legend.json` (Task 4), `data/regions.json`.
- Produces: a single-file viewer with: labels from ALL visible tiles, citation-ranked with greedy screen-space collision culling and citation-scaled font size; region names shown at z2-6 from regions.json; hover tooltip + cursor change at browser zoom >= 10 (nearest hit within 14 px from z9 label-tile data, shows name/citations); click opens `https://openalex.org/{id}` in a new tab (zoom >= 10 only); collapsible legend showing top-25 legend.json entries (color swatch, region name or majority field, member count). No build step, vanilla JS only.

There is no pytest here; verification is Step 3 (manual/scripted browser check).

- [ ] **Step 1: Implement `web/dev.html`**

Complete file:

```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>mapademic dev preview</title>
  <script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
  <link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet">
  <style>
    html, body, #map { margin: 0; height: 100%; background: #000; }
    .lbl { color: #fff; font-family: system-ui, sans-serif;
           text-shadow: 0 0 4px #000, 0 0 8px #000; pointer-events: none;
           white-space: nowrap; }
    .region { color: #9ab; font: 600 15px system-ui; letter-spacing: .08em;
              text-transform: uppercase; text-shadow: 0 0 6px #000;
              pointer-events: none; opacity: .85; }
    #tip { position: fixed; display: none; background: #111c; color: #eee;
           font: 12px system-ui; padding: 4px 8px; border-radius: 4px;
           pointer-events: none; z-index: 10; }
    #legend { position: fixed; right: 10px; top: 10px; background: #0b0b0bd9;
              color: #ddd; font: 12px system-ui; border-radius: 6px;
              max-height: 70vh; overflow-y: auto; z-index: 9; width: 230px; }
    #legend summary { cursor: pointer; padding: 6px 10px; font-weight: 600; }
    #legend .row { display: flex; gap: 6px; align-items: center; padding: 2px 10px; }
    #legend .sw { width: 10px; height: 10px; border-radius: 2px; flex: none; }
    #legend .n { color: #888; margin-left: auto; }
  </style>
</head>
<body>
<div id="map"></div>
<div id="tip"></div>
<details id="legend" open><summary>Legend</summary><div id="legend-rows"></div></details>
<script>
const HIT_ZOOM = 10;                 // hover/click active at or past this zoom
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

const tileCache = new Map();         // "z/x/y" -> {l:[...]} or null
async function fetchTile(z, x, y) {
  const k = `${z}/${x}/${y}`;
  if (!tileCache.has(k)) {
    tileCache.set(k, fetch(`/data/index/labels/${k}.json`)
      .then(r => r.ok ? r.json() : null).catch(() => null));
    if (tileCache.size > 600) tileCache.delete(tileCache.keys().next().value);
  }
  return tileCache.get(k);
}
function visibleTiles(lz) {
  const n = 1 << lz, b = map.getBounds();
  const m0 = maplibregl.MercatorCoordinate.fromLngLat(b.getNorthWest());
  const m1 = maplibregl.MercatorCoordinate.fromLngLat(b.getSouthEast());
  const tiles = [];
  const x0 = Math.max(0, Math.floor(m0.x * n)), x1 = Math.min(n - 1, Math.floor(m1.x * n));
  const y0 = Math.max(0, Math.floor(m0.y * n)), y1 = Math.min(n - 1, Math.floor(m1.y * n));
  for (let x = x0; x <= x1; x++) for (let y = y0; y <= y1; y++) tiles.push([x, y]);
  return tiles;
}
const toLngLat = (xw, yw) => new maplibregl.MercatorCoordinate(xw, 1 - yw, 0).toLngLat();

let markers = [];
async function drawLabels() {
  const z = map.getZoom();
  markers.forEach(m => m.remove()); markers = [];
  if (z < 5.5) return;
  const lz = Math.max(6, Math.min(9, Math.round(z)));
  const lists = await Promise.all(visibleTiles(lz).map(([x, y]) => fetchTile(lz, x, y)));
  const entries = lists.filter(Boolean).flatMap(t => t.l);
  entries.sort((a, b) => b[4] - a[4]);          // by citations desc
  const placed = [];                            // screen-space rects
  const cap = 60;
  for (const [name, id, xw, yw, cited] of entries) {
    if (markers.length >= cap) break;
    const pt = map.project(toLngLat(xw, yw));
    if (pt.x < -50 || pt.y < -20 || pt.x > innerWidth + 50 || pt.y > innerHeight + 20) continue;
    const size = 11 + Math.min(6, Math.log10(1 + cited));       // 11-17 px
    const w = name.length * size * 0.58, h = size + 6;
    if (placed.some(r => Math.abs(r.x - pt.x) * 2 < r.w + w
                      && Math.abs(r.y - pt.y) * 2 < r.h + h)) continue;
    placed.push({ x: pt.x, y: pt.y, w, h });
    const el = document.createElement("div");
    el.className = "lbl"; el.textContent = name;
    el.style.fontSize = size + "px";
    markers.push(new maplibregl.Marker({ element: el })
      .setLngLat(toLngLat(xw, yw)).addTo(map));
  }
}

let regions = [];
fetch("/data/regions.json").then(r => r.json()).then(j => { regions = j; drawRegions(); });
let regionMarkers = [];
function drawRegions() {
  regionMarkers.forEach(m => m.remove()); regionMarkers = [];
  const z = map.getZoom();
  for (const r of regions) {
    if (z < r.zmin || z > r.zmax + 1) continue;
    const el = document.createElement("div");
    el.className = "region"; el.textContent = r.name;
    regionMarkers.push(new maplibregl.Marker({ element: el })
      .setLngLat(toLngLat(r.xw, r.yw)).addTo(map));
  }
}

fetch("/data/index/legend.json").then(r => r.ok ? r.json() : []).then(entries => {
  const box = document.getElementById("legend-rows");
  for (const e of entries.slice(0, 25)) {
    const row = document.createElement("div"); row.className = "row";
    row.innerHTML = `<span class="sw" style="background:${e.color}"></span>` +
      `<span>${e.name || e.field || "Community " + e.community}</span>` +
      `<span class="n">${(e.members / 1e6).toFixed(1)}M</span>`;
    box.appendChild(row);
  }
});

const tip = document.getElementById("tip");
async function hitTest(pt) {
  if (map.getZoom() < HIT_ZOOM) return null;
  const merc = maplibregl.MercatorCoordinate.fromLngLat(map.unproject(pt));
  const xw = merc.x, yw = 1 - merc.y, n = 1 << 9;
  const t = await fetchTile(9, Math.floor(xw * n), Math.floor(yw * n));
  if (!t) return null;
  let best = null, bestD = 14;                  // px radius
  for (const e of t.l) {
    const p = map.project(toLngLat(e[2], e[3]));
    const d = Math.hypot(p.x - pt.x, p.y - pt.y);
    if (d < bestD) { bestD = d; best = e; }
  }
  return best;
}
map.on("mousemove", async ev => {
  const hit = await hitTest(ev.point);
  map.getCanvas().style.cursor = hit ? "pointer" : "";
  if (hit) {
    tip.style.display = "block";
    tip.style.left = (ev.point.x + 14) + "px";
    tip.style.top = (ev.point.y + 14) + "px";
    tip.textContent = `${hit[0]} - ${hit[4].toLocaleString()} citations`;
  } else tip.style.display = "none";
});
map.on("click", async ev => {
  const hit = await hitTest(ev.point);
  if (hit) window.open("https://openalex.org/" + hit[1], "_blank", "noopener");
});
map.on("moveend", () => { drawLabels(); drawRegions(); });
map.on("zoomend", () => { drawLabels(); drawRegions(); });
</script>
</body>
</html>
```

- [ ] **Step 2: Serve and verify**

Run: `bash scripts/preview.sh` then open `http://localhost:8123/web/dev.html`.
Check (browser or chrome-devtools MCP): labels appear over the whole viewport
(not just center), grow in number while zooming, no overlaps; region names at
z2-6; legend panel lists colored rows; at zoom >= 10 hovering a dot shows the
tooltip and clicking opens openalex.org; below zoom 10 clicks do nothing.

- [ ] **Step 3: Commit**

```bash
git add web/dev.html
git commit -m "feat(web): viewer v2 - full-viewport labels, hover/click, legend"
```

---

### Task 9: full CPU re-run + QA composites

**Files:**
- Create: `data/qa/overhaul-z3.png`, `data/qa/overhaul-z5.png`, `data/qa/overhaul-edges-z9.png` (data outputs, not committed)

No code; execution + judgment. Budget note: full z0-9 tile render took hours
previously; QA gate renders z0-5 only (~1.4k tiles) plus targeted z8/9 crops.

- [ ] **Step 1: Re-run webcoords + edgepx**

```bash
cd /Users/eric/Downloads/mapademic
MAPADEMIC_THREADS=8 MAPADEMIC_MEMORY_LIMIT=24GB .venv/bin/python -m pipeline webcoords --spread 1.35
MAPADEMIC_THREADS=8 MAPADEMIC_MEMORY_LIMIT=24GB .venv/bin/python -m pipeline edgepx
```

Expected: webcoords prints `ring=207,450` (about); edgepx prints the drawable
edge count.

- [ ] **Step 2: Render QA zooms into a SEPARATE dir (do not clobber prod tiles)**

```bash
MAPADEMIC_THREADS=8 MAPADEMIC_MEMORY_LIMIT=24GB .venv/bin/python -m pipeline tiles \
  --zooms 0-5 --out data/tiles_qa --force-aggregate
```

Then z8-9 for a dense crop + Hinton area, with edges (render_zoom skips
existing files, so restrict by rendering into the same qa dir; the z8/9 full
set is large - accept the time or temporarily bound: it is fine to run z8-9
fully overnight if needed; otherwise present edges via a direct
render-to-array snippet over the crop region).

- [ ] **Step 3: Build composites**

```bash
.venv/bin/python - <<'EOF'
from PIL import Image
for z, name in ((3, "overhaul-z3"), (5, "overhaul-z5")):
    n = 1 << z
    out = Image.new("RGB", (256 * n, 256 * n))
    for x in range(n):
        for y in range(n):
            try:
                out.paste(Image.open(f"data/tiles_qa/{z}/{x}/{y}.png"), (x*256, y*256))
            except FileNotFoundError:
                pass
    out.save(f"data/qa/{name}.png")
    print(name, "written")
EOF
```

- [ ] **Step 4: QA gate (Eric)**

Present composites + a viewer screenshot to Eric: ring gone? clusters spaced?
colors coherent? splats visible? edges tasteful? Tune `--spread`,
`EDGE_ALPHA`, `--splat-min-cited` per feedback and re-render QA zooms
(cheap). Full z0-9 + index re-render happens only after approval, in one
batch (tiles are stale as a set: delete `data/tiles/` and `data/index/`
first, then `tiles --zooms 0-9 --edges`, `regions` (required - xw/yw stale after webcoords --spread),
`index`).

- [ ] **Step 5: Update AGENTS.md progress + commit docs**

```bash
git add docs/superpowers/plans/2026-07-10-visual-overhaul.md AGENTS.md
git commit -m "docs: visual overhaul plan + progress"
```

---

## Self-Review

- Spec coverage: ring (T1), spacing (T2), community color + field-as-text (T3/T4/T8 tooltip+legend), citation dot size (T4 splats), edges (T5/T6), labels (T7/T8), hover/click (T8), QA (T9). All five complaints + two additions covered.
- Placeholders: none; all code inline.
- Type consistency: `build_webcoords(coords_path, out_path, spread, ring_comm_max)` used consistently; `load_level9(pixels_path, palette)`; `render_zoom(level, z, out_dir, bloom, splats, edges)`; parquet schema `(px, py, cnt, community)` matches between T4 tests and code; `is_ring` column consumed by T4/T5/T7 matches T1's output.
