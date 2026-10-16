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
    stats = build_webcoords(coords_file, out, ring_comm_max=0)
    assert stats["n"] == 7
    lo, hi = duckdb.sql(
        f"SELECT least(min(xw), min(yw)), greatest(max(xw), max(yw)) FROM '{out}'"
    ).fetchone()
    assert lo >= 0.02 - 1e-9 and hi <= 0.98 + 1e-9
    schema = {
        name: typ
        for name, typ, *_ in duckdb.sql(
            f"DESCRIBE SELECT xw, yw FROM '{out}'"
        ).fetchall()
    }
    assert schema == {"xw": "DOUBLE", "yw": "DOUBLE"}


def test_radius_order_preserved_and_angles_kept(coords_file, tmp_path):
    out = str(tmp_path / "coords_web.parquet")
    build_webcoords(coords_file, out, ring_comm_max=0)
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
    stats = build_webcoords(coords_file, out, ring_comm_max=0)
    meta = json.loads((tmp_path / "coords_web.parquet.meta.json").read_text())
    assert meta == stats
    assert set(stats) == {"n", "cx", "cy", "s", "r2max", "ring_n", "spread", "tighten", "fill",
                          "split_communities"}


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
        assert 0.60 - 1e-9 <= ru <= 0.98 + 1e-9


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


@pytest.fixture
def blob_and_embedded_small_comm_file(tmp_path):
    # community 1: 1000-member blob centered at (+10, 0) -> qualifies for spread
    # community 2: 200-member blob near the center, well within the core
    # radius; also size >= default ring_comm_max=100 so the ring rule can't
    # touch it either way
    rows = []
    for k in range(1000):
        dx, dy = math.cos(k) * (k % 10) / 10, math.sin(k) * (k % 10) / 10
        rows.append((f"B{k}", 10 + dx, dy, 1))
    for k in range(200):
        dx, dy = math.cos(k) * (k % 5) / 20, math.sin(k) * (k % 5) / 20
        rows.append((f"S{k}", 1.0 + dx, 1.0 + dy, 2))
    p = tmp_path / "coords.parquet"
    write_coords_comm(p, rows)
    return str(p)


def test_small_core_community_not_displaced(blob_and_embedded_small_comm_file, tmp_path):
    a, b = str(tmp_path / "a.parquet"), str(tmp_path / "b.parquet")
    build_webcoords(blob_and_embedded_small_comm_file, a, spread=1.0)
    build_webcoords(blob_and_embedded_small_comm_file, b, spread=1.6)

    def centroid_ratio_and_theta(p, community, maxr):
        cxw, cyw = duckdb.sql(
            f"SELECT avg(xw), avg(yw) FROM '{p}' WHERE community = {community}"
        ).fetchone()
        r = math.hypot(cxw - 0.5, cyw - 0.5)
        theta = math.atan2(cyw - 0.5, cxw - 0.5)
        return r / maxr, theta

    def maxr(p):
        (v,) = duckdb.sql(
            f"""SELECT max(sqrt((xw-0.5)*(xw-0.5)+(yw-0.5)*(yw-0.5))) FROM '{p}'
                WHERE NOT is_ring"""
        ).fetchone()
        return v

    small_ratio_a, small_theta_a = centroid_ratio_and_theta(a, 2, maxr(a))
    small_ratio_b, small_theta_b = centroid_ratio_and_theta(b, 2, maxr(b))
    big_ratio_a, _ = centroid_ratio_and_theta(a, 1, maxr(a))
    big_ratio_b, _ = centroid_ratio_and_theta(b, 1, maxr(b))

    # the small community isn't shifted - only globally rescaled - so its
    # centroid direction from center is unchanged
    assert small_theta_b == pytest.approx(small_theta_a, abs=1e-6)
    # ... and its relative radius doesn't grow with spread
    assert small_ratio_b <= small_ratio_a * 1.01
    # while the big (shifted) community's relative radius does grow
    assert big_ratio_b > big_ratio_a * 1.05


@pytest.fixture
def negative_median_file(tmp_path):
    # x/y medians clearly negative; a few spread-out points keep the median
    # from landing exactly on any single row's coordinate
    rows = [("N0", -50, -30), ("N1", -49, -31), ("N2", -51, -29),
            ("N3", -48, -30), ("N4", -52, -30), ("N5", -50, -28),
            ("N6", -50, -32), ("F1", -500, -30), ("F2", -50, -400)]
    p = tmp_path / "coords.parquet"
    write_coords(p, rows)
    return str(p)


def test_negative_median_center_does_not_crash(negative_median_file, tmp_path):
    out = str(tmp_path / "coords_web.parquet")
    stats = build_webcoords(negative_median_file, out, spread=1.0, ring_comm_max=0)
    assert stats["n"] == len(
        duckdb.sql(f"SELECT * FROM '{negative_median_file}'").fetchall()
    )


def test_unplaced_authors_become_ring_dust_with_their_own_community(tmp_path):
    # cuGraph sizes the graph by its highest vertex id with an edge, so an
    # edgeless author numbered above it gets no position and no community
    p = tmp_path / "coords.parquet"
    rows = [(f"C{k}", math.cos(k) * (1 + k % 7), math.sin(k) * (1 + k % 7), 1)
            for k in range(150)]
    write_coords_comm(p, rows)
    duckdb.sql(
        f"COPY (SELECT * FROM read_parquet('{p}') UNION ALL"
        f" SELECT 'U1', 'N U1', NULL::FLOAT, NULL::FLOAT, NULL::INTEGER, 20, 100, 'I', 'Biology'"
        f" UNION ALL"
        f" SELECT 'U2', 'N U2', NULL::FLOAT, NULL::FLOAT, NULL::INTEGER, 20, 100, 'I', 'Biology')"
        f" TO '{tmp_path / 'with_unplaced.parquet'}' (FORMAT PARQUET)"
    )
    out = str(tmp_path / "web.parquet")
    stats = build_webcoords(str(tmp_path / "with_unplaced.parquet"), out, spread=1.0)
    assert stats["n"] == 152                       # nobody dropped
    rows = duckdb.sql(f"SELECT id, xw, yw, is_ring, community FROM '{out}'"
                      f" WHERE id IN ('U1', 'U2') ORDER BY id").fetchall()
    assert [r[3] for r in rows] == [True, True]
    assert all(0.0 <= r[1] <= 1.0 and 0.0 <= r[2] <= 1.0 for r in rows)
    communities = {r[4] for r in rows}
    assert None not in communities and len(communities) == 2
    assert not communities & {1}                   # not merged into a real community


def test_linkless_singletons_are_dust_and_do_not_set_the_map_edge(tmp_path):
    # an author with no coauthor link is its own one-member community; its
    # layout position carries no information (FA2 only pushes it away). A
    # warm-started layout leaves such authors at arbitrary radii, some far
    # beyond the network, which used to stretch the normalization so the
    # whole core shrank and they piled up on the rim.
    p = tmp_path / "coords.parquet"
    rows = [(f"C{k}", math.cos(k) * (1 + k % 7), math.sin(k) * (1 + k % 7), 1)
            for k in range(150)]
    rows += [("S_far", 900.0, 0.0, 2), ("S_mid", 3.0, 3.0, 3)]   # singletons
    write_coords_comm(p, rows)
    with_singletons, without = str(tmp_path / "a.parquet"), str(tmp_path / "b.parquet")
    build_webcoords(str(p), with_singletons, spread=1.0)
    write_coords_comm(tmp_path / "core.parquet", rows[:150])
    build_webcoords(str(tmp_path / "core.parquet"), without, spread=1.0)
    ring = dict(duckdb.sql(f"SELECT id, is_ring FROM '{with_singletons}'"
                           f" WHERE id IN ('S_far', 'S_mid')").fetchall())
    assert ring == {"S_far": True, "S_mid": True}
    core_a = duckdb.sql(f"SELECT id, xw, yw FROM '{with_singletons}' WHERE id LIKE 'C%' ORDER BY id").fetchall()
    core_b = duckdb.sql(f"SELECT id, xw, yw FROM '{without}' ORDER BY id").fetchall()
    for (_, xa, ya), (_, xb, yb) in zip(core_a, core_b):
        assert xa == pytest.approx(xb, abs=1e-9) and ya == pytest.approx(yb, abs=1e-9)


def _spread_ratio(p, community):
    """blob radius / distance between the two blob centroids"""
    return duckdb.sql(
        f"""WITH c AS (SELECT community, avg(xw) cx, avg(yw) cy FROM '{p}' GROUP BY 1),
                 r AS (SELECT max(sqrt((w.xw - c.cx)**2 + (w.yw - c.cy)**2)) rad
                       FROM '{p}' w JOIN c USING (community) WHERE community = {community})
            SELECT (SELECT rad FROM r) / (SELECT abs(max(cx) - min(cx)) FROM c)""").fetchone()[0]


def test_tighten_pulls_big_communities_toward_their_centres(two_blob_file, tmp_path):
    loose, tight = str(tmp_path / "a.parquet"), str(tmp_path / "b.parquet")
    build_webcoords(two_blob_file, loose, spread=1.0, tighten=1.0)
    build_webcoords(two_blob_file, tight, spread=1.0, tighten=0.5)
    assert _spread_ratio(tight, 1) == pytest.approx(0.5 * _spread_ratio(loose, 1), rel=0.05)
    lo, hi = duckdb.sql(f"SELECT least(min(xw), min(yw)), greatest(max(xw), max(yw)) FROM '{tight}'").fetchone()
    assert lo >= 0.02 - 1e-9 and hi <= 0.98 + 1e-9          # re-fitted to the map, no dead margin


def test_tighten_leaves_small_communities_alone(tmp_path):
    # below the 1000-member threshold (the same one spread uses) nothing moves
    p = tmp_path / "coords.parquet"
    rows = [(f"C{k}", math.cos(k) * (1 + k % 7), math.sin(k) * (1 + k % 7), 1) for k in range(150)]
    write_coords_comm(p, rows)
    a, b = str(tmp_path / "a.parquet"), str(tmp_path / "b.parquet")
    build_webcoords(str(p), a, spread=1.0, tighten=1.0)
    build_webcoords(str(p), b, spread=1.0, tighten=0.5)
    assert duckdb.sql(f"SELECT id, xw, yw FROM '{a}' ORDER BY id").fetchall() == \
        duckdb.sql(f"SELECT id, xw, yw FROM '{b}' ORDER BY id").fetchall()


def test_fill_magnifies_the_middle_and_keeps_the_rim(two_blob_file, tmp_path):
    # tightening leaves the dense cores small inside a dark band; the fill
    # stretch r' = k r / (1 + (k - 1) r) magnifies the middle by k, keeps
    # r = 1 in place and is monotone, so order and gaps survive
    plain, filled = str(tmp_path / "a.parquet"), str(tmp_path / "b.parquet")
    build_webcoords(two_blob_file, plain, spread=1.0, tighten=0.5, fill=1.0)
    build_webcoords(two_blob_file, filled, spread=1.0, tighten=0.5, fill=2.0)
    q = ("SELECT id, sqrt((xw - 0.5)^2 + (yw - 0.5)^2) / 0.48 r, atan2(yw - 0.5, xw - 0.5) a"
         " FROM '{}' ORDER BY id")
    a = duckdb.sql(q.format(plain)).fetchall()
    b = duckdb.sql(q.format(filled)).fetchall()
    assert max(r for _, r, _ in b) == pytest.approx(max(r for _, r, _ in a), abs=1e-9)
    for (_, ra, aa), (_, rb, ab) in zip(a, b):
        assert rb == pytest.approx(2 * ra / (1 + ra), abs=1e-9)
        assert ab == pytest.approx(aa, abs=1e-9)          # radial only


# ---- sub-group areas (pipeline/subcommunities.py feeds these) ----
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.webcoords import pack_subareas


def _overlapping_subs(seed=0):
    rng = np.random.default_rng(seed)
    sizes = [600, 300, 100]
    sub = np.repeat(np.arange(3), sizes)
    x, y = rng.normal(0, 1, len(sub)), rng.normal(0, 1, len(sub))   # all three on top of each other
    return x, y, sub


def test_pack_subareas_separates_overlapping_subgroups():
    x, y, sub = _overlapping_subs()
    nx, ny = pack_subareas(x, y, sub)
    cen = np.array([[nx[sub == s].mean(), ny[sub == s].mean()] for s in range(3)])
    spread = [np.sqrt(((nx[sub == s] - cen[s, 0]) ** 2 + (ny[sub == s] - cen[s, 1]) ** 2).mean()) for s in range(3)]
    for a in range(3):
        for b in range(a + 1, 3):
            # centres now further apart than the two sub-groups' own spreads
            assert np.hypot(*(cen[a] - cen[b])) > spread[a] + spread[b]


def test_pack_subareas_keeps_the_community_centre_and_spread():
    x, y, sub = _overlapping_subs(1)
    x = x + 5.0
    nx, ny = pack_subareas(x, y, sub)
    assert nx.mean() == pytest.approx(x.mean()) and ny.mean() == pytest.approx(y.mean())
    rms = lambda a, b: np.sqrt(((a - a.mean()) ** 2 + (b - b.mean()) ** 2).mean())
    assert rms(nx, ny) == pytest.approx(rms(x, y))


def test_pack_subareas_is_deterministic_and_draws_linked_groups_together():
    x, y, sub = _overlapping_subs(2)
    a1, b1 = pack_subareas(x, y, sub)
    a2, b2 = pack_subareas(x, y, sub)
    assert (a1 == a2).all() and (b1 == b2).all()
    links = np.zeros((3, 3)); links[1, 2] = links[2, 1] = 1.0
    lx, ly = pack_subareas(x, y, sub, links)
    d = lambda X, Y, s, t: np.hypot(X[sub == s].mean() - X[sub == t].mean(), Y[sub == s].mean() - Y[sub == t].mean())
    assert d(lx, ly, 1, 2) <= d(a1, b1, 1, 2) + 1e-9


def test_build_webcoords_adds_sub_and_only_moves_split_communities(tmp_path):
    rng = np.random.default_rng(3)
    ids, xs, ys, comms = [], [], [], []
    for c, (cx, n) in {1: (-50.0, 1500), 2: (50.0, 1200)}.items():
        for i in range(n):
            ids.append(f"C{c}_{i}"); xs.append(cx + rng.normal(0, 5)); ys.append(rng.normal(0, 5)); comms.append(c)
    coords = tmp_path / "coords.parquet"
    pq.write_table(pa.table({"id": ids, "display_name": ids, "x": pa.array(xs, pa.float32()),
                             "y": pa.array(ys, pa.float32()), "community": pa.array(comms, pa.int32()),
                             "works_count": pa.array([10] * len(ids), pa.int32()),
                             "cited_by_count": pa.array([100] * len(ids), pa.int32()),
                             "institution": ["Inst"] * len(ids), "field": ["Medicine"] * len(ids)}), coords)
    split_ids = [i for i in ids if i.startswith("C1_")]
    subs = tmp_path / "subcommunities.parquet"
    pq.write_table(pa.table({"id": split_ids, "community": pa.array([1] * 1500, pa.int32()),
                             "sub": pa.array([0] * 900 + [1] * 600, pa.int16())}), subs)
    plain = str(tmp_path / "plain.parquet"); refined = str(tmp_path / "refined.parquet")
    build_webcoords(str(coords), plain)
    stats = build_webcoords(str(coords), refined, subs_path=str(subs))
    assert stats["split_communities"] == 1
    q = lambda f: dict(duckdb.sql(f"SELECT id, (xw, yw, sub) FROM '{f}'").fetchall())
    a, b = q(plain), q(refined)
    assert {v[2] for k, v in a.items()} == {-1}
    assert b["C1_0"][2] == 0 and b["C1_1499"][2] == 1 and b["C2_0"][2] == -1
    moved = [k for k in ids if abs(a[k][0] - b[k][0]) + abs(a[k][1] - b[k][1]) > 1e-9]
    assert moved and all(k.startswith("C1_") for k in moved)


def test_is_dust_marks_ring_small_and_near_empty_communities(tmp_path):
    rng = np.random.default_rng(5)
    rows = []
    for i in range(1200):        # big, real community
        rows.append((f"B{i}", 30 + rng.normal(0, 5), rng.normal(0, 5), 1, "Inst", 300))
    for i in range(1100):        # big but near-empty: no institutions, no citations
        rows.append((f"E{i}", -30 + rng.normal(0, 5), rng.normal(0, 5), 2, None, 0))
    for i in range(40):          # small group
        rows.append((f"S{i}", rng.normal(0, 5), 30 + rng.normal(0, 5), 3, "Inst", 300))
    rows.append(("L0", 500.0, 500.0, 4, "Inst", 5))     # linkless singleton -> outer ring
    coords = tmp_path / "coords.parquet"
    pq.write_table(pa.table({"id": [r[0] for r in rows], "display_name": [r[0] for r in rows],
                             "x": pa.array([r[1] for r in rows], pa.float32()), "y": pa.array([r[2] for r in rows], pa.float32()),
                             "community": pa.array([r[3] for r in rows], pa.int32()),
                             "works_count": pa.array([5] * len(rows), pa.int32()),
                             "cited_by_count": pa.array([r[5] for r in rows], pa.int32()),
                             "institution": pa.array([r[4] for r in rows], pa.string()),
                             "field": ["Medicine"] * len(rows)}), coords)
    out = str(tmp_path / "web.parquet")
    build_webcoords(str(coords), out)
    dust = dict(duckdb.sql(f"SELECT community, bool_and(is_dust) FROM '{out}' GROUP BY 1").fetchall())
    assert dust == {1: False, 2: True, 3: True, 4: True}
