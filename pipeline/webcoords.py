"""asinh radial compression: layout coords -> unit-square web coords."""
import json
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa

from pipeline.config import apply_resource_limits, data_dir
from pipeline.palette import sparse_communities

MARGIN = 0.02  # points land in [MARGIN, 1-MARGIN]
SPREAD = 1.6   # community-centroid expansion used for the published map
# Pull members of big communities part-way toward their community centre
# (1.0 = off). Chosen 2026-10-10 from renders of the 2026-09-23 layout: the
# map centre is where big communities overlap in 2D (81% of coauthor link
# weight is within communities, so the groups are real), and 0.5 opens clear
# gaps between them so their colours stop interleaving.
TIGHTEN = 0.5
# Radial fill after tightening: r' = FILL * r / (1 + (FILL - 1) * r) on the
# normalized core radius. Tightening packs each community's dense core toward
# its centre, which left the outer third of the disk to faint tails (a dark
# band inside the dust ring). This stretch magnifies the middle by FILL,
# keeps the rim fixed and is smooth and monotone, so gaps and order survive
# with no seams (exact radial CDF matching was tried and drew arcs).
FILL = 2.0
BIG_COMMUNITY = 1000   # spread and tighten apply to communities this big
# Sub-group areas (pipeline/subcommunities.py). Inside a split community the
# layout leaves its sub-groups overlapping, so each gets its own circle:
# circle area ~ members (radius SUB_AREA * community RMS radius *
# sqrt(share)), circles packed apart from their current centres, linked
# sub-groups drawn together, and each sub-group's members shrunk to fill
# SUB_FIT of its circle. The community is then rescaled to its original
# centre and RMS radius, so its neighbours do not move. Eric chose "own areas"
# over legend-only splits on 2026-10-11 from a preview of community 0; after
# seeing it built at 0.55 ("works, but much more patchy") the cells were
# softened to 0.75 from whole-map renders at 0.55 / 0.75 / 0.95.
SUB_AREA = 1.5
SUB_FIT = 0.75
SUB_SPRING = 0.02     # pull back toward the sub-group's current centre
SUB_LINK = 0.004      # pull between linked sub-groups (scaled by link share)
SUB_ITERS = 2000


def pack_subareas(x, y, sub, links=None, iters: int = SUB_ITERS):
    """One community's members -> new (x, y) with each sub-group in its own
    area. `sub` holds 0..K-1; `links` is a KxK symmetric weight matrix."""
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64); sub = np.asarray(sub)
    k = int(sub.max()) + 1
    n_s = np.bincount(sub, minlength=k).astype(np.float64)
    px, py = x.mean(), y.mean()
    rms = np.sqrt(((x - px) ** 2 + (y - py) ** 2).mean()) or 1e-12
    c0 = np.stack([np.bincount(sub, weights=x, minlength=k) / n_s,
                   np.bincount(sub, weights=y, minlength=k) / n_s], axis=1)
    sd = np.sqrt(np.bincount(sub, weights=(x - c0[sub, 0]) ** 2 + (y - c0[sub, 1]) ** 2,
                             minlength=k) / n_s)
    sd = np.where(sd > 0, sd, rms * 1e-3)
    r = SUB_AREA * rms * np.sqrt(n_s / n_s.sum())
    w = np.zeros((k, k)) if links is None else np.asarray(links, np.float64)
    w = w / w.max() if w.max() > 0 else w
    ang = np.arange(k) * 2.399963                                     # golden-angle jitter
    q = c0 + 1e-6 * rms * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    mass = n_s[None, :] / (n_s[:, None] + n_s[None, :])              # lighter circles move more
    eye = np.eye(k, dtype=bool)
    for _ in range(iters):
        d = q[:, None, :] - q[None, :, :]
        dist = np.sqrt((d ** 2).sum(-1)) + 1e-12
        gap = dist - (r[:, None] + r[None, :])
        over = np.where(eye, 0.0, np.maximum(-gap, 0.0))
        rep = ((0.5 * over * mass / dist)[:, :, None] * d).sum(1)
        pull = (np.where(eye | (gap <= 0), 0.0, w)[:, :, None] * -d).sum(1) * SUB_LINK
        q = q + rep + pull + SUB_SPRING * (c0 - q)
    scale = SUB_FIT * r / sd
    nx = q[sub, 0] + (x - c0[sub, 0]) * scale[sub]
    ny = q[sub, 1] + (y - c0[sub, 1]) * scale[sub]
    nrms = np.sqrt(((nx - nx.mean()) ** 2 + (ny - ny.mean()) ** 2).mean()) or 1e-12
    f = rms / nrms
    return px + (nx - nx.mean()) * f, py + (ny - ny.mean()) * f


def _refine_subareas(con, subs_path: str, links_path: str | None) -> int:
    """Rewrite `tightened` so split communities' sub-groups get their own
    areas; adds the `sub` column (-1 outside split communities)."""
    t = con.execute(
        f"""SELECT t.id, t.community, t.sx, t.sy, s.sub
            FROM tightened t JOIN read_parquet('{subs_path}') s ON s.id = t.id
            ORDER BY t.community, t.id"""
    ).fetchnumpy()
    links = {}
    if links_path and Path(links_path).exists():
        for c, a, b, wt in con.execute(
                f"SELECT community, a, b, weight FROM read_parquet('{links_path}')").fetchall():
            links.setdefault(int(c), []).append((int(a), int(b), float(wt)))
    comm = t["community"]
    nsx = np.empty(len(comm)); nsy = np.empty(len(comm))
    bounds = np.flatnonzero(np.diff(comm)) + 1
    for lo, hi in zip(np.concatenate([[0], bounds]), np.concatenate([bounds, [len(comm)]])):
        c, sub = int(comm[lo]), t["sub"][lo:hi].astype(np.int64)
        k = int(sub.max()) + 1
        w = np.zeros((k, k))
        for a, b, wt in links.get(c, []):
            if a < k and b < k:
                w[a, b] = w[b, a] = wt
        nsx[lo:hi], nsy[lo:hi] = pack_subareas(t["sx"][lo:hi], t["sy"][lo:hi], sub, w)
    con.register("refined", pa.table({"id": t["id"], "nsx": nsx, "nsy": nsy,
                                      "sub": t["sub"].astype(np.int16)}))
    con.execute("""CREATE TEMP TABLE tightened_sub AS
                   SELECT t.* REPLACE (coalesce(r.nsx, t.sx) AS sx, coalesce(r.nsy, t.sy) AS sy),
                          coalesce(r.sub, -1)::SMALLINT AS sub
                   FROM tightened t LEFT JOIN refined r ON r.id = t.id""")
    con.unregister("refined")
    con.execute("DROP TABLE tightened")
    con.execute("ALTER TABLE tightened_sub RENAME TO tightened")
    return len(np.unique(comm))


def build_webcoords(coords_path: str, out_path: str, spread: float = SPREAD,
                    ring_comm_max: int = 100, tighten: float = TIGHTEN,
                    fill: float = FILL, subs_path: str | None = None,
                    links_path: str | None = None) -> dict:
    con = duckdb.connect()
    apply_resource_limits(con)
    # An edgeless author numbered above the graph's highest connected vertex
    # gets no layout position or community from cuGraph (renumber=False sizes
    # the graph by that vertex). Such rows are scattered as ring dust, each
    # in its own negative singleton community, instead of being dropped.
    con.execute(
        f"""CREATE TEMP TABLE src AS
            SELECT * REPLACE (coalesce(community, -CAST(row_number() OVER (ORDER BY id)
                                                         AS INTEGER)) AS community),
                   (x IS NULL OR y IS NULL) AS unplaced
            FROM read_parquet('{coords_path}')"""
    )
    # An author with no coauthor link is a one-member community whose layout
    # position carries no information: FA2 only pushes it away. A warm-started
    # layout leaves such authors at arbitrary radii (50,737 in the 2026-09-23
    # run, some ~20x the median radius), so they are dust, and the map's
    # centre, scale and edge come from the linked network alone.
    con.execute("""CREATE TEMP TABLE net AS
                   SELECT *, count(*) OVER (PARTITION BY community) AS comm_n FROM src""")
    # Dust (R35b, Eric: "an option to turn it off completely"): the outer
    # ring, communities too small for the legend, and near-empty communities
    # (palette.sparse_communities). Tiles and the point cloud leave it out;
    # the viewer draws it as its own switchable layer from dust.bin.
    faded = sparse_communities(con, coords_path, min_members=BIG_COMMUNITY)
    faded_sql = ",".join(str(c) for c in sorted(faded)) or "NULL"
    linked = "(NOT unplaced AND comm_n > 1)"
    if not con.execute(f"SELECT count(*) FROM net WHERE {linked}").fetchone()[0]:
        linked = "(NOT unplaced)"          # degenerate input: no links anywhere
    # median center per controller decision - mean is halo-sensitive
    cx, cy = con.execute(f"SELECT median(x), median(y) FROM net WHERE {linked}").fetchone()
    # median radius as the asinh scale; guard degenerate all-at-center inputs
    s = con.execute(
        f"""SELECT greatest(median(sqrt((x-({cx}))*(x-({cx})) + (y-({cy}))*(y-({cy})))), 1e-9)
            FROM net WHERE {linked}"""
    ).fetchone()[0]
    con.execute(
        f"""CREATE TEMP TABLE polar AS
            SELECT *,
                   sqrt((x-({cx}))*(x-({cx})) + (y-({cy}))*(y-({cy}))) / {s} AS rs,
                   atan2(y-({cy}), x-({cx})) AS theta
            FROM net"""
    )
    # asinh(v) = ln(v + sqrt(v*v + 1)); DuckDB lacks asinh
    con.execute(
        """CREATE TEMP TABLE au AS
           SELECT *, ln(rs + sqrt(rs*rs + 1)) AS a FROM polar"""
    )
    amax = con.execute(f"SELECT max(a) FROM au WHERE {linked}").fetchone()[0] or 1e-9
    # ring rule: unplaced or linkless authors, and small disconnected
    # communities parked out near the max radius
    dust = "(unplaced OR comm_n = 1)" if linked != "(NOT unplaced)" else "unplaced"
    ring_pred = (f"({dust} OR (comm_n < {int(ring_comm_max)} AND a > 0.9 * {amax}))"
                 if ring_comm_max > 0 else dust)
    a2max = con.execute(
        f"SELECT max(a) FROM au WHERE NOT {ring_pred}"
    ).fetchone()[0] or 1e-9
    half = 0.5 - MARGIN
    # hash-based uniforms in [0,1): DuckDB hash() is a stable UBIGINT
    con.execute(
        f"""CREATE TEMP TABLE placed AS
            SELECT id, display_name, community, works_count, cited_by_count,
                   institution, field, {ring_pred} AS is_ring, comm_n,
                   ({ring_pred} OR comm_n < {BIG_COMMUNITY} OR community IN ({faded_sql})) AS is_dust,
                   CASE WHEN {ring_pred}
                        THEN (0.60 + 0.38 * ((hash(id) % 100000) / 100000.0))
                        ELSE least(a / {a2max}, 1.0) END AS ru,
                   CASE WHEN {ring_pred}
                        THEN 2 * pi() * ((hash(id || '/t') % 100000) / 100000.0)
                        ELSE theta END AS th
            FROM au"""
    )
    con.execute(
        f"""CREATE TEMP TABLE core AS
            SELECT *, ru * cos(th) AS ex, ru * sin(th) AS ey
            FROM placed WHERE NOT is_ring"""
    )
    con.execute(
        f"""CREATE TEMP TABLE shifted AS
            SELECT *,
                   ex + CASE WHEN comm_n >= {BIG_COMMUNITY}
                        THEN ({spread} - 1.0) * avg(ex) OVER (PARTITION BY community)
                        ELSE 0.0 END AS sx0,
                   ey + CASE WHEN comm_n >= {BIG_COMMUNITY}
                        THEN ({spread} - 1.0) * avg(ey) OVER (PARTITION BY community)
                        ELSE 0.0 END AS sy0
            FROM core"""
    )
    # tighten: x' = centre + tighten * (x - centre), big communities only; the
    # normalization below then re-fits the result to the map
    con.execute(
        f"""CREATE TEMP TABLE tightened AS
            SELECT * EXCLUDE (sx0, sy0),
                   CASE WHEN comm_n >= {BIG_COMMUNITY}
                        THEN avg(sx0) OVER w + {tighten} * (sx0 - avg(sx0) OVER w)
                        ELSE sx0 END AS sx,
                   CASE WHEN comm_n >= {BIG_COMMUNITY}
                        THEN avg(sy0) OVER w + {tighten} * (sy0 - avg(sy0) OVER w)
                        ELSE sy0 END AS sy
            FROM shifted WINDOW w AS (PARTITION BY community)"""
    )
    split = 0
    if subs_path and Path(subs_path).exists():
        split = _refine_subareas(con, subs_path, links_path)
    else:
        con.execute("ALTER TABLE tightened ADD COLUMN sub SMALLINT DEFAULT -1")
    smax = con.execute(
        "SELECT greatest(max(sqrt(sx*sx + sy*sy)), 1e-9) FROM tightened"
    ).fetchone()[0]
    n = con.execute(
        f"""COPY (
              SELECT id, display_name,
                     0.5 + {half} * sx / {smax} * g AS xw,
                     0.5 + {half} * sy / {smax} * g AS yw,
                     community, sub, works_count, cited_by_count, institution,
                     field, is_ring, is_dust
              FROM (SELECT *, {fill} / (1 + ({fill} - 1) * sqrt(sx*sx + sy*sy) / {smax}) AS g
                    FROM tightened)
              UNION ALL
              SELECT id, display_name,
                     0.5 + {half} * ru * cos(th) AS xw,
                     0.5 + {half} * ru * sin(th) AS yw,
                     community, CAST(-1 AS SMALLINT) AS sub, works_count, cited_by_count,
                     institution, field, is_ring, is_dust
              FROM placed WHERE is_ring
            ) TO '{out_path}' (FORMAT PARQUET)"""
    ).fetchone()[0]
    ring_n = con.execute("SELECT count(*) FROM placed WHERE is_ring").fetchone()[0]
    stats = {"n": n, "cx": cx, "cy": cy, "s": s, "r2max": a2max,
             "ring_n": ring_n, "spread": spread, "tighten": tighten, "fill": fill,
             "split_communities": split}
    Path(out_path + ".meta.json").write_text(json.dumps(stats))
    return stats


def add_parser(parser) -> None:
    parser.add_argument("--coords", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--spread", type=float, default=SPREAD)
    parser.add_argument("--tighten", type=float, default=TIGHTEN,
                        help="pull big-community members toward their centre (1.0 = off)")
    parser.add_argument("--fill", type=float, default=FILL,
                        help="radial magnification of the map centre, rim fixed (1.0 = off)")
    parser.add_argument("--ring-comm-max", type=int, default=100)
    parser.add_argument("--subs", default=None,
                        help="subcommunities.parquet (default: the release's, when present)")


def run(args) -> int:
    coords = args.coords or str(data_dir() / "coords.parquet")
    out = args.out or str(data_dir() / "coords_web.parquet")
    subs = args.subs or str(data_dir() / "subcommunities.parquet")
    stats = build_webcoords(coords, out, spread=args.spread,
                            ring_comm_max=args.ring_comm_max, tighten=args.tighten,
                            fill=args.fill, subs_path=subs,
                            links_path=str(Path(subs).with_name("subcommunity_links.parquet")))
    print(f"{stats['n']:,} points -> {out} (s={stats['s']:.1f}, "
          f"r2max={stats['r2max']:.3f}, ring={stats['ring_n']:,}, "
          f"spread={stats['spread']}, tighten={stats['tighten']}, fill={stats['fill']}, "
          f"split communities={stats['split_communities']})")
    return 0
