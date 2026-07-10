"""Auto-name the largest communities from their members' dominant topics."""
import glob as globmod
import json
import os
from pathlib import Path

import duckdb

from pipeline.config import apply_resource_limits, data_dir

DEFAULT_AUTHORS = "/Volumes/Untitled/mapademic/snapshot/authors/*/*.parquet"

# distinctive-topic candidates kept per community for greedy unique naming
CANDIDATES_PER_COMMUNITY = 5


def _assign_unique_names(comm_rows, candidates):
    """Greedy: in community size-rank order (biggest first), each community
    takes its highest-scoring candidate topic not already claimed by a bigger
    community. If all candidates are taken, fall back to a rank-suffixed name so
    every kept region gets a distinct label. Deterministic given ordered input."""
    taken: set = set()
    names: dict = {}
    for community, _members, _xw, _yw, _spread, rank in comm_rows:
        chosen = None
        for topic in candidates.get(community, []):
            if topic not in taken:
                chosen = topic
                break
        if chosen is None:
            base = (candidates.get(community) or ["Unnamed"])[0] or "Unnamed"
            suffix = rank
            chosen = f"{base} ({suffix})"
            while chosen in taken:               # base+rank is already unique per
                suffix += 1                      # community; loop only guards the
                chosen = f"{base} ({suffix})"    # astronomically-unlikely collision
        taken.add(chosen)
        names[community] = chosen
    return names


def build_regions(webcoords_path: str, authors_glob: str, out_path: str,
                  top_n: int = 300, keep: int = 120) -> int:
    if not globmod.glob(authors_glob):
        raise SystemExit(
            f"authors snapshot not found at {authors_glob!r} - plug in the "
            "external drive or pass --authors <glob>"
        )
    con = duckdb.connect()
    apply_resource_limits(con)
    # top_n communities by size define the distinctiveness scope; rank them and
    # keep the biggest `keep` for output.
    con.execute(
        f"""
        CREATE TEMP TABLE comms AS
        WITH top_comms AS (
            SELECT community, count(*) AS members,
                   avg(xw) AS xw, avg(yw) AS yw,
                   sqrt(var_pop(xw) + var_pop(yw)) AS spread
            FROM read_parquet('{webcoords_path}')
            GROUP BY community
            ORDER BY members DESC, community
            LIMIT {int(top_n)}
        )
        SELECT community, members, xw, yw, spread,
               row_number() OVER (ORDER BY members DESC, community) AS rank
        FROM top_comms
        """
    )
    # per-community distinctive-topic ranking over the top_n scope
    con.execute(
        f"""
        CREATE TEMP TABLE best AS
        WITH member_topics AS (
            SELECT w.community, a.topics[1].display_name AS topic
            FROM read_parquet('{webcoords_path}') w
            JOIN read_parquet('{authors_glob}') a ON a.id = w.id
            JOIN comms c ON c.community = w.community
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
        )
        SELECT community, topic,
               row_number() OVER (PARTITION BY community
                                  ORDER BY score DESC, c DESC, topic) AS rn
        FROM scored
        """
    )
    comm_rows = con.execute(
        f"SELECT community, members, xw, yw, spread, rank FROM comms "
        f"WHERE rank <= {int(keep)} ORDER BY rank"
    ).fetchall()
    cand_rows = con.execute(
        f"""SELECT c.rank, b.community, b.topic
            FROM best b JOIN comms c ON c.community = b.community
            WHERE c.rank <= {int(keep)}
              AND b.rn <= {int(CANDIDATES_PER_COMMUNITY)}
            ORDER BY c.rank, b.rn"""
    ).fetchall()
    candidates: dict = {}
    for _rank, community, topic in cand_rows:
        candidates.setdefault(community, []).append(topic)
    names = _assign_unique_names(comm_rows, candidates)
    regions = []
    for community, members, xw, yw, spread, rank in comm_rows:
        regions.append({
            "name": names[community], "xw": xw, "yw": yw,
            "spread": spread, "members": members, "rank": rank,
            "community": community,
            "zmin": 2 if rank <= 30 else 4,
            "zmax": 4 if rank <= 30 else 6,
        })
    tmp = str(out_path) + ".tmp"
    Path(tmp).write_text(json.dumps(regions, indent=1))
    os.replace(tmp, out_path)
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
