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
