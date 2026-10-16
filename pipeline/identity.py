"""Coauthor links that probably join two different people who share a name.

OpenAlex sometimes merges namesakes into one author record: in 2026-09-23
"Weiwei Sun" (A5050009113, 384 works across engineering, computer science,
environmental science and chemistry) links Geoffrey Hinton, through a
computer-vision paper, to landscape ecologists. Shortest-path search is
drawn to such records, because a merged record is a shortcut between
fields: 36% of the shortest paths between 300 random author pairs used at
least one link flagged below, and every one of them had a path avoiding
flagged links that was the same length (78%) or one step longer (22%).

A link X-Y is suspect when, from either side, the person has a common name
(at least COMMON_NAME author records share its "first initial + surname")
and the other person's map region (community, pooled below LEGEND_MIN
members) carries under RARE_REGION of that person's coauthorship weight:
a rare excursion of a common name is what a merged namesake looks like.
Hinton's links are never flagged on his account ("g hinton": 3 records),
while Hinton-Weiwei Sun is (997 records; Hinton's region holds 4.2% of that
record's links). About 4.5% of all links are flagged.
"""
import numpy as np

COMMON_NAME = 200
RARE_REGION = 0.05
LEGEND_MIN = 1000


def suspect_links(con, nodes_path: str, edges_path: str, web_path: str,
                  common_name: int = COMMON_NAME, rare_region: float = RARE_REGION,
                  legend_min: int = LEGEND_MIN) -> tuple[np.ndarray, np.ndarray]:
    """(src, dst) node indices of suspect links, as edges_path stores them."""
    con.execute(f"""CREATE OR REPLACE TEMP TABLE id_node AS
        WITH sizes AS (SELECT community, count(*) n FROM read_parquet('{web_path}') GROUP BY 1)
        SELECT n.node_idx,
               CASE WHEN s.n >= {int(legend_min)} THEN w.community ELSE -1 END AS comm,
               lower(strip_accents(left(trim(n.display_name), 1) || ' '
                     || regexp_extract(trim(n.display_name), '(\\S+)$', 1))) AS name_key
        FROM read_parquet('{nodes_path}') n
        JOIN read_parquet('{web_path}') w ON w.id = n.id
        JOIN sizes s ON s.community = w.community""")
    con.execute("""CREATE OR REPLACE TEMP TABLE id_common AS
        SELECT node_idx, comm, count(*) OVER (PARTITION BY name_key) >= ? AS common
        FROM id_node""", [int(common_name)])
    con.execute(f"""CREATE OR REPLACE TEMP TABLE id_share AS
        WITH d AS (SELECT src AS u, dst AS v, weight AS w FROM read_parquet('{edges_path}')
                   UNION ALL SELECT dst, src, weight FROM read_parquet('{edges_path}')),
             uc AS (SELECT d.u, nv.comm AS c, sum(d.w) AS w FROM d JOIN id_common nv ON nv.node_idx = d.v GROUP BY 1, 2)
        SELECT u, c, w / sum(w) OVER (PARTITION BY u) AS share FROM uc""")
    pairs = con.execute(f"""
        SELECT e.src, e.dst FROM read_parquet('{edges_path}') e
        JOIN id_common a ON a.node_idx = e.src JOIN id_common b ON b.node_idx = e.dst
        JOIN id_share sa ON sa.u = e.src AND sa.c = b.comm
        JOIN id_share sb ON sb.u = e.dst AND sb.c = a.comm
        WHERE (a.common AND sa.share < {float(rare_region)}) OR (b.common AND sb.share < {float(rare_region)})""").fetchnumpy()
    return pairs["src"].astype(np.int64), pairs["dst"].astype(np.int64)
