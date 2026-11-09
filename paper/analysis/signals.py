"""Identity and link signals of the network in use, written next to its pathgraph.

identity_signals.parquet, per node: display_name; name_key (first initial + surname, lowercased, accents
stripped) and key_n, the researchers in the network who share it; exact_n, those who share the whole display name;
total link weight; ncomm, the communities among its coauthors; share1 and share2, the shares of its link weight
in its two leading communities; big5, the communities holding at least 5% of it.

link_signals.parquet, per link (u < v as stored): ku, kv, the name-key counts of both ends; s_uv, the share of
u's link weight in v's community, and s_vu the reverse. common.suspect_mask flags a link when a common name
(key_n >= 200) meets a community holding under 5% of that person's link weight (pipeline/identity.py).

Communities of under 1,000 members are pooled as one (-1). Communities: the layout's on the map's 15-work
network, leiden.py's on the 5-work network.
"""
import time

import numpy as np
import pyarrow as pa

from common import CACHE, D, G, NET, PG, con, log

t0 = time.time()
c = con()
c.execute("SET memory_limit='10GB'")
if NET == "min15":
    c.execute(f"""CREATE TABLE cm AS SELECT n.node_idx, w.community
                  FROM read_parquet('{G}/nodes_int32.parquet') n JOIN read_parquet('{D}/coords_web.parquet') w ON w.id = n.id""")
else:
    lab = np.load(CACHE / "leiden.npy")
    c.register("lab", pa.table({"node_idx": np.arange(len(lab), dtype=np.int32), "community": lab}))
    c.execute("CREATE TABLE cm AS SELECT * FROM lab")
c.execute("CREATE TABLE sizes AS SELECT community, count(*) n FROM cm GROUP BY 1")
c.execute(f"""CREATE TABLE node AS
  SELECT n.node_idx, n.display_name,
         CASE WHEN s.n >= 1000 THEN cm.community ELSE -1 END AS comm,
         lower(strip_accents(left(trim(n.display_name), 1) || ' ' || regexp_extract(trim(n.display_name), '(\\S+)$', 1))) AS name_key
  FROM read_parquet('{G}/nodes_int32.parquet') n JOIN cm USING (node_idx) JOIN sizes s USING (community)""")
c.execute("CREATE TABLE namecount AS SELECT name_key, count(*) AS key_n FROM node GROUP BY 1")
c.execute("CREATE TABLE exactcount AS SELECT display_name, count(*) AS exact_n FROM node GROUP BY 1")
log(f"nodes ready ({time.time() - t0:.0f}s)")
c.execute(f"CREATE TABLE e AS SELECT src AS u, dst AS v, weight AS w FROM read_parquet('{G}/edges_int32.parquet')")
c.execute("""CREATE TABLE uc AS
  WITH d AS (SELECT u, v, w FROM e UNION ALL SELECT v, u, w FROM e)
  SELECT d.u, nv.comm c, sum(d.w) w FROM d JOIN node nv ON nv.node_idx = d.v GROUP BY 1, 2""")
c.execute("CREATE TABLE ushare AS SELECT u, c, w / sum(w) OVER (PARTITION BY u) AS share FROM uc")
log(f"shares ready ({time.time() - t0:.0f}s)")
c.execute("""CREATE TABLE spread AS
  WITH tot AS (SELECT u, sum(w) total, count(*) ncomm FROM uc GROUP BY 1),
       ranked AS (SELECT u, c, w, row_number() OVER (PARTITION BY u ORDER BY w DESC, c) rn FROM uc)
  SELECT t.u, t.total, t.ncomm,
         max(CASE WHEN r.rn = 1 THEN r.w END) / t.total AS share1,
         coalesce(max(CASE WHEN r.rn = 2 THEN r.w END), 0) / t.total AS share2,
         count(*) FILTER (WHERE r.w / t.total >= 0.05) AS big5
  FROM tot t JOIN ranked r USING (u) GROUP BY t.u, t.total, t.ncomm""")
c.execute(f"""COPY (
  SELECT n.node_idx, n.display_name, n.name_key, nc.key_n, ec.exact_n, s.total, s.ncomm, s.share1, s.share2, s.big5
  FROM node n JOIN namecount nc USING (name_key) JOIN exactcount ec USING (display_name) LEFT JOIN spread s ON s.u = n.node_idx
  ORDER BY n.node_idx) TO '{PG}/identity_signals.parquet' (FORMAT PARQUET)""")
log(f"identity signals written ({time.time() - t0:.0f}s)")
c.execute("DROP TABLE uc")
c.execute(f"""COPY (
  SELECT e.u, e.v, ku.key_n ku, kv.key_n kv, su.share s_uv, sv.share s_vu
  FROM e JOIN node nu ON nu.node_idx = e.u JOIN namecount ku ON ku.name_key = nu.name_key
  JOIN node nv ON nv.node_idx = e.v JOIN namecount kv ON kv.name_key = nv.name_key
  JOIN ushare su ON su.u = e.u AND su.c = nv.comm
  JOIN ushare sv ON sv.u = e.v AND sv.c = nu.comm) TO '{PG}/link_signals.parquet' (FORMAT PARQUET)""")
log(f"link signals written ({time.time() - t0:.0f}s)")
