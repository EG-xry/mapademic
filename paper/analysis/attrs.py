"""Per-node attributes aligned with the graph's node index, cached as cache/node_attrs.parquet.

node_idx, id, name, works, works_kept, cited, institution, field, country,
community, sub, is_dust, is_ring, on_map, flagged (links flagged as probable
merges), degree. Country: the most common country_code among institutions
with the author's last-known institution name (the same mapping regions.py
uses).

Communities: on the map's 15-work network, the layout's Leiden run
(coords_web.parquet); on the 5-work network, leiden.py's (cache/leiden.npy).
is_dust marks members of communities the map draws as dust: under 1,000
members, or near-empty records (palette.sparse_communities: three quarters
without an institution and a median of at most 10 citations); on the map
itself also its outer ring. sub and is_ring exist only for researchers on
the map (on_map: at least 15 works).
"""
import numpy as np
import pyarrow as pa

from common import CACHE, D, ND, NET, PG, con, load_csr, log

c = con()
ids = np.load(PG / "ids.npy")
offsets, _ = load_csr()
deg = np.diff(np.asarray(offsets))
flagged = np.load(CACHE / "flagged_per_node.npy")
g = {"node_idx": np.arange(len(ids), dtype=np.int32), "num": ids.astype(np.int64),
     "degree": deg.astype(np.int32), "flagged": flagged}
if NET != "min15":
    g["leiden"] = np.load(CACHE / "leiden.npy")
c.register("g", pa.table(g))
comm = "w.community" if NET == "min15" else "g.leiden"
inst_glob = f"{D}/snapshot/institutions/**/*.parquet"
log("joining node attributes")
c.execute(f"""
    CREATE TABLE t AS
      WITH ic AS (SELECT display_name, mode(country_code) AS cc FROM read_parquet('{inst_glob}') GROUP BY 1),
           n AS (SELECT CAST(regexp_extract(id, '(\\d+)$', 1) AS BIGINT) AS num, id, display_name, works_count,
                        cited_by_count, institution, field, works_kept FROM read_parquet('{ND}/nodes.parquet')),
           w AS (SELECT CAST(regexp_extract(id, '(\\d+)$', 1) AS BIGINT) AS num, community, sub, is_dust, is_ring
                 FROM read_parquet('{D}/coords_web.parquet'))
      SELECT g.node_idx, n.id, n.display_name AS name, n.works_count AS works, n.works_kept, n.cited_by_count AS cited,
             n.institution, n.field, ic.cc AS country, {comm} AS community, w.sub, w.is_dust AS map_dust, w.is_ring,
             w.num IS NOT NULL AS on_map, g.flagged, g.degree
      FROM g LEFT JOIN n USING (num) LEFT JOIN w USING (num) LEFT JOIN ic ON ic.display_name = n.institution""")
c.execute("""
    CREATE TABLE faded AS
      SELECT community FROM t GROUP BY 1
      HAVING count(*) < 1000 OR (avg((institution IS NULL)::INTEGER) >= 0.75 AND median(cited) <= 10)""")
dust = "t.map_dust" if NET == "min15" else "f.community IS NOT NULL"
c.execute(f"""
    COPY (SELECT t.* EXCLUDE (map_dust), {dust} AS is_dust
          FROM t LEFT JOIN faded f USING (community) ORDER BY t.node_idx)
    TO '{CACHE / 'node_attrs.parquet'}' (FORMAT PARQUET)""")
r = c.execute(f"""SELECT count(*), count(id), count(country), count(community), count(field), sum(on_map::INT),
                         sum(is_dust::INT)
                  FROM read_parquet('{CACHE / 'node_attrs.parquet'}')""").fetchone()
log(f"rows {r[0]:,}; with id {r[1]:,}, country {r[2]:,}, community {r[3]:,}, field {r[4]:,}; "
    f"on the map {r[5]:,}; dust {r[6]:,}")
