"""A5b: the 36 big communities, profiled -- cohesion, country, fields, citations, and where the errors sit.

Per community with 1,000+ members: size; share of members' link weight that
stays inside (cohesion); top country and its share; top fields; median degree
and citations; share of members carrying a flagged (probable-merge) link;
share with a common name (identity key shared by 200+ records); share of
members with a placeholder-like or non-person name pattern.
"""
from common import CACHE, G, PG, con, log, save_result

c = con()
A = f"read_parquet('{CACHE / 'node_attrs.parquet'}')"
E = f"read_parquet('{G / 'edges_int32.parquet'}')"
log("cohesion")
c.execute(f"""CREATE TABLE coh AS
    WITH e AS (SELECT a.community AS ca, b.community AS cb, w FROM
                 (SELECT src, dst, weight AS w FROM {E}) x
                 JOIN {A} a ON a.node_idx = x.src JOIN {A} b ON b.node_idx = x.dst)
    SELECT community, sum(w) FILTER (WHERE inside) / sum(w) AS cohesion, sum(w) AS weight FROM (
        SELECT ca AS community, ca = cb AS inside, w FROM e UNION ALL SELECT cb, ca = cb, w FROM e)
    GROUP BY 1""")
log("profiles")
df = c.execute(f"""
    WITH m AS (SELECT a.*, s.key_n FROM {A} a JOIN read_parquet('{PG / 'identity_signals.parquet'}') s USING (node_idx)),
         big AS (SELECT community FROM m GROUP BY 1 HAVING count(*) >= 1000),
         f AS (SELECT community, field, count(*) n FROM m WHERE field IS NOT NULL GROUP BY 1, 2),
         ftop AS (SELECT community, string_agg(field || ' ' || round(100 * share)::INT || '%', '; ' ORDER BY share DESC) AS fields
                  FROM (SELECT community, field, n / sum(n) OVER (PARTITION BY community) AS share,
                               row_number() OVER (PARTITION BY community ORDER BY n DESC) AS r FROM f) WHERE r <= 3 GROUP BY 1),
         ctry AS (SELECT community, country, count(*) n FROM m WHERE country IS NOT NULL GROUP BY 1, 2),
         ctop AS (SELECT community, arg_max(country, n) AS top_country, max(n) / sum(n) AS top_country_share FROM ctry GROUP BY 1)
    SELECT m.community, count(*) AS size, any_value(coh.cohesion) AS cohesion, any_value(ctop.top_country) AS top_country,
           any_value(ctop.top_country_share) AS top_country_share, any_value(ftop.fields) AS top_fields,
           median(m.degree) AS median_degree, median(m.cited) AS median_cited,
           avg(CASE WHEN m.flagged > 0 THEN 1 ELSE 0 END) AS share_with_flagged_link,
           avg(CASE WHEN m.key_n >= 200 THEN 1 ELSE 0 END) AS share_common_name,
           avg(CASE WHEN m.country IS NULL THEN 1 ELSE 0 END) AS share_no_country
    FROM m JOIN big USING (community) LEFT JOIN coh USING (community) LEFT JOIN ctop USING (community) LEFT JOIN ftop USING (community)
    GROUP BY m.community ORDER BY size DESC""").fetchdf()
df.to_csv(CACHE / "a5b_communities.csv", index=False)
save_result("a5b_communities", {"communities": df.to_dict("records")})
with __import__("pandas").option_context("display.width", 250, "display.max_colwidth", 70):
    print(df.round(3).to_string())
