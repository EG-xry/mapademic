"""The field and subfield of every researcher on the map (the 15-work layout that Figs. 1 and 9 draw), for Fig. 5, and
the field groups' shares in the map's large communities.

A researcher's field is that of the first OpenAlex topic, as everywhere in the paper; its subfield comes from
OpenAlex's topics table, as in A9d and pipeline/regions.py. Records whose first topic is the fallback "Legal and
Regulatory Analysis" are marked and get no field (Section 2). Fields are grouped as in common.FIELD_GROUPS.

Writes cache/map_fields.parquet (id, field and subfield, both null for the fallback topic or no topic, fallback) and
results/map_fields.json: for every community with at least 100,000 researchers on the map who have a field, the
groups' shares among them; where the map's arts and humanities researchers sit.
"""
import json

import pandas as pd
import pyarrow.parquet as pq

from common import CACHE, D, FIELD_GROUPS, con, log, save_result

FALLBACK_TOPIC = "Legal and Regulatory Analysis"
MAP = CACHE / "map"
OUT = CACHE / "map_fields.parquet"
BIG = 100_000

c = con()
c.execute(f"""
    COPY (
        WITH tp AS (SELECT display_name, mode(subfield.display_name) AS subfield
                    FROM read_parquet('{D}/snapshot/topics/*/*.parquet') GROUP BY 1),
             m AS (SELECT id FROM read_parquet('{MAP}/coords_web.parquet')),
             a AS (SELECT id, topics[1].display_name AS topic, topics[1].field.display_name AS field
                   FROM read_parquet('{D}/authors_extract/*/*.parquet') WHERE id IN (SELECT id FROM m))
        SELECT m.id,
               CASE WHEN a.topic = '{FALLBACK_TOPIC}' THEN NULL ELSE a.field END AS field,
               CASE WHEN a.topic = '{FALLBACK_TOPIC}' THEN NULL ELSE tp.subfield END AS subfield,
               coalesce(a.topic = '{FALLBACK_TOPIC}', false) AS fallback
        FROM m LEFT JOIN a USING (id) LEFT JOIN tp ON tp.display_name = a.topic
        ORDER BY m.id
    ) TO '{OUT}' (FORMAT PARQUET)""")
n, fb, nofield, arts = c.execute(f"""SELECT count(*), sum(fallback::INT), count(*) FILTER (WHERE field IS NULL),
                                        count(*) FILTER (WHERE field = 'Arts and Humanities')
                                 FROM read_parquet('{OUT}')""").fetchone()
log(f"{n:,} researchers on the map; {fb:,} with the fallback topic; {nofield:,} without a field; {arts:,} in the arts "
    f"and humanities")
check = c.execute(f"""SELECT count(*) FROM read_parquet('{OUT}') f JOIN read_parquet('{MAP}/coords_web.parquet') w
                      USING (id) WHERE NOT f.fallback AND f.field IS DISTINCT FROM w.field""").fetchone()[0]
log(f"{check:,} disagree with the map's own field column outside the fallback topic")
assert check == 0
c.close()

# the groups' shares in the large communities, among the map's researchers (outer ring left out) with a field
w = pq.read_table(MAP / "coords_web.parquet", columns=["id", "community", "is_ring"]).to_pandas()
f = pq.read_table(OUT, columns=["id", "field", "fallback"]).to_pandas().set_index("id")
assert f.index.is_unique and len(f) == len(w)
w = w.assign(field=f.field.reindex(w.id).to_numpy(), fallback=f.fallback.reindex(w.id).to_numpy())
w = w[~w.is_ring]
group_of = {field: g for g, fields in FIELD_GROUPS.items() for field in fields}
assert len(group_of) == 26 and set(w.field.dropna()) <= set(group_of)
w["group"] = w.field.map(group_of)
known = w[w.group.notna()]
names = {r["community"]: r["name"] for r in json.loads((MAP / "regions.json").read_text())}
a5b = pd.read_csv(CACHE / "a5b_communities.csv").set_index("community")
national = set(a5b.index[(a5b.top_country_share >= 0.5) & (a5b.share_no_country <= 0.75)])
tab = pd.crosstab(known.community, known.group).reindex(columns=list(FIELD_GROUPS), fill_value=0)
tab = tab[tab.sum(axis=1) >= BIG]
share = tab.div(tab.sum(axis=1), axis=0)
arts = known[known.group == "Arts & humanities"].community
where = arts.value_counts(normalize=True)
out = {
    "researchers": int(len(w)), "with_field": int(len(known)), "fallback": int(w.fallback.sum()),
    "groups": {g: int((known.group == g).sum()) for g in FIELD_GROUPS},
    "communities": {str(k): {"name": names.get(k), "national": k in national, "with_field": int(tab.loc[k].sum()),
                             "shares": {g: round(float(v), 4) for g, v in share.loc[k].items()},
                             "groups_at_5_percent": int((share.loc[k] >= 0.05).sum()),
                             "top_share": round(float(share.loc[k].max()), 4)} for k in tab.index},
    "arts_and_humanities": {"researchers": int(len(arts)), "in_national": round(float(arts.isin(national).mean()), 4),
                            "by_community": {str(k): {"name": names.get(k), "share": round(float(v), 4)}
                                             for k, v in where.head(8).items()}},
}
save_result("map_fields", out)
for k, row in out["communities"].items():
    log(k, row["name"], "national" if row["national"] else "", row["groups_at_5_percent"], row["top_share"])
log("arts and humanities:", out["arts_and_humanities"])
