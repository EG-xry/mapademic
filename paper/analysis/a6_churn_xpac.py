"""A6c follow-up: is the drop in works counts between the snapshot and the live API real, or xpac?

The snapshot's author works_count includes expansion-pack (xpac) deposits; the live API hides them by
default. For the sampled authors that are still alive, count their works in the snapshot with and without
xpac, and compare the snapshot's works_count with the first and the live works_count with the second.
"""
from common import CACHE, D, con, log, save_result

c = con()
df = c.execute(f"""
    WITH s AS (SELECT aid, CAST(substr(aid, 2) AS BIGINT) AS a, works_count, works_live, cited_by_count, cited_live
               FROM read_csv('{CACHE}/a6_churn_alive.csv')),
    w AS (SELECT a, count(*) AS works_all, count(*) FILTER (WHERE NOT coalesce(is_xpac, FALSE)) AS works_noxpac
          FROM (SELECT unnest(author_ids) AS a, is_xpac FROM read_parquet('{D}/works_authorships/*/*.parquet'))
          WHERE a IN (SELECT a FROM s) GROUP BY a)
    SELECT s.*, coalesce(works_all, 0) AS works_all, coalesce(works_noxpac, 0) AS works_noxpac FROM s LEFT JOIN w USING (a)
""").fetchdf()
df.to_csv(CACHE / "a6_churn_xpac.csv", index=False)


def rel(x, y):
    return (x - y) / y.clip(lower=1)


snap_vs_all = rel(df.works_count, df.works_all)
live_vs_noxpac = rel(df.works_live, df.works_noxpac)
live_vs_snap = rel(df.works_live, df.works_count)
out = {"authors": int(len(df)), "xpac_share_of_their_works": float(1 - df.works_noxpac.sum() / df.works_all.sum()),
       "authors_with_any_xpac": float((df.works_all > df.works_noxpac).mean()),
       "snapshot_count_vs_snapshot_works_incl_xpac": {"median": float(snap_vs_all.median()), "within_5pct": float((snap_vs_all.abs() <= 0.05).mean())},
       "live_count_vs_snapshot_works_excl_xpac": {"median": float(live_vs_noxpac.median()), "within_5pct": float((live_vs_noxpac.abs() <= 0.05).mean()),
                                                   "share_down_10pct": float((live_vs_noxpac < -0.1).mean()), "share_up_50pct": float((live_vs_noxpac > 0.5).mean())},
       "live_count_vs_snapshot_count": {"median": float(live_vs_snap.median()), "share_down_10pct": float((live_vs_snap < -0.1).mean())}}
save_result("a6_churn_xpac", out)
log(out)
