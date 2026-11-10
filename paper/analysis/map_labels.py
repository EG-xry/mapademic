"""Names for the paper network's communities, by the map's own rules (pipeline/regions.py), for the deposit and the
figures. Run after map_recolour.py.

A community of at least 1,000 members is named by its field(s), then its country when the country is a reason for
the cluster ("Medicine (Italy)", but "Physics & Astronomy" for an international field), from the shares of all its
members, not only those drawn on the map: fields leave out records on OpenAlex's fallback topic, countries use the
members with a known institution country, against shares among communities of at least 1,000 members. Script communities keep the
names they are given in common.SCRIPT_COMMUNITIES.

Writes cache/community_labels.json ({community: name}, for export_deposit.py) and cache/map/regions.json, the map's
label file for the communities drawn in colour (positions: the median map position of their members on the map,
cache/map/positions.parquet), with a short name for the ties figure.
"""
import json
import sys

import duckdb
import pandas as pd

from common import CACHE, D, ROOT, SCRIPT_COMMUNITIES, log

sys.path.insert(0, str(ROOT))
from pipeline.regions import FALLBACK_TOPIC, country_label, field_label  # noqa: E402

MIN_MEMBERS, OVERVIEW = 1000, 30
SCRIPTS = ("Chinese characters", "Japanese script", "Japanese script (small)", "Mixed CJK script", "Arabic script")

def grouped_shares(frame, key):
    """Small aggregate rows, descending by count with explicit first-row tie breaking."""
    result = {}
    for community, rows in frame.groupby("community", sort=False):
        total = int(rows.n.sum())
        result[int(community)] = [(value, int(count) / total) for value, count in zip(rows[key], rows.n)]
    return result


def community_statistics(attrs_path, authors_path, min_members=MIN_MEMBERS):
    """Read aggregate counts only; exact fallback strings never form a Python set.

    node_idx is the original row order. It preserves community-size tie order.
    Within-community category ties use that explicit order, rather than the
    incidental grouped tie order of a particular pandas release.
    """
    db = duckdb.connect(config={"memory_limit": "4GB", "threads": "2",
                                "temp_directory": str(attrs_path.parent / "duck_map_labels"),
                                "max_temp_directory_size": "8GB"})
    try:
        db.execute("""CREATE TEMP TABLE big AS
            SELECT community, count(*) AS n, min(node_idx) AS first_row,
                   coalesce(bool_or(is_dust), false) AS dust
            FROM read_parquet(?) WHERE community IS NOT NULL GROUP BY community HAVING count(*) >= ?""",
            [str(attrs_path), min_members])
        sizes = db.execute("SELECT * FROM big ORDER BY n DESC, first_row").fetchdf()
        country_counts = db.execute("""SELECT a.community, a.country, count(*) AS n, min(a.node_idx) AS first_row
            FROM read_parquet(?) a JOIN big b USING (community)
            WHERE a.country IS NOT NULL GROUP BY a.community, a.country
            ORDER BY a.community, n DESC, first_row""", [str(attrs_path)]).fetchdf()
        field_counts = db.execute("""SELECT a.community, a.field, count(*) AS n, min(a.node_idx) AS first_row
            FROM read_parquet(?) a JOIN big b USING (community)
            WHERE a.field IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM read_parquet(?) f
                WHERE f.id = a.id AND f.topics[1].display_name = ?)
            GROUP BY a.community, a.field ORDER BY a.community, n DESC, first_row""",
            [str(attrs_path), str(authors_path), FALLBACK_TOPIC]).fetchdf()
    finally:
        db.close()
    world_counts = country_counts.groupby("country", sort=False).n.sum()
    world = world_counts / world_counts.sum()
    return (sizes.set_index("community").n, world.to_dict(), grouped_shares(field_counts, "field"),
            grouped_shares(country_counts, "country"), sizes.set_index("community").dust.to_dict())


def community_names(big, world, fields, countries, dust, scripts=SCRIPT_COMMUNITIES):
    script = {v: k for k, v in scripts.items() if k in SCRIPTS}
    labels, info = {}, {}
    for c in big.index:
        fl = field_label(fields[c]) if c in fields else "Unclassified"
        cl = [(cc, s, s / world.get(cc, 1e-9)) for cc, s in countries.get(c, [])]
        country = country_label(cl)
        name = script.get(c) or (f"{fl} ({country})" if country else fl)
        if name in labels.values():
            k = 2
            while f"{name} {k}" in labels.values():
                k += 1
            name = f"{name} {k}"
        labels[int(c)] = name
        short = (country.split(",")[0] if country else fl) if c not in script else name
        info[int(c)] = {"name": name, "kind": "script" if c in script else ("national" if country else "field"),
                        "short": short.replace(" & ", "\n& ") if len(short) > 14 else short,
                        "members": int(big[c]), "dust": bool(dust[c])}
    return labels, info


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=1, ensure_ascii=False))
    temporary.replace(path)


def main():
    log("community labels: bounded count aggregation (DuckDB 4GB, 2 threads)")
    stats = community_statistics(CACHE / "node_attrs.parquet", D / "authors_extract/*/*.parquet")
    big = stats[0]
    labels, info = community_names(*stats)
    atomic_json(CACHE / "community_labels.json", labels)
    log(f"{len(labels)} communities named -> community_labels.json")
    pos = pd.read_parquet(CACHE / "map" / "positions.parquet").set_index("community")
    coloured = [c for c in big.index if not info[int(c)]["dust"] and c in pos.index]
    regions = []
    for rank, c in enumerate(sorted(coloured, key=lambda c: -big[c]), start=1):
        r = info[int(c)]
        regions.append({"name": r["name"], "short": r["short"], "kind": r["kind"], "xw": float(pos.xw[c]),
                        "yw": float(pos.yw[c]), "members": r["members"], "on_map": int(pos.on_map[c]), "rank": rank,
                        "community": int(c), "zmin": 2 if rank <= OVERVIEW else 4})
    atomic_json(CACHE / "map" / "regions.json", regions)
    for r in regions[:OVERVIEW]:
        log(f"{r['rank']:>3} {r['community']:>6} {r['members']:>9,} {r['on_map']:>9,}  {r['name']}")


if __name__ == "__main__":
    main()
