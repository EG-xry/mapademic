"""A6b: catalogue of OpenAlex record problems and how common they are.

Works: xpac records, excluded types, hyperauthorship (works with > 50 authors
and the coauthor pairs they would create). Authors: placeholder names, uncited
bulk records, works_count inflation against the deduplicated, quality-filtered
count. Topics: cancer topics filed outside Oncology; the fallback topic given
wholesale to non-English records.
"""
import glob
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from pipeline.filter_authors import PLACEHOLDER_NAMES, is_person_sql  # noqa: E402
from pipeline.works import EXCLUDED_TYPES, id_key_sql, quality_predicate, qualifying_works_sql  # noqa: E402

from common import CACHE, D, MIN_WORKS, ND, con, log, save_result  # noqa: E402
from identities import file_identity, same_identity  # noqa: E402


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=1))
    temporary.replace(path)


def configure_quality_connection(db, cache_dir):
    db.execute("SET memory_limit='6GB'")
    db.execute("SET threads=4")
    db.execute("SET preserve_insertion_order=false")
    spill = Path(cache_dir) / "duck_quality"
    spill.mkdir(parents=True, exist_ok=True)
    if db.execute("SELECT current_setting('temp_directory')").fetchone()[0] != str(spill):
        db.execute(f"SET temp_directory='{spill}'")
    db.execute("SET max_temp_directory_size='20GB'")


def qualifying_author_census(db, works_glob, authors_glob, nodes, cache_dir, min_works):
    """Reproduce the eligibility census without loading author IDs into Python.

    Every distinct, non-null author ID on a qualifying work counts once, with
    no team-size cap. Work rows are not deduplicated here: the result remains
    conditional on the final ALL-qualifying-work-ID uniqueness check, including
    solo-authored works. The compact count cache is tied to source identities,
    the shared quality predicate and a file checksum. Eligibility is recomputed
    from the author extract each time using the pipeline's exact person rule.
    """
    cache_dir, nodes = Path(cache_dir), Path(nodes)
    cache_dir.mkdir(parents=True, exist_ok=True)
    work_files = sorted(glob.glob(str(works_glob)))
    author_files = sorted(glob.glob(str(authors_glob)))
    if not work_files or not author_files:
        raise FileNotFoundError("Author census requires both work and author extracts")
    configure_quality_connection(db, cache_dir)
    counts = cache_dir / "qualifying_author_counts.parquet"
    manifest_path = cache_dir / "qualifying_author_counts.provenance.json"
    identity = {"version": 1, "works_inputs": [file_identity(p) for p in work_files],
                "query": qualifying_works_sql("INPUT", dedupe=False),
                "author_deduplication": "list_distinct(author_ids); null author keys omitted"}
    fingerprint = json_digest(identity)
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    # Counts made before the project moved stay valid: their record may hold paths from the old folder, and
    # results and the population record refer to its fingerprint.
    reusable = (previous.get("fingerprint") == json_digest(previous.get("identity"))
                and same_identity(previous.get("identity"), identity) and counts.exists()
                and previous.get("cache", {}).get("sha256") == file_digest(counts))
    if not reusable:
        log("qualifying-author census: aggregating all qualifying works; DuckDB 6GB, spill at most 20GB")
        query = qualifying_works_sql(str(works_glob), dedupe=False)
        temporary = counts.with_name(counts.name + ".tmp")
        temporary.unlink(missing_ok=True)
        rows = db.execute(f"""COPY (
            SELECT author_key, count(*)::BIGINT AS works_kept
            FROM (SELECT unnest(list_distinct(author_ids)) AS author_key FROM ({query}))
            WHERE author_key IS NOT NULL GROUP BY author_key
        ) TO '{temporary}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)""").fetchone()[0]
        temporary.replace(counts)
        previous = {"fingerprint": fingerprint, "identity": identity,
                    "cache": {"path": str(counts.resolve()), "author_rows": int(rows),
                              "sha256": file_digest(counts)},
                    "work_id_uniqueness": "Required final validation across ALL qualifying works; not established by this census"}
        atomic_json(manifest_path, previous)
    else:
        fingerprint = previous["fingerprint"]
        log("qualifying-author census: reused source-fingerprinted, checksum-verified author counts")

    population, authorships, one, eligible, eligible_authorships = db.execute(f"""
        SELECT count(*), sum(works_kept), count(*) FILTER (WHERE works_kept=1),
               count(*) FILTER (WHERE works_kept >= {int(min_works)}),
               sum(works_kept) FILTER (WHERE works_kept >= {int(min_works)})
        FROM read_parquet('{counts}')""").fetchone()
    assert population > 0 and authorships > 0 and eligible > 0
    assert population == previous["cache"]["author_rows"]
    bands = {}
    for band, records, works in db.execute(f"""SELECT CASE WHEN works_kept=1 THEN '1'
            WHEN works_kept <= 4 THEN '2_to_4' WHEN works_kept <= 14 THEN '5_to_14' ELSE '15_plus' END AS band,
            count(*), sum(works_kept) FROM read_parquet('{counts}') GROUP BY band""").fetchall():
        bands[band] = {"author_ids": int(records), "authorships": int(works),
                       "share_of_author_ids": records / population, "share_of_authorships": works / authorships}
    for band in ("1", "2_to_4", "5_to_14", "15_plus"):
        bands.setdefault(band, {"author_ids": 0, "authorships": 0, "share_of_author_ids": 0.0, "share_of_authorships": 0.0})

    # A right join lets the compact candidate table be the build side. Only
    # integer IDs/counts/status enter the temporary table, never display names.
    db.execute("DROP TABLE IF EXISTS census_eligible")
    db.execute(f"""CREATE TEMP TABLE census_eligible AS
        SELECT c.author_key, c.works_kept,
               CASE WHEN a.id IS NULL THEN 0 WHEN {is_person_sql('c.works_kept')} THEN 2 ELSE 1 END::TINYINT AS status
        FROM read_parquet('{authors_glob}') a RIGHT JOIN
             (SELECT * FROM read_parquet('{counts}') WHERE works_kept >= {int(min_works)}) c
             ON {id_key_sql('a.id')} = c.author_key""")
    try:
        joined, missing, excluded, retained, retained_authorships = db.execute("""SELECT count(*),
            count(*) FILTER (WHERE status=0), count(*) FILTER (WHERE status=1), count(*) FILTER (WHERE status=2),
            sum(works_kept) FILTER (WHERE status=2) FROM census_eligible""").fetchone()
        assert joined == eligible, "duplicate author records produce multiple matches for an eligible ID"
        assert missing + excluded + retained == eligible
        # Full outer join verifies both directions, not merely the final size.
        node_rows, matched_eligible_rows, missing_nodes, unexpected_nodes, count_mismatches = db.execute(f"""
            SELECT count(*) FILTER (WHERE n.id IS NOT NULL), count(*) FILTER (WHERE e.author_key IS NOT NULL),
                   count(*) FILTER (WHERE n.id IS NULL), count(*) FILTER (WHERE e.author_key IS NULL),
                   count(*) FILTER (WHERE n.id IS NOT NULL AND e.author_key IS NOT NULL
                                    AND n.works_kept IS DISTINCT FROM e.works_kept)
            FROM read_parquet('{nodes}') n FULL JOIN (SELECT * FROM census_eligible WHERE status=2) e
              ON {id_key_sql('n.id')} = e.author_key""").fetchone()
        assert missing_nodes == unexpected_nodes == count_mismatches == 0, (
            f"census disagrees with delivered nodes: missing={missing_nodes}, unexpected={unexpected_nodes}, works={count_mismatches}")
        assert node_rows == matched_eligible_rows == retained, "retained node IDs are not unique"
    finally:
        db.execute("DROP TABLE census_eligible")

    provenance = {"version": 1, "counts_fingerprint": fingerprint,
                  "authors_inputs": [file_identity(p) for p in author_files], "nodes": file_identity(nodes),
                  "min_works": int(min_works), "person_predicate": is_person_sql("c.works_kept")}
    population_fingerprint = json_digest(provenance)
    atomic_json(cache_dir / "qualifying_author_census.provenance.json",
                {"fingerprint": population_fingerprint, "identity": provenance})
    out = {"author_ids_with_qualifying_work": int(population), "authorships_total": int(authorships),
           "exactly_one_work": {"author_ids": int(one), "authorships": int(one), "share_of_author_ids": one / population},
           "work_count_bands": bands,
           "threshold": {"min_works": int(min_works), "eligible_author_ids": int(eligible),
                         "eligible_authorships": int(eligible_authorships), "missing_author_records": int(missing),
                         "excluded_nonperson_records": int(excluded), "retained_author_ids": int(retained),
                         "retained_authorships": int(retained_authorships), "retained_author_share": retained / population,
                         "retained_authorship_share": retained_authorships / authorships},
           "validation": {"node_works_kept_exact": True, "retained_id_set_exact": True,
                          "eligible_ids_partition_exact": True, "eligible_author_records_unique": True},
           "counts_fingerprint": fingerprint, "population_fingerprint": population_fingerprint,
           "count_cache_sha256": previous["cache"]["sha256"],
           "definition": "Distinct non-null author IDs per qualifying work, without a team-size cap. Coverage uses retained researchers' qualifying authorships over all qualifying authorships, including IDs without author metadata.",
           "work_id_uniqueness_dependency": "Pending validate_min5.py check of ALL qualifying work IDs, including solo-authored works; counts are not final if duplicate qualifying work rows exist."}
    log("qualifying-author census:", {k: v for k, v in out.items() if k in ("author_ids_with_qualifying_work", "authorships_total", "threshold")})
    return out

def main():
    c = con()
    configure_quality_connection(c, CACHE)
    W = f"read_parquet('{D}/works_authorships/*/*.parquet')"
    AU = f"read_parquet('{D}/authors_extract/*/*.parquet')"
    T = f"read_parquet('{D}/snapshot/topics/*/*.parquet')"
    out = {}

    log("works")
    types = ", ".join(f"'{t}'" for t in EXCLUDED_TYPES)
    quality = quality_predicate()
    r = c.execute(f"""SELECT count(*), count(*) FILTER (WHERE is_xpac), count(*) FILTER (WHERE type IN ({types})),
                             count(*) FILTER (WHERE is_paratext), count(*) FILTER (WHERE n_authors > 50),
                             sum(n_authors) FILTER (WHERE n_authors > 50),
                             sum(n_authors::HUGEINT * (n_authors - 1) / 2) FILTER (WHERE n_authors BETWEEN 2 AND 50),
                             sum(n_authors::HUGEINT * (n_authors - 1) / 2) FILTER (WHERE n_authors > 50),
                             max(n_authors),
                             count(*) FILTER (WHERE {quality}),
                             count(*) FILTER (WHERE ({quality}) AND n_authors > 50),
                             max(n_authors) FILTER (WHERE {quality}),
                             sum(n_authors::HUGEINT * (n_authors - 1) / 2)
                                 FILTER (WHERE ({quality}) AND n_authors BETWEEN 2 AND 50),
                             sum(n_authors::HUGEINT * (n_authors - 1) / 2)
                                 FILTER (WHERE ({quality}) AND n_authors > 50)
                      FROM {W}""").fetchone()
    out["works"] = {"total": r[0], "xpac": r[1], "excluded_types": r[2], "paratext": r[3], "over_50_authors": r[4],
                    "authorships_in_over_50": int(r[5]), "pairs_from_2_to_50": int(r[6]), "pairs_from_over_50": int(r[7]),
                    "max_authors": r[8]}
    out["qualifying_works"] = {
        "total": r[9], "over_50_authors": r[10], "max_authors": r[11],
        "pair_occurrences_all_authors_2_to_50": int(r[12]),
        "pair_occurrences_all_authors_over_50": int(r[13]),
        "definition": "Quality-filtered works across the full extract, before selecting eligible researchers; pair occurrences count listed authors and are not distinct edges.",
        "work_id_uniqueness": "Checked separately by validate_min5.py; these aggregate scans do not deduplicate."}
    out["works"]["types_excluded_breakdown"] = dict(c.execute(f"""SELECT type, count(*) FROM {W} WHERE type IN ({types})
                                                                  GROUP BY 1 ORDER BY 2 DESC""").fetchall())
    log(out["works"])

    log("qualifying author eligibility and coverage")
    out["qualifying_author_census"] = qualifying_author_census(
        c, str(D / "works_authorships/*/*.parquet"), str(D / "authors_extract/*/*.parquet"),
        ND / "nodes.parquet", CACHE, MIN_WORKS)

    log("authors")
    names = ", ".join(f"'{n}'" for n in PLACEHOLDER_NAMES)
    r = c.execute(f"""SELECT count(*),
                             count(*) FILTER (WHERE regexp_replace(lower(trim(display_name)), '[\\s\\p{{P}}\\p{{S}}]', '', 'g') IN ({names})),
                             count(*) FILTER (WHERE coalesce(cited_by_count, 0) = 0 AND works_count >= 1000),
                             count(*) FILTER (WHERE works_count >= {MIN_WORKS})
                      FROM {AU}""").fetchone()
    out["authors"] = {"records": r[0], "placeholder_names": r[1], "uncited_bulk_1000_plus_works": r[2], "raw_works_at_least_min": r[3], "min_works": MIN_WORKS}
    r = c.execute(f"""SELECT count(*), median(works_count / greatest(works_kept, 1)),
                             avg(CASE WHEN works_count >= 2 * works_kept THEN 1 ELSE 0 END),
                             avg(CASE WHEN works_count >= 4 * works_kept THEN 1 ELSE 0 END)
                      FROM read_parquet('{ND}/nodes.parquet')""").fetchone()
    out["works_count_inflation"] = {"mapped_authors": r[0], "median_raw_over_kept": float(r[1]),
                                    "share_raw_at_least_2x": float(r[2]), "share_raw_at_least_4x": float(r[3])}
    log(out["authors"], out["works_count_inflation"])

    log("topics")
    cancer = c.execute(f"""SELECT display_name, subfield.display_name AS subfield, works_count FROM {T}
                           WHERE regexp_matches(lower(display_name), 'cancer|carcinoma|tumou?r|leuk|lymphoma|melanoma|glioma|sarcoma|myeloma|neoplas|oncolog')
                           ORDER BY works_count DESC""").fetchdf()
    cancer.to_csv(CACHE / "a6_cancer_topics.csv", index=False)
    by_sub = cancer.groupby("subfield")["works_count"].agg(["count", "sum"]).sort_values("sum", ascending=False)
    out["cancer_topics"] = {"topics": int(len(cancer)), "works": int(cancer.works_count.sum()),
                            "outside_oncology_topics": int((cancer.subfield != "Oncology").sum()),
                            "outside_oncology_works": int(cancer.loc[cancer.subfield != "Oncology", "works_count"].sum()),
                            "by_subfield": {k: {"topics": int(v["count"]), "works": int(v["sum"])} for k, v in by_sub.iterrows()}}
    fb = c.execute(f"""SELECT count(*) FILTER (WHERE topics[1].display_name = 'Legal and Regulatory Analysis'), count(*)
                       FROM {AU} WHERE works_count >= {MIN_WORKS}""").fetchone()
    out["fallback_topic"] = {"authors_min_plus_with_it_first": fb[0], "authors_min_plus": fb[1], "share": fb[0] / fb[1],
                            "min_works": MIN_WORKS}
    log(out["cancer_topics"]["outside_oncology_topics"], "of", out["cancer_topics"]["topics"], "cancer topics outside Oncology;",
        out["fallback_topic"])
    save_result("a6_quality", out)

    c.close()


if __name__ == "__main__":
    main()
