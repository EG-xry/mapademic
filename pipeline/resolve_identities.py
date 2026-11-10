"""Conservatively partition mixed author profiles before coauthor aggregation.

This stage consumes deposited authorship evidence, never profile-level ORCIDs.
It preserves untouched OpenAlex IDs and assigns deterministic application IDs
(`M<number>`, represented by negative BIGINT keys) to repaired identities.
Neither a different topic nor a disconnected coauthor component alone is a
split. Independent, repeated, contemporaneous affiliation and field evidence
is required. Records that cannot be assigned after a split are kept visible
in a quarantined partition rather than reconnecting the separated identities.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.config import apply_resource_limits, data_dir, spill_dir
from pipeline.sql import sql_literal
from pipeline.works import quality_predicate

COLLECTION_TITLES = {"book reviews", "reviews", "contributors", "editorial board"}
MAX_CLUSTER_COAUTHORS = 20


def canonical_orcid(value: str | None) -> str | None:
    value = re.sub(r"^(?:https?://(?:www\.)?orcid\.org/|orcid:)", "", (value or "").strip(), flags=re.I).upper()
    digits = value.replace("-", "")
    if not re.fullmatch(r"\d{15}[\dX]", digits):
        return None
    total = 0
    for d in digits[:-1]:
        total = (total + int(d)) * 2
    check = (12 - total % 11) % 11
    if digits[-1] != ("X" if check == 10 else str(check)):
        return None
    return "-".join(digits[i:i + 4] for i in range(0, 16, 4))


@dataclass(frozen=True)
class Authorship:
    work_id: int
    author_id: int
    raw_name: str
    raw_orcid: str | None
    institution_ids: tuple[str, ...]
    field_id: str | None
    year: int | None
    coauthors: tuple[int, ...]
    citations: int = 0
    weak: bool = False


@dataclass
class PartitionPlan:
    assignments: dict[int, str]
    groups: dict[str, dict]


class _Union:
    def __init__(self, count):
        self.parent = list(range(count))

    def root(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def join(self, a, b):
        a, b = self.root(a), self.root(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _summary(records: list[Authorship], minimum: int) -> dict:
    strong = [r for r in records if not r.weak]
    institutions = Counter(i for r in strong for i in set(r.institution_ids))
    fields = Counter(r.field_id for r in strong if r.field_id)
    return {
        "records": records,
        "institutions": {i for i, n in institutions.items() if n >= minimum},
        "fields": {f for f, n in fields.items() if n >= minimum},
        "years": {r.year for r in strong if r.year is not None},
        "coauthors": {a for r in strong for a in r.coauthors},
        "orcids": {o for r in strong if (o := canonical_orcid(r.raw_orcid))},
        "strong_count": len(strong),
    }


def partition_profile(records: list[Authorship], min_component_works: int = 2) -> PartitionPlan:
    """Return stable source-scoped partitions for one profile's authorships.

    Conservative automatic split: repeated evidence in different institutions
    AND fields, with overlapping years and disjoint coauthor neighborhoods.
    A trusted deposited ORCID is an identity anchor; a profile-level ORCID
    never propagates to unmarked papers. Career moves and interdisciplinary
    work remain intact when the required contradictions are absent.
    """
    if not records:
        return PartitionPlan({}, {})
    records = sorted(records, key=lambda r: r.work_id)
    source = records[0].author_id
    union = _Union(len(records))
    owners: dict[tuple, int] = {}
    for i, r in enumerate(records):
        if r.weak:
            continue
        tokens = [("institution", x) for x in r.institution_ids]
        tokens += [("coauthor", x) for x in r.coauthors]
        # Within one existing profile, keep compatible topic records together.
        # Topic is never a sufficient reason to create separate identities.
        if r.field_id:
            tokens.append(("field", r.field_id))
        for token in tokens:
            if token in owners:
                union.join(i, owners[token])
            else:
                owners[token] = i
    # Establish independent structural neighborhoods before considering ORCID.
    # A single incorrectly deposited identifier must not bridge two otherwise
    # well-supported identities. Repeated anchors in both neighborhoods still
    # protect one researcher's joint appointments and different fields.
    initial = defaultdict(list)
    for i, r in enumerate(records):
        initial[union.root(i)].append((i, r))
    anchors = defaultdict(list)
    for root, members in initial.items():
        counts = Counter(o for _, r in members if not r.weak and (o := canonical_orcid(r.raw_orcid)))
        for oid, count in counts.items():
            if count >= min_component_works:
                anchors[oid].append(root)
    for roots in anchors.values():
        for other in roots[1:]:
            union.join(roots[0], other)
    components = defaultdict(list)
    for i, r in enumerate(records):
        components[union.root(i)].append(r)
    summaries = []
    contradictory = []
    for root, rs in components.items():
        counts = Counter(o for r in rs if not r.weak and (o := canonical_orcid(r.raw_orcid)))
        retained = []
        for r in rs:
            oid = canonical_orcid(r.raw_orcid)
            if (oid and counts[oid] < min_component_works and oid in anchors
                    and all(union.root(anchor) != root for anchor in anchors[oid])):
                contradictory.append(r)
            else:
                retained.append(r)
        if retained:
            summaries.append(_summary(retained, min_component_works))
    summaries.extend(_summary([r], min_component_works) for r in contradictory)
    supported = set()
    for i, a in enumerate(summaries):
        if a["strong_count"] < min_component_works or not a["fields"] or not a["institutions"]:
            continue
        for j in range(i):
            b = summaries[j]
            if (b["strong_count"] >= min_component_works and b["fields"] and b["institutions"]
                    and not a["fields"] & b["fields"]
                    and not a["institutions"] & b["institutions"]
                    and not a["coauthors"] & b["coauthors"]
                    and a["years"] & b["years"]):
                supported.update((i, j))
    if len(supported) < 2:
        token = f"source:{source}"
        return PartitionPlan({r.work_id: token for r in records}, {token: {
            "status": "unchanged", "records": records, "source_ids": [source],
            "orcid": next(iter(summaries[0]["orcids"])) if len(summaries) == 1 and len(summaries[0]["orcids"]) == 1 else None,
        }})
    groups, assignments = {}, {}
    for i in sorted(supported):
        s = summaries[i]
        # A scoped ORCID anchor remains stable when later works arrive. When
        # the same ORCID occurs in contradictory clusters, include a work seed.
        unique = len(s["orcids"]) == 1 and sum(bool(s["orcids"] & other["orcids"]) for other in summaries) == 1
        anchor = "orcid:" + next(iter(s["orcids"])) if unique else "work:" + str(min(r.work_id for r in s["records"]))
        token = f"source:{source}:{anchor}"
        groups[token] = {"status": "split", "records": list(s["records"]), "source_ids": [source],
                         "orcid": next(iter(s["orcids"])) if len(s["orcids"]) == 1 else None}
        assignments.update((r.work_id, token) for r in s["records"])
    quarantine = []
    for i, s in enumerate(summaries):
        if i in supported:
            continue
        for r in s["records"]:
            oid = canonical_orcid(r.raw_orcid)
            matches = [t for t, g in groups.items() if oid and g["orcid"] == oid
                       and not r.weak and (not r.field_id or any(x.field_id == r.field_id for x in g["records"]))]
            if len(matches) == 1:
                groups[matches[0]]["records"].append(r)
                assignments[r.work_id] = matches[0]
            else:
                quarantine.append(r)
    if quarantine:
        token = f"source:{source}:quarantine"
        groups[token] = {"status": "quarantined", "records": quarantine, "source_ids": [source], "orcid": None}
        assignments.update((r.work_id, token) for r in quarantine)
    return PartitionPlan(assignments, groups)


def _name_words(name):
    normalized = unicodedata.normalize("NFKD", name or "")
    normalized = "".join(c for c in normalized if not unicodedata.category(c).startswith("M"))
    return re.findall(r"[^\W\d_]+", normalized.casefold())


def _name_key(name):
    words = _name_words(name)
    return (words[0], words[-1]) if len(words) >= 2 else None


def _names_compatible(a, b):
    x, y = _name_words(a), _name_words(b)
    if len(x) < 2 or len(y) < 2 or x[-1] != y[-1]:
        return False
    compatible = lambda left, right: left == right or (min(len(left), len(right)) == 1 and left[0] == right[0])
    return compatible(x[0], y[0]) and all(compatible(left, right) for left, right in zip(x[1:-1], y[1:-1]))


def _deduplicate_bylines(records):
    # Providers sometimes repeat one byline under a full and abbreviated name.
    # Collapse only when every other deposited evidence field agrees. Different
    # affiliations or identifiers must remain a reviewable ambiguous seat.
    variants = defaultdict(list)
    for record in dict.fromkeys(records):
        variants[replace(record, raw_name="")].append(record)
    result = []
    for group in variants.values():
        representative = max(group, key=lambda r: (len(_name_words(r.raw_name)), len(r.raw_name), r.raw_name))
        if all(_names_compatible(representative.raw_name, r.raw_name) for r in group):
            result.append(representative)
        else:
            result.extend(group)
    return result


def _identity_key(token: str, registry: dict[str, int], used: dict[int, str] | None = None) -> int:
    if token in registry:
        return registry[token]
    if used is None:
        used = {v: k for k, v in registry.items()}
    attempt = 0
    while True:
        digest = hashlib.sha256((token + (f":{attempt}" if attempt else "")).encode()).digest()
        key = -(int.from_bytes(digest[:7], "big") & ((1 << 52) - 1) or 1)
        if key not in used or used[key] == token:
            registry[token] = key
            used[key] = token
            return key
        attempt += 1


def _canonical_sql(column: str) -> str:
    digits = f"upper(replace(regexp_replace(trim({column}), '^https?://(www[.])?orcid[.]org/|^orcid:', '', 'i'), '-', ''))"
    weighted = " + ".join(f"CAST(substr(d, {i + 1}, 1) AS INTEGER) * {2 ** (15 - i)}" for i in range(15))
    # CASE shields numeric casts from malformed external identifiers.
    return (f"(SELECT CASE WHEN regexp_full_match(d, '[0-9]{{15}}[0-9X]') THEN "
            f"CASE WHEN right(d, 1) = CASE WHEN (12 - (({weighted}) % 11)) % 11 = 10 THEN 'X' "
            f"ELSE CAST((12 - (({weighted}) % 11)) % 11 AS VARCHAR) END THEN "
            "substr(d, 1, 4) || '-' || substr(d, 5, 4) || '-' || substr(d, 9, 4) || '-' || substr(d, 13, 4) END END "
            f"FROM (SELECT {digits} AS d))")


def resolve_identities(works_glob: str, nodes_path: str, out_dir: Path, *,
                       min_component_works: int = 2, max_profile_works: int = 50000,
                       registry_path: Path | None = None) -> dict:
    """Write a repaired release subset without mutating the input artifacts.

    Evidence is scanned and sorted in DuckDB (which can spill to disk); Python
    receives one candidate profile at a time. Profiles above the configured
    bound are retained unchanged and counted in diagnostics. A persisted token
    registry keeps keys stable and detects/disambiguates hash collisions.
    """
    out_dir = Path(out_dir)
    if min_component_works < 2 or max_profile_works < 2:
        raise ValueError("min_component_works and max_profile_works must be at least 2")
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    apply_resource_limits(con)
    con.execute(f"SET temp_directory={sql_literal(spill_dir(out_dir))}")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"CREATE VIEW source_works AS SELECT * FROM read_parquet({sql_literal(works_glob)})")
    duplicate_works = con.execute("SELECT count(*) - count(DISTINCT work_id) FROM source_works").fetchone()[0]
    if duplicate_works:
        conflict = con.execute("""SELECT work_id FROM source_works w
            WHERE work_id IN (SELECT work_id FROM source_works GROUP BY work_id HAVING count(*) > 1)
            GROUP BY work_id HAVING count(DISTINCT to_json(w)) > 1 LIMIT 1""").fetchone()
        if conflict:
            con.close()
            raise ValueError(f"work {conflict[0]} has conflicting versions; use one release-consistent extract")
    con.execute("CREATE VIEW works AS SELECT * FROM source_works" +
                (" QUALIFY row_number() OVER (PARTITION BY work_id ORDER BY work_id) = 1" if duplicate_works else ""))
    column_types = {row[0]: row[1] for row in con.execute("DESCRIBE works").fetchall()}
    columns = set(column_types)
    if "authorship_evidence" not in columns:
        raise ValueError("works extract lacks authorship_evidence; re-extract with deposited byline evidence")
    if "source_author_ids" in columns:
        raise ValueError("input is already identity-resolved; rerun against the original works and persisted registry")
    evidence_type = column_types["authorship_evidence"]
    institution_names = "e.institution_names" if re.search(r"\binstitution_names\b", evidence_type) else "[]::VARCHAR[]"
    field_name = "e.field_name" if re.search(r"\bfield_name\b", evidence_type) else "NULL::VARCHAR"
    con.execute(f"CREATE VIEW nodes AS SELECT *, CAST(regexp_extract(id, '(\\d+)$', 1) AS BIGINT) AS source_key FROM read_parquet({sql_literal(nodes_path)})")
    title = "lower(trim(coalesce(title, '')))" if "title" in columns else "''"
    titles = ", ".join(sql_literal(t) for t in sorted(COLLECTION_TITLES))
    con.execute(f"CREATE VIEW eligible AS SELECT *, ({title} IN ({titles}) AND n_authors >= 3) AS collection FROM works")
    con.execute(f"""CREATE TEMP TABLE evidence AS
        WITH raw AS (
          SELECT w.work_id, CASE WHEN w.collection OR w.n_authors > {MAX_CLUSTER_COAUTHORS}
                    THEN []::BIGINT[] ELSE w.author_ids END AS author_ids,
                 w.year, w.cited_by_count, w.n_authors, w.collection,
                 e.author_id, e.raw_name, e.raw_orcid, e.institution_ids, e.field_id,
                 {institution_names} AS institution_names, {field_name} AS field_name
          FROM eligible w, unnest(w.authorship_evidence) AS a(e)
          WHERE {quality_predicate()} AND list_contains(w.author_ids, e.author_id)
        ), normalized AS (SELECT * EXCLUDE(raw_orcid), {_canonical_sql('raw_orcid')} AS raw_orcid FROM raw),
        checked AS (SELECT *, count(*) OVER (PARTITION BY work_id, raw_orcid) AS orcid_occurrences FROM normalized)
        SELECT c.* EXCLUDE(raw_orcid, orcid_occurrences),
               CASE WHEN orcid_occurrences = 1 THEN raw_orcid END AS raw_orcid
        FROM checked c JOIN nodes n ON n.source_key = c.author_id
    """)
    con.execute(f"""CREATE TEMP TABLE candidates AS
        SELECT author_id FROM evidence GROUP BY author_id
        HAVING (count(*) >= {2 * min_component_works}
           AND count(DISTINCT field_id) > 1
           AND count(DISTINCT list_sort(institution_ids)) > 1)
           OR count(*) > count(DISTINCT work_id)
    """)
    registry = {}
    if registry_path:
        loaded = json.loads(Path(registry_path).read_text())
        registry = {k: int(v) for k, v in loaded.get("tokens", loaded).items()}
        if any(v >= 0 or abs(v) >= 2 ** 53 for v in registry.values()) or len(set(registry.values())) != len(registry):
            raise ValueError("identity registry contains invalid or colliding keys")
    used_keys = {v: k for k, v in registry.items()}
    groups = {}
    overrides = []
    split_profiles = oversized_profiles = ambiguous_profiles = 0
    # The sort is performed in DuckDB and may spill; no all-authorship Python list.
    reader = con.sql("SELECT * FROM evidence WHERE author_id IN (SELECT author_id FROM candidates) ORDER BY author_id, work_id").to_arrow_reader(8192)
    current = None
    records = []
    oversized = False

    def finish():
        nonlocal split_profiles, oversized_profiles, ambiguous_profiles
        if current is None:
            return
        if oversized:
            oversized_profiles += 1
            return
        unique = _deduplicate_bylines(records)
        repeated = {wid for wid, count in Counter(r.work_id for r in unique).items() if count > 1}
        clean = [r for r in unique if r.work_id not in repeated]
        plan = partition_profile(clean, min_component_works)
        if repeated:
            ambiguous_profiles += 1
            # Same source ID on distinct bylines: the compact work format
            # cannot identify the individual seat. Do not choose one person.
            ambiguous = [r for r in unique if r.work_id in repeated]
            if len(plan.groups) == 1:
                # Preserve clean collaborations even if a few publications
                # contain unresolved duplicate seats. This retains the source
                # profile rather than claiming to identify a new person.
                next(iter(plan.groups.values()))["status"] = "retained"
            token = f"source:{current}:quarantine"
            plan.groups[token] = {"status": "quarantined", "records": ambiguous,
                                  "source_ids": [current], "orcid": None}
            plan.assignments.update((r.work_id, token) for r in ambiguous)
        if len(plan.groups) < 2 and not repeated:
            return
        split_profiles += int(len(plan.groups) > 1 and not any(g["status"] == "retained" for g in plan.groups.values()))
        for token, group in plan.groups.items():
            group["key"] = current if group["status"] == "retained" else _identity_key(token, registry, used_keys)
            # Keep only the repair delta; complete publication records can be
            # rejoined from DuckDB when writing profile metadata.
            group["records"] = []
            groups[token] = group
        overrides.extend((wid, current, groups[token]["key"], token) for wid, token in plan.assignments.items())

    for batch in reader:
        for row in batch.to_pylist():
            if current != row["author_id"]:
                finish()
                current, records, oversized = row["author_id"], [], False
            if len(records) >= max_profile_works:
                oversized = True
                continue
            records.append(Authorship(row["work_id"], current, row["raw_name"] or "", row["raw_orcid"],
                                      tuple(row["institution_ids"] or []), row["field_id"], row["year"],
                                      tuple(a for a in row["author_ids"] if a != current), row["cited_by_count"] or 0,
                                      bool(row["collection"] or row["n_authors"] > MAX_CLUSTER_COAUTHORS)))
    finish()
    override_schema = pa.schema([("work_id", pa.int64()), ("source_author_id", pa.int64()),
                                 ("identity_key", pa.int64()), ("token", pa.string())])
    con.register("overrides", pa.Table.from_pylist([dict(zip(override_schema.names, r)) for r in overrides], schema=override_schema))
    con.execute("""CREATE TEMP TABLE memberships AS
        WITH flat AS MATERIALIZED (
          SELECT DISTINCT w.work_id, a.source_author_id
          FROM eligible w, unnest(list_distinct(w.author_ids)) AS a(source_author_id))
        SELECT f.work_id, f.source_author_id, coalesce(o.identity_key, f.source_author_id) AS identity_key,
               o.token FROM flat f LEFT JOIN overrides o
                 ON o.work_id = f.work_id AND o.source_author_id = f.source_author_id
    """)
    # A source profile that has been split must never silently reconnect its
    # partitions through a quality-excluded work lacking evidence.
    affected = {source for _, source, _, _ in overrides}
    for source in sorted(affected):
        missing = con.execute("SELECT work_id FROM memberships WHERE source_author_id = ? AND token IS NULL", [source]).fetchall()
        if missing:
            token = f"source:{source}:quarantine"
            key = _identity_key(token, registry, used_keys)
            groups.setdefault(token, {"status": "quarantined", "records": [], "source_ids": [source], "orcid": None, "key": key})
            con.execute("UPDATE memberships SET identity_key = ?, token = ? WHERE source_author_id = ? AND token IS NULL", [key, token, source])
    # Find duplicate source profiles by deposited ORCID plus compatible name
    # and shared affiliation. Candidate comparison stays within each ORCID.
    duplicate_rows = con.execute(f"""SELECT e.author_id, e.raw_orcid,
        first(e.raw_name), list_distinct(flatten(list(e.institution_ids))) AS institutions
        FROM evidence e WHERE e.raw_orcid IS NOT NULL AND NOT e.collection
        AND e.n_authors <= {MAX_CLUSTER_COAUTHORS}
        AND e.raw_orcid IN (SELECT raw_orcid FROM evidence WHERE raw_orcid IS NOT NULL
            GROUP BY raw_orcid HAVING count(DISTINCT author_id) > 1)
        AND e.author_id NOT IN (SELECT source_author_id FROM overrides)
        GROUP BY e.author_id, e.raw_orcid HAVING count(*) >= 2
        ORDER BY e.raw_orcid, e.author_id""").fetchall()
    duplicate_groups = defaultdict(list)
    for source, oid, name, inst in duplicate_rows:
        duplicate_groups[oid].append((source, name, set(inst or [])))
    merge_sources = sorted({r[0] for r in duplicate_rows})
    con.register("merge_sources", pa.Table.from_pydict({"author_id": merge_sources}, schema=pa.schema([("author_id", pa.int64())])))
    con.execute("CREATE TEMP TABLE merge_evidence AS SELECT e.* FROM evidence e JOIN merge_sources s ON s.author_id = e.author_id")
    con.execute("CREATE INDEX merge_evidence_author ON merge_evidence(author_id)")
    merged_profiles = 0
    for oid, candidates in sorted(duplicate_groups.items()):
        if len(candidates) < 2 or len(candidates) > 1000:
            continue
        union = _Union(len(candidates))
        for i, a in enumerate(candidates):
            for j in range(i):
                b = candidates[j]
                if _names_compatible(a[1], b[1]) and a[2] & b[2]:
                    union.join(i, j)
        components = defaultdict(list)
        for i, item in enumerate(candidates):
            components[union.root(i)].append(item)
        for component in components.values():
            if len(component) < 2:
                continue
            sources = sorted(x[0] for x in component)
            # A different trusted ORCID elsewhere on a source profile makes a
            # whole-profile merge unsafe; leave that case unresolved.
            n_orcids = con.execute("SELECT count(DISTINCT raw_orcid) FROM merge_evidence WHERE author_id IN (SELECT unnest(?)) AND raw_orcid IS NOT NULL", [sources]).fetchone()[0]
            if n_orcids != 1:
                continue
            supported = True
            for source, anchor_name, _ in component:
                count = con.execute("SELECT count(*) FROM merge_evidence WHERE author_id = ?", [source]).fetchone()[0]
                if count > max_profile_works:
                    supported = False
                    break
                bylines = con.execute("""SELECT raw_name, raw_orcid, institution_ids, field_id, author_ids
                    FROM merge_evidence WHERE author_id = ?""", [source]).fetchall()
                anchors = [r for r in bylines if r[1] == oid]
                anchor_institutions = {i for r in anchors for i in (r[2] or [])}
                anchor_fields = {r[3] for r in anchors if r[3]}
                anchor_coauthors = {a for r in anchors for a in r[4] if a != source}
                for name, raw_orcid, institutions, field, authors in bylines:
                    if not _names_compatible(name, anchor_name):
                        supported = False
                        break
                    if raw_orcid is None and field and anchor_fields and field not in anchor_fields:
                        shared_institution = set(institutions or []) & anchor_institutions
                        shared_coauthor = (set(authors) - {source}) & anchor_coauthors
                        if not shared_institution and not shared_coauthor:
                            supported = False
                            break
                if not supported:
                    break
            if not supported:
                continue
            token = f"duplicate-orcid:{oid}:name:{_name_key(component[0][1])}:source:{sources[0]}"
            key = _identity_key(token, registry, used_keys)
            groups[token] = {"status": "merged", "records": [], "source_ids": sources, "orcid": oid, "key": key}
            con.execute("UPDATE memberships SET identity_key = ?, token = ? WHERE source_author_id IN (SELECT unnest(?))", [key, token, sources])
            affected.update(sources)
            merged_profiles += len(sources) - 1
    out_works = out_dir / "works.parquet"
    if Path(works_glob).resolve() == out_works.resolve() or Path(nodes_path).resolve() == (out_dir / "nodes.parquet").resolve():
        raise ValueError("identity resolution must write to a different directory from its inputs")
    con.execute("""CREATE TEMP TABLE resolved_lists AS
        SELECT w.work_id, list(m.identity_key ORDER BY array_position(w.author_ids, m.source_author_id)) AS ids
        FROM eligible w JOIN memberships m ON m.work_id = w.work_id GROUP BY w.work_id""")
    con.execute(f"""COPY (
        SELECT w.* EXCLUDE(author_ids, collection), w.author_ids AS source_author_ids,
          coalesce(r.ids, []::BIGINT[]) AS author_ids,
          CASE WHEN w.collection THEN []::BIGINT[] ELSE list_distinct(coalesce(r.ids, []::BIGINT[])) END AS edge_author_ids
        FROM eligible w LEFT JOIN resolved_lists r ON r.work_id = w.work_id
    ) TO {sql_literal(out_works)} (FORMAT PARQUET)""")
    # Restrict metadata joins and indexes to the repair delta. Repeated per-
    # identity scans of the entire snapshot would make a small correction O(N*M).
    con.execute("CREATE TEMP TABLE repaired_memberships AS SELECT * FROM memberships WHERE token IS NOT NULL")
    con.execute("CREATE INDEX repaired_memberships_key ON repaired_memberships(identity_key)")
    con.execute("""CREATE TEMP TABLE repaired_evidence AS SELECT m.identity_key, e.* FROM evidence e
        JOIN repaired_memberships m ON m.work_id = e.work_id AND m.source_author_id = e.author_id""")
    con.execute("CREATE INDEX repaired_evidence_key ON repaired_evidence(identity_key)")
    con.execute("""CREATE TEMP TABLE stats AS SELECT m.identity_key, count(DISTINCT m.work_id) AS works_count,
        sum(w.cited_by_count) AS cited_by_count FROM (SELECT DISTINCT identity_key, work_id FROM repaired_memberships) m
        JOIN works w ON w.work_id = m.work_id GROUP BY m.identity_key""")
    con.execute("CREATE INDEX stats_identity_key ON stats(identity_key)")
    source_cache = {row.pop("source_key"): row for row in con.execute("""SELECT n.* FROM nodes n
        JOIN (SELECT DISTINCT source_author_id FROM repaired_memberships) r ON r.source_author_id = n.source_key""").to_arrow_table().to_pylist()}
    metadata, node_rows = {}, []
    for token, group in sorted(groups.items()):
        key = group["key"]
        base = dict(source_cache[min(group["source_ids"])])
        stats = con.execute("SELECT works_count, cited_by_count FROM stats WHERE identity_key = ?", [key]).fetchone()
        work_ids = [r[0] for r in con.execute("SELECT DISTINCT work_id FROM repaired_memberships WHERE identity_key = ? ORDER BY work_id", [key]).fetchall()]
        evidence_rows = con.execute("""SELECT raw_name, institution_ids, field_id, institution_names, field_name
            FROM repaired_evidence WHERE identity_key = ?""", [key]).fetchall()
        inst = Counter(i for _, ids, _, _, _ in evidence_rows for i in set(ids or []))
        field = Counter(fid for _, _, fid, _, _ in evidence_rows if fid)
        names = Counter(name for name, _, _, _, _ in evidence_rows if name)
        inst_labels, field_labels = defaultdict(Counter), defaultdict(Counter)
        for _, ids, fid, labels, label in evidence_rows:
            for iid, iname in zip(ids or [], labels or []):
                if iname:
                    inst_labels[iid][iname] += 1
            if fid and label:
                field_labels[fid][label] += 1
        dominant_inst = inst.most_common(1)[0][0] if inst else None
        dominant_field = field.most_common(1)[0][0] if field else None
        source_ids = [f"A{s}" for s in group["source_ids"]]
        profile = {"id": f"M{-key}" if key < 0 else f"A{key}", "source_ids": source_ids,
                   "display_name": names.most_common(1)[0][0] if names else base["display_name"],
                   "institution": inst_labels[dominant_inst].most_common(1)[0][0] if inst_labels[dominant_inst] else dominant_inst,
                   "institution_ids": sorted(inst),
                   "field": field_labels[dominant_field].most_common(1)[0][0] if field_labels[dominant_field] else dominant_field,
                   "field_ids": sorted(field), "field_id": dominant_field,
                   "works_count": int(stats[0]) if stats else 0, "cited_by_count": int(stats[1]) if stats else 0,
                   "orcid": group["orcid"], "status": group["status"], "identity_status": group["status"],
                   "work_ids": [f"W{w}" for w in work_ids]}
        metadata[profile["id"]] = profile
        base.update(id=profile["id"] if key < 0 else base["id"], display_name=profile["display_name"],
                    works_count=profile["works_count"], works_kept=profile["works_count"],
                    cited_by_count=profile["cited_by_count"], institution=profile["institution"], field=profile["field"],
                    institution_ids=profile["institution_ids"], field_ids=profile["field_ids"],
                    field_id=profile["field_id"], source_ids=source_ids, orcid=group["orcid"], identity_status=group["status"])
        node_rows.append(base)
    original = con.execute("SELECT * EXCLUDE(source_key) FROM nodes WHERE source_key NOT IN (SELECT DISTINCT source_author_id FROM repaired_memberships) ORDER BY id").to_arrow_table()
    if node_rows:
        additions = pa.Table.from_pylist(node_rows)
        output = pa.concat_tables([original, additions], promote_options="permissive")
    else:
        output = original
    pq.write_table(output, out_dir / "nodes.parquet")
    con.execute(f"""COPY (SELECT *, CASE WHEN identity_key < 0 THEN 'M' || CAST(-identity_key AS VARCHAR)
        ELSE 'A' || CAST(identity_key AS VARCHAR) END AS identity_id FROM memberships)
        TO {sql_literal(out_dir / 'memberships.parquet')} (FORMAT PARQUET)""")
    aliases = defaultdict(list)
    for pid, profile in metadata.items():
        for source in profile["source_ids"]:
            if source not in metadata:  # retained A IDs still resolve to themselves
                aliases[source].append(pid)
    (out_dir / "profiles.json").write_text(json.dumps({"version": 1, "profiles": metadata, "aliases": dict(aliases)}, indent=2))
    (out_dir / "identity_registry.json").write_text(json.dumps({"version": 1, "tokens": registry}, indent=2))
    diagnostics = {"version": 1, "split_profiles": split_profiles, "merged_profiles": merged_profiles,
                   "ambiguous_profiles": ambiguous_profiles,
                   "repaired_identities": sum(pid.startswith("M") for pid in metadata),
                   "retained_profiles": sum(pid.startswith("A") for pid in metadata), "quarantined_identities": sum(p["status"] == "quarantined" for p in metadata.values()),
                   "oversized_profiles_unchanged": oversized_profiles, "input_nodes": original.num_rows + len(affected),
                   "output_nodes": output.num_rows, "work_count": con.execute("SELECT count(*) FROM works").fetchone()[0],
                   "duplicate_works_removed": duplicate_works,
                   "collection_works_without_edges": con.execute("SELECT count(*) FROM eligible WHERE collection").fetchone()[0]}
    (out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    con.close()
    return diagnostics


def add_parser(parser):
    parser.add_argument("--works", required=True, help="enriched original works extract glob")
    parser.add_argument("--nodes", required=True, help="original selected nodes.parquet")
    parser.add_argument("--out", default=None, help="new repaired directory (default DATA_DIR/identities)")
    parser.add_argument("--registry", default=None, help="identity registry from the previous run")
    parser.add_argument("--min-component-works", type=int, default=2)
    parser.add_argument("--max-profile-works", type=int, default=50000)


def run(args):
    report = resolve_identities(args.works, args.nodes, Path(args.out) if args.out else data_dir() / "identities",
                                registry_path=Path(args.registry) if args.registry else None,
                                min_component_works=args.min_component_works, max_profile_works=args.max_profile_works)
    print(json.dumps(report, indent=2))
    return 0
