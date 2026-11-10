"""Export static, work-scoped metadata for repaired identities and their neighbours.

Only repaired identities and their immediate graph neighbours receive profiles.
All large joins and sorting run in DuckDB; JSON profiles are written one at a
time, with paper evidence streamed in batches. A failed generation never
replaces the previously completed output directory.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa

from pipeline.config import apply_resource_limits, data_dir, spill_dir
from pipeline.identifiers import bare_author_id, decode_author_id, encode_author_id
from pipeline.sql import sql_literal
from pipeline.works import quality_predicate

LEGACY_ID_BASE = 5_000_000_000


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")), encoding="utf-8")


def _legacy_flags(path: Path | None) -> dict[str, dict]:
    if path is None or not Path(path).exists():
        return {}
    raw = json.loads(Path(path).read_text())
    flags = {}
    for row in raw.get("entries", []):
        aid = bare_author_id(row["id"])
        flags[aid] = {k: v for k, v in row.items() if k not in ("id", "n")}
    for kind, cols in raw.get("flags", {}).items():
        for index, value in enumerate(cols.get("id", [])):
            aid = bare_author_id(value) if isinstance(value, str) else "A" + str(raw.get("id_base", LEGACY_ID_BASE) + int(value))
            flags[aid] = {"kind": kind, **{key: vals[index] for key, vals in cols.items() if key not in ("id", "n")}}
    return flags


def _flag_numbers(paths_dir: Path, manifest: dict, wanted: set[str]) -> dict[str, int]:
    """Find only flagged IDs in the current numbering, without an 8M-entry dict."""
    if not wanted:
        return {}
    filename = manifest["files"]["ids"]
    if Path(filename).name != filename:
        raise ValueError("path IDs filename must be a basename")
    tagged = manifest.get("id_encoding") == "tagged-u64"
    ids = np.memmap(paths_dir / filename, dtype="<u8" if tagged else "<u4", mode="r")
    requested = np.array([encode_author_id(aid) for aid in wanted], dtype=np.uint64)
    found = {}
    for first in range(0, len(ids), 1_000_000):
        chunk = np.asarray(ids[first:first + 1_000_000], dtype=np.uint64)
        if not tagged:
            chunk = chunk + np.uint64(manifest.get("id_base", LEGACY_ID_BASE))
        for index in np.flatnonzero(np.isin(chunk, requested)):
            found[decode_author_id(int(chunk[index]))] = first + int(index)
    return found


def _publish(stage: Path, output: Path) -> None:
    """Replace a complete generation; restore the old one if the rename fails."""
    backup = None
    if output.exists():
        backup = Path(tempfile.mkdtemp(prefix=output.name + ".previous-", dir=output.parent))
        backup.rmdir()
        os.replace(output, backup)
    try:
        os.replace(stage, output)
    except BaseException:
        if backup is not None:
            os.replace(backup, output)
        raise
    if backup is not None:
        if backup.is_symlink():
            backup.unlink()
        else:
            shutil.rmtree(backup)


def export_identities(repair_dir: Path, edges_path: str, web_path: str,
                      paths_dir: Path, out_dir: Path, *, flags_path: Path | None = None) -> dict:
    repair_dir, paths_dir, out_dir = Path(repair_dir), Path(paths_dir), Path(out_dir)
    output = out_dir.resolve()
    inputs = [repair_dir.resolve(), paths_dir.resolve(), Path(edges_path).resolve(), Path(web_path).resolve()]
    if any(output == source or output in source.parents for source in inputs):
        raise ValueError("identity metadata output must be separate from its inputs")
    metadata = json.loads((repair_dir / "profiles.json").read_text())
    profiles = {bare_author_id(aid): profile for aid, profile in metadata.get("profiles", {}).items()}
    if any(aid.startswith("A") and (profile.get("status") != "retained" or profile.get("source_ids") != [aid])
           for aid, profile in profiles.items()):
        raise ValueError("OpenAlex repair profiles must retain their own source ID")
    aliases = {}
    for source, values in metadata.get("aliases", {}).items():
        source = bare_author_id(source)
        values = sorted({bare_author_id(value) for value in values})
        if not source.startswith("A") or source in profiles or any(
                not value.startswith("M") or value not in profiles for value in values):
            raise ValueError("aliases must map source A IDs to known M identities")
        aliases[source] = values
    paths_manifest = json.loads((paths_dir / "manifest.json").read_text())
    paths_ids_file = paths_manifest["files"]["ids"]
    if Path(paths_ids_file).name != paths_ids_file:
        raise ValueError("path IDs filename must be a basename")
    if not (paths_dir / paths_ids_file).exists():
        raise ValueError("path IDs file is missing")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=out_dir.name + ".build-", dir=out_dir.parent))
    con = duckdb.connect()
    profile_cursor = None
    try:
        apply_resource_limits(con)
        con.execute(f"SET temp_directory={sql_literal(spill_dir(stage))}")
        con.execute("SET preserve_insertion_order=false")
        con.execute(f"CREATE VIEW web AS SELECT * REPLACE (regexp_replace(id, '^https://openalex.org/', '') AS id) FROM read_parquet({sql_literal(web_path)})")
        con.execute(f"CREATE VIEW edges AS SELECT regexp_replace(src, '^https://openalex.org/', '') AS src, regexp_replace(dst, '^https://openalex.org/', '') AS dst FROM read_parquet({sql_literal(edges_path)})")
        con.execute(f"CREATE VIEW works AS SELECT * FROM read_parquet({sql_literal(repair_dir / 'works.parquet')})")
        con.execute(f"CREATE VIEW memberships AS SELECT * FROM read_parquet({sql_literal(repair_dir / 'memberships.parquet')})")
        con.register("changed", pa.table({"id": pa.array(sorted(profiles), type=pa.string())}))
        con.execute("""CREATE TABLE affected AS
            SELECT id FROM changed UNION SELECT dst AS id FROM edges JOIN changed c ON c.id = src
            UNION SELECT src AS id FROM edges JOIN changed c ON c.id = dst""")
        missing = con.execute("SELECT id FROM affected EXCEPT SELECT id FROM web").fetchall()
        if missing:
            raise ValueError(f"repaired graph identities missing map coordinates: {missing[:5]}")
        con.execute("CREATE TABLE profile_data AS SELECT w.* FROM web w JOIN affected a ON a.id = w.id")
        con.execute("CREATE INDEX profile_data_id ON profile_data(id)")
        con.execute("CREATE TABLE affected_memberships AS SELECT DISTINCT m.identity_id, m.work_id FROM memberships m JOIN affected a ON a.id = m.identity_id")
        con.execute("CREATE INDEX affected_membership_id ON affected_memberships(identity_id)")
        con.execute("CREATE TABLE directed AS SELECT DISTINCT src, dst FROM edges UNION SELECT dst AS src, src AS dst FROM edges")
        # Memberships and author lists retain papers for visibility. Only the
        # corrected edge_author_ids can supply collaboration evidence.
        con.execute("""CREATE TABLE relevant_ids AS SELECT DISTINCT m.work_id
            FROM affected_memberships m""")
        columns = {row[0] for row in con.execute("DESCRIBE works").fetchall()}
        if "edge_author_ids" not in columns:
            raise ValueError("repaired works must include edge_author_ids")
        title = "title" if "title" in columns else "NULL::VARCHAR AS title"
        doi = "doi" if "doi" in columns else "NULL::VARCHAR AS doi"
        venue = "venue" if "venue" in columns else "NULL::VARCHAR AS venue"
        con.execute(f"""CREATE TABLE eligible AS SELECT w.work_id, w.edge_author_ids, w.year, w.cited_by_count,
            {title}, {doi}, {venue} FROM works w JOIN relevant_ids r ON r.work_id = w.work_id
            WHERE {quality_predicate()} AND w.n_authors BETWEEN 2 AND 50""")
        con.execute("""CREATE TABLE seats AS SELECT DISTINCT w.work_id,
            CASE WHEN key < 0 THEN 'M' || CAST(-key AS VARCHAR) ELSE 'A' || CAST(key AS VARCHAR) END AS id
            FROM eligible w, unnest(list_distinct(edge_author_ids)) t(key) WHERE key IS NOT NULL""")
        con.execute("""CREATE TABLE shared AS SELECT a.id, b.id AS neighbour, w.* EXCLUDE(edge_author_ids)
            FROM seats a JOIN affected target ON target.id = a.id
            JOIN directed d ON d.src = a.id JOIN seats b ON b.id = d.dst AND b.work_id = a.work_id
            JOIN eligible w ON w.work_id = a.work_id""")
        con.execute("""CREATE TABLE counts AS SELECT id, neighbour, count(DISTINCT work_id) AS n
            FROM shared GROUP BY id, neighbour""")
        con.execute("""CREATE TABLE papers AS SELECT * FROM shared
            QUALIFY row_number() OVER (PARTITION BY id, neighbour ORDER BY cited_by_count DESC NULLS LAST, work_id) = 1""")
        con.execute("CREATE TABLE ray_counts AS SELECT c.id, c.neighbour, c.n, w.display_name FROM counts c LEFT JOIN web w ON w.id = c.neighbour QUALIFY row_number() OVER (PARTITION BY c.id ORDER BY c.n DESC, c.neighbour) <= 25")
        con.execute("CREATE INDEX rays_profile_id ON ray_counts(id)")
        con.execute("CREATE INDEX papers_profile_id ON papers(id)")
        web_columns = {row[0] for row in con.execute("DESCRIBE profile_data").fetchall()}
        fields = ["display_name", "works_count", "cited_by_count", "institution", "field"]
        selection = ", ".join(f"w.{name}" if name in web_columns else f"NULL AS {name}" for name in fields)
        ids = con.execute("SELECT id FROM affected ORDER BY id").to_arrow_reader(8192)
        profile_count = 0
        institutions = {}
        # A separate cursor keeps the ID stream alive during per-profile reads.
        profile_cursor = con.cursor()
        for batch in ids:
            for aid in batch.column(0).to_pylist():
                row = profile_cursor.execute(f"SELECT {selection} FROM profile_data w WHERE w.id = ?", [aid]).fetchone()
                base = dict(zip(fields, row))
                base.update(id=aid, source_ids=[aid], orcid=None, status="unchanged")
                base.update(profiles.get(aid, {}))
                base["source_ids"] = [bare_author_id(value) for value in base.get("source_ids", [])]
                base["institution_ids"] = sorted({"I" + match.group(1)
                    for value in base.get("institution_ids", [])
                    if (match := re.fullmatch(r"(?:https://openalex\.org/)?I(\d+)", str(value)))})
                if aid.startswith("M"):
                    candidate = {key: base.get(key) for key in ("id", "display_name", "cited_by_count",
                        "works_count", "institution", "institution_ids", "source_ids", "orcid", "status")}
                    for institution in base["institution_ids"]:
                        institutions.setdefault(institution, []).append(candidate)
                base["work_ids"] = ["W" + str(row[0]) for row in profile_cursor.execute(
                    "SELECT DISTINCT work_id FROM affected_memberships WHERE identity_id = ? ORDER BY work_id", [aid]).fetchall()]
                base["coauthors"] = [{"id": neighbour, "name": name or neighbour, "count": int(count)}
                    for neighbour, name, count in profile_cursor.execute("""SELECT c.neighbour, c.display_name, c.n
                        FROM ray_counts c WHERE c.id = ?
                        ORDER BY c.n DESC, c.neighbour""", [aid]).fetchall()]
                target = stage / "profiles" / (aid + ".json")
                target.parent.mkdir(parents=True, exist_ok=True)
                # Stream the all-neighbour evidence map, avoiding a huge Python
                # dict for prolific collaborators. One work proves each edge.
                with target.open("w", encoding="utf-8") as handle:
                    prefix = json.dumps(base, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                    handle.write(prefix[:-1] + ',"papers":{')
                    first = True
                    reader = profile_cursor.execute("SELECT neighbour, work_id, title, year, doi, venue FROM papers WHERE id = ? ORDER BY neighbour", [aid]).to_arrow_reader(8192)
                    for evidence_batch in reader:
                        for neighbour, wid, paper_title, year, paper_doi, paper_venue in zip(*[evidence_batch.column(i).to_pylist() for i in range(6)]):
                            paper = {"title": paper_title or f"OpenAlex work W{wid}", "year": year,
                                     "url": paper_doi if isinstance(paper_doi, str) and paper_doi.startswith(("https://", "http://")) else f"https://openalex.org/W{wid}",
                                     "work_id": "W" + str(wid)}
                            if paper_venue:
                                paper["venue"] = paper_venue
                            if not first:
                                handle.write(",")
                            handle.write(json.dumps(neighbour) + ":" + json.dumps(paper, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
                            first = False
                    handle.write("}}")
                profile_count += 1
        grouped_aliases = {}
        for aid, values in sorted(aliases.items()):
            grouped_aliases.setdefault(int(aid[1:]) % 1000, {})[aid] = values
        for bucket, values in grouped_aliases.items():
            _write_json(stage / "aliases" / f"{bucket}.json", values)
        for institution, candidates in institutions.items():
            candidates.sort(key=lambda row: (-int(row.get("cited_by_count") or 0), row["id"]))
            _write_json(stage / "institutions" / (institution + ".json"), candidates)
        flagged = _legacy_flags(flags_path)
        for aid, profile in profiles.items():
            if profile.get("status") == "quarantined":
                flagged[aid] = {"kind": "quarantined"}
        numbers = _flag_numbers(paths_dir, paths_manifest, set(flagged) | set(profiles))
        if any(aid not in numbers for aid in profiles):
            raise ValueError("repaired identities are missing from the current path graph")
        entries = [{"id": aid, "n": numbers[aid], **flagged[aid]}
                   for aid in sorted(set(numbers) & set(flagged), key=numbers.get)]
        _write_json(stage / "flags.json", {"version": 1, "paths_ids_file": paths_ids_file, "entries": entries})
        manifest = {"version": 1, "graph_repaired": True, "paths_ids_file": paths_ids_file,
                    "profiles_path": "profiles/{id}.json", "aliases_path": "aliases/{bucket}.json",
                    "institutions_path": "institutions/{id}.json", "institution_count": len(institutions),
                    "flags_file": "flags.json", "profile_count": profile_count, "alias_count": len(aliases)}
        _write_json(stage / "manifest.json", manifest)
        profile_cursor.close()
        profile_cursor = None
        con.close()
        if (stage / ".duckdb_tmp").exists():
            shutil.rmtree(stage / ".duckdb_tmp")
        _publish(stage, out_dir)
        return manifest
    finally:
        if profile_cursor is not None:
            profile_cursor.close()
        con.close()
        if stage.exists():
            shutil.rmtree(stage)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair", required=True, type=Path)
    parser.add_argument("--edges", default=None)
    parser.add_argument("--web", default=None)
    parser.add_argument("--paths", default=None, type=Path)
    parser.add_argument("--out", default=None, type=Path)
    parser.add_argument("--flags", default=None, type=Path)
    args = parser.parse_args()
    base = data_dir()
    result = export_identities(args.repair, args.edges or str(base / "edges.parquet"),
        args.web or str(base / "coords_web.parquet"), args.paths or base / "paths",
        args.out or base / "identities", flags_path=args.flags)
    print(f"{result['profile_count']:,} static profiles, {result['alias_count']:,} source aliases exported")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
