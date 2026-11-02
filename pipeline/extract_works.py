"""Extract work authorships + quality columns from the works snapshot (~20GB of ~707GB)."""
from pathlib import Path

import pyarrow as pa

from pipeline import extract, snapshot
from pipeline.config import data_dir
from pipeline.works import id_key_sql

DEFAULT_SRC = f"{snapshot.BUCKET_ROOT}/works"

# Leaf columns fetched from each source file (see pipeline/extract.py for why
# the nested author id is named down to its leaf).
COLUMNS = (
    "id",
    "authors_count",
    "authorships.list.element.author.id",
    "publication_year",
    "type",
    "is_xpac",
    "is_paratext",
    "is_retracted",
    "cited_by_count",
    "doi",
    "title",
    "authorships.list.element.raw_author_name",
    "authorships.list.element.raw_orcid",
    "authorships.list.element.institutions.list.element.id",
    "authorships.list.element.institutions.list.element.display_name",
    "primary_topic.field.id",
    "primary_topic.field.display_name",
)

# Ids become numeric keys (works.id_key_sql); the quality columns feed
# works.quality_predicate, so the filter can change without re-extracting.
SELECT_SQL = f"""
    SELECT
        {id_key_sql('id')} AS work_id,
        authors_count AS n_authors,
        [{id_key_sql('a.author.id')} FOR a IN authorships] AS author_ids,
        CAST(publication_year AS SMALLINT) AS year,
        type,
        is_xpac,
        is_paratext,
        is_retracted,
        cited_by_count
    FROM src
"""


def evidence_select(schema) -> str:
    """Project available original byline evidence; never inherit a profile ORCID.

    Older/sparse snapshots have no such fields. They still extract, with null
    evidence, so the resolver leaves those memberships alone rather than guessing.
    """
    authorship = schema.field("authorships").type.value_type
    fields = set(authorship.names)
    name = "a.raw_author_name" if "raw_author_name" in fields else "NULL::VARCHAR"
    orcid = "a.raw_orcid" if "raw_orcid" in fields else "NULL::VARCHAR"
    inst = "[i.id FOR i IN a.institutions]" if "institutions" in fields else "[]::VARCHAR[]"
    inst_names = "[]::VARCHAR[]"
    if "institutions" in fields:
        inst_type = authorship.field("institutions").type
        if pa.types.is_list(inst_type) and pa.types.is_struct(inst_type.value_type) and "display_name" in inst_type.value_type.names:
            inst_names = "[i.display_name FOR i IN a.institutions]"
    field = "NULL::VARCHAR"
    field_name = "NULL::VARCHAR"
    if "primary_topic" in schema.names:
        topic_type = schema.field("primary_topic").type
        if pa.types.is_struct(topic_type) and "field" in topic_type.names:
            field_type = topic_type.field("field").type
            if pa.types.is_struct(field_type) and "id" in field_type.names:
                field = "primary_topic.field.id"
                if "display_name" in field_type.names:
                    field_name = "primary_topic.field.display_name"
    doi = "doi" if "doi" in schema.names else "NULL::VARCHAR"
    title = "title" if "title" in schema.names else "NULL::VARCHAR"
    extra = f""",
        {doi} AS doi, {title} AS title,
        [struct_pack(author_id := {id_key_sql('a.author.id')}, raw_name := {name},
                     raw_orcid := {orcid}, institution_ids := {inst}, field_id := {field},
                     institution_names := {inst_names}, field_name := {field_name})
         FOR a IN authorships] AS authorship_evidence
    FROM src"""
    return SELECT_SQL.replace("FROM src", extra)


def extract_file(con, fs, src_path: str, out_path: Path,
                 expected_rows: int | None = None) -> bool:
    return extract.extract_file(con, fs, src_path, COLUMNS, evidence_select, out_path,
                                expected_rows)


def add_parser(parser) -> None:
    parser.add_argument("--src", default=DEFAULT_SRC,
                        help="works parquet root (S3, or a local mirror of the same release)")
    parser.add_argument("--out", default=None, help="override output dir")
    parser.add_argument("--manifest", default=None,
                        help="release manifest (default: DATA_DIR/snapshot/manifest.json "
                             "from `download`, else the bucket's current one)")
    parser.add_argument("--jobs", type=int, default=4,
                        help="files fetched concurrently; S3 reads are latency-bound, "
                             "so a few in flight use the link far better than one")
    parser.add_argument("--force", action="store_true",
                        help="restamp an output dir that holds another release")


def run(args) -> int:
    out_dir = Path(args.out) if args.out else data_dir() / "works_authorships"
    manifest = extract.load_manifest(args.manifest, data_dir())
    n = extract.extract_release("works", COLUMNS, evidence_select, args.src, out_dir,
                                manifest, jobs=args.jobs, force=args.force)
    print(f"{n} files extracted -> {out_dir} (release {manifest['date']})")
    return 0
