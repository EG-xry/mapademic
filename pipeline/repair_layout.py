"""Preserve the published map while placing automatically repaired identities.

This CPU-only stage reuses existing unit-square web positions. Unchanged IDs
keep their positions and communities exactly. New identities start near their
actual repaired collaborators, with source-profile positions as a fallback.
Zero-degree new identities are searchable dust (is_ring/is_dust), matching the
existing viewer's optional dust display rather than discarding small partitions.
"""
from pathlib import Path
import os

import duckdb
import pyarrow.parquet as pq

from pipeline.config import apply_resource_limits, spill_dir

JITTER = 0.00001


def repair_webcoords(previous_web: str, repaired_nodes: str, edges_path: str,
                     out_path: str, hops: int = 2) -> dict:
    """Write a full compatible coords_web table; return placement counts.

    All joins and averaging run in DuckDB; the full map never becomes Python
    dictionaries. Output is atomically replaced only after validation succeeds.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".tmp")
    con = duckdb.connect()
    try:
        apply_resource_limits(con)
        if not os.environ.get("MAPADEMIC_THREADS"):
            con.execute("SET threads=2")
        con.execute("SET temp_directory=?", [str(spill_dir(out.parent))])
        con.read_parquet(str(previous_web)).create_view("previous")
        con.read_parquet(str(repaired_nodes)).create_view("node_source")
        con.read_parquet(str(edges_path)).create_view("edges")
        if "id" not in pq.read_schema(repaired_nodes).names:
            raise ValueError("repaired nodes lack identity IDs")
        if not {"src", "dst", "weight"} <= set(pq.read_schema(edges_path).names):
            raise ValueError("repaired edges lack src, dst, or weight")
        required = {"id", "xw", "yw", "community"}
        previous_columns = set(pq.read_schema(previous_web).names)
        if not required <= previous_columns:
            raise ValueError("previous web coordinates lack id, xw, yw, or community")
        for table in ("previous", "node_source"):
            if con.execute(f"SELECT count(*) != count(DISTINCT id) FROM {table}").fetchone()[0]:
                raise ValueError(f"duplicate or null node IDs in {table}")
        if con.execute("""SELECT count(*) FROM previous WHERE xw IS NULL OR yw IS NULL
                          OR NOT isfinite(xw) OR NOT isfinite(yw)
                          OR xw < 0 OR xw > 1 OR yw < 0 OR yw > 1""").fetchone()[0]:
            raise ValueError("previous positions must be finite and inside the unit square")
        if con.execute("""SELECT count(*) FROM edges e
                          LEFT JOIN node_source a ON a.id=e.src LEFT JOIN node_source b ON b.id=e.dst
                          WHERE a.id IS NULL OR b.id IS NULL OR e.weight IS NULL
                             OR NOT isfinite(e.weight) OR e.weight <= 0 OR e.src=e.dst""").fetchone()[0]:
            raise ValueError("repaired graph has dangling, self, or invalid-weight edges")
        columns = set(pq.read_schema(repaired_nodes).names)
        defaults = {"display_name": "id", "works_count": "0::BIGINT", "cited_by_count": "0::BIGINT",
                    "institution": "NULL::VARCHAR", "field": "NULL::VARCHAR", "source_ids": "[id]"}
        extras = [f"{expr} AS {name}" for name, expr in defaults.items() if name not in columns]
        con.execute("CREATE TEMP VIEW nodes AS SELECT *" + (", " + ", ".join(extras) if extras else "") + " FROM node_source")
        sub = "coalesce(p.sub,-1)" if "sub" in previous_columns else "-1"
        ring = "coalesce(p.is_ring,false)" if "is_ring" in previous_columns else "false"
        dust = "coalesce(p.is_dust,false)" if "is_dust" in previous_columns else ring
        con.execute(f"""CREATE TEMP TABLE pos AS
            SELECT n.id, p.xw::DOUBLE xw, p.yw::DOUBLE yw, coalesce(p.community,-1)::BIGINT community,
                   {sub}::BIGINT sub, {ring}::BOOLEAN is_ring, {dust}::BOOLEAN is_dust
            FROM nodes n JOIN previous p ON p.id=n.id""")
        carried = con.execute("SELECT count(*) FROM pos").fetchone()[0]
        con.execute("""CREATE TEMP VIEW adjacency AS
                       SELECT src AS v,dst AS u,weight FROM edges
                       UNION ALL SELECT dst,src,weight FROM edges""")
        con.execute("""CREATE TEMP VIEW jitter AS SELECT id,
                       ((hash(id) % 2001)::DOUBLE / 1000 - 1) * ? AS dx,
                       ((hash(id || ':y') % 2001)::DOUBLE / 1000 - 1) * ? AS dy FROM nodes""".replace("?", str(JITTER)))
        neighbours = 0
        for _ in range(max(0, int(hops))):
            con.execute("""CREATE OR REPLACE TEMP TABLE additions AS
                WITH touch AS (
                    SELECT a.v,p.*,a.weight,
                           row_number() OVER (PARTITION BY a.v ORDER BY a.weight DESC,a.u) AS preference
                    FROM adjacency a JOIN pos p ON p.id=a.u
                    WHERE NOT EXISTS (SELECT 1 FROM pos done WHERE done.id=a.v)),
                means AS (SELECT v,sum(weight*xw)/sum(weight) xw,sum(weight*yw)/sum(weight) yw FROM touch GROUP BY v)
                SELECT m.v AS id,least(1.,greatest(0.,m.xw+j.dx)) xw,least(1.,greatest(0.,m.yw+j.dy)) yw,
                       t.community,t.sub,false AS is_ring,false AS is_dust
                FROM means m JOIN touch t ON t.v=m.v AND t.preference=1 JOIN jitter j ON j.id=m.v""")
            count = con.execute("SELECT count(*) FROM additions").fetchone()[0]
            if not count:
                break
            con.execute("INSERT INTO pos SELECT * FROM additions")
            neighbours += count
        # Alias source IDs are bare A IDs; the previous release stores full URLs.
        con.execute(f"""CREATE TEMP TABLE sources AS
            WITH old AS (SELECT *,regexp_replace(id,'^https?://openalex[.]org/','') AS bare_id FROM previous),
            source_links AS (SELECT n.id,unnest(n.source_ids) AS source FROM nodes n
                             WHERE NOT EXISTS (SELECT 1 FROM pos p WHERE p.id=n.id)),
            matched AS (SELECT s.id,p.*,row_number() OVER (PARTITION BY s.id ORDER BY p.id) preference
                        FROM source_links s JOIN old p ON p.bare_id=regexp_replace(s.source,'^https?://openalex[.]org/','')),
            means AS (SELECT id,avg(xw) xw,avg(yw) yw FROM matched GROUP BY id)
            SELECT m.id,least(1.,greatest(0.,m.xw+j.dx)) xw,least(1.,greatest(0.,m.yw+j.dy)) yw,
                   coalesce(p.community,-1)::BIGINT community,{"coalesce(p.sub,-1)" if "sub" in previous_columns else "-1"}::BIGINT sub,
                   NOT EXISTS(SELECT 1 FROM adjacency a WHERE a.v=m.id) AS is_ring,
                   NOT EXISTS(SELECT 1 FROM adjacency a WHERE a.v=m.id) AS is_dust
            FROM means m JOIN matched p ON p.id=m.id AND p.preference=1 JOIN jitter j ON j.id=m.id""")
        source_count = con.execute("SELECT count(*) FROM sources").fetchone()[0]
        con.execute("INSERT INTO pos SELECT * FROM sources")
        scattered = con.execute("SELECT count(*) FROM nodes n WHERE NOT EXISTS(SELECT 1 FROM pos p WHERE p.id=n.id)").fetchone()[0]
        con.execute("""INSERT INTO pos
            SELECT n.id,.5+j.dx*100,.5+j.dy*100,-1,-1,
                   NOT EXISTS(SELECT 1 FROM adjacency a WHERE a.v=n.id),
                   NOT EXISTS(SELECT 1 FROM adjacency a WHERE a.v=n.id)
            FROM nodes n JOIN jitter j ON j.id=n.id WHERE NOT EXISTS(SELECT 1 FROM pos p WHERE p.id=n.id)""")
        overridden = columns & {"xw", "yw", "community", "sub", "is_ring", "is_dust"}
        exclude = " EXCLUDE (" + ",".join(sorted(overridden)) + ")" if overridden else ""
        query = f"SELECT n.*{exclude},p.xw,p.yw,p.community,p.sub,p.is_ring,p.is_dust FROM nodes n JOIN pos p ON p.id=n.id ORDER BY n.id"
        con.execute("COPY (" + query + ") TO ? (FORMAT PARQUET)", [str(temporary)])
        size = con.execute("SELECT count(*) FROM pos").fetchone()[0]
        expected = con.execute("SELECT count(*) FROM nodes").fetchone()[0]
        if size != expected:
            raise ValueError("placement did not produce exactly one position per repaired identity")
        os.replace(temporary, out)
        return {"carried": carried, "from_neighbours": neighbours, "from_sources": source_count,
                "scattered": scattered, "nodes": size}
    finally:
        con.close()
        temporary.unlink(missing_ok=True)
