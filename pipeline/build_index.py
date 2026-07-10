"""Label/hit tiles (zooms 6-9) and prefix search shards. Zero backend."""
import json
import unicodedata
from collections import defaultdict
from pathlib import Path

import duckdb

from pipeline.config import apply_resource_limits, data_dir

LABEL_ZOOMS = {6: 50, 7: 50, 8: 200, 9: 200}   # zoom -> per-tile capacity


def normalize(name: str) -> str:
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = "".join(c if (c.isalnum() or c == " ") else " " for c in s)
    return " ".join(s.split())


def build_label_tiles(web: str, out_dir: Path) -> int:
    con = duckdb.connect()
    apply_resource_limits(con)
    written = 0
    for z, cap in LABEL_ZOOMS.items():
        ntiles = 1 << z
        rows = con.execute(
            f"""
            SELECT tx, ty_up, display_name, id, xw, yw, cited_by_count FROM (
                SELECT least({ntiles - 1}, CAST(floor(CAST(xw AS DOUBLE) * {ntiles}) AS INT)) AS tx,
                       least({ntiles - 1}, CAST(floor(CAST(yw AS DOUBLE) * {ntiles}) AS INT)) AS ty_up,
                       display_name, id, CAST(xw AS DOUBLE) AS xw, CAST(yw AS DOUBLE) AS yw, cited_by_count,
                       row_number() OVER (
                           PARTITION BY least({ntiles - 1}, CAST(floor(CAST(xw AS DOUBLE) * {ntiles}) AS INT)),
                                        least({ntiles - 1}, CAST(floor(CAST(yw AS DOUBLE) * {ntiles}) AS INT))
                           ORDER BY cited_by_count DESC, id
                       ) AS rn
                FROM read_parquet('{web}')
            ) WHERE rn <= {cap}
            ORDER BY tx, ty_up, cited_by_count DESC, id
            """
        ).fetchall()
        tiles = defaultdict(list)
        for tx, ty_up, name, aid, xw, yw, cited in rows:
            tiles[(tx, ty_up)].append(
                [name, aid, round(xw, 6), round(yw, 6), int(cited)]
            )
        for (tx, ty_up), entries in tiles.items():
            ty = (ntiles - 1) - ty_up               # XYZ y-flip
            p = out_dir / "labels" / str(z) / str(tx) / f"{ty}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"l": entries}, ensure_ascii=False))
            written += 1
    return written


def build_search_shards(web: str, out_dir: Path) -> int:
    con = duckdb.connect()
    apply_resource_limits(con)
    shards = defaultdict(list)
    rows = con.execute(
        f"""SELECT display_name, id, CAST(xw AS DOUBLE) AS xw, CAST(yw AS DOUBLE) AS yw, cited_by_count
            FROM read_parquet('{web}') ORDER BY cited_by_count DESC, id"""
    ).fetchall()
    for name, aid, xw, yw, cited in rows:
        norm = normalize(name or "")
        head = norm.replace(" ", "")[:2]
        key = head if len(head) == 2 and head.isascii() and head.isalpha() else "_"
        shards[key].append([norm, name, aid, round(xw, 6), round(yw, 6), int(cited)])
    sdir = out_dir / "search"
    sdir.mkdir(parents=True, exist_ok=True)
    total = 0
    for key, entries in shards.items():
        (sdir / f"{key}.json").write_text(json.dumps(entries, ensure_ascii=False))
        total += len(entries)
    return total


def add_parser(parser) -> None:
    parser.add_argument("--web", default=None)
    parser.add_argument("--out", default=None)


def run(args) -> int:
    web = args.web or str(data_dir() / "coords_web.parquet")
    out = Path(args.out) if args.out else data_dir() / "index"
    t = build_label_tiles(web, out)
    s = build_search_shards(web, out)
    print(f"{t:,} label tiles, {s:,} searchable names -> {out}")
    return 0
