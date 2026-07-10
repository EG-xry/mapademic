"""Label/hit tiles (zooms 6-9) and prefix search shards. Zero backend.

Search shard fallback order (viewer): 3-char shard -> 2-char shard -> "_".
"""
import json
import os
import shutil
import unicodedata
from collections import defaultdict
from pathlib import Path

import duckdb

from pipeline.config import apply_resource_limits, data_dir

LABEL_ZOOMS = {6: 50, 7: 50, 8: 200, 9: 4000}   # zoom -> per-tile capacity
SHARD_SPLIT_BYTES = 4_000_000                   # 2-char shards larger than this get 3-char split


def normalize(name: str) -> str:
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = "".join(c if (c.isalnum() or c == " ") else " " for c in s)
    return " ".join(s.split())


def build_label_tiles(web: str, out_dir: Path) -> int:
    labels_dir = out_dir / "labels"
    if labels_dir.exists():
        shutil.rmtree(labels_dir)
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
                FROM read_parquet('{web}') WHERE NOT is_ring
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
            tmp = p.parent / (p.name + ".tmp")
            tmp.write_text(json.dumps({"l": entries}, ensure_ascii=False))
            os.replace(tmp, p)
            written += 1
    return written


def _write_json_atomic(path: Path, obj) -> None:
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False))
    os.replace(tmp, path)


def build_search_shards(web: str, out_dir: Path) -> int:
    search_dir = out_dir / "search"
    if search_dir.exists():
        shutil.rmtree(search_dir)
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
        total += len(entries)
        size = len(json.dumps(entries, ensure_ascii=False).encode("utf-8"))
        if key != "_" and size > SHARD_SPLIT_BYTES:
            sub = defaultdict(list)
            parent = []
            for e in entries:
                concat = e[0].replace(" ", "")
                head3 = concat[:3]
                if len(concat) >= 3 and head3.isascii() and head3.isalpha():
                    sub[head3].append(e)
                else:
                    parent.append(e)
            for subkey, subentries in sub.items():
                _write_json_atomic(sdir / f"{subkey}.json", subentries)
            _write_json_atomic(sdir / f"{key}.json", parent)  # always written: viewer fallback
        else:
            _write_json_atomic(sdir / f"{key}.json", entries)
    return total


def build_id_shards(web: str, out_dir: Path) -> int:
    ids_dir = out_dir / "ids"
    if ids_dir.exists():
        shutil.rmtree(ids_dir)
    con = duckdb.connect()
    apply_resource_limits(con)
    buckets = defaultdict(dict)
    rows = con.execute(
        f"""SELECT id, CAST(xw AS DOUBLE) AS xw, CAST(yw AS DOUBLE) AS yw
            FROM read_parquet('{web}')"""
    ).fetchall()
    for aid, xw, yw in rows:
        digits = "".join(filter(str.isdigit, aid))
        bucket = int(digits) % 1000 if digits else 0
        buckets[bucket][aid] = [round(xw, 6), round(yw, 6)]
    ids_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for bucket, entries in buckets.items():
        _write_json_atomic(ids_dir / f"{bucket}.json", entries)
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
    i = build_id_shards(web, out)
    print(f"{t:,} label tiles, {s:,} searchable names, {i:,} id-shard entries -> {out}")
    return 0
