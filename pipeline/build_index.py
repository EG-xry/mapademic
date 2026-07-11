"""Label/hit tiles (zooms 6-9) and prefix search shards. Zero backend.

Search shard fallback order (viewer): try prefix shards from
min(len(normalized concatenated name), 5) chars down to 2 chars, then the
codepoint shard `_<ord(first char of normalized name) mod 128>`, then `_`.
That lookup is unchanged. What changed is which shards an author's entry
lives in: each author is indexed under BOTH the first token and the last
token of their normalized name (skipped for single-token names, when
first token == last token, and when the last token is under 2 chars, e.g.
an initial). Both insertions use the same entry (the full
spaced norm, not a rotated one) -- an entry is just findable via either
token's shard family. An author's id can therefore appear in up to two
shard files; the viewer dedupes by id client-side.
(The docstring doubles as argparse help, so no percent signs here.)
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
SHARD_SPLIT_BYTES = 4_000_000                   # shards larger than this split by one more prefix char
MAX_PREFIX_LEN = 5                              # deepest prefix-shard key length
CATCHALL_BUCKETS = 128                          # codepoint-modulus for the `_` catch-all split (JS mirror: web/dev.html)


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


def _shard_bytes(entries) -> int:
    return len(json.dumps(entries, ensure_ascii=False).encode("utf-8"))


def _bucket_key(token: str) -> str:
    """2-char ascii-alpha head of a name token, else the catch-all key.

    Shared by both the first-token and the last-token insertion so the two
    families are bucketed by identical rules.
    """
    head = token[:2]
    if len(head) == 2 and head.isascii() and head.isalpha():
        return head
    return "_"


def _split_prefix_shard(key: str, items: list, out: dict) -> None:
    """Recursively split an alpha-ascii prefix shard by one more prefix char.

    items: list of (sort_token, entry) pairs. sort_token is whichever name
    token (first or last) routed this entry into the shard family -- the
    split key is derived from that token, not from the entry's full norm,
    so a last-token (surname) family recurses on the surname's own chars.

    A split parent is always written (possibly small) so the viewer fallback
    chain is deterministic. Entries whose sort_token is exactly len(key)
    chars, or whose next-char prefix is not alpha-ascii, stay at the parent
    level.
    """
    k = len(key)
    if _shard_bytes([e for _, e in items]) <= SHARD_SPLIT_BYTES or k >= MAX_PREFIX_LEN:
        out[key] = [e for _, e in items]
        return
    sub = defaultdict(list)
    parent = []
    for token, e in items:
        child = token[:k + 1]
        if len(token) > k and child.isascii() and child.isalpha():
            sub[child].append((token, e))
        else:
            parent.append((token, e))
    out[key] = [e for _, e in parent]           # always written: viewer fallback
    for child_key, child_items in sub.items():
        _split_prefix_shard(child_key, child_items, out)


def _split_catchall_shard(items: list, out: dict) -> None:
    """Split the `_` catch-all by first-char codepoint when oversized.

    items: list of (sort_token, entry) pairs (see _split_prefix_shard).
    Entries go to `_<ord(first char of sort_token) % CATCHALL_BUCKETS>` (JS
    mirror: "_" + (norm.codePointAt(0) % 128)); entries with an empty
    sort_token stay in `_`, which is always written.
    """
    if _shard_bytes([e for _, e in items]) <= SHARD_SPLIT_BYTES:
        out["_"] = [e for _, e in items]
        return
    sub = defaultdict(list)
    parent = []
    for token, e in items:
        if token:
            sub[f"_{ord(token[0]) % CATCHALL_BUCKETS}"].append((token, e))
        else:
            parent.append((token, e))
    out["_"] = [e for _, e in parent]            # always written: viewer fallback
    for child_key, child_items in sub.items():
        out[child_key] = [e for _, e in child_items]


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
        entry = [norm, name, aid, round(xw, 6), round(yw, 6), int(cited)]
        tokens = norm.split(" ")
        first_tok, last_tok = tokens[0], tokens[-1]
        shards[_bucket_key(first_tok)].append((first_tok, entry))
        if len(tokens) > 1 and first_tok != last_tok and len(last_tok) >= 2:
            shards[_bucket_key(last_tok)].append((last_tok, entry))
    final = {}
    for key, items in shards.items():
        if key == "_":
            _split_catchall_shard(items, final)
        else:
            _split_prefix_shard(key, items, final)
    sdir = out_dir / "search"
    sdir.mkdir(parents=True, exist_ok=True)
    for key, entries in final.items():
        _write_json_atomic(sdir / f"{key}.json", entries)
    return len(rows)


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
