"""Coauthor graph for the browser's own path search (web/dev.html, Connect).

Eric asked for Connect to work with no server and no bill (2026-10-11), so
the graph lives on the data host (R2) beside the tiles and the browser
reads only the coauthor lists its bidirectional search touches: a median of
~160 authors (~35k links) per search, ~900 at the 90th percentile, fetched
with HTTP Range requests that Cloudflare serves from its cache.

To fit R2's free 10GB with the rest of the site, the 732M directed links
(2.9GB as int32) are stored compactly:
  - authors are renumbered along a Z-order curve of their map position, so
    coauthors (placed close together by the layout) get close numbers;
  - each author's list is sorted and stored as gaps: the first entry as a
    zigzag varint of (neighbour - author), the rest as varint gaps, each
    shifted left one bit to carry a "suspect" flag (pipeline.identity: the
    link probably joins two namesakes merged by OpenAlex), so the browser
    can prefer paths between real people (2026-10-15);
  - the byte stream is split at author boundaries into files of at most
    CHUNK_BYTES (under Cloudflare's 512MB cacheable-object limit).

Writes <release>/paths/: neighbors-<k>-<hash>.bin, offsets-<hash>.bin
(uint32 byte offset of each author's list, n+1 entries, read in blocks),
lengths-<hash>.bin (the same as varint list lengths, ~12MB: the browser
downloads it once so hard searches skip the offset reads), ids-<hash>.bin
(uint32 OpenAlex numeric id minus ID_BASE, by new author number) and
manifest.json, the only fixed name, listing them (served with a short
cache lifetime; meta.json was the unhashed v1 layout). Content-hashed names mean a
re-export never collides with a cached copy of the last one. The index's id
shards carry each author's new number (see build_index), which is how the
browser starts a search.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import duckdb
import numpy as np

from pipeline.config import data_dir
from pipeline.identity import suspect_links

ID_BASE = 5_000_000_000      # every author id in the 2026-09-23 graph is A5xxxxxxxxx
CHUNK_BYTES = 200_000_000
BATCH = 40_000_000           # directed links per encoding batch (bounds memory)


def morton16(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Interleave the bits of two uint16 arrays into a uint32 Z-order code."""
    def spread(v):
        v = v.astype(np.uint32)
        v = (v | (v << 8)) & 0x00FF00FF
        v = (v | (v << 4)) & 0x0F0F0F0F
        v = (v | (v << 2)) & 0x33333333
        return (v | (v << 1)) & 0x55555555
    return spread(x) | (spread(y) << 1)


def varint_lengths(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.uint64)
    return (1 + (v >= 1 << 7) + (v >= 1 << 14) + (v >= 1 << 21) + (v >= 1 << 28)
            + (v >= 1 << 35)).astype(np.uint8)


def varint_encode(v: np.ndarray) -> bytes:
    """LEB128-style: 7 bits per byte, low first, high bit = more bytes."""
    v = v.astype(np.uint64)
    lens = varint_lengths(v).astype(np.int64)
    out = np.empty(int(lens.sum()), np.uint8)
    starts = np.cumsum(lens) - lens
    for k in range(int(lens.max(initial=1))):
        sel = lens > k
        byte = ((v[sel] >> np.uint64(7 * k)) & np.uint64(0x7F)).astype(np.uint8)
        byte[lens[sel] > k + 1] |= 0x80
        out[starts[sel] + k] = byte
    return out.tobytes()


def zigzag(d: np.ndarray) -> np.ndarray:
    d = d.astype(np.int64)
    return ((d << 1) ^ (d >> 63)).astype(np.uint64)


def export_paths(pathgraph_dir: Path, web_path: str, out_dir: Path,
                 chunk_bytes: int = CHUNK_BYTES, batch: int = BATCH,
                 suspect: tuple[np.ndarray, np.ndarray] | None = None) -> dict:
    """`suspect`: (src, dst) old node indices of links that probably join two
    namesakes (pipeline.identity.suspect_links); each is stored with its flag
    bit set, so the viewer can route around them."""
    t0 = time.time()
    pathgraph_dir, out_dir = Path(pathgraph_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in list(out_dir.glob("*.bin")) + list(out_dir.glob("*.json")):
        old.unlink()
    offsets = np.load(pathgraph_dir / "offsets.npy", mmap_mode="r")
    neighbors = np.load(pathgraph_dir / "neighbors.npy", mmap_mode="r")
    ids = np.load(pathgraph_dir / "ids.npy")
    n = len(ids)
    assert int(ids.min()) >= ID_BASE and int(ids.max()) - ID_BASE < 2 ** 32, "ids outside the uint32 window"
    # renumber along a Z-order curve of map position (authors without one go last)
    con = duckdb.connect()
    pos = con.execute(f"""SELECT CAST(regexp_extract(id, '(\\d+)$', 1) AS UBIGINT) AS k,
                                 CAST(round(xw * 65535) AS INTEGER) AS x, CAST(round(yw * 65535) AS INTEGER) AS y
                          FROM read_parquet('{web_path}')""").fetchnumpy()
    sk = np.argsort(pos["k"])
    where = np.searchsorted(pos["k"][sk], ids)
    found = (where < len(sk)) & (pos["k"][sk][np.minimum(where, len(sk) - 1)] == ids)
    code = np.full(n, np.uint64(2 ** 33), np.uint64)
    src = sk[where[found]]
    code[found] = morton16(np.clip(pos["x"][src], 0, 65535), np.clip(pos["y"][src], 0, 65535)).astype(np.uint64)
    order = np.lexsort((ids, code))                     # new number -> old node
    new_of_old = np.empty(n, np.int64)
    new_of_old[order] = np.arange(n)
    deg_old = np.diff(np.asarray(offsets))
    deg_new = deg_old[order]
    if suspect is not None and len(suspect[0]):
        a, b = new_of_old[suspect[0]], new_of_old[suspect[1]]
        suspect_keys = np.unique(np.minimum(a, b) * n + np.maximum(a, b))
    else:
        suspect_keys = np.empty(0, np.int64)
    # encode each new row: sorted neighbours as gaps, batched over rows
    row_bytes = np.zeros(n, np.int64)
    n_suspect = 0
    chunks, cur, cur_bytes, cur_first = [], bytearray(), 0, 0
    lo = 0
    while lo < n:
        hi, links = lo, 0
        while hi < n and (links + deg_new[hi] <= batch or hi == lo):
            links += int(deg_new[hi]); hi += 1
        rows = np.arange(lo, hi)
        olds = order[lo:hi]
        starts = np.asarray(offsets)[olds]
        deg = deg_new[lo:hi]
        idx = np.repeat(starts - (np.cumsum(deg) - deg), deg) + np.arange(int(deg.sum()))
        vals = new_of_old[np.asarray(neighbors[idx])]
        rid = np.repeat(rows, deg)
        o = np.lexsort((vals, rid))
        vals, rid = vals[o], rid[o]
        first = np.ones(len(vals), bool)
        first[1:] = rid[1:] != rid[:-1]
        gaps = np.empty(len(vals), np.uint64)
        gaps[first] = zigzag(vals[first] - rid[first])
        gaps[~first] = (vals[~first] - np.concatenate([[0], vals[:-1]])[~first]).astype(np.uint64)
        keys = np.minimum(rid, vals) * n + np.maximum(rid, vals)
        at = np.searchsorted(suspect_keys, keys)
        flag = (at < len(suspect_keys)) & (suspect_keys[np.minimum(at, len(suspect_keys) - 1)] == keys)
        gaps = (gaps << np.uint64(1)) | flag.astype(np.uint64)
        n_suspect += int(flag.sum())
        lens = varint_lengths(gaps).astype(np.int64)
        row_bytes[lo:hi] = np.bincount(rid - lo, weights=lens, minlength=hi - lo).astype(np.int64)
        enc = varint_encode(gaps)
        # split the batch at row boundaries into chunk files
        cum = np.cumsum(row_bytes[lo:hi])
        r0, used = 0, 0                                   # rows / bytes of this batch already placed
        while r0 < hi - lo:
            r1 = int(np.searchsorted(cum, used + (chunk_bytes - cur_bytes), side="right"))
            if r1 <= r0:
                if cur_bytes:                             # next row does not fit: close the file
                    chunks.append((cur_first, lo + r0, bytes(cur)))
                    cur, cur_bytes, cur_first = bytearray(), 0, lo + r0
                    continue
                r1 = r0 + 1                               # a single row larger than a whole file
            cur += enc[used:int(cum[r1 - 1])]
            cur_bytes += int(cum[r1 - 1]) - used
            used, r0 = int(cum[r1 - 1]), r1
        lo = hi
    chunks.append((cur_first, n, bytes(cur)))
    # every data file is named after its content hash, so a re-export never
    # collides with week-long cached copies of the old one; only manifest.json
    # (served with a short cache lifetime) has a fixed name
    def write(stem: str, blob: bytes) -> str:
        name = f"{stem}-{hashlib.sha1(blob).hexdigest()[:10]}.bin"
        (out_dir / name).write_bytes(blob)
        return name
    table, byte_start = [], 0
    for k, (first_row, end_row, blob) in enumerate(chunks):
        table.append({"file": write(f"neighbors-{k}", blob), "first": int(first_row), "end": int(end_row),
                      "byte_start": byte_start, "bytes": len(blob)})
        byte_start += len(blob)
    byte_off = np.zeros(n + 1, np.uint64)
    np.cumsum(row_bytes, out=byte_off[1:])
    assert int(byte_off[-1]) < 2 ** 32
    files = {"offsets": write("offsets", byte_off.astype("<u4").tobytes()),
             "lengths": write("lengths", varint_encode(row_bytes.astype(np.uint64))),
             "ids": write("ids", (ids[order] - ID_BASE).astype("<u4").tobytes())}
    np.save(pathgraph_dir / "new_of_old.npy", new_of_old)      # for build_index's id shards
    meta = {"version": 2, "authors": int(n), "directed_links": int(deg_new.sum()),
            "suspect_links": n_suspect, "id_base": ID_BASE, "files": files,
            "encoding": "per author: sorted neighbours as varints of (delta << 1 | suspect); the first "
                        "delta is zigzag(neighbour - author), the rest are gaps; varint = 7 bits per byte, "
                        "low first, high bit = continue; suspect = probably two namesakes (pipeline.identity)",
            "offsets": "uint32 LE byte offset of each author's list in the concatenated chunks, authors+1 entries",
            "lengths": "the same as varint byte lengths, one per author (the browser's one-time index)",
            "ids": "uint32 LE OpenAlex id minus id_base, by author number",
            "chunks": table, "seconds": round(time.time() - t0, 1)}
    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=1))
    return meta


def read_neighbors(out_dir: Path, node: int) -> list[tuple[int, bool]]:
    """Reference decoder (web/dev.html's decodePathList mirrors it):
    [(neighbour, suspect)] in neighbour order."""
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "manifest.json").read_text())
    off = np.fromfile(out_dir / meta["files"]["offsets"], dtype="<u4")
    a, b = int(off[node]), int(off[node + 1])
    chunk = next(c for c in meta["chunks"] if c["first"] <= node < c["end"])
    with open(out_dir / chunk["file"], "rb") as f:
        f.seek(a - chunk["byte_start"])
        buf = f.read(b - a)
    out, val, shift, prev = [], 0, 0, None
    for byte in buf:
        val |= (byte & 0x7F) << shift
        if byte & 0x80:
            shift += 7
            continue
        flag, val = bool(val & 1), val >> 1
        if prev is None:
            prev = node + ((val >> 1) ^ -(val & 1))
        else:
            prev = prev + val
        out.append((prev, flag))
        val, shift = 0, 0
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pathgraph", default=None)
    p.add_argument("--web", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    d = data_dir()
    web = args.web or str(d / "coords_web.parquet")
    suspect = suspect_links(duckdb.connect(), str(d / "graph" / "nodes_int32.parquet"),
                            str(d / "graph" / "edges_int32.parquet"), web)
    meta = export_paths(Path(args.pathgraph or d / "pathgraph"), web, Path(args.out or d / "paths"),
                        suspect=suspect)
    total = sum(c["bytes"] for c in meta["chunks"])
    print(f"{meta['authors']:,} authors, {meta['directed_links']:,} links -> {len(meta['chunks'])} chunks, "
          f"{total / 1e9:.2f} GB ({total / meta['directed_links']:.2f} bytes/link), "
          f"{meta['suspect_links']:,} directed links flagged as likely namesakes, in {meta['seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
