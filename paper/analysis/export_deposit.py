"""Stage the paper's reproducible deposit beside the selected network, with checksums.

researchers.parquet retains the original node order and repair mapping; links.parquet is a hard link
(or verified fresh copy) of graph/edges_int32.parquet. Flagged links are emitted once, src < dst.
Twin/split CSVs retain all selected pairs and say whether removed endpoints prevented joining.

Researcher attributes stream in 250,000-row batches. Only names needed by selected pairs remain in
memory. Flagged links stream over CSR entry slices without constructing a billion-entry source array.
Five-work researchers also retain the exact fallback-topic exclusion and core-layout coordinates.
Fallback membership uses a memory-limited exact-string semi-join; coordinates use numeric ID lookup.
Every completed file replaces its predecessor atomically; SHA256SUMS is written last.

--small-files rewrites only communities.csv and README.md, then SHA256SUMS, after a full export (for README or
label edits); it stops with an error if a large file no longer matches its previous checksum.
"""
import argparse
import hashlib
import json
import os
import shutil

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common import CACHE, D, G, ND, NET, PG, RESULTS, ROOT, load_csr, log, split_keep

BATCH = 250_000
ENTRY_BATCH = 2_000_000
SCRIPT_LABELS = (("zh", "Chinese characters"), ("ja", "Japanese script"), ("ko", "Hangul"), ("ru", "Cyrillic"))
FLAG_LABELS = ("", "merged", "split", "not_a_person")
FALLBACK_TOPIC = "Legal and Regulatory Analysis"
PACKAGE_FILES = ("researchers.parquet", "links.parquet", "flagged_links.parquet", "twin_pairs.csv",
                 "split_pairs.csv", "communities.csv", "README.md")
SMALL_FILES = ("communities.csv", "README.md")
# The deposit's name for a script community whose code key predates A5c, which found it is Japanese written surname
# first (Section 4, Table 3); the code keeps the key "Mixed CJK script"
DEPOSIT_LABELS = {"Mixed CJK script": "Japanese, surname first"}


def temporary_path(path):
    return path.with_name("." + path.name + ".export-tmp")


def atomic_csv(frame, path):
    temporary = temporary_path(path)
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def flag_codes(n, flags, rep):
    """Compact flag array with the original later-kind-wins priority."""
    codes = np.zeros(n, np.uint8)
    for code, kind in enumerate(("merged", "split", "notPerson"), start=1):
        indices = np.asarray(flags[kind], dtype=np.int64)
        assert np.all((indices >= 0) & (indices < n)), f"invalid {kind} flag index"
        codes[indices] = code
    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        assert np.array_equal(rep[start:end] < 0, codes[start:end] != 0), "flags and the repaired network disagree"
        assert np.all(rep[start:end] >= -1), "repair mapping has invalid removed indices"
    return codes


def selected_pairs(repair, offsets, cache=CACHE):
    """Apply the original twin and split rules; read only the columns the deposit uses."""
    twins = []
    for script, label in SCRIPT_LABELS:
        threshold = repair["merged_pairs"][f"twins_{script}"]["threshold"]
        columns = ["focal_idx", "twin_idx", "shared", "coauthors"]
        frame = pd.read_csv(cache / f"a6_twins_{script}.csv", usecols=columns,
                            dtype={column: np.int64 for column in columns})
        frame = frame[frame.shared >= threshold].copy()
        frame["script"] = label
        frame["threshold"] = threshold
        twins.append(frame)
    twins = pd.concat(twins, ignore_index=True)
    splits = []
    for batch in pq.ParquetFile(cache / "split_pairs.parquet").iter_batches(
            batch_size=BATCH, columns=["a", "b", "shared", "group_size"]):
        frame = batch.to_pandas()
        a, b = frame.a.to_numpy(dtype=np.int64), frame.b.to_numpy(dtype=np.int64)
        small = np.minimum(offsets[a + 1] - offsets[a], offsets[b + 1] - offsets[b])
        keep = split_keep(frame.group_size, frame.shared, small)
        splits.append(frame[keep])
    splits = (pd.concat(splits, ignore_index=True) if splits else
              pd.DataFrame({key: pd.Series(dtype="int64") for key in ("a", "b", "shared", "group_size")}))
    assert len(splits) == repair["merged_pairs"]["splits"]
    selected = np.unique(np.concatenate([twins.focal_idx.to_numpy(), twins.twin_idx.to_numpy(),
                                        splits.a.to_numpy(), splits.b.to_numpy()]).astype(np.int64))
    assert np.all((selected >= 0) & (selected < len(offsets) - 1)), "pair endpoint outside node index"
    return twins, splits, selected


def openalex_ids(ids):
    return pc.binary_join_element_wise(pa.scalar("A", type=pa.large_string()),
                                      pa.array(ids).cast(pa.large_string()), pa.scalar("", type=pa.large_string()))


def matching_nodes(source_ids, keys, order):
    """attrs.py's numeric coordinate join, preserving coordinate-source order."""
    numeric = pc.struct_field(pc.extract_regex(source_ids, r"(?P<num>\d+)$"), "num").cast(pa.int64())
    rows = np.flatnonzero(pc.is_valid(numeric).to_numpy(zero_copy_only=False))
    values = pc.take(numeric, pa.array(rows)).to_numpy()
    positions = np.minimum(np.searchsorted(keys, values), len(keys) - 1)
    present = keys[positions] == values
    return rows[present], order[positions[present]]


def fallback_membership(attrs_path, author_paths, n, batch_size=BATCH):
    """A9's exact ID-string set membership, streamed from a bounded DuckDB semi-join.

    The source uses full URLs. URL and short aliases are deliberately not normalized: a record
    matches only the string present in node_attrs.id. EXISTS preserves any-record semantics for
    duplicate extracts while emitting each attribute row once. No all-author Python strings persist.
    """
    assert pq.ParquetFile(attrs_path).metadata.num_rows == n, "attribute row count differs from node IDs"
    result = np.zeros(n, bool)
    log(f"deposit fallback topics: exact-ID join over {len(author_paths):,} files (DuckDB 4GB, 2 threads)")
    db = duckdb.connect(config={"memory_limit": "4GB", "threads": "2",
                                "temp_directory": str(attrs_path.parent / "duck_export")})
    reader = None
    try:
        reader = db.sql("""SELECT a.node_idx FROM read_parquet(?) a
            WHERE EXISTS (SELECT 1 FROM read_parquet(?) f
                          WHERE f.id = a.id AND f.topics[1].display_name = ?)""",
            params=[str(attrs_path), [str(path) for path in author_paths], FALLBACK_TOPIC]).to_arrow_reader(batch_size)
        for batch in reader:
            nodes = batch.column("node_idx").to_numpy()
            assert np.all((nodes >= 0) & (nodes < n)), "fallback join returned an invalid node index"
            result[nodes] = True
    finally:
        if reader is not None:
            reader.close()
        db.close()
    return result


def reproducibility_arrays(ids, attrs_path, author_paths, coords_path, batch_size=BATCH):
    """Align fallback membership and exact nullable double coordinates with original node order.

    Duplicate author-extract records retain the existing set-membership semantics: any matching
    first fallback topic excludes the exactly matching node_attrs.id string. Coordinate IDs must be unique. Layout membership is
    independent of missing/nonfinite coordinate values, as in attrs.py's original join.
    """
    author_paths = list(author_paths)
    if not author_paths:
        raise FileNotFoundError("No authors_extract parquet files for fallback-topic membership")
    n = len(ids)
    assert n > 0 and batch_size > 0
    order = np.argsort(ids)
    keys = ids[order]
    assert np.all(keys[1:] > keys[:-1]), "duplicate researcher IDs"
    # Close/release the string join before allocating coordinate arrays.
    fallback = fallback_membership(attrs_path, author_paths, n, batch_size)
    out = {"is_fallback_topic": fallback, "on_map": np.zeros(n, bool),
           "xw": np.full(n, np.nan, np.float64), "yw": np.full(n, np.nan, np.float64),
           "xw_valid": np.zeros(n, bool), "yw_valid": np.zeros(n, bool)}
    count = 0
    for batch in pq.ParquetFile(coords_path).iter_batches(batch_size=batch_size, columns=["id", "xw", "yw"]):
        rows, nodes = matching_nodes(batch.column("id"), keys, order)
        assert len(np.unique(nodes)) == len(nodes) and not out["on_map"][nodes].any(), "duplicate map coordinate IDs"
        out["on_map"][nodes] = True
        for key in ("xw", "yw"):
            values = pc.take(batch.column(key), pa.array(rows)).cast(pa.float64())
            out[key][nodes] = values.to_numpy(zero_copy_only=False)
            out[key + "_valid"][nodes] = pc.is_valid(values).to_numpy(zero_copy_only=False)
        count += len(batch)
    log(f"deposit metadata: {int(out['is_fallback_topic'].sum()):,} fallback records; "
        f"{int(out['on_map'].sum()):,} mapped of {count:,} coordinate rows")
    return out


def export_researchers(attrs_path, path, ids, rep, codes, selected, network=NET, batch_size=BATCH,
                       supplementary=None):
    """Write original-order rows and retain a small node-indexed name lookup for pair CSVs."""
    n = len(ids)
    reader = pq.ParquetFile(attrs_path)
    assert reader.metadata.num_rows == n == len(rep) == len(codes)
    columns = ["node_idx", "name", "works_kept", "works", "cited", "degree", "community", "country", "institution", "field"]
    if network != "min15":
        columns.append("on_map")
        assert supplementary is not None, "five-work export needs fallback topics and layout coordinates"
        assert all(len(supplementary[key]) == n for key in
                   ("is_fallback_topic", "on_map", "xw", "yw", "xw_valid", "yw_valid"))
    labels = pa.array(FLAG_LABELS, type=pa.large_string())
    temporary = temporary_path(path)
    writer = None
    names = []
    start = 0
    try:
        for batch in reader.iter_batches(batch_size=batch_size, columns=columns):
            end = start + len(batch)
            nodes = np.arange(start, end, dtype=np.int32)
            assert np.array_equal(batch.column("node_idx").to_numpy(), nodes), "attributes are not in node order"
            data = {"node": pa.array(nodes), "openalex_id": openalex_ids(ids[start:end]),
                    "name": batch.column("name").cast(pa.large_string()), "works": batch.column("works_kept"),
                    "works_openalex": batch.column("works"), "citations": batch.column("cited"),
                    "coauthors": batch.column("degree"), "community": batch.column("community")}
            if network != "min15":
                data["on_map"] = batch.column("on_map").cast(pa.bool_())
                assert np.array_equal(data["on_map"].to_numpy(zero_copy_only=False), supplementary["on_map"][start:end]), "map membership differs from node attributes"
            data.update({"country": batch.column("country").cast(pa.large_string()),
                         "institution": batch.column("institution").cast(pa.large_string()),
                         "field": batch.column("field").cast(pa.large_string()),
                         "flag": pc.take(labels, pa.array(codes[start:end])),
                         "repaired_node": pa.array(rep[start:end].astype(np.int32))})
            if network != "min15":
                data["is_fallback_topic"] = pa.array(supplementary["is_fallback_topic"][start:end], type=pa.bool_())
                for key in ("xw", "yw"):
                    data[key] = pa.array(supplementary[key][start:end], type=pa.float64(),
                                         mask=~supplementary[key + "_valid"][start:end])
            table = pa.table(data)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
            wanted = selected[np.searchsorted(selected, start):np.searchsorted(selected, end)]
            if len(wanted):
                names.append(pa.table({"node": pa.array(wanted), "name": pc.take(data["name"], pa.array(wanted - start))}))
            start = end
            if start % 2_000_000 < len(batch) or start == n:
                log(f"deposit researchers: {start:,}/{n:,}")
    finally:
        if writer is not None:
            writer.close()
    assert start == n and writer is not None
    temporary.replace(path)
    if not names:
        return pd.Series(dtype="str", name="name", index=pd.Index([], dtype="int64", name="node"))
    lookup = pa.concat_tables(names).to_pandas().set_index("node").name
    assert lookup.index.is_unique and len(lookup) == len(selected)
    return lookup


def stage_links(source, target):
    """Reuse only the same inode; replace an existing stale file through a fresh hard link/copy."""
    if target.exists() and not target.is_symlink() and os.path.samefile(source, target):
        return "existing hard link"
    before = source.stat()
    temporary = temporary_path(target)
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    try:
        os.link(source, temporary)
        method = "hard link"
    except OSError:
        shutil.copy2(source, temporary)
        method = "copy"
    after = source.stat()
    assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns), "link source changed during export"
    assert temporary.stat().st_size == after.st_size
    temporary.replace(target)
    return method


def export_flagged_links(offsets, neighbors, mask, path, batch_size=ENTRY_BATCH, repair=None):
    """Emit selected CSR entries in their original order, with one row per undirected link; with `repair` (a subset
    mask of the same entries), a boolean column marks the links the repaired network removes."""
    assert len(mask) == len(neighbors) == offsets[-1]
    assert repair is None or len(repair) == len(mask)
    fields = [("src", pa.int32()), ("dst", pa.int32())] + ([("repair", pa.bool_())] if repair is not None else [])
    schema = pa.schema(fields)
    temporary = temporary_path(path)
    count = entries = 0
    with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
        for start in range(0, len(neighbors), batch_size):
            end = min(start + batch_size, len(neighbors))
            selected = np.flatnonzero(mask[start:end]) + start
            entries += len(selected)
            if end % 100_000_000 < end - start or end == len(neighbors):
                log(f"deposit flagged links: scanned {end:,}/{len(neighbors):,} directed entries")
            if not len(selected):
                continue
            src = (np.searchsorted(offsets, selected, side="right") - 1).astype(np.int32)
            dst = np.asarray(neighbors[selected], np.int32)
            keep = src < dst
            columns = {"src": src[keep], "dst": dst[keep]}
            if repair is not None:
                columns["repair"] = np.asarray(repair[selected], bool)[keep]
            table = pa.table(columns, schema=schema)
            writer.write_table(table)
            count += len(table)
    assert entries == 2 * count, "flagged CSR mask must be symmetric and contain no self links"
    temporary.replace(path)
    return count, entries


def pair_frames(twins, splits, ids, rep, names):
    """Construct the CSV columns in their original order, including unjoined removed endpoints."""
    removed = rep < 0
    a, b = twins.focal_idx.to_numpy(dtype=np.int64), twins.twin_idx.to_numpy(dtype=np.int64)
    twin_frame = pd.DataFrame({
        "script": twins.script, "native_node": a, "latin_node": b,
        "native_id": openalex_ids(ids[a]).to_pandas(), "latin_id": openalex_ids(ids[b]).to_pandas(),
        "native_name": names.reindex(a).to_numpy(), "latin_name": names.reindex(b).to_numpy(),
        "shared_coauthors": twins.shared, "native_coauthors": twins.coauthors, "threshold": twins.threshold,
        "joined": ~(removed[a] | removed[b])})
    a, b = splits.a.to_numpy(dtype=np.int64), splits.b.to_numpy(dtype=np.int64)
    split_frame = pd.DataFrame({
        "node_a": a, "node_b": b, "id_a": openalex_ids(ids[a]).to_pandas(), "id_b": openalex_ids(ids[b]).to_pandas(),
        "name": names.reindex(a).to_numpy(), "shared_coauthors": splits.shared,
        "profiles_with_name": splits.group_size, "joined": ~(removed[a] | removed[b])})
    return twin_frame, split_frame


def write_checksums(out):
    sums = []
    for name in sorted(PACKAGE_FILES):
        path = out / name
        log(f"deposit checksum: {name} ({path.stat().st_size / 1e6:,.1f} MB)")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1 << 24), b""):
                digest.update(chunk)
        sums.append(f"{digest.hexdigest()}  {name}")
    temporary = temporary_path(out / "SHA256SUMS")
    temporary.write_text("\n".join(sums) + "\n")
    temporary.replace(out / "SHA256SUMS")


def write_small_files(out, cache=CACHE, readme=None):
    """communities.csv from the community profiles and names, and README.md from the paper's deposit README."""
    if NET == "min15":
        labels = {r["community"]: r["name"] for r in json.loads((D / "regions.json").read_text()) if r.get("zmin", 9) <= 2}
    else:
        labels = {int(k): DEPOSIT_LABELS.get(v, v)
                  for k, v in json.loads((cache / "community_labels.json").read_text()).items()}
    communities = pd.read_csv(cache / "a5b_communities.csv")
    frame = pd.DataFrame({"community": communities.community, "map_label" if NET == "min15" else "label": communities.community.map(labels),
                          "members": communities["size"], "national": communities.top_country_share >= 0.5,
                          "top_country": communities.top_country, "top_country_share": communities.top_country_share.round(4),
                          "top_fields": communities.top_fields, "weight_inside": communities.cohesion.round(4),
                          "median_coauthors": communities.median_degree, "median_citations": communities.median_cited})
    atomic_csv(frame, out / "communities.csv")
    log(f"communities: {len(frame)}")
    readme = readme or ROOT / "paper" / ("deposit_README.md" if NET != "min15" else "deposit_README_min15.md")
    temporary = temporary_path(out / "README.md")
    shutil.copy2(readme, temporary)
    temporary.replace(out / "README.md")


def checksum_entries(out):
    return {name: digest for digest, name in (line.split() for line in (out / "SHA256SUMS").read_text().splitlines())}


def refresh_small_files(out, cache=CACHE, readme=None):
    """Rewrite the small files and SHA256SUMS; the large files must keep the checksums of the last full export."""
    before = checksum_entries(out)
    write_small_files(out, cache=cache, readme=readme)
    write_checksums(out)
    after = checksum_entries(out)
    changed = sorted(name for name in PACKAGE_FILES if name not in SMALL_FILES and before.get(name) != after.get(name))
    if changed:
        raise RuntimeError(f"large deposit files changed since the last full export: {changed}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--small-files", action="store_true",
                        help="rewrite only communities.csv and README.md, then SHA256SUMS")
    args = parser.parse_args()
    out = ND / "deposit"
    if args.small_files:
        refresh_small_files(out)
        log({name: f"{(out / name).stat().st_size:,} bytes" for name in (*SMALL_FILES, "SHA256SUMS")})
        return
    out.mkdir(exist_ok=True)
    ids = np.load(PG / "ids.npy", mmap_mode="r")
    rep = np.load(CACHE / "repair_rep.npy", mmap_mode="r")
    n = len(ids)
    assert n <= np.iinfo(np.int32).max and rep.max() <= np.iinfo(np.int32).max
    with np.load(CACHE / "flags.npz") as flags:
        codes = flag_codes(n, flags, rep)
    offsets, neighbors = load_csr()
    repair = json.loads((RESULTS / "a8_repair.json").read_text())
    twins, splits, selected = selected_pairs(repair, offsets, cache=CACHE)
    supplementary = (reproducibility_arrays(ids, CACHE / "node_attrs.parquet", sorted((D / "authors_extract").glob("*/*.parquet")),
                                            D / "coords_web.parquet") if NET != "min15" else None)
    names = export_researchers(CACHE / "node_attrs.parquet", out / "researchers.parquet", ids, rep, codes, selected,
                              network=NET, supplementary=supplementary)
    log(f"researchers: {n:,}; flagged {int(np.count_nonzero(codes)):,}; repaired nodes {int(rep.max()) + 1:,}")
    del codes, selected, supplementary
    log("links:", stage_links(G / "edges_int32.parquet", out / "links.parquet"))
    mask = np.load(CACHE / "suspect_mask.npy", mmap_mode="r")
    repair = np.load(CACHE / "repair_link_mask.npy", mmap_mode="r")
    count, entries = export_flagged_links(offsets, neighbors, mask, out / "flagged_links.parquet", repair=repair)
    log(f"flagged links: {count:,} (mask entries {entries:,})")
    twin_frame, split_frame = pair_frames(twins, splits, ids, rep, names)
    atomic_csv(twin_frame, out / "twin_pairs.csv")
    atomic_csv(split_frame, out / "split_pairs.csv")
    log(f"twin pairs: {len(twin_frame):,}", twin_frame.groupby("script").size().to_dict())
    log(f"split pairs: {len(split_frame):,}")
    write_small_files(out)
    write_checksums(out)
    log({name: f"{(out / name).stat().st_size / 1e6:,.1f} MB" for name in (*PACKAGE_FILES, "SHA256SUMS")})


if __name__ == "__main__":
    main()
