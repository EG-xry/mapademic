"""A6i: institution-history checks, pinned to the 2026-10-14 comparison snapshot.

Reuse ID-keyed institution features from the archived min15 run, recalculate
all network-specific name and degree signals, and acquire missing features
only. The legacy features were acquired 2026-10-24; their release is recorded
in project memory, but no manifest hash survives. New reads require both the
expected live S3 release and works-count agreement with the cached comparison
snapshot, and checkpoint each file. A changed bucket fails rather than mixing
releases. Results and a sidecar record this provenance limitation explicitly.
"""
import hashlib
import json
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.fs as pafs
import pyarrow.parquet as pq
from scipy.stats import rankdata

from common import CACHE, PAPER, PG, con, log, save_result

HOME_YEARS = 3
RELEASE = "2026-10-14"
LEAVES = ["affiliations.list.element.institution.id", "affiliations.list.element.years.list.element"]
FEATURES = ["id", "peak", "homes", "concurrent", "home_share"]


def features(affs):
    years_of = {}
    for inst, years in affs:
        if inst:
            years_of.setdefault(inst, set()).update(y for y in years if y)
    per_year = {}
    for years in years_of.values():
        for year in years:
            per_year[year] = per_year.get(year, 0) + 1
    if not per_year:
        return 0, 0, 0, 0.0
    homes = [(min(years), max(years)) for years in years_of.values() if len(years) >= HOME_YEARS]
    concurrent = max((sum(1 for a, b in homes if a <= year <= b) for year in per_year), default=0)
    home_share = max(len(years) for years in years_of.values()) / len(per_year)
    return max(per_year.values()), len(homes), concurrent, home_share


def auc_pairwise(positive, negative):
    """P(positive > negative) + half ties, without an n-by-m matrix."""
    if not len(positive) or not len(negative):
        return None
    rank = rankdata(np.concatenate([positive, negative]))
    n, m = len(positive), len(negative)
    return float((rank[:n].sum() - n * (n + 1) / 2) / (n * m))


def release_state(s3):
    with s3.open_input_file("openalex/RELEASE_NOTES.txt") as f:
        notes = f.read().decode("utf8")
    match = re.search(r"RELEASE (\d{4}-\d{2}-\d{2})", notes)
    if not match:
        raise RuntimeError("cannot identify OpenAlex snapshot release")
    if match.group(1) != RELEASE:
        raise RuntimeError(f"OpenAlex S3 is {match.group(1)}, expected {RELEASE}; refusing to mix snapshots")
    with s3.open_input_file("openalex/data/parquet/manifest.json") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return {"release": RELEASE, "manifest_sha256": digest}


def read_features(s3, path, wanted, expected_works, attempts=5):
    for attempt in range(attempts):
        try:
            with s3.open_input_file(path) as f:
                parquet = pq.ParquetFile(f, pre_buffer=False)
                table = parquet.read(columns=["id", "works_count"])
                numeric = pc.cast(pc.utf8_slice_codeunits(table.column("id"), 22), "int64").to_numpy()
                keep = np.flatnonzero(np.isin(numeric, wanted))
                if not len(keep):
                    return []
                works = table.column("works_count").to_numpy()[keep]
                for identifier, count in zip(numeric[keep], works):
                    if expected_works[int(identifier)] != count:
                        raise RuntimeError(f"snapshot mismatch for A{identifier}: cached and S3 works counts differ")
                affs = parquet.read(columns=LEAVES).column("affiliations").take(keep).to_pylist()
                return [(int(identifier), *features([((x.get("institution") or {}).get("id"), x.get("years") or [])
                                                    for x in (aff or [])])) for identifier, aff in zip(numeric[keep], affs)]
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(5 * (attempt + 1))


def acquire_missing(wanted, expected_works):
    """At most four leaf-column reads active, resumable per source Parquet file."""
    s3 = pafs.S3FileSystem(anonymous=True, region="us-east-1")
    state = release_state(s3)
    identity = hashlib.sha256(np.sort(wanted).astype("<i8").tobytes()).hexdigest()
    checkpoint = CACHE / "a6_institutions_parts" / f"{RELEASE}_{identity[:16]}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    state.update({"requested_ids_sha256": identity, "requested": len(wanted)})
    manifest_path = checkpoint / "provenance.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if any(previous[k] != state[k] for k in state):
            raise RuntimeError("institution checkpoint belongs to a different snapshot or target population")
    else:
        state["started_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(state, indent=2) + "\n")
    files = sorted(f.path for f in s3.get_file_info(pafs.FileSelector("openalex/data/parquet/authors", recursive=True))
                   if f.path.endswith(".parquet"))
    parts, pending = [], []
    for path in files:
        part = checkpoint / (hashlib.sha256(path.encode()).hexdigest() + ".json")
        if part.exists():
            cached = json.loads(part.read_text())
            assert cached["path"] == path
            parts.extend(cached["rows"])
        else:
            pending.append((path, part))
    log(f"institutions: {len(wanted):,} missing profiles, {len(files) - len(pending)}/{len(files)} files checkpointed")
    iterator = iter(pending)
    with ThreadPoolExecutor(max_workers=4) as pool:
        active = {}
        def submit_one():
            item = next(iterator, None)
            if item is None:
                return
            path, part = item
            active[pool.submit(read_features, s3, path, wanted, expected_works)] = item
        for _ in range(4):
            submit_one()
        completed = len(files) - len(pending)
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                path, part = active.pop(future)
                rows = future.result()
                temp = part.with_suffix(".tmp")
                temp.write_text(json.dumps({"path": path, "rows": rows}) + "\n")
                temp.replace(part)
                parts.extend(rows)
                completed += 1
                if completed % 50 == 0 or completed == len(files):
                    log(f"institutions: {completed}/{len(files)} files, {len(parts):,}/{len(wanted):,} profiles")
                submit_one()
    final_state = release_state(s3)
    if final_state["manifest_sha256"] != state["manifest_sha256"]:
        raise RuntimeError("OpenAlex manifest changed during institution scan; results not accepted")
    result = pd.DataFrame(parts, columns=FEATURES).drop_duplicates("id", keep="last")
    missing = np.setdiff1d(wanted, result.id.to_numpy())
    if len(missing):
        raise RuntimeError(f"{len(missing)} institution profiles absent although present in comparison cache")
    state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    state["source_files"] = len(files)
    state["checkpoint"] = str(checkpoint)
    return result, state


def main():
    offsets = np.load(PG / "offsets.npy", mmap_mode="r")
    degree = np.diff(offsets)
    top = np.argsort(-degree, kind="stable")[:50_000]
    ids = np.load(PG / "ids.npy", mmap_mode="r")
    target = pd.DataFrame({"id": ids[top].astype(np.int64), "node_idx": top, "degree": degree[top]})
    del degree, top, ids, offsets
    c = con()
    c.execute("SET memory_limit='6GB'")
    c.execute("SET threads=4")
    c.register("targets", target)
    target = c.execute(f"""SELECT t.*, s.works_count, k.key_n, f.full_n FROM targets t
        LEFT JOIN read_parquet('{CACHE / 'snapshot_20261014.parquet'}') s USING (id)
        LEFT JOIN read_parquet('{PG / 'identity_signals.parquet'}') k USING (node_idx)
        LEFT JOIN read_parquet('{CACHE / 'full_name_counts.parquet'}') f USING (node_idx)""").fetchdf()
    c.close()
    alive = target[target.works_count.notna()].copy()
    want = alive.id.to_numpy()
    cache_path = CACHE / "a6_institutions.parquet"
    sidecar = CACHE / "a6_institutions.provenance.json"
    provenance = {"comparison_release": RELEASE, "home_years": HOME_YEARS, "target_profiles": len(target),
                  "present_in_comparison_snapshot": len(alive), "sources": []}
    blocks = []
    if cache_path.exists():
        if not sidecar.exists():
            raise RuntimeError("existing min5 institution cache lacks provenance; inspect before reusing")
        old_provenance = json.loads(sidecar.read_text())
        if old_provenance["comparison_release"] != RELEASE or old_provenance["home_years"] != HOME_YEARS:
            raise RuntimeError("institution feature cache has incompatible release or definition")
        old = pd.read_parquet(cache_path, columns=FEATURES)
        blocks.append(old[old.id.isin(want)])
        provenance["sources"] = old_provenance["sources"]
        provenance["reused_current_cache_profiles"] = len(blocks[-1])
    else:
        previous = PAPER / "cache_min15" / "a6_institutions.parquet"
        if previous.exists():
            prior_result = json.loads((PAPER / "results_min15" / "a6_institutions.json").read_text())
            if prior_result["home_years"] != HOME_YEARS:
                raise RuntimeError("archived institution features use a different home-year definition")
            old = pd.read_parquet(previous, columns=FEATURES)
            blocks.append(old[old.id.isin(want)])
            provenance["sources"].append({"cache": str(previous), "profiles": len(blocks[-1]),
                "acquired_date": "2026-10-24", "release": RELEASE,
                "release_evidence": "project memory; original run has no manifest hash",
                "limitation": "legacy snapshot provenance cannot be independently reverified from saved feature columns"})
    known = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame(columns=FEATURES)
    missing = np.setdiff1d(want, known.id.to_numpy())
    log(f"institutions: reusing {len(known):,}; {len(missing):,} profiles require snapshot reads")
    if len(missing):
        new, source = acquire_missing(missing, dict(zip(alive.id, alive.works_count)))
        known = pd.concat([known, new], ignore_index=True)
        provenance["sources"].append(source)
    feat = known.merge(alive.drop(columns="works_count"), on="id", validate="one_to_one")
    assert len(feat) == len(alive)
    temporary = cache_path.with_suffix(".tmp.parquet")
    feat.to_parquet(temporary, index=False)
    temporary.replace(cache_path)
    sidecar.write_text(json.dumps(provenance, indent=2) + "\n")
    groups = {"merged (strict)": (feat.peak >= 25) & (feat.key_n >= 200) & (feat.full_n >= 10),
              "busy, rare name": (feat.peak >= 25) & (feat.key_n < 50), "control (peak <= 5)": feat.peak <= 5}
    out = {"profiles": len(feat), "home_years": HOME_YEARS, "provenance": provenance, "groups": {}}
    for name, group in groups.items():
        x = feat[group]
        out["groups"][name] = {"n": len(x), **{f"{col}_median": float(x[col].median()) if len(x) else None
            for col in ("peak", "homes", "concurrent", "home_share")}, **{f"concurrent_ge_{k}":
            float((x.concurrent >= k).mean()) if len(x) else None for k in (3, 5, 10)}}
    merged, busy = feat[groups["merged (strict)"]], feat[groups["busy, rare name"]]
    for col in ("concurrent", "peak"):
        out[f"auc_{col}_merged_vs_busy"] = auc_pairwise(merged[col].to_numpy(), busy[col].to_numpy())
    save_result("a6_institutions", out)
    log(out)


if __name__ == "__main__":
    main()
