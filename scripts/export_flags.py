"""Write docs/flags-<release>.json: profiles the site warns about and Connect routes around (R46).

Three kinds, from the paper's analysis of the 2026-09-23 snapshot (paper/analysis, outputs in paper/cache):
  merged     25+ institutions in a single year in OpenAlex's own affiliation list (a6_affiliations.py,
             the 50,000 highest-degree profiles), a common name (200+ OpenAlex records share its initial
             and surname, pipeline/identity.py) and a common full name (10+ mapped profiles carry it):
             several people sharing a name.
  split      a common name, and OpenAlex has broken the profile up since the snapshot: it lost at least
             half its works by the live API a month later (a6_live_works.py) or in OpenAlex's next
             snapshot, 2026-10-14 (a6_churn_snapshot.py, which covers all 50,000 hubs). On the map it
             still mixes several people.
  notPerson  the name is an organization, team, typesetter, editorial office or placeholder
             (a6_nonperson.py's name patterns, people with malformed names left out).
A profile in several kinds keeps the last of merged, split, notPerson. Uncommon names are left out of
merged and split: there, 25+ institutions mostly means a prolific clinician credited with trial sites
(Eugene Braunwald, Michael Gnant; with a common surname but a rare full name, Manesh R. Patel or George
Davey Smith), and a lost half means OpenAlex moved a duplicate's works into the person's main profile.

Each kind is stored as columns: n (Connect's author number, scripts/export_paths_bin.py's Z-order
renumbering), id (OpenAlex numeric id minus the paths manifest's id_base) and the kind's details.

    python -m scripts.export_flags [--release 2026-09-23]
"""
import argparse
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MERGED_PEAK = 25          # institutions in one year
SPLIT_RATIO = 0.5         # live works / snapshot works
COMMON_NAME = 200         # records sharing the name key (pipeline/identity.py's threshold)
COMMON_FULL_NAME = 10     # mapped profiles sharing the whole name


def full_name_counts(cache: Path) -> pd.DataFrame:
    """node_idx -> how many mapped profiles carry the same name (ASCII-folded, lowercase, letters only)."""
    a = pd.read_parquet(cache / "node_attrs.parquet", columns=["node_idx", "name"])
    folded = a.name.fillna("").map(lambda s: unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower())
    key = folded.str.replace(r"[^a-z ]", " ", regex=True).str.split().str.join(" ")
    return pd.DataFrame({"node_idx": a.node_idx, "full_n": key.map(key.value_counts())})


def author_numbers(release_dir: Path, node_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pathgraph node indices -> (Connect author numbers, ids minus id_base)."""
    ids = np.load(release_dir / "pathgraph" / "ids.npy", mmap_mode="r")
    manifest = json.loads((release_dir / "paths" / "manifest.json").read_text())
    base = manifest["id_base"]
    by_number = np.fromfile(release_dir / "paths" / manifest["files"]["ids"], dtype="<u4")
    order = np.argsort(by_number)
    want = (np.asarray(ids[node_idx]) - base).astype(np.uint32)
    pos = np.searchsorted(by_number, want, sorter=order)
    numbers = order[np.minimum(pos, len(order) - 1)]
    assert (by_number[numbers] == want).all(), "a flagged profile is missing from the paths ids file"
    return numbers.astype(np.int64), want.astype(np.int64)


def next_snapshot_splits(release_dir: Path, cache: Path, key_n: pd.DataFrame, exclude: set) -> pd.DataFrame:
    """Common-name hubs (the 50,000 highest-degree) with under half their works in OpenAlex's next snapshot
    (cache/snapshot_20261014.parquet, from paper/analysis/a6_churn_snapshot.py): node_idx, works, works_live."""
    path = cache / "snapshot_20261014.parquet"
    if not path.exists():
        return pd.DataFrame({c: pd.Series(dtype="int64") for c in ("node_idx", "works", "works_live")})
    offsets = np.load(release_dir / "pathgraph" / "offsets.npy", mmap_mode="r")
    top = np.argsort(-np.diff(offsets), kind="stable")[:50_000]
    ids = np.load(release_dir / "pathgraph" / "ids.npy", mmap_mode="r")
    old = pd.read_parquet(release_dir / "nodes.parquet", columns=["id", "works_count"])
    old = pd.Series(old.works_count.to_numpy(), index=old["id"].str.extract(r"A(\d+)$")[0].astype("int64"))
    new = pd.read_parquet(path).set_index("id").works_count
    hub_ids = np.asarray(ids[top]).astype("int64")
    df = pd.DataFrame({"node_idx": top, "works": old.reindex(hub_ids).to_numpy(), "works_live": new.reindex(hub_ids).to_numpy()})
    common = key_n.set_index("node_idx").key_n.reindex(top).to_numpy() >= COMMON_NAME
    df = df[common & (df.works_live < SPLIT_RATIO * df.works) & ~df.node_idx.isin(exclude)]
    return df.astype({"works": int, "works_live": int})


def build(release: str, cache: Path) -> dict:
    release_dir = ROOT / "data" / release
    key_n = pd.read_parquet(release_dir / "pathgraph" / "identity_signals.parquet", columns=["node_idx", "key_n"])
    aff = pd.read_csv(cache / "a6_affiliations.csv").merge(key_n, on="node_idx").merge(full_name_counts(cache), on="node_idx")
    aff = aff[(aff.group == "top") & aff.alive & (aff.peak_insts >= MERGED_PEAK) & (aff.key_n >= COMMON_NAME)
              & (aff.full_n >= COMMON_FULL_NAME)]
    live = pd.read_csv(cache / "a6_live_works.csv").merge(key_n, on="node_idx")
    live = live[live.alive & (live.works_ratio < SPLIT_RATIO) & (live.key_n >= COMMON_NAME)]
    census = next_snapshot_splits(release_dir, cache, key_n, exclude=set(live.node_idx))
    live = pd.concat([live[["node_idx", "works", "works_live"]], census], ignore_index=True).astype("int64")
    nonperson = np.load(cache / "nonperson_idx.npy")
    kinds = {
        "merged": (aff.node_idx.to_numpy(), {"peak": aff.peak_insts.astype(int), "year": aff.peak_year.astype(int)}),
        "split": (live.node_idx.to_numpy(), {"then": live.works.astype(int), "now": live.works_live.astype(int)}),
        "notPerson": (nonperson, {}),
    }
    seen, flags = set(), {}
    for kind in reversed(list(kinds)):                     # later kinds win: drop earlier duplicates
        idx, cols = kinds[kind]
        keep = np.array([i not in seen for i in idx.tolist()], bool)
        seen.update(idx.tolist())
        numbers, ids = author_numbers(release_dir, idx[keep])
        order = np.argsort(numbers)
        flags[kind] = {"n": numbers[order].tolist(), "id": ids[order].tolist(),
                       **{k: np.asarray(v)[keep][order].tolist() for k, v in cols.items()}}
    return {"release": release, "flags": {k: flags[k] for k in kinds}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--release", default="2026-09-23")
    ap.add_argument("--cache", type=Path, default=ROOT / "paper" / "cache")
    args = ap.parse_args()
    out = build(args.release, args.cache)
    path = ROOT / "docs" / f"flags-{args.release}.json"
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(path, {k: len(v["n"]) for k, v in out["flags"].items()}, f"{path.stat().st_size / 1e3:.0f} kB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
