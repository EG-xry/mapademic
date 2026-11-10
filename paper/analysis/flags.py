"""Profiles flagged as merged, split or not a person, on the network in use: cache/flags.npz (node indices).

The rules of scripts/export_flags.py, which writes the same flags for the site from the map's network:
  merged     25+ institutions in a single year in OpenAlex's affiliation list (a6_affiliations.py, the 50,000
             best connected profiles), a common name key (200+ researchers in the network share its initial and
             surname) and a common full name (10+ profiles in the network carry it);
  split      a common name key, and the profile lost at least half its works by the live API a month later
             (a6_live_works.py) or in OpenAlex's next snapshot (a6_churn_snapshot.py), among the 50,000 best
             connected;
  notPerson  a6_nonperson.py's name patterns.
A profile in several kinds keeps the last of merged, split, notPerson.
"""
import numpy as np
import pandas as pd

from common import CACHE, ND, PG, full_name_counts, load_csr, log

MERGED_PEAK, SPLIT_RATIO, COMMON_NAME, COMMON_FULL_NAME = 25, 0.5, 200, 10


def build():
    key_n = pd.read_parquet(PG / "identity_signals.parquet", columns=["node_idx", "key_n"])
    aff = pd.read_csv(CACHE / "a6_affiliations.csv").merge(key_n, on="node_idx")
    aff = aff.merge(full_name_counts().rename("full_n").reset_index(), on="node_idx")
    merged = aff[(aff.group == "top") & aff.alive & (aff.peak_insts >= MERGED_PEAK) & (aff.key_n >= COMMON_NAME)
                 & (aff.full_n >= COMMON_FULL_NAME)].node_idx.to_numpy()
    live = pd.read_csv(CACHE / "a6_live_works.csv").merge(key_n, on="node_idx")
    split_live = live[live.alive & (live.works_ratio < SPLIT_RATIO) & (live.key_n >= COMMON_NAME)].node_idx.to_numpy()
    offsets, _ = load_csr()
    top = np.argsort(-np.diff(offsets), kind="stable")[:50_000]
    ids = np.load(PG / "ids.npy").astype(np.int64)
    old = pd.read_parquet(ND / "nodes.parquet", columns=["id", "works_count"])
    old = pd.Series(old.works_count.to_numpy(), index=old["id"].str.extract(r"A(\d+)$")[0].astype("int64"))
    new = pd.read_parquet(CACHE / "snapshot_20261014.parquet").set_index("id").works_count
    works, works_next = old.reindex(ids[top]).to_numpy(), new.reindex(ids[top]).to_numpy()
    common = key_n.set_index("node_idx").key_n.reindex(top).to_numpy() >= COMMON_NAME
    split_next = top[common & (works_next < SPLIT_RATIO * works)]
    split = np.union1d(split_live, split_next)
    nonperson = np.load(CACHE / "nonperson_idx.npy")
    kinds = {"merged": merged, "split": split, "notPerson": nonperson}
    seen, out = set(), {}
    for kind in reversed(list(kinds)):                   # later kinds win
        idx = np.asarray(kinds[kind], np.int64)
        out[kind] = np.array(sorted(set(idx.tolist()) - seen), np.int64)
        seen.update(idx.tolist())
    return out


if __name__ == "__main__":
    f = build()
    np.savez(CACHE / "flags.npz", **f)
    log({k: len(v) for k, v in f.items()})
