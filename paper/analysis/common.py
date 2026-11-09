"""Shared loaders and graph kernels for the paper's analyses.

The coauthor network is a pathgraph: compressed sparse rows (offsets int64
n+1, neighbors int32, both directions of every link). Node index i is author
ids[i] (numeric part of the OpenAlex id). Kernels are numba-compiled loops
over the CSR.

NET picks the network (environment variable, default min5):
- min5: the paper's network, the 19,848,920 researchers with at least 5
  works, built into data/2026-09-23/min5/; cache in paper/cache/, results
  in paper/results/.
- min15: the map's network, the 8,396,458 researchers with at least 15
  works (data/2026-09-23/ itself); cache_min15/ and results_min15/.
"""
import json
import os
import time
from pathlib import Path

import duckdb
import numpy as np
from numba import njit, prange

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data" / "2026-09-23"                 # the release: snapshot, extracts, map
NET = os.environ.get("NET", "min5")
if NET not in {"min5", "min15"}:
    raise ValueError(f"Unknown network {NET!r}; expected min5 or min15")
PAPER = ROOT / "paper"
if NET == "min15":
    ND, CACHE, RESULTS = D, PAPER / "cache_min15", PAPER / "results_min15"
else:
    ND, CACHE, RESULTS = D / "min5", PAPER / "cache", PAPER / "results"
MIN_WORKS = 15 if NET == "min15" else 5
PG = ND / "pathgraph"
G = ND / "graph"
FIGURES = PAPER / "figures"
for d in (CACHE, RESULTS, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

# identity.py's rule for a probable namesake merge
COMMON_NAME, RARE_REGION = 200, 0.05
# communities followed in A8: script-separated ones and their national counterparts. On the map's network, those of
# the 2026-09-23 layout; on the paper's, those of leiden.py, identified from their profiles (A5b) in
# cache/script_communities.json
if NET == "min15":
    SCRIPT_COMMUNITIES = {"Chinese characters": 17, "Japanese script": 18, "Japanese script (small)": 36,
                          "Arabic script": 48, "Russia": 22, "Korea": 14, "China (main)": 7, "Japan (main)": 6}
else:
    _sc = CACHE / "script_communities.json"
    SCRIPT_COMMUNITIES = json.loads(_sc.read_text()) if _sc.exists() else {}
# the script a display name is recorded in (B1, A4b); a name matching none is Latin
SCRIPTS = {"Chinese characters": r"^[一-鿿]{2,4}$", "Japanese kanji/kana": r"^[一-鿿々〆ヵヶぁ-ゟ゠-ヿ]+\s+[一-鿿々〆ヵヶぁ-ゟ゠-ヿ]+$",
           "Hangul": r"[가-힣]", "Cyrillic": r"[Ѐ-ӿ]", "Arabic script": r"[؀-ۿ]"}


# groups of OpenAlex's 26 fields for the field map (Fig. 5, map_fields.py): its four domains, with the physical
# sciences split four ways and the arts and humanities apart from the rest of the social sciences
FIELD_GROUPS = {
    "Medicine & health": ("Medicine", "Nursing", "Dentistry", "Health Professions", "Veterinary"),
    "Life sciences": ("Biochemistry, Genetics and Molecular Biology", "Immunology and Microbiology", "Neuroscience",
                      "Pharmacology, Toxicology and Pharmaceutics", "Agricultural and Biological Sciences"),
    "Earth & environment": ("Environmental Science", "Earth and Planetary Sciences"),
    "Chemistry & materials": ("Chemistry", "Materials Science", "Chemical Engineering"),
    "Engineering & computing": ("Engineering", "Computer Science", "Energy"),
    "Physics & mathematics": ("Physics and Astronomy", "Mathematics"),
    "Social sciences": ("Social Sciences", "Psychology", "Economics, Econometrics and Finance",
                        "Business, Management and Accounting", "Decision Sciences"),
    "Arts & humanities": ("Arts and Humanities",),
}


def name_script(names):
    """The script of each name in a pandas Series of display names."""
    out = names.map(lambda _: "Latin")
    for label, pat in SCRIPTS.items():
        out[names.fillna("").str.contains(pat, regex=True) & (out == "Latin")] = label
    return out


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save_result(name: str, obj) -> None:
    (RESULTS / f"{name}.json").write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def load_csr(mmap=True):
    mode = "r" if mmap else None
    return np.load(PG / "offsets.npy", mmap_mode=mode), np.load(PG / "neighbors.npy", mmap_mode=mode)


def load_ids():
    return np.load(PG / "ids.npy")


class Names:
    """Display name of node i, read lazily from names.bin."""
    def __init__(self):
        self.off = np.load(PG / "names_offsets.npy")
        self.blob = (PG / "names.bin").read_bytes()

    def __getitem__(self, i):
        return self.blob[self.off[i]:self.off[i + 1]].decode("utf-8", "replace")


def con():
    c = duckdb.connect()
    c.execute("SET memory_limit='10GB'")
    c.execute(f"SET threads={int(os.environ.get('MAPADEMIC_THREADS', '4'))}")
    c.execute(f"SET temp_directory='{CACHE / 'duck'}'")
    return c


# ---------------------------------------------------------------- kernels

@njit(cache=True)
def _find(parent, x):
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        nxt = parent[x]
        parent[x] = root
        x = nxt
    return root


@njit(cache=True)
def components(offsets, neighbors, skip):
    """Union-find over all links (skip[j] true drops directed entry j; empty skip = keep all).
    Returns the root label of every node."""
    n = offsets.shape[0] - 1
    parent = np.arange(n, dtype=np.int32)
    use_skip = skip.shape[0] > 0
    for u in range(n):
        for j in range(offsets[u], offsets[u + 1]):
            if use_skip and skip[j]:
                continue
            v = neighbors[j]
            if v > u:
                ru = _find(parent, u)
                rv = _find(parent, v)
                if ru != rv:
                    if ru < rv:
                        parent[rv] = ru
                    else:
                        parent[ru] = rv
    for u in range(n):
        parent[u] = _find(parent, u)
    return parent


@njit(cache=True)
def bfs(offsets, neighbors, skip, src, dist, queue):
    """Hop distance from src to every node (-1 unreachable) into dist (int16, reset here).
    Returns (number reached, farthest distance, one farthest node)."""
    n = offsets.shape[0] - 1
    for i in range(n):
        dist[i] = -1
    use_skip = skip.shape[0] > 0
    dist[src] = 0
    queue[0] = src
    head, tail = 0, 1
    last = src
    while head < tail:
        u = queue[head]
        head += 1
        du = dist[u] + 1
        for j in range(offsets[u], offsets[u + 1]):
            if use_skip and skip[j]:
                continue
            v = neighbors[j]
            if dist[v] < 0:
                dist[v] = du
                queue[tail] = v
                tail += 1
                last = v
    return tail, dist[last], last


@njit(cache=True)
def hist_int16(dist, maxd):
    h = np.zeros(maxd + 1, np.int64)
    for i in range(dist.shape[0]):
        d = dist[i]
        if d >= 0:
            h[d] += 1
    return h


@njit(parallel=True, cache=True)
def mark_pairs(offsets, neighbors, us, vs_sorted_by_u, starts, mask):
    """mask[j] = True for every directed CSR entry (u -> v) whose (u, v) is listed.
    us: distinct u values; for us[k], its listed v's are vs_sorted_by_u[starts[k]:starts[k+1]] (sorted)."""
    for k in prange(us.shape[0]):
        u = us[k]
        lo, hi = starts[k], starts[k + 1]
        vs = vs_sorted_by_u[lo:hi]
        for j in range(offsets[u], offsets[u + 1]):
            v = neighbors[j]
            p = np.searchsorted(vs, v)
            if p < vs.shape[0] and vs[p] == v:
                mask[j] = True


def suspect_mask(offsets, neighbors):
    """Directed-entry mask of links flagged as probable namesake merges (both directions)."""
    path = CACHE / "suspect_mask.npy"
    if path.exists():
        return np.load(path, mmap_mode="r")
    log("suspect links: reading link_signals")
    c = con()
    t = c.execute(f"""SELECT u, v FROM read_parquet('{PG / 'link_signals.parquet'}')
                      WHERE (ku >= {COMMON_NAME} AND s_uv < {RARE_REGION})
                         OR (kv >= {COMMON_NAME} AND s_vu < {RARE_REGION})""").fetchnumpy()
    u = np.concatenate([t["u"], t["v"]]).astype(np.int32)
    v = np.concatenate([t["v"], t["u"]]).astype(np.int32)
    log(f"suspect links: {len(t['u']):,} undirected")
    order = np.lexsort((v, u))
    u, v = u[order], v[order]
    us, starts = np.unique(u, return_index=True)
    starts = np.append(starts, len(u)).astype(np.int64)
    mask = np.zeros(neighbors.shape[0], np.bool_)
    mark_pairs(offsets, neighbors, us, v, starts, mask)
    np.save(path, mask)
    log(f"suspect mask: {int(mask.sum()):,} directed entries marked")
    return mask


# Repair rules (revised 2026-11-03 after the five-work review).
# Links: the repair removes a suspect link only when its common-name endpoint (initial-and-surname key shared by
# COMMON_NAME+ profiles) also has a full name shared by COMMON_FULL_NAME+ retained profiles, flags.py's merge
# threshold. A distinctive full name makes a merge of namesakes unlikely, so the links of such a researcher into
# communities holding little of their link weight are kept as collaboration (worldwide consortia). Names that fold
# to no Latin letters share one empty full-name key and so keep the broad rule. A merged profile can still display a
# distinctive name, so among the 50,000 highest-degree profiles one whose own OpenAlex name record lists two or more
# given names that cannot be its displayed one (a6_name_records.py) is treated like a common full name.
# Same-name joins: name groups of at most SPLIT_GROUP_MAX profiles need SPLIT_MIN_SHARED+ shared coauthors making up
# SPLIT_MIN_SHARE of the smaller profile's; groups of up to SPLIT_GROUP_LARGE, where a prolific researcher's
# fragments outnumber the five-profile cap, need SPLIT_LARGE_SHARE.
COMMON_FULL_NAME = 10
SPLIT_GROUP_MAX, SPLIT_MIN_SHARED, SPLIT_MIN_SHARE = 5, 3, 0.2
SPLIT_GROUP_LARGE, SPLIT_LARGE_SHARE = 20, 0.5


def split_keep(group_size, shared, small):
    """The same-name pairs the repair joins (arrays or Series of name-group size, shared coauthors and the smaller
    profile's coauthor count)."""
    return (shared >= SPLIT_MIN_SHARED) & (((group_size <= SPLIT_GROUP_MAX) & (shared >= SPLIT_MIN_SHARE * small))
                                           | ((group_size <= SPLIT_GROUP_LARGE) & (shared >= SPLIT_LARGE_SHARE * small)))


def repair_link_mask(offsets, neighbors):
    """Directed-entry mask of the suspect links the repair removes (both directions); a subset of suspect_mask."""
    path = CACHE / "repair_link_mask.npy"
    if path.exists():
        return np.load(path, mmap_mode="r")
    full = full_name_counts()
    fn = np.zeros(offsets.shape[0] - 1, np.int64)
    fn[full.index.to_numpy()] = full.to_numpy()
    import pandas as pd
    records = pd.read_csv(CACHE / "a6_name_records.csv", usecols=["node_idx", "name_record_merge"])
    merged_name = np.zeros(offsets.shape[0] - 1, np.bool_)
    merged_name[records.node_idx[records.name_record_merge].to_numpy()] = True
    common_full = (fn >= COMMON_FULL_NAME) | merged_name
    log(f"repair links: {int(merged_name.sum()):,} name-record merges among the top {len(records):,}")
    log("repair links: reading suspect link signals")
    c = con()
    t = c.execute(f"""SELECT u, v, ku, kv, coalesce(s_uv, 1.0) AS s_uv, coalesce(s_vu, 1.0) AS s_vu
                      FROM read_parquet('{PG / 'link_signals.parquet'}')
                      WHERE (ku >= {COMMON_NAME} AND s_uv < {RARE_REGION})
                         OR (kv >= {COMMON_NAME} AND s_vu < {RARE_REGION})""").fetchnumpy()
    c.close()
    u, v = np.asarray(t["u"], np.int64), np.asarray(t["v"], np.int64)
    keep = (((np.asarray(t["ku"]) >= COMMON_NAME) & (np.asarray(t["s_uv"]) < RARE_REGION) & common_full[u])
            | ((np.asarray(t["kv"]) >= COMMON_NAME) & (np.asarray(t["s_vu"]) < RARE_REGION) & common_full[v]))
    log(f"repair links: {int(keep.sum()):,} of {len(u):,} suspect links")
    u, v = u[keep], v[keep]
    a = np.concatenate([u, v]).astype(np.int32)
    b = np.concatenate([v, u]).astype(np.int32)
    order = np.lexsort((b, a))
    a, b = a[order], b[order]
    us, starts = np.unique(a, return_index=True)
    starts = np.append(starts, len(a)).astype(np.int64)
    mask = np.zeros(neighbors.shape[0], np.bool_)
    mark_pairs(offsets, neighbors, us, b, starts, mask)
    np.save(path, mask)
    log(f"repair link mask: {int(mask.sum()):,} directed entries marked")
    return mask


NO_SKIP = np.zeros(0, np.bool_)


def full_name_counts():
    """node_idx -> how many mapped profiles carry the same name (ASCII-folded, lowercase, letters only), cached;
    the same key as scripts/export_flags.full_name_counts."""
    import unicodedata
    import pandas as pd
    path = CACHE / "full_name_counts.parquet"
    if not path.exists():
        a = pd.read_parquet(CACHE / "node_attrs.parquet", columns=["node_idx", "name"])
        folded = a.name.fillna("").map(lambda s: unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower())
        key = folded.str.replace(r"[^a-z ]", " ", regex=True).str.split().str.join(" ")
        pd.DataFrame({"node_idx": a.node_idx, "full_n": key.map(key.value_counts())}).to_parquet(path, index=False)
    return pd.read_parquet(path).set_index("node_idx").full_n


def flagged_mask(n):
    """Node mask of the profiles flagged as merged, split or not a person (flags.py, cache/flags.npz)."""
    f = np.load(CACHE / "flags.npz")
    mask = np.zeros(n, np.bool_)
    for kind in f.files:
        mask[f[kind]] = True
    return mask
