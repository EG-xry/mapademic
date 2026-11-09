"""A9b: field and country distances on the repaired network (A8's cache/repair_rep.npy), to check A9's
matrices against merged hubs, non-person records and duplicates. A merged node takes the field and country of
its highest-degree member; sources are 100 random giant-component nodes per group, as in A9.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import CACHE, D, con, load_csr, log, save_result, repair_link_mask
from center_cache import verified_largest_component
from kernels import _bfs, contract, group_distances, highest_degree_members
from repair_metrics import group_summary

PER_GROUP = 100
rng = np.random.default_rng(20261025)
offsets, neighbors = load_csr(mmap=False)
deg = np.diff(offsets)
skip = np.asarray(repair_link_mask(offsets, neighbors))
rep = np.load(CACHE / "repair_rep.npy")
m = int(rep.max()) + 1
off2, nb2 = contract(offsets, neighbors, skip, rep, m)
del neighbors, offsets, skip
attrs = pq.read_table(CACHE / "node_attrs.parquet", columns=["node_idx", "field", "country"]).to_pandas(
    categories=["field", "country"])
# Match fallback-topic IDs inside bounded DuckDB; do not retain every author ID
# as a Python string, a fallback DataFrame, and a Python set alongside the CSR.
db = con()
db.execute("SET memory_limit='6GB'")
fallback = db.execute(f"""SELECT a.node_idx FROM read_parquet('{CACHE / 'node_attrs.parquet'}') a
    SEMI JOIN (SELECT id FROM read_parquet('{D}/authors_extract/*/*.parquet')
               WHERE topics[1].display_name = 'Legal and Regulatory Analysis') f ON a.id = f.id""").fetchnumpy()["node_idx"]
db.close()
attrs.loc[attrs.node_idx.isin(fallback), "field"] = None
first = highest_degree_members(rep, np.argsort(-deg, kind="stable"), m)
del deg, rep, fallback
reach = np.empty(m, np.int16)
_bfs(off2, nb2, int(np.argmax(np.diff(off2))), reach, np.empty(m, np.int32))
lcc = verified_largest_component(reach)
out = {"component": {"size": int(lcc.sum()), "nodes": m, "verified_largest": True,
                     "selection": "component of the maximum-degree repaired node; size exceeds half of all repaired nodes"}}
log("repaired field/country component", out["component"])
for key, top in (("field", None), ("country", 16)):
    labels = pd.Series(attrs[key].to_numpy()[first]).fillna("")
    names = [x for x in labels.value_counts().index if x][:top] if top else [x for x in labels.value_counts().index if x]
    code = {x: i for i, x in enumerate(names)}
    group = labels.map(code).fillna(-1).astype(np.int64).to_numpy()
    sources, src_group = [], []
    for i in range(len(names)):
        members = np.flatnonzero((group == i) & lcc)
        pick = rng.choice(members, size=min(PER_GROUP, len(members)), replace=False)
        sources.extend(pick.tolist())
        src_group.extend([i] * len(pick))
    sm, c = group_distances(off2, nb2, np.array(sources, np.int64), group, len(names))
    src_group = np.array(src_group)
    out[key] = group_summary(sm, c, src_group, names)
    log(key, {name: round(value, 2) for name, value in out[key]["within"].items()})
save_result("a9b_repaired", out)
