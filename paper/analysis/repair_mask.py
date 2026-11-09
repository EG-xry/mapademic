"""The revised repair rules (common.repair_link_mask, common.split_keep), measured before the repair reruns.

Builds cache/repair_link_mask.npy and reports how many suspect links the repair now removes, how many same-name
pairs it now joins, and the effect on the profiles that showed the old rules' limits (Section 6). The broad suspect
links (common.suspect_mask) remain the delivered-network diagnostic; only the repair uses the stricter subset.
Writes results/repair_mask.json.
"""
import numpy as np
import pyarrow.parquet as pq

from common import (CACHE, ND, SPLIT_GROUP_LARGE, SPLIT_GROUP_MAX, load_csr, log, repair_link_mask, save_result,
                    split_keep, suspect_mask)

offsets, neighbors = load_csr()
broad = np.asarray(suspect_mask(offsets, neighbors))
repair = np.asarray(repair_link_mask(offsets, neighbors))
assert not np.any(repair & ~broad), "repair links must be a subset of the suspect links"
out = {"suspect_links": int(broad.sum() // 2), "repair_links": int(repair.sum() // 2), "links": int(len(neighbors) // 2)}
out["repair_share_of_suspect"] = out["repair_links"] / out["suspect_links"]
deg = np.diff(offsets)
sp = pq.read_table(CACHE / "split_pairs.parquet").to_pandas()
small = np.minimum(deg[sp.a.to_numpy()], deg[sp.b.to_numpy()])
old = (sp.group_size <= 5) & (sp.shared >= 3) & (sp.shared >= 0.2 * small)
new = split_keep(sp.group_size, sp.shared, small)
assert not np.any(old & ~new), "the revised same-name rule must keep every pair the old one joined"
out["split_pairs"] = {"old_rule": int(old.sum()), "revised_rule": int(new.sum()),
                      f"added_in_groups_{SPLIT_GROUP_MAX + 1}_to_{SPLIT_GROUP_LARGE}": int((new & ~old).sum())}
names = pq.read_table(ND / "deposit" / "researchers.parquet", columns=["node", "name", "coauthors"]).to_pandas()
examples = {}
for name in ("Paul M. Thompson", "Henrik Zetterberg", "Luigi Ferrucci", "Kaj Blennow", "Yue Wang", "Kai You Wang"):
    rows = names[names.name == name].sort_values("coauthors", ascending=False)
    node = int(rows.node.iloc[0])
    lo, hi = offsets[node], offsets[node + 1]
    nodes = set(rows.node.tolist())
    pairs = sp[sp.a.isin(nodes) & sp.b.isin(nodes)]
    examples[name] = {"node": node, "coauthors": int(hi - lo), "suspect_links": int(broad[lo:hi].sum()),
                      "repair_links": int(repair[lo:hi].sum()), "profiles": int(len(rows)),
                      "joined_pairs_old": int(old[pairs.index].sum()), "joined_pairs_revised": int(new[pairs.index].sum())}
out["examples"] = examples
save_result("repair_mask", out)
log(out)
