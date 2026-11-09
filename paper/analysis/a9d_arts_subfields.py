"""A9d: the arts and humanities, split into OpenAlex's 13 subfields.

The paper's opening comparison uses OpenAlex's field "Arts and Humanities", which spans history, philosophy,
languages and literature, religious studies, music, the visual and performing arts and more. Here each subfield of an
author's first topic is a group of its own: from 100 random giant-component members of each subfield (all members
when fewer), as in A9, the mean distance to the members of every subfield and of the immunology field; and from 100
random immunologists back to each subfield, so that the two directions can be averaged as in the field matrix.

Writes results/a9d_arts_subfields.json.
"""
import numpy as np
import pyarrow.parquet as pq

from common import CACHE, D, con, load_csr, log, save_result
from kernels import group_distances

PER_GROUP = 100
IMMUNO = "Immunology and Microbiology"
rng = np.random.default_rng(20261103)

attrs = pq.read_table(CACHE / "node_attrs.parquet", columns=["node_idx", "id", "field"]).to_pandas()
n = len(attrs)
assert np.array_equal(attrs.node_idx.to_numpy(), np.arange(n))
# the extract's topics carry the field but not the subfield: look the first topic up in OpenAlex's topics table,
# as pipeline/regions.py does
c = con()
sub = c.execute(f"""
    WITH tp AS (SELECT display_name, mode(subfield.display_name) AS subfield
                FROM read_parquet('{D}/snapshot/topics/*/*.parquet') GROUP BY 1)
    SELECT a.id, tp.subfield
    FROM read_parquet('{D}/authors_extract/*/*.parquet') a JOIN tp ON tp.display_name = a.topics[1].display_name
    WHERE a.topics[1].field.display_name = 'Arts and Humanities'""").fetchdf()
c.close()
sub = attrs[["node_idx", "id", "field"]].merge(sub, on="id", how="inner")
log(f"{len(sub):,} researchers whose first topic is in the arts and humanities; "
    f"{int((sub.field != 'Arts and Humanities').sum())} disagree with node_attrs' field")
sub = sub[sub.field == "Arts and Humanities"]
names = sub.subfield.value_counts().index.tolist()
code = {s: i for i, s in enumerate(names)}
group = np.full(n, -1, np.int64)
group[sub.node_idx.to_numpy()] = sub.subfield.map(code).to_numpy()
imm = len(names)
group[(attrs.field == IMMUNO).to_numpy()] = imm
lcc = np.load(CACHE / "lcc_all.npy")
sources, src_group = [], []
for g in range(imm + 1):
    members = np.flatnonzero((group == g) & lcc)
    pick = rng.choice(members, size=min(PER_GROUP, len(members)), replace=False)
    sources.extend(pick.tolist())
    src_group.extend([g] * len(pick))
sources, src_group = np.array(sources, np.int64), np.array(src_group)
log(f"{len(sources)} sources over {len(names)} subfields and immunology")
offsets, neighbors = load_csr(mmap=True)
s, c = group_distances(offsets, neighbors, sources, group, imm + 1)
mean = np.array([[s[src_group == i, j].sum() / max(c[src_group == i, j].sum(), 1) for j in range(imm + 1)]
                 for i in range(imm + 1)])
members = np.bincount(group[group >= 0], minlength=imm + 1)
rows = []
for i, nm in enumerate(names):
    rows.append({"subfield": nm, "members": int(members[i]), "sources": int((src_group == i).sum()),
                 "within": float(mean[i, i]), "to_immunology": float(mean[i, imm]), "from_immunology": float(mean[imm, i]),
                 "immunology_both_directions": float((mean[i, imm] + mean[imm, i]) / 2)})
    rows[-1]["within_exceeds_immunology"] = rows[-1]["within"] > rows[-1]["immunology_both_directions"]
save_result("a9d_arts_subfields", {"subfields": rows, "immunology_within": float(mean[imm, imm]),
                                   "names": names + [IMMUNO], "mean_distance": mean.round(4).tolist()})
for r in sorted(rows, key=lambda r: r["within"]):
    log(f"{r['subfield']:40s} {r['members']:>8,} within {r['within']:.2f}  immunology {r['immunology_both_directions']:.2f}")
