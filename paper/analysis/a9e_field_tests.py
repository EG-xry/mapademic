"""A9e: which differences between fields and between countries exceed sampling error? (A9's per-source sums.)

A9 measures each group's distances from 100 random members. With each source as one cluster of pairs, a group mean
has the standard error of a ratio of sums (A9's ratio_se). Two groups' within-group means come from different
sources, so their difference has standard error sqrt(se_i^2 + se_j^2), and it counts as significant at the 5% level
when it exceeds 1.96 of those. For the abstract's comparison, the distance among arts-and-humanities researchers and
their distance to immunologists share the arts sources, so that part of the difference is paired; the immunologists'
distance to the arts comes from other sources and enters as independent (the paper averages the two directions).
"""
import json

import numpy as np

from common import CACHE, RESULTS, log, save_result

Z = 1.96
a9 = json.loads((RESULTS / "a9_structure.json").read_text())
raw = np.load(CACHE / "a9_per_source.npz")


def within_group(key):
    """Each group's within-group mean and its standard error, from the per-source sums (the same as A9's)."""
    group, s, c = raw[f"{key}_src_group"], raw[f"{key}_s"], raw[f"{key}_c"]
    names = [str(x) for x in raw[f"{key}_names"]]
    means, ses = [], []
    for i in range(len(names)):
        rows = group == i
        means.append(ratio(s[rows, i], c[rows, i]))
        ses.append(ratio_se(s[rows, i], c[rows, i]))
    return names, np.array(means), np.array(ses)


def ratio_se(s, c):
    r = ratio(s, c)
    return float(np.sqrt(len(s) / (len(s) - 1) * ((s - r * c) ** 2).sum()) / c.sum())


def tests(key):
    names, within, se = within_group(key)
    assert names == a9[key]["names"] and np.allclose(within, [a9[key]["within"][x] for x in names], atol=1e-9)
    z = (within[:, None] - within[None, :]) / np.sqrt(se[:, None] ** 2 + se[None, :] ** 2)
    upper = np.triu_indices(len(names), 1)
    order = np.argsort(within)
    return {"within": dict(zip(names, within.round(4).tolist())), "within_se": dict(zip(names, se.round(4).tolist())),
            "se_range": [float(se.min()), float(se.max())], "pairs": int(len(upper[0])),
            "pairs_significant": int((np.abs(z[upper]) > Z).sum()),
            "neighbours_in_rank": [{"closer": names[a], "farther": names[b], "difference": float(within[b] - within[a]),
                                    "z": float(z[b, a])} for a, b in zip(order, order[1:])]}


def ratio(s, c):
    return s.sum() / c.sum()


def hook():
    names = list(raw["field_names"])
    arts, imm = names.index("Arts and Humanities"), names.index("Immunology and Microbiology")
    group, s, c = raw["field_src_group"], raw["field_s"], raw["field_c"]
    from_arts, from_imm = group == arts, group == imm
    s_aa, c_aa = s[from_arts, arts], c[from_arts, arts]
    s_ai, c_ai = s[from_arts, imm], c[from_arts, imm]
    s_ia, c_ia = s[from_imm, arts], c[from_imm, arts]
    r_aa, r_ai, r_ia = ratio(s_aa, c_aa), ratio(s_ai, c_ai), ratio(s_ia, c_ia)
    paired = (s_aa - r_aa * c_aa) / c_aa.sum() - 0.5 * (s_ai - r_ai * c_ai) / c_ai.sum()   # one term per arts source
    other = 0.5 * (s_ia - r_ia * c_ia) / c_ia.sum()                                        # one per immunologist
    var = len(paired) / (len(paired) - 1) * (paired ** 2).sum() + len(other) / (len(other) - 1) * (other ** 2).sum()
    difference = r_aa - (r_ai + r_ia) / 2
    return {"within": float(r_aa), "to_immunology_both_directions": float((r_ai + r_ia) / 2),
            "difference": float(difference), "se": float(np.sqrt(var)), "z": float(difference / np.sqrt(var))}


out = {"field": tests("field"), "country": tests("country"), "arts_versus_immunology": hook()}
save_result("a9e_field_tests", out)
for key in ("field", "country"):
    log(key, {k: v for k, v in out[key].items() if k != "neighbours_in_rank"})
    log(key, "closest neighbours:", [(r["closer"], r["farther"], round(r["difference"], 3), round(r["z"], 2))
                                     for r in out[key]["neighbours_in_rank"][:4]])
log(out["arts_versus_immunology"])
