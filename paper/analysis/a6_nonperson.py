"""A6e: mapped author records that are not people, by kind.

Name patterns for organizations, collectives (teams, consortia, study groups), placeholders, role words,
printers and typesetters, and records whose name is a sentence; Latin-script and CJK/Cyrillic/Korean
organization words. Each kind is counted with the coauthor links it holds, and a random sample of each is
written out for inspection. Names carrying a URL, an ORCID or dates are counted too, but they are mostly
people, so they are left out of cache/nonperson_idx.npy (the records removed in A8).
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import CACHE, RESULTS, save_result

KINDS = {
    "organization": r"(?i)\b(?:universit\w*|universidad|université|università|universidade|institut\w*|hospital\w*|klinik\w*|"
                    r"college|faculty|facultad|department|dept|laborator\w*|ministry|ministerio|ministère|agency|"
                    r"council|bureau of|museum|library|society|association|academy|akademie|academia|company|corporation|"
                    r"inc|ltd|gmbh|llc|foundation|fundação|fundación|office of|office for|center for|centre for|centre de|"
                    r"school of|clinic of|division of|unit of)\b",
    "collective": r"(?i)\b(?:group|groupe|grupo|gruppe|team|consortium|consortia|collaborat\w+|investigators|"
                  r"network|committee|comité|comision|commission|working party|task force|trialists|"
                  r"study|project|initiative|programme|members|panel|alliance|partnership|registry|"
                  r"upgrade|experiment)\b",
    "placeholder": r"(?i)^(?:.*\b)?(?:unknown|anonymous|anonym|et al|not available|no name|null null|"
                   r"authors?|various|and others|varios|anónimo|anonimo)\b|^(?:n\.?\s?a\.?|none|others|null)$",
    "role": r"(?i)^(?:editor|editors|editorial|the editors|reviewer|translator|guest editor|editor in chief|"
            r"corresponding author|redaktion|redacción|rédaction|مؤطر|المؤلف|редакция|редколлегия|編集部|編集委員会)",
    "printing": r"(?i)(?:^satz\b|^druck\b|\bdruck\b.*\b(?:ag|gmbh|kg|medien|basel)\b|\bdruckerei\b|\bverlag\b|\btypesetting\b|\bimprimerie\b|\bprinting (?:office|house|press)\b)",
    "sentence": r"(?i)(?:please (?:correct|note|update|change)|thank you|my correct|is wrong|not my)",
    "url_or_digits": r"(?i)(?:https?://|www\.|@|\d{3,})",
    "cjk_cyr_ko_org": r"(?:大学|大學|学会|學會|委員会|委員會|研究所|研究会|センター|協会|協會|学院|學院|医院|醫院|"
                      r"研究室|編集|株式会社|有限公司|集团|集團|中心|学校|学部|"
                      r"университет|институт|академия|министерство|редакция|комитет|"
                      r"대학교|학회|연구소|위원회|협회|센터)",
}
# People whose surname is one of the pattern words: "Sophie Panel", "Axel Gruppe", "P. Panel", and the
# astronomer "Marc Huertas-Company" (a pattern word inside a hyphenated name).
PERSON_LIKE = (r"^(?:[A-Z]\.|[A-Z][a-zà-ÿ]+)\s+(?:Panel|Gruppe|Company|Bureau|Alliance|Study|Project|Network|Members|"
               r"Society|Library|Office|College|Academy|Council|Agency|Museum|Foundation|Association|Commission|"
               r"Committee|Initiative|Registry|Partnership|Programme|Experiment|Druck)$|\w-(?:Company|Bureau|Society)\b")
a = pq.read_table(CACHE / "node_attrs.parquet", columns=["node_idx", "name", "degree", "works", "cited", "community"]).to_pandas()
a = a[a.degree > 0].copy()
a["name"] = a.name.fillna("")
person_like = a.name.str.contains(PERSON_LIKE, regex=True)
out, samples = {"mapped": int(len(a)), "total_degree": int(a.degree.sum())}, []
hit = pd.Series(False, index=a.index)
for kind, pat in KINDS.items():
    m = a.name.str.contains(pat, regex=True) & ~person_like
    if kind != "url_or_digits":
        hit |= m
    g = a[m]
    out[kind] = {"profiles": int(m.sum()), "degree": int(g.degree.sum()), "max_degree": int(g.degree.max()) if len(g) else 0,
                 "with_100_coauthors": int((g.degree >= 100).sum()),
                 "top": g.nlargest(12, "degree")[["name", "degree"]].values.tolist()}
    samples.append(g.sample(min(40, len(g)), random_state=7).assign(kind=kind))
g = a[hit]
np.save(CACHE / "nonperson_idx.npy", g.node_idx.to_numpy())
out["person_like_excluded"] = a.name[person_like].head(40).tolist()
out["any"] = {"profiles": int(hit.sum()), "share": float(hit.mean()), "degree_share": float(g.degree.sum() / a.degree.sum())}
pd.concat(samples)[["kind", "node_idx", "name", "degree", "works", "cited"]].to_csv(RESULTS / "a6_nonperson_sample.csv", index=False)
save_result("a6_nonperson", out)
for k in KINDS:
    print(k, {x: y for x, y in out[k].items() if x != "top"})
    print("   ", out[k]["top"][:8])
print(out["any"])
