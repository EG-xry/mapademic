"""A5c: is the mixed Chinese/Japanese-script community Japanese? (Section 4's script communities.)

common.SCRIPTS separates Chinese from Japanese names by spacing: "Chinese characters" are two to four Han characters
with no space, "Japanese kanji/kana" two runs of Han characters or kana separated by a space. A Japanese name recorded
in kanji without a space therefore counts as Chinese. For the Japanese-script, Chinese-character and mixed communities
(cache/script_communities.json): how their members' names are written, the share containing kana (which only Japanese
uses), the length of the unspaced Han names, the share of those that split into a JMnedict surname followed by a
JMnedict given name (the Chinese-character community gives the rate for Chinese names), the order of the two parts of
spaced names, and the countries of the few members with a known institution.
"""
import json
import re

from common import CACHE, SCRIPT_COMMUNITIES, SCRIPTS, con, log, save_result

KANA = re.compile(r"[ぁ-ゟ゠-ヿ]")
UNSPACED = SCRIPTS["Chinese characters"]
SPACED = SCRIPTS["Japanese kanji/kana"]
jm = json.loads((CACHE / "jmnedict_readings.json").read_text())
surnames, given = set(jm["surname"]), set(jm["given"])


def japanese_split(name):
    """A surname from JMnedict followed directly by a given name from JMnedict."""
    return any(name[:k] in surnames and name[k:] in given for k in range(1, len(name)))


groups = {label: SCRIPT_COMMUNITIES[label] for label in ("Mixed CJK script", "Japanese script", "Chinese characters")}
ids = ", ".join(str(int(c)) for c in groups.values())
members = con().execute(f"""SELECT community, coalesce(name, '') AS name, country
                            FROM read_parquet('{CACHE / "node_attrs.parquet"}') WHERE community IN ({ids})""").fetchdf()
out = {}
for label, community in groups.items():
    g = members[members.community == community]
    unspaced = g.name[g.name.str.match(UNSPACED)]
    spaced = g.name[g.name.str.match(SPACED)]
    parts = spaced.str.split()
    countries = g.country.dropna().value_counts(normalize=True).head(3)
    out[label] = {"community": int(community), "members": int(len(g)),
                  "unspaced_han": len(unspaced) / len(g), "spaced_han_or_kana": len(spaced) / len(g),
                  "neither": 1 - (len(unspaced) + len(spaced)) / len(g),
                  "with_kana": float(g.name.str.contains(KANA).mean()),
                  "unspaced_length_shares": {str(k): round(float(v), 4) for k, v in
                                             unspaced.str.len().value_counts(normalize=True).sort_index().items()},
                  "unspaced_japanese_split": float(unspaced.map(japanese_split).mean()) if len(unspaced) else None,
                  "spaced_japanese_parts": float(parts.map(lambda p: p[0] in surnames and p[-1] in given).mean())
                  if len(spaced) else None,
                  "spaced_japanese_parts_given_first": float(parts.map(lambda p: p[0] in given and p[-1] in surnames).mean())
                  if len(spaced) else None,
                  "known_country": float(g.country.notna().mean()),
                  "top_countries": {k: round(float(v), 3) for k, v in countries.items()}}
    log(label, out[label])
save_result("a5c_mixed_script", out)
