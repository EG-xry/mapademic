"""Kanji -> romanized readings of Japanese surnames and given names, from JMnedict (EDRDG, CC BY-SA 4.0).

Run isolated, since it reads a downloaded file:

    python -I jmnedict_readings.py ../cache/ext/jmnedict/JMnedict.xml.gz ../cache/jmnedict_readings.json
"""
import gzip
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import pykakasi

src, dst = sys.argv[1], sys.argv[2]
kks = pykakasi.kakasi()
romaji = {}
readings = {"surname": defaultdict(set), "given": defaultdict(set)}
types = Counter()
for _, e in ET.iterparse(gzip.open(src), events=("end",)):
    if e.tag != "entry":
        continue
    kinds = {t.text for t in e.iter("name_type")}
    types.update(kinds)
    kind = ("surname" if any("surname" in k for k in kinds) else
            "given" if any("given name" in k for k in kinds) else None)
    if kind:
        for reb in (r.text for r in e.iter("reb")):
            if reb not in romaji:
                romaji[reb] = "".join(x["hepburn"] for x in kks.convert(reb)).lower()
            for keb in (k.text for k in e.iter("keb")):
                readings[kind][keb].add(romaji[reb])
    e.clear()
json.dump({k: {kb: sorted(v) for kb, v in d.items()} for k, d in readings.items()}, open(dst, "w"), ensure_ascii=False)
print({k: len(d) for k, d in readings.items()}, types.most_common(12))
