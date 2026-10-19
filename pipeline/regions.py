"""Name the map's regions (communities) and their sub-areas (sub-groups).

Level 1, a community: its field(s), then the country when the country is a
reason for the cluster, e.g. "Medicine (Italy)" but "Computer Science &
Engineering" for an international field (Eric, 2026-10-10). Coauthorship
clusters split by country because clinical research is organised nationally:
in 2026-09-23, 81 to 93 percent of Italian, French, Spanish, Dutch, Polish
and Russian medical researchers sit in their own country's cluster, but
only 5 to 9 percent of their physicists. China and Japan form whole national science systems. Kind
"national" or "field" sections the legend.

Level 2, a sub-group of a split community (pipeline/subcommunities.py): the
OpenAlex subfields over-represented in it relative to its community, then a
country when one other than the community's own dominates it, e.g.
"Cardiology & Surgery" or "Medicine (UK)" inside the US-led medicine region.

The old names were the single most distinctive OpenAlex topic (of ~4,500),
typically held by about 1 percent of a community, so "Battery Materials" labelled China
and "Legal and Regulatory Analysis" labelled Japan. Near-empty communities
(palette.sparse_communities) fade to dust and get no name.

Only what is drawn in colour is named (R36, 2026-10-16): 84 of 113 names
went to communities under 1,000 members, which webcoords draws as dust (a
37-member "Social Sciences, Business & Economics (Pakistan)" floated over
the economists). And a label sits at its group's average or median
position, which for a group spread over the map (the Iran and West African
clusters) can land in another field's territory: "Optics & Radiation", a
compact Iranian sub-group, sat among economists, where its group was 8 percent of
the dots. A label whose group is under SCATTER_SHARE of the coloured dots
around it, at the zoom it first shows, moves to its group's densest spot if
it is a level-1 name, and otherwise is marked "scattered": the viewer shows
it only while that group is filtered in the legend. regions.json is
data: hand-edit freely.
"""
import glob as globmod
import json
import math
import os
from pathlib import Path

import duckdb
import numpy as np

from pipeline import config
from pipeline.config import apply_resource_limits, data_dir
from pipeline.palette import sparse_communities

# OpenAlex's topic classifier gives non-English records (Japanese, Chinese,
# Korean, Russian, Persian titles) this topic wholesale; it says nothing
# about their field, so those authors are left out of field and subfield
# shares.
FALLBACK_TOPIC = "Legal and Regulatory Analysis"

# level 1: field part
MULTI_BELOW = 0.28      # top field under this share -> "Multidisciplinary"
EXTRA_FIELD = 0.15      # further fields need this share ...
EXTRA_FIELD_REL = 0.4   # ... and this fraction of the top field's share (max 3 fields)
# level 1: country part, from members with a known institution country
COUNTRY_SHARE, COUNTRY_LIFT = 0.12, 4.0         # a country qualifies at >= 12% and 4x its map share,
SMALL_SHARE, SMALL_LIFT = 0.07, 10.0            # or >= 7% and 10x (language areas: Spain + Mexico)
COUNTRY_TOTAL = 0.45                            # qualifying countries (max 3) must hold 45% together
# level 2
SUB_LIFT = 2.0          # subfield over-represented at least 2x vs its community ...
SUB_SHARE = 0.05        # ... with 5% of the sub-group, or
SUB_SHARE_RARE, SUB_LIFT_RARE = 0.02, 8.0       # 2% at 8x (dentistry is split over 3 subfields)
SUB_MIN_COUNT = 20
SUB_COUNTRY_FIRST, SUB_COUNTRY_NEXT = 0.12, 0.08
LABEL_PX = 60           # a sub-area's label shows once its spread is this many pixels
SCATTER_SHARE = 0.3     # below this share of the coloured dots around it, a label is "scattered"

FIELD_LABELS = {
    "Biochemistry, Genetics and Molecular Biology": "Molecular Biology",
    "Arts and Humanities": "Humanities", "Physics and Astronomy": "Physics & Astronomy",
    "Agricultural and Biological Sciences": "Biology & Agriculture",
    "Earth and Planetary Sciences": "Earth Sciences",
    "Business, Management and Accounting": "Business",
    "Economics, Econometrics and Finance": "Economics",
    "Immunology and Microbiology": "Immunology & Microbiology",
    "Veterinary": "Veterinary Science", "Pharmacology, Toxicology and Pharmaceutics": "Pharmacology",
}
SUBFIELD_LABELS = {
    "Cellular and Molecular Neuroscience": "Neuroscience",
    "Public Health, Environmental and Occupational Health": "Public Health",
    "Cardiology and Cardiovascular Medicine": "Cardiology",
    "Pulmonary and Respiratory Medicine": "Respiratory Medicine",
    "Pediatrics, Perinatology and Child Health": "Pediatrics",
    "Radiology, Nuclear Medicine and Imaging": "Radiology",
    "Obstetrics and Gynecology": "Obstetrics & Gynecology",
    "Orthopedics and Sports Medicine": "Orthopedics", "Psychiatry and Mental health": "Psychiatry",
    "Endocrinology, Diabetes and Metabolism": "Endocrinology",
    "Pathology and Forensic Medicine": "Pathology", "General Health Professions": "Health Professions",
    "Health, Toxicology and Mutagenesis": "Toxicology", "Animal Science and Zoology": "Animal Science",
    "Agronomy and Crop Science": "Agronomy", "Industrial and Manufacturing Engineering": "Manufacturing",
    "Electrical and Electronic Engineering": "Electrical Engineering",
    "Developmental and Educational Psychology": "Developmental Psychology",
    "Political Science and International Relations": "Political Science",
    "Computer Vision and Pattern Recognition": "Computer Vision",
    "Computer Networks and Communications": "Computer Networks",
    "Artificial Intelligence": "AI", "Ecology, Evolution, Behavior and Systematics": "Ecology & Evolution",
    "Nutrition and Dietetics": "Nutrition", "Management, Monitoring, Policy and Law": "Environmental Policy",
    "Renewable Energy, Sustainability and the Environment": "Renewable Energy",
    "Atomic and Molecular Physics, and Optics": "Optics",
    "Nuclear and High Energy Physics": "High Energy Physics",
    "Astronomy and Astrophysics": "Astronomy", "Condensed Matter Physics": "Condensed Matter",
    "Materials Chemistry": "Materials Chemistry", "Electronic, Optical and Magnetic Materials": "Electronic Materials",
    "Surfaces, Coatings and Films": "Surfaces & Coatings", "Mechanics of Materials": "Materials Mechanics",
    "Statistics, Probability and Uncertainty": "Statistics",
    "Earth-Surface Processes": "Earth Surface", "Global and Planetary Change": "Climate Change",
    "Economics and Econometrics": "Economics", "Sociology and Political Science": "Sociology",
    "Language and Linguistics": "Linguistics", "Literature and Literary Theory": "Literature",
    "Library and Information Sciences": "Information Science",
    "Molecular Biology": "Molecular Biology", "Infectious Diseases": "Infectious Diseases",
    "Organizational Behavior and Human Resource Management": "Organizational Behavior",
    "Strategy and Management": "Management", "Civil and Structural Engineering": "Civil Engineering",
    "Control and Systems Engineering": "Control Engineering", "Statistics and Probability": "Statistics",
    "Linguistics and Language": "Linguistics", "Safety, Risk, Reliability and Quality": "Safety & Reliability",
    "Building and Construction": "Construction", "Polymers and Plastics": "Polymers",
    "General Social Sciences": "Social Sciences", "Religious studies": "Religious Studies",
    "Ocean Engineering": "Ocean Engineering", "Aerospace Engineering": "Aerospace Engineering",
    "Fluid Flow and Transfer Processes": "Fluid Dynamics", "Process Chemistry and Technology": "Process Chemistry",
    "Physical and Theoretical Chemistry": "Physical Chemistry", "Spectroscopy": "Spectroscopy",
    "Computational Theory and Mathematics": "Theory of Computation",
    "Hardware and Architecture": "Computer Hardware", "Human-Computer Interaction": "HCI",
    "Information Systems and Management": "Information Systems", "Media Technology": "Media",
    "Geography, Planning and Development": "Geography & Planning", "Waste Management and Disposal": "Waste Management",
    "Water Science and Technology": "Water Science", "Environmental Chemistry": "Environmental Chemistry",
    "Food Science": "Food Science", "Horticulture": "Horticulture", "Insect Science": "Entomology",
    "Aquatic Science": "Aquatic Science", "Forestry": "Forestry", "Soil Science": "Soil Science",
    "Plant Science": "Plant Science", "Cancer Research": "Cancer Research", "Hepatology": "Hepatology",
    "Physical Therapy, Sports Therapy and Rehabilitation": "Rehabilitation",
    "Otorhinolaryngology": "ENT", "Emergency Medicine": "Emergency Medicine",
    "Critical Care and Intensive Care Medicine": "Intensive Care", "Anesthesiology and Pain Medicine": "Anesthesiology",
    "Reproductive Medicine": "Reproductive Medicine", "Health Information Management": "Health Informatics",
    "Biomedical Engineering": "Biomedical Engineering", "Clinical Biochemistry": "Clinical Biochemistry",
    "Experimental and Cognitive Psychology": "Experimental Psychology",
    "Social Psychology": "Social Psychology", "Applied Psychology": "Applied Psychology",
    "Visual Arts and Performing Arts": "Arts", "Philosophy": "Philosophy", "Archeology": "Archaeology",
    "Classics": "Classics", "Conservation": "Conservation", "Museology": "Museum Studies",
}
COUNTRY_NAMES = {
    "US": "US", "GB": "UK", "CN": "China", "JP": "Japan", "KR": "South Korea", "TW": "Taiwan",
    "HK": "Hong Kong", "SG": "Singapore", "IN": "India", "PK": "Pakistan", "BD": "Bangladesh",
    "MY": "Malaysia", "ID": "Indonesia", "TH": "Thailand", "VN": "Vietnam", "PH": "Philippines",
    "IR": "Iran", "IQ": "Iraq", "SA": "Saudi Arabia", "AE": "UAE", "QA": "Qatar", "JO": "Jordan",
    "IL": "Israel", "TR": "Turkey", "EG": "Egypt", "MA": "Morocco", "DZ": "Algeria", "TN": "Tunisia",
    "NG": "Nigeria", "GH": "Ghana", "KE": "Kenya", "ET": "Ethiopia", "ZA": "South Africa",
    "FR": "France", "BE": "Belgium", "NL": "Netherlands", "LU": "Luxembourg", "DE": "Germany",
    "AT": "Austria", "CH": "Switzerland", "IT": "Italy", "ES": "Spain", "PT": "Portugal",
    "IE": "Ireland", "DK": "Denmark", "SE": "Sweden", "NO": "Norway", "FI": "Finland",
    "IS": "Iceland", "EE": "Estonia", "LV": "Latvia", "LT": "Lithuania", "PL": "Poland",
    "CZ": "Czechia", "SK": "Slovakia", "HU": "Hungary", "RO": "Romania", "BG": "Bulgaria",
    "GR": "Greece", "CY": "Cyprus", "SI": "Slovenia", "HR": "Croatia", "RS": "Serbia",
    "BA": "Bosnia", "MK": "North Macedonia", "ME": "Montenegro", "AL": "Albania", "UA": "Ukraine",
    "BY": "Belarus", "RU": "Russia", "KZ": "Kazakhstan", "GE": "Georgia", "AM": "Armenia",
    "AZ": "Azerbaijan", "CA": "Canada", "MX": "Mexico", "CU": "Cuba", "CO": "Colombia",
    "VE": "Venezuela", "EC": "Ecuador", "PE": "Peru", "CL": "Chile", "AR": "Argentina",
    "UY": "Uruguay", "BR": "Brazil", "AU": "Australia", "NZ": "New Zealand",
}


def _lab(name: str, table: dict) -> str:
    if table is SUBFIELD_LABELS and name.startswith("General ") and name not in table:
        return _lab(name[len("General "):], FIELD_LABELS)   # OpenAlex's per-field catch-all subfield
    return table.get(name, name)


def field_label(fields: list[tuple[str, float]]) -> str:
    """[(field, share)] by share desc -> "A", "A & B", "A, B & C" or "Multidisciplinary"."""
    if not fields:
        return "Unclassified"
    top, s1 = fields[0]
    if s1 < MULTI_BELOW:
        return "Multidisciplinary"
    parts = [_lab(top, FIELD_LABELS)] + [_lab(f, FIELD_LABELS) for f, s in fields[1:3]
                                         if s >= EXTRA_FIELD and s >= EXTRA_FIELD_REL * s1]
    if len(parts) == 1:
        return parts[0]
    flat = [w for p in parts for w in p.split(" & ")]        # "Biology & Agriculture" joins the list
    return ", ".join(flat[:-1]) + " & " + flat[-1]


def country_label(countries: list[tuple[str, float, float]]) -> str | None:
    """[(code, share, lift)] by share desc -> "Japan", "Spain, Mexico, Argentina" or None."""
    picked, total = [], 0.0
    for cc, s, lift in countries:
        if len(picked) < 3 and ((s >= COUNTRY_SHARE and lift >= COUNTRY_LIFT)
                                or (s >= SMALL_SHARE and lift >= SMALL_LIFT)):
            picked.append(cc); total += s
    return ", ".join(_lab(c, COUNTRY_NAMES) for c in picked) if picked and total >= COUNTRY_TOTAL else None


def sub_label(subfields: list[tuple[str, float, float]], countries: list[tuple[str, float]],
              parent_country: str | None, parent_field: str, topic: str | None) -> str:
    """Level-2 name. subfields: [(subfield, share, lift vs community)] best first."""
    picked = [_lab(sf, SUBFIELD_LABELS) for sf, s, lift in subfields
              if lift >= SUB_LIFT and (s >= SUB_SHARE or (s >= SUB_SHARE_RARE and lift >= SUB_LIFT_RARE))][:2]
    country = None
    if countries and countries[0][0] != parent_country:
        chosen, total = [], 0.0
        for cc, s in countries:
            if len(chosen) < 3 and cc != parent_country and s >= (SUB_COUNTRY_NEXT if chosen else SUB_COUNTRY_FIRST):
                chosen.append(cc); total += s
        if total >= COUNTRY_TOTAL:
            country = ", ".join(_lab(c, COUNTRY_NAMES) for c in chosen)
    picked = list(dict.fromkeys(picked))            # "Linguistics" twice after shortening
    if picked:
        # "Obstetrics & Gynecology, Pediatrics" rather than a run of ampersands
        base = (", " if any(" & " in p for p in picked) else " & ").join(picked)
    elif country:
        base = parent_field
    else:
        base = topic or parent_field
    return f"{base} ({country})" if country else base


def _unique(name: str, taken: set, extras: list[str], rank: int) -> str:
    for cand in [name] + [f"{name}: {t}" for t in extras]:
        if cand not in taken:
            return cand
    k = rank
    while f"{name} ({k})" in taken:
        k += 1
    return f"{name} ({k})"


def _need(pattern: str, what: str, how: str) -> str:
    if not globmod.glob(pattern):
        raise SystemExit(f"{what} not found at {pattern!r} - {how}")
    return pattern


def build_regions(webcoords_path: str, authors_glob: str, out_path: str,
                  institutions_glob: str, topics_glob: str,
                  top_n: int = 300, keep: int = 120, fade_min: int = 1000) -> int:
    _need(authors_glob, "authors snapshot",
          "run `python -m pipeline authors` (or `download`), or pass --authors <glob>")
    _need(institutions_glob, "institutions table",
          "run `python -m pipeline download --entities institutions topics`")
    _need(topics_glob, "topics table",
          "run `python -m pipeline download --entities institutions topics`")
    con = duckdb.connect()
    apply_resource_limits(con)
    cols = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{webcoords_path}')").fetchall()}
    sub_col = "coalesce(w.sub, -1)" if "sub" in cols else "-1"
    ring = "WHERE NOT w.is_ring" if "is_ring" in cols else ""
    dust = "w.is_dust" if "is_dust" in cols else "FALSE"
    con.execute(f"""CREATE TEMP TABLE ic AS SELECT display_name, mode(country_code) AS cc
                    FROM read_parquet('{institutions_glob}') WHERE country_code IS NOT NULL GROUP BY 1""")
    con.execute(f"""CREATE TEMP TABLE tp AS SELECT display_name, mode(subfield.display_name) AS subfield
                    FROM read_parquet('{topics_glob}') GROUP BY 1""")
    con.execute(f"""CREATE TEMP TABLE m AS
        SELECT w.community, {sub_col} AS sub, CAST(w.xw AS DOUBLE) AS xw, CAST(w.yw AS DOUBLE) AS yw,
               ic.cc, t.topic,
               CASE WHEN t.topic = '{FALLBACK_TOPIC}' THEN NULL ELSE w.field END AS field,
               CASE WHEN t.topic = '{FALLBACK_TOPIC}' THEN NULL ELSE tp.subfield END AS subfield,
               {dust} AS dust
        FROM read_parquet('{webcoords_path}') w
        LEFT JOIN ic ON ic.display_name = w.institution
        LEFT JOIN (SELECT id, topics[1].display_name AS topic FROM read_parquet('{authors_glob}')) t
               ON t.id = w.id
        LEFT JOIN tp ON tp.display_name = t.topic
        {ring}""")
    faded = sparse_communities(con, webcoords_path, min_members=fade_min)
    comms = con.execute(f"""
        SELECT community, count(*) AS members, avg(xw), avg(yw), sqrt(var_pop(xw) + var_pop(yw)),
               count(*) FILTER (WHERE NOT dust) AS coloured
        FROM m GROUP BY community ORDER BY members DESC, community LIMIT {int(top_n)}""").fetchall()
    ranked = [(int(c), int(n), x, y, s or 0.0, r + 1) for r, (c, n, x, y, s, k) in enumerate(comms) if k]
    named = [row for row in ranked if row[5] <= keep and row[0] not in faded]
    ids = ",".join(str(r[0]) for r in named) or "NULL"

    def shares(col: str, by: str, extra: str = "") -> dict:
        out: dict = {}
        for *key, v, k in con.execute(
                f"""SELECT {by}, {col}, count(*) FROM m WHERE {col} IS NOT NULL AND community IN ({ids}) {extra}
                    GROUP BY ALL ORDER BY count(*) DESC, {col}""").fetchall():
            out.setdefault(tuple(key), []).append((v, k))
        return {key: [(v, k / sum(c for _, c in vals)) for v, k in vals] for key, vals in out.items()}

    world = dict(con.execute("SELECT cc, count(*) * 1.0 / sum(count(*)) OVER () FROM m "
                             "WHERE cc IS NOT NULL GROUP BY 1").fetchall())
    fields, countries = shares("field", "community"), shares("cc", "community")
    topics = shares("topic", "community", f"AND topic <> '{FALLBACK_TOPIC}'")
    regions, taken, top_country, label_of = [], set(), {}, {}
    for c, n, x, y, spread, rank in named:
        cl = [(cc, s, s / world.get(cc, 1e-9)) for cc, s in countries.get((c,), [])]
        fl = field_label(fields.get((c,), []))
        country = country_label(cl)
        top_country[c] = cl[0][0] if cl else None
        label_of[c] = fl
        name = _unique(f"{fl} ({country})" if country else fl, taken,
                       [t for t, _ in topics.get((c,), [])[:3]], rank)
        taken.add(name)
        regions.append({"name": name, "kind": "national" if country else "field",
                        "xw": x, "yw": y, "spread": spread, "members": n, "rank": rank,
                        "community": c, "zmin": 2 if rank <= 30 else 4, "zmax": 4 if rank <= 30 else 6})
    if "sub" in cols:
        regions += _sub_regions(con, ids, label_of, top_country, {r["community"]: r["rank"] for r in regions})
    _mark_scattered(con, regions)
    tmp = str(out_path) + ".tmp"
    Path(tmp).write_text(json.dumps(regions, indent=1, ensure_ascii=False))
    os.replace(tmp, out_path)
    return len(regions)


def _mark_scattered(con, regions: list[dict], grid: int = 2048) -> None:
    """Flag labels whose group holds under SCATTER_SHARE of the coloured dots
    in a LABEL_PX box around the label at the zoom it first shows. A level-1
    label first moves to its group's densest box (a group spread over the
    map averages to somewhere it barely is); a sub-area is compact, so where
    it is outnumbered it stays put and is flagged."""
    if not regions:
        return
    p = con.execute("SELECT xw, yw, community FROM m WHERE NOT dust").fetchnumpy()
    gx = np.clip((p["xw"] * grid).astype(np.int64), 0, grid - 1)
    gy = np.clip((p["yw"] * grid).astype(np.int64), 0, grid - 1)

    def table(cx, cy):                                               # summed-area table of dot counts
        t = np.zeros((grid + 1, grid + 1), np.int64)
        t[1:, 1:] = np.bincount(cx * grid + cy, minlength=grid * grid).reshape(grid, grid).cumsum(0).cumsum(1)
        return t

    def box(t, x0, x1, y0, y1):
        return t[x1, y1] - t[x0, y1] - t[x1, y0] + t[x0, y0]

    total = table(gx, gy)
    comm = p["community"]
    order = np.argsort(comm, kind="stable")
    by_comm = comm[order]
    for r in regions:
        lo, hi = np.searchsorted(by_comm, [r["community"], r["community"] + 1])
        own = order[lo:hi]
        if not len(own):
            r["scattered"] = True
            continue
        mine = table(gx[own], gy[own])
        k = max(1, int(min(0.03, max(0.001, LABEL_PX / (512 * 2 ** r["zmin"]))) * grid))
        cx, cy = int(r["xw"] * grid), int(r["yw"] * grid)
        span = lambda c: (max(0, c - k), min(grid, c + k + 1))

        def share(cx, cy):
            (x0, x1), (y0, y1) = span(cx), span(cy)
            n = box(total, x0, x1, y0, y1)
            return box(mine, x0, x1, y0, y1) / n if n else 0.0

        if share(cx, cy) >= SCATTER_SHARE:
            continue
        if not r.get("level"):
            idx = np.arange(grid)
            x0, x1 = np.maximum(0, idx - k), np.minimum(grid, idx + k + 1)
            dense = (mine[x1][:, x1] - mine[x0][:, x1] - mine[x1][:, x0] + mine[x0][:, x0])
            bx, by = np.unravel_index(int(dense.argmax()), dense.shape)
            (x0, x1), (y0, y1) = span(bx), span(by)
            inside = own[(gx[own] >= x0) & (gx[own] < x1) & (gy[own] >= y0) & (gy[own] < y1)]
            mx, my = float(p["xw"][inside].mean()), float(p["yw"][inside].mean())   # centre on its dots
            if share(int(mx * grid), int(my * grid)) >= SCATTER_SHARE:
                r["xw"], r["yw"] = mx, my
                continue
        r["scattered"] = True


def _sub_regions(con, ids: str, label_of: dict, top_country: dict, rank_of: dict) -> list[dict]:
    subs = con.execute(f"""
        SELECT community, sub, count(*), median(xw), median(yw), sqrt(var_pop(xw) + var_pop(yw))
        FROM m WHERE sub >= 0 AND community IN ({ids}) GROUP BY 1, 2 ORDER BY 1, 2""").fetchall()
    if not subs:
        return []
    sf = con.execute(f"""
        WITH s AS (SELECT community, sub, subfield, count(*) c FROM m
                   WHERE sub >= 0 AND subfield IS NOT NULL AND community IN ({ids}) GROUP BY 1, 2, 3),
             st AS (SELECT community, sub, sum(c) n FROM s GROUP BY 1, 2),
             p AS (SELECT community, subfield, sum(c) c FROM s GROUP BY 1, 2),
             pt AS (SELECT community, sum(c) n FROM p GROUP BY 1)
        SELECT s.community, s.sub, s.subfield, s.c * 1.0 / st.n AS share,
               (s.c * 1.0 / st.n) / (p.c * 1.0 / pt.n) AS lift, s.c
        FROM s JOIN st USING (community, sub) JOIN p USING (community, subfield) JOIN pt USING (community)
        WHERE s.c >= {SUB_MIN_COUNT}
        ORDER BY s.community, s.sub, s.c * ln(1 + (s.c * 1.0 / st.n) / (p.c * 1.0 / pt.n)) DESC, s.subfield""").fetchall()
    subfields: dict = {}
    for c, s, name, share, lift, _ in sf:
        subfields.setdefault((c, s), []).append((name, share, lift))
    cc_rows = con.execute(f"""
        SELECT community, sub, cc, count(*) * 1.0 / sum(count(*)) OVER (PARTITION BY community, sub)
        FROM m WHERE sub >= 0 AND cc IS NOT NULL AND community IN ({ids})
        GROUP BY 1, 2, 3 ORDER BY 1, 2, 4 DESC, 3""").fetchall()
    ccs: dict = {}
    for c, s, cc, share in cc_rows:
        ccs.setdefault((c, s), []).append((cc, share))
    top_topic = dict(((c, s), t) for c, s, t in con.execute(f"""
        SELECT community, sub, arg_max(topic, k) FROM (
            SELECT community, sub, topic, count(*) k FROM m
            WHERE sub >= 0 AND topic IS NOT NULL AND topic <> '{FALLBACK_TOPIC}' AND community IN ({ids})
            GROUP BY 1, 2, 3) GROUP BY 1, 2""").fetchall())
    out, taken = [], {}
    for c, s, n, x, y, spread in subs:
        name = sub_label(subfields.get((c, s), []), ccs.get((c, s), []), top_country.get(c),
                         label_of.get(c, ""), top_topic.get((c, s)))
        seen = taken.setdefault(c, set())
        if name in seen:
            name = _unique(name, seen, [t for t in [top_topic.get((c, s))] if t], s + 1)
        seen.add(name)
        zmin = min(8.0, max(3.5, math.log2(LABEL_PX / (512 * max(spread or 0.0, 1e-6)))))
        out.append({"level": 2, "community": int(c), "sub": int(s), "name": name,
                    "xw": x, "yw": y, "spread": spread or 0.0, "members": int(n),
                    "rank": rank_of.get(c), "zmin": round(zmin, 2)})
    return out


def add_parser(parser) -> None:
    parser.add_argument("--authors", default=None,
                        help="authors glob (default: the `authors` extract, else the snapshot mirror)")
    parser.add_argument("--institutions", default=None,
                        help="institutions glob (default: <data>/snapshot/institutions/*/*.parquet)")
    parser.add_argument("--topics", default=None,
                        help="topics glob (default: <data>/snapshot/topics/*/*.parquet)")
    parser.add_argument("--web", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--top-n", type=int, default=300)
    parser.add_argument("--keep", type=int, default=120)


def run(args) -> int:
    web = args.web or str(data_dir() / "coords_web.parquet")
    out = args.out or str(data_dir() / "regions.json")
    authors = args.authors or config.authors_glob()
    snap = data_dir() / "snapshot"
    n = build_regions(web, authors, out,
                      args.institutions or str(snap / "institutions" / "*" / "*.parquet"),
                      args.topics or str(snap / "topics" / "*" / "*.parquet"),
                      top_n=args.top_n, keep=args.keep)
    print(f"{n} regions and sub-areas named -> {out} (hand-edit freely; it is data)")
    return 0
