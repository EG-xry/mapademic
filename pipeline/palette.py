"""Dark-cosmic palette: 26 field hue families, community shade jitter."""
import colorsys

# Hue anchors (degrees) spread around the wheel for perceptual separation on
# black. The 26 OpenAlex fields (topics[1].field.display_name vocabulary) as
# they appear verbatim in coords.parquet. Hues use a coprime step so overall
# spacing is ~13.8 deg (no collisions) while adjacent-name fields land ~97 deg
# apart, minimizing look-alike neighbours.
FIELD_HUES: dict[str, float] = {
    "Medicine": 0.0,
    "Social Sciences": 96.9,
    "Engineering": 193.8,
    "Biochemistry, Genetics and Molecular Biology": 290.8,
    "Physics and Astronomy": 27.7,
    "Agricultural and Biological Sciences": 124.6,
    "Computer Science": 221.5,
    "Environmental Science": 318.5,
    "Materials Science": 55.4,
    "Arts and Humanities": 152.3,
    "Chemistry": 249.2,
    "Earth and Planetary Sciences": 346.2,
    "Neuroscience": 83.1,
    "Psychology": 180.0,
    "Economics, Econometrics and Finance": 276.9,
    "Business, Management and Accounting": 13.8,
    "Health Professions": 110.8,
    "Immunology and Microbiology": 207.7,
    "Mathematics": 304.6,
    "Energy": 41.5,
    "Decision Sciences": 138.5,
    "Dentistry": 235.4,
    "Nursing": 332.3,
    "Pharmacology, Toxicology and Pharmaceutics": 69.2,
    "Chemical Engineering": 166.2,
    "Veterinary": 263.1,
}


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def field_community_rgb(field: str | None, community: int) -> tuple[float, float, float]:
    """Base RGB in [0,1] at full brightness; deterministic."""
    h = splitmix64(int(community))
    j1 = ((h & 0xFFFF) / 0xFFFF) * 2 - 1          # [-1, 1]
    j2 = (((h >> 16) & 0xFFFF) / 0xFFFF) * 2 - 1  # [-1, 1]
    if field is None or field not in FIELD_HUES:
        light = 0.62 + 0.18 * j1                   # neutral grey family
        return colorsys.hls_to_rgb(0.0, light, 0.03)
    hue = (FIELD_HUES[field] + 9.0 * j1) % 360.0   # small hue wobble in-family
    # deep, saturated colors so the dense core stays vivid instead of washing
    # to white when rendered at full brightness (Task 4 QA tuning)
    sat = min(1.0, max(0.80, 0.90 + 0.10 * j2))
    light = min(0.58, max(0.40, 0.48 + 0.08 * j2))
    return colorsys.hls_to_rgb(hue / 360.0, light, sat)


def community_rgb(community: int, majority_field: str | None,
                  members: int, min_members: int = 1000) -> tuple[float, float, float]:
    """Community base RGB in [0,1]; deterministic. Big communities take their
    majority field's hue family; the tail is dim grey dust."""
    h = splitmix64(int(community))
    j1 = ((h & 0xFFFF) / 0xFFFF) * 2 - 1
    j2 = (((h >> 16) & 0xFFFF) / 0xFFFF) * 2 - 1
    if members < min_members or majority_field is None \
            or majority_field not in FIELD_HUES:
        return colorsys.hls_to_rgb(0.0, 0.30 + 0.06 * j1, 0.03)
    hue = (FIELD_HUES[majority_field] + 12.0 * j1) % 360.0
    sat = min(1.0, max(0.85, 0.925 + 0.075 * j2))
    light = min(0.56, max(0.44, 0.50 + 0.06 * j2))
    return colorsys.hls_to_rgb(hue / 360.0, light, sat)


def load_community_stats(con, web_path: str) -> list[tuple[int, str | None, int]]:
    """Per-community (majority_field, member_count), one parquet scan.
    Deterministic order: members DESC, community ASC."""
    rows = con.execute(
        f"""WITH fc AS (SELECT community, field, count(*) c
                        FROM read_parquet('{web_path}') GROUP BY 1, 2),
             tot AS (SELECT community, sum(c) n FROM fc GROUP BY 1),
             maj AS (SELECT community, field,
                            row_number() OVER (PARTITION BY community
                                ORDER BY c DESC, field NULLS LAST) rn
                     FROM fc)
            SELECT t.community, m.field, t.n
            FROM tot t JOIN maj m ON m.community = t.community AND m.rn = 1
            ORDER BY t.n DESC, t.community"""
    ).fetchall()
    return [(int(c), f, int(n)) for c, f, n in rows]


def build_community_palette(con, web_path: str,
                            min_members: int = 1000) -> dict[int, tuple]:
    stats = load_community_stats(con, web_path)
    return {c: community_rgb(c, f, n, min_members) for c, f, n in stats}


def add_parser(parser) -> None:  # not a CLI stage; keeps import-shape uniform
    raise SystemExit("palette is a library module, not a stage")


def run(args) -> int:
    raise SystemExit("palette is a library module, not a stage")
