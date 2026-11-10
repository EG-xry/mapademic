"""Bounded local delivery checks shared by build, validation and provenance.

Small scientific outputs and code are content hashed. Large data, arrays and
dictionary caches have explicit size/mtime inventories; those are identities,
not retrospective content checksums. No graph is read or materialized here.
"""
import hashlib
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

FIGURES = {"1": "fig_map.png", "2": "fig_distances.pdf", "3": "fig_far_edge.png",
           "4": "fig_field_distances.pdf", "5": "fig_fields.png", "6": "fig_national_heatmap.pdf",
           "7": "fig_countries.pdf", "8": "fig_hub_definitions.pdf", "9": "fig_ties.png", "10": "fig_uncited.pdf",
           "A1": "fig_affiliations.pdf", "A2": "fig_twins.pdf"}
RASTER_FIGURES = {"1", "3", "5", "9"}
MAP_FIGURES = {"1", "5", "9"}                   # drawn over the recoloured map (figures.map_communities)
BUILD_OUTPUTS = ("manuscript.docx", "manuscript.pdf", "manuscript.tex", "header.tex",
                 "supplement.docx", "supplement.pdf", "supplement.tex")
RECEIPT = "build-receipt.json"
VERSION = 1
REQUIRED_FINAL_CHECKS = {
    "dependency completeness and finite JSON", "five-work population and CSR dimensions",
    "cap-50 parity and sensitivity checkpoints", "flags and exported joins reconstruct representatives",
    "deposited researcher rows", "CSR neighbors and deposited flagged links", "deposit SHA256 checksums",
    "unique eligible work identifiers", "exact repaired edge reconstruction", "repaired center selection",
    "manuscript placeholders and build", "delivery fingerprint capture", "delivery inputs unchanged during validation"}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def record(root, path, content_hash=False):
    root, path = Path(root), Path(path)
    before = path.stat()
    result = {"path": str(path.relative_to(root)), "bytes": before.st_size, "mtime_ns": before.st_mtime_ns}
    if content_hash:
        result["sha256"] = digest(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"file changed while recording: {path}")
    return result


def verify_records(root, records):
    for expected in records:
        path = Path(root) / expected["path"]
        if not path.is_file() or record(root, path, "sha256" in expected) != expected:
            raise ValueError(f"file changed since receipt: {expected['path']}")


def figure_paths(root):
    directory = Path(root) / "paper/figures"
    paths = []
    for number, embedded in FIGURES.items():
        stem = Path(embedded).stem
        paths.extend((directory / f"{stem}.pdf", directory / f"{stem}.png",
                      directory / "submission" / f"Fig{number}.tif"))
        if number not in RASTER_FIGURES:
            paths.append(directory / "submission" / f"Fig{number}.eps")
    return paths


def figure_inputs(root, number):
    root = Path(root)
    common = ["paper/analysis/figures.py", "paper/analysis/common.py"]
    data = {
        "1": ["paper/cache/map/regions.json", "paper/cache/map/coords_web.parquet", "paper/cache/node_attrs.parquet"],
        "2": ["paper/results/a2_distances.json", "paper/results/a8_repair.json", "paper/cache/a2c_hist_cap50.npz",
              "paper/cache/repair_distance_summaries.npz", "paper/cache/closeness_1000.npy",
              "paper/results/a1b_clustering.json"],
        "3": ["paper/results/a3b_far_edge.json"],
        "4": ["paper/results/a9_structure.json", "paper/results/a9d_arts_subfields.json"],
        "5": ["paper/cache/map_fields.parquet", "paper/analysis/map_fields.py", "paper/cache/map/regions.json",
              "paper/cache/node_attrs.parquet"],
        "6": ["paper/cache/a5_field_country.csv", "paper/cache/node_attrs.parquet"],
        "7": ["paper/results/b4_countries.json"],
        "8": ["paper/analysis/fig_concepts.py"],
        "9": ["paper/results/b2c_repeat_ties.json", "paper/cache/b2_long_links.parquet", "paper/cache/map/regions.json",
              "paper/cache/node_attrs.parquet"],
        "10": ["paper/results/b1_uncited.json", "paper/cache/closeness_1000.npy", "paper/cache/node_attrs.parquet"],
        "A1": ["paper/cache/a6_affiliations.csv"],
        "A2": ["paper/results/a6_twins_zh.json", "paper/results/a6_twins_ja.json"],
    }
    paths = [root / name for name in common + data[number]]
    if number in MAP_FIGURES:
        paths.extend(root / name for name in ("paper/cache/map/coords_web.parquet", "paper/cache/a5b_communities.csv",
                                              "paper/cache/script_communities.json"))
        paths.extend(root / name for name in ("paper/analysis/map_recolour.py", "paper/analysis/map_labels.py",
                                              "pipeline/regions.py", "pipeline/tiles.py", "pipeline/palette.py",
                                              "pipeline/sizing.py", "pipeline/config.py"))
        tiles = sorted((root / "paper/cache/map/tiles/4").glob("*/*.png"))
        if not tiles:
            raise ValueError("missing zoom-4 map tiles used by map figures")
        paths.extend(tiles)
    return paths


def validate_format(path):
    """Header/container checks, not a replacement for rendering/visual review."""
    path = Path(path)
    if not path.is_file() or not path.stat().st_size:
        raise ValueError(f"missing or empty delivery file: {path}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        with path.open("rb") as handle:
            head = handle.read(8)
            handle.seek(max(0, path.stat().st_size - 2048))
            tail = handle.read()
        if not head.startswith(b"%PDF-") or b"%%EOF" not in tail:
            raise ValueError(f"invalid/truncated PDF container: {path}")
    elif suffix == ".docx":
        with zipfile.ZipFile(path) as archive:
            if not {"[Content_Types].xml", "word/document.xml"} <= set(archive.namelist()):
                raise ValueError(f"invalid DOCX container: {path}")
            if archive.testzip() is not None:
                raise ValueError(f"corrupt DOCX member: {path}")
    elif suffix in {".png", ".tif"}:
        from PIL import Image
        # Open reads headers only; no 600-dpi image arrays are allocated.
        with Image.open(path) as picture:
            expected = "PNG" if suffix == ".png" else "TIFF"
            if picture.format != expected or min(picture.size) <= 0:
                raise ValueError(f"invalid figure image: {path}")
            if suffix == ".tif" and min(picture.info.get("dpi", (0, 0))) < 599.5:
                raise ValueError(f"submission TIFF is not 600 dpi: {path}")
    elif suffix == ".eps":
        with path.open("rb") as handle:
            header = handle.read(8192)
        if not header.startswith(b"%!PS-Adobe") or not re.search(rb"%%BoundingBox: [-\d ]+", header):
            raise ValueError(f"invalid EPS header: {path}")


def validate_figures(root):
    map_root = Path(root) / "paper/cache/map"
    completion = json.loads((map_root / ".recolour-complete.json").read_text())
    counts = {str(z): len(list((map_root / "tiles" / str(z)).glob("*/*.png"))) for z in range(6)}
    if completion.get("network") != "min5" or completion.get("tiles_by_zoom") != counts or not all(counts.values()):
        raise ValueError("recoloured map tile inventory differs from its successful publication receipt")
    paths = figure_paths(root)
    for number, embedded in FIGURES.items():
        newest = max(path.stat().st_mtime_ns for path in figure_inputs(root, number))
        stem = Path(embedded).stem
        selected = [p for p in paths if p.stem == stem or p.stem == f"Fig{number}"]
        for path in selected:
            validate_format(path)
            if path.stat().st_mtime_ns < newest:
                raise ValueError(f"figure predates its code or data: {path}")
    formats = {"pdf": len(FIGURES), "png": len(FIGURES), "tif": len(FIGURES), "eps": len(FIGURES) - len(RASTER_FIGURES)}
    return {"figures": len(FIGURES), "files": len(paths), "formats": formats, "recoloured_tiles_by_zoom": counts,
            "freshness": "each format follows its explicitly listed figure code/data inputs",
            "limits": "container/header and TIFF resolution checks do not certify visual appearance"}


def build_inputs(root):
    root = Path(root)
    paths = [root / "paper/draft2.md", root / "paper/build.py", root / "paper/analysis/delivery_checks.py"]
    # The PDF embeds FIGURES; the DOCX embeds each one's PNG, as Word cannot show PDF figures.
    names = dict.fromkeys([*FIGURES.values(), *(f"{Path(name).stem}.png" for name in FIGURES.values())])
    paths.extend(root / "paper/figures" / name for name in names)
    return [record(root, path, True) for path in paths]


def write_build_receipt(root, inputs, started, *, locale=None):
    root = Path(root)
    verify_records(root, inputs)
    if inputs != build_inputs(root):
        raise ValueError("build inputs changed while pandoc was running")
    outputs = [root / "paper/build" / name for name in BUILD_OUTPUTS]
    for path in outputs:
        validate_format(path)
    value = {"version": VERSION, "status": "success", "started": started,
             "completed": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
             "inputs": inputs, "outputs": [record(root, path, True) for path in outputs]}
    if locale is not None:
        # Never persist an entire subprocess environment: it can contain secrets.
        value["subprocess_locale"] = {key: locale.get(key) for key in ("LANG", "LC_ALL")}
    target = root / "paper/build" / RECEIPT
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(target)
    return value


def validate_build_receipt(root):
    root = Path(root)
    receipt = json.loads((root / "paper/build" / RECEIPT).read_text())
    if receipt.get("version") != VERSION or receipt.get("status") != "success":
        raise ValueError("missing successful current-version manuscript build receipt")
    if receipt["inputs"] != build_inputs(root):
        raise ValueError("manuscript build inputs changed; rebuild the manuscript")
    expected = {str((root / "paper/build" / name).relative_to(root)) for name in BUILD_OUTPUTS}
    if {r["path"] for r in receipt["outputs"]} != expected or len(receipt["outputs"]) != len(expected):
        raise ValueError("manuscript build receipt omits required outputs")
    if any(not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", "")) for row in receipt["outputs"]):
        raise ValueError("manuscript build receipt omits output content hashes")
    verify_records(root, receipt["outputs"])
    for row in receipt["outputs"]:
        validate_format(root / row["path"])
    return {"receipt": record(root, root / "paper/build" / RECEIPT, True),
            "completed": receipt["completed"], "input_hashes_verified": len(receipt["inputs"]),
            "output_hashes_verified": len(receipt["outputs"])}


def code_paths(root):
    root = Path(root)
    paths = set((root / "paper/analysis").rglob("*.py")) | set((root / "pipeline").rglob("*.py"))
    paths.update((root / "paper/build.py", root / "pyproject.toml"))
    return sorted(path for path in paths if path.is_file())


def supporting_paths(root):
    """Only named scientific selections/receipts, never arbitrary recursive caches."""
    root = Path(root)
    cache = root / "paper/cache"
    paths = set(cache.glob("*.provenance.json")) | set(cache.glob("*_manifest.json"))
    paths.update(cache / name for name in (
        "script_communities.json", "community_labels.json", "a2_sources.npy", "a6_affiliations.csv", "a6_live_works.csv",
        "a6_twins_zh.csv", "a6_twins_ja.csv", "a6_twins_ko.csv", "a6_twins_ru.csv", "a5_field_country.csv",
        "a5_domestic_by_field.csv", "map/regions.json", "map/.recolour-complete.json", "ties/manifest.json",
        "repair_center.json", "repair_center.json.provenance.json"))
    paths.update(cache.glob("repair_community_*.json"))
    paths.update((cache / "a6_institutions_parts").glob("*/provenance.json"))
    paths.update((root / "data/2026-09-23").glob("coords_web.parquet.meta.json"))
    return sorted(path for path in paths if path.is_file())


def large_source_paths(root):
    root = Path(root)
    data, cache = root / "data/2026-09-23", root / "paper/cache"
    paths = set()
    for folder in ("works_authorships", "authors_extract"):
        paths.update((data / folder).glob("*/*.parquet"))
    for folder in ("snapshot/institutions", "snapshot/topics"):
        paths.update((data / folder).glob("**/*.parquet"))
    for folder in ("min5/graph", "min5/pathgraph", "graph", "pathgraph"):
        paths.update(path for path in (data / folder).glob("*") if path.is_file())
    paths.update((data / "min5").glob("*.parquet"))
    paths.update(data / name for name in ("coords_web.parquet", "nodes.parquet"))
    for name in ("node_attrs.parquet", "snapshot_20261014.parquet", "a6_institutions.parquet",
                 "full_name_counts.parquet", "qualifying_author_counts.parquet", "hubs.parquet", "split_pairs.parquet",
                 "b2_long_links.parquet", "lcc_all.npy", "lcc_clean.npy", "clustering_all.npy", "clustering_flagged.npy",
                 "closeness_1000.npy", "leiden.npy", "nonperson_idx.npy", "suspect_mask.npy", "repair_link_mask.npy", "flags.npz",
                 "flagged_per_node.npy", "repair_rep.npy", "a7_sets_peak25_strict.npz", "repair_distance_summaries.npz",
                 "jmnedict_readings.json", "ext/jmnedict/JMnedict.xml.gz", "map/coords_web.parquet", "map/positions.parquet",
                 "map_fields.parquet", "a5b_communities.csv"):
        paths.add(cache / name)
    for pattern in ("bipartite_*.npy", "a2c_hist_cap*.npz", "closeness_*_peak25_strict.npy", "ties/*.npy"):
        paths.update(cache.glob(pattern))
    # API-part caches can be numerous; enumerate only the named acquisition folder.
    paths.update((cache / "a6_institutions_parts").glob("*/*.json"))
    paths.update((cache / "map/tiles").glob("*/*/*.png"))
    for name in ("a6_institutions.parquet", "a6_affiliations.csv", "a6_live_works.csv", "snapshot_20261014.parquet"):
        paths.add(root / "paper/cache_min15" / name)
    return sorted(path for path in paths if path.is_file())


def validation_fingerprint(root):
    """Bind science/delivery state, excluding this validation's own output and logs."""
    root = Path(root)
    hashed = set(code_paths(root)) | set(supporting_paths(root))
    for folder in ("paper/results", "paper/results_min15"):
        hashed.update(path for path in (root / folder).glob("*.json") if path.name != "validation_min5.json")
    hashed.update(figure_paths(root))
    hashed.update(root / "paper/build" / name for name in (*BUILD_OUTPUTS, RECEIPT))
    hashed.update(root / name for name in ("paper/draft2.md", "data/2026-09-23/min5/deposit/README.md",
                                         "data/2026-09-23/min5/deposit/SHA256SUMS"))
    # Partial runs may lack delivery files; --require-manuscript separately requires all.
    hashed = {p for p in hashed if p.is_file()}
    metadata = set(large_source_paths(root))
    metadata.update(p for p in (root / "data/2026-09-23/min5/deposit").glob("*") if p.is_file())
    metadata.difference_update(hashed)
    content = {"version": VERSION, "content_hashed": [record(root, p, True) for p in sorted(hashed)],
               "metadata_only": [record(root, p) for p in sorted(metadata)]}
    content["sha256"] = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    return content


def verify_fingerprint(root, expected):
    if not isinstance(expected, dict) or expected.get("version") != VERSION:
        raise ValueError("missing current-version scientific/delivery fingerprint")
    current = validation_fingerprint(root)
    if expected != current:
        old = {r["path"]: r for category in ("content_hashed", "metadata_only") for r in expected.get(category, [])}
        new = {r["path"]: r for category in ("content_hashed", "metadata_only") for r in current[category]}
        changed = sorted(p for p in old.keys() | new.keys() if old.get(p) != new.get(p))
        raise ValueError(f"scientific/delivery inputs changed since validation capture: {changed[:15]}")
    return {"sha256": current["sha256"], "content_hashed_files": len(current["content_hashed"]),
            "metadata_only_files": len(current["metadata_only"])}
