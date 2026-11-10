"""Build a complete repaired, backward-compatible map release without a GPU.

Usage: python -m scripts.repair_map --works 'enriched/*/*.parquet'
       --nodes data/nodes.parquet --previous-web data/coords_web.parquet
       --out data/repaired --zooms 0-10

The target must not exist. All artifacts are built in a sibling staging
directory, validated by their builders, and published together by rename.
"""
import argparse
from contextlib import contextmanager
import glob
import json
import os
from pathlib import Path
import shutil
import resource
import sys
import time
import tempfile

import duckdb

from pipeline.config import apply_resource_limits, spill_dir
from pipeline.identity import suspect_links
from pipeline.resolve_identities import resolve_identities
from pipeline.build_edges import build_edges
from pipeline.prep_graph import prep_graph
from pipeline.repair_layout import repair_webcoords
from pipeline.pathgraph import build_pathgraph
from pipeline import build_index, tiles
from scripts.export_paths_bin import export_paths
from scripts.export_points_bin import export_points
from scripts.export_ids_bin import export_ids
from scripts.export_edges_bin import export_edges
from scripts.export_identities import export_identities


@contextmanager
def release_environment(path):
    previous = os.environ.get("MAPADEMIC_DATA")
    os.environ["MAPADEMIC_DATA"] = str(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("MAPADEMIC_DATA", None)
        else:
            os.environ["MAPADEMIC_DATA"] = previous


def repair_map(works: str, nodes: str, previous_web: str, out_dir: Path, *,
               registry_path: Path | None = None, flags_path: Path | None = None,
               zooms: str = "0-10", min_component_works: int = 2,
               max_profile_works: int = 50000) -> dict:
    """Resolve identities and build every artifact needed by the static viewer.

    Accepts an enriched works glob, original selected nodes, and the previous
    published web coordinates. No source API, GPU, or manual per-person edit is
    required. Existing releases and caller environment are never overwritten.
    """
    started = time.perf_counter()
    output = Path(out_dir).expanduser().resolve()
    if output.exists():
        raise ValueError("repaired release output already exists; choose a new directory")
    zoom_values = tiles._parse_zooms(zooms)
    if not zoom_values or min(zoom_values) < 0 or max(zoom_values) > tiles.MAXZ:
        raise ValueError(f"zooms must be within 0-{tiles.MAXZ}")
    matches = glob.glob(str(Path(works).expanduser()), recursive=True)
    if not matches:
        raise ValueError("works glob matches no source files")
    inputs = [Path(p).expanduser().resolve() for p in [*matches, nodes, previous_web]]
    for path in inputs:
        if not path.is_file():
            raise ValueError(f"input is not a file: {path}")
        if path == output or path.is_relative_to(output):
            raise ValueError("release output cannot contain or overwrite its inputs")
    if glob.has_magic(works):
        prefix = works
        for marker in "*?[":
            prefix = prefix.split(marker, 1)[0]
        search_root = Path(prefix).expanduser()
        if not prefix.endswith(os.sep):
            search_root = search_root.parent
        if output.is_relative_to(search_root.resolve()):
            raise ValueError("release output cannot be nested inside its source works glob")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".build-", dir=output.parent))
    try:
        report = {"version": 1}
        with release_environment(stage):
            report["identity_resolution"] = resolve_identities(
                works, nodes, stage, registry_path=registry_path,
                min_component_works=min_component_works, max_profile_works=max_profile_works)
            report["edges"] = build_edges(str(stage / "works.parquet"), str(stage / "nodes.parquet"), str(stage / "edges.parquet"))
            graph = stage / "graph"
            prep_graph(str(stage / "nodes.parquet"), str(stage / "edges.parquet"), graph)
            web = str(stage / "coords_web.parquet")
            report["layout"] = repair_webcoords(previous_web, str(stage / "nodes.parquet"), str(stage / "edges.parquet"), web)
            report["pathgraph"] = build_pathgraph(str(graph / "nodes_int32.parquet"), str(graph / "edges_int32.parquet"), stage / "pathgraph")
            with duckdb.connect() as con:
                apply_resource_limits(con)
                con.execute("SET temp_directory=?", [str(spill_dir(stage))])
                suspect = suspect_links(con, str(graph / "nodes_int32.parquet"),
                                        str(graph / "edges_int32.parquet"), web)
            report["paths"] = export_paths(stage / "pathgraph", web, stage / "paths", suspect=suspect)
            regions = Path(previous_web).parent / "regions.json"
            if regions.exists():
                shutil.copy2(regions, stage / "regions.json")
            build_index.run(argparse.Namespace(web=web, out=str(stage / "index")))
            export_identities(stage, str(stage / "edges.parquet"), web, stage / "paths", stage / "identities", flags_path=flags_path)
            tiles.run(argparse.Namespace(web=web, out=str(stage / "tiles"), zooms=zooms, force_aggregate=True))
            vectors = stage / "vector_experiment"
            report["points"] = export_points(web, vectors)
            report["point_ids"] = export_ids(web, vectors)
            report["visible_edges"] = export_edges(str(stage / "edges.parquet"), web, vectors)
            report["zooms"] = zooms
            report["wall_seconds"] = round(time.perf_counter() - started, 3)
            # Linux reports KiB; macOS reports bytes. This is the process high-water mark.
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            report["peak_rss_bytes"] = rss if sys.platform == "darwin" else rss * 1024
            (stage / "repair_report.json").write_text(json.dumps(report, indent=2))
            source_stamp = Path(previous_web).parent / "RELEASE"
            if source_stamp.exists():
                shutil.copy2(source_stamp, stage / "RELEASE")
        # Renaming a directory on one filesystem publishes all artifacts at once.
        if output.exists():
            raise ValueError("release output appeared during build; refusing to overwrite it")
        os.rename(stage, output)
        return report
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--works", required=True)
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--previous-web", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--registry")
    parser.add_argument("--flags")
    parser.add_argument("--zooms", default="0-10")
    parser.add_argument("--min-component-works", type=int, default=2)
    parser.add_argument("--max-profile-works", type=int, default=50000)
    args = parser.parse_args(argv)
    try:
        report = repair_map(args.works, args.nodes, args.previous_web, Path(args.out),
                            registry_path=Path(args.registry) if args.registry else None,
                            flags_path=Path(args.flags) if args.flags else None, zooms=args.zooms,
                            min_component_works=args.min_component_works, max_profile_works=args.max_profile_works)
    except (ValueError, OSError) as error:
        parser.exit(1, f"repair-map: {error}\n")
    print(json.dumps({"output": str(Path(args.out).resolve()), "identities": report["identity_resolution"], "layout": report["layout"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
