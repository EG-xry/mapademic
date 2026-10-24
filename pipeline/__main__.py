"""mapademic pipeline CLI: python -m pipeline <stage> [options]."""
import argparse

from pipeline import (build_edges, build_index, download, extract_authors,
                      extract_works, filter_authors, pathgraph, prep_graph, regions,
                      subcommunities, tiles, warm_start, webcoords)

STAGES: dict = {
    "download": download,
    "authors": extract_authors,
    "extract": extract_works,
    "filter": filter_authors,
    "edges": build_edges,
    "prep": prep_graph,
    "warmstart": warm_start,
    "subcommunities": subcommunities,
    "webcoords": webcoords,
    "tiles": tiles,
    "regions": regions,
    "index": build_index,
    "pathgraph": pathgraph,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pipeline", description="mapademic batch pipeline"
    )
    sub = parser.add_subparsers(dest="stage", required=True)
    for name, mod in STAGES.items():
        mod.add_parser(sub.add_parser(name, help=(mod.__doc__ or "").strip()))
    args = parser.parse_args(argv)
    return STAGES[args.stage].run(args)


if __name__ == "__main__":
    raise SystemExit(main())
