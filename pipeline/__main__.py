"""mapademic pipeline CLI: python -m pipeline <stage> [options]."""
import argparse

from pipeline import build_edges, download, extract_works, filter_authors, prep_graph

STAGES: dict = {
    "download": download,
    "extract": extract_works,
    "filter": filter_authors,
    "edges": build_edges,
    "prep": prep_graph,
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
