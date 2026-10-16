"""Mirror one OpenAlex parquet release (authors ~59GB) from S3 and verify it."""
import subprocess
from pathlib import Path

from pipeline import snapshot
from pipeline.config import data_dir


def sync_command(dest: Path, entity: str = "authors",
                 root: str = snapshot.BUCKET_ROOT) -> list[str]:
    return [
        "aws", "s3", "sync",
        f"{root}/{entity}/",
        str(dest / "snapshot" / entity),
        "--no-sign-request",
        "--no-progress",
        "--delete",
    ]


def add_parser(parser) -> None:
    parser.add_argument(
        "--dest", default=None,
        help="download root (default: DATA_DIR; put this on the external drive)",
    )
    parser.add_argument(
        "--entities", nargs="+", default=["authors"],
        help="snapshot entities to mirror; works is ~707GB and not needed "
             "(extract streams the columns it uses)",
    )
    parser.add_argument("--verify-only", action="store_true",
                        help="skip the sync; check the local mirror against the manifest")
    parser.add_argument("--deep", action="store_true",
                        help="also decode every row of every file (a full read; catches "
                             "damaged pages that sizes and row counts cannot)")
    parser.add_argument("--force", action="store_true",
                        help="restamp a directory that holds another release")


def run(args) -> int:
    dest = Path(args.dest) if args.dest else data_dir()
    snap = dest / "snapshot"
    manifest_path = snap / "manifest.json"
    con = snapshot.connect()
    if args.verify_only:
        manifest = snapshot.load_manifest(con, str(manifest_path))
    else:
        manifest = snapshot.load_manifest(con, f"{snapshot.BUCKET_ROOT}/manifest.json")
        snapshot.ensure_release(snap, manifest["date"], force=args.force)
        snapshot.write_manifest(manifest, manifest_path)
        for entity in args.entities:
            cmd = sync_command(dest, entity)
            print("+", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
    release = manifest["date"]
    problems = [p for e in args.entities
                for p in snapshot.verify_entity(manifest, e, snap, deep=args.deep)]
    if problems:
        for p in problems[:50]:
            print(p)
        print(f"{len(problems)} problem(s) against release {release}; re-run to resume")
        return 1
    counts = {e: len(snapshot.entity_files(manifest, e)) for e in args.entities}
    print(f"release {release} verified: " +
          ", ".join(f"{e} {n} files" for e, n in counts.items()) + f" -> {snap}")
    return 0
