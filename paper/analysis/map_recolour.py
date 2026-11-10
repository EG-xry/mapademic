"""Recolour the existing 15-work layout with the five-work communities.

Build coordinates, positions and tiles in a fresh staging directory. The tile renderer
skips existing PNGs even with --force-aggregate, so it must not render into the previous
map directory. Publish only after every requested zoom is present; a journal restores
the previous directory if a process stops between the two directory renames.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from common import CACHE, D, NET, ROOT, con, log

COMPLETE = ".recolour-complete.json"


def _journal_path(destination):
    return destination.parent / f".{destination.name}-recolour-swap.json"


def _read_complete(directory):
    marker = directory / COMPLETE
    return json.loads(marker.read_text()) if marker.exists() else {}


def recover_publish(destination):
    """Finish our completed swap or restore our previous map after an interruption.

    Cleanup is restricted to the backup named by this destination's journal. A new
    map must carry the matching completion token before its old backup is removed.
    """
    destination = Path(destination)
    journal = _journal_path(destination)
    if not journal.exists():
        return
    record = json.loads(journal.read_text())
    if record.get("destination") != destination.name:
        raise ValueError(f"unexpected map destination in {journal}")
    for key, prefix in (("stage", f".{destination.name}-recolour-stage-"),
                        ("backup", f".{destination.name}-recolour-backup-")):
        name = record[key]
        if Path(name).name != name or not name.startswith(prefix):
            raise ValueError(f"unrecognized {key} path in {journal}")
    stage, backup = (destination.parent / record[key] for key in ("stage", "backup"))
    if destination.exists() and _read_complete(destination).get("token") == record["token"]:
        if backup.exists():
            shutil.rmtree(backup)
        journal.unlink()
    elif not destination.exists() and backup.exists():
        backup.rename(destination)
        journal.unlink()
    elif destination.exists() and not backup.exists():
        # Interrupted before the original directory was moved: it is still intact.
        journal.unlink()
    elif not destination.exists() and not backup.exists() and stage.exists():
        # First-ever map, interrupted before stage promotion. Leave staging for inspection.
        journal.unlink()
    else:
        raise RuntimeError(f"ambiguous interrupted map publication; inspect {journal}")


def publish_map(stage, destination):
    stage, destination = Path(stage), Path(destination)
    recover_publish(destination)
    complete = _read_complete(stage)
    if not complete.get("token"):
        raise ValueError("a map staging directory needs its completion marker before publication")
    if stage.parent != destination.parent or not stage.name.startswith(f".{destination.name}-recolour-stage-"):
        raise ValueError("map staging directory must be an owned sibling of its destination")
    backup = destination.parent / f".{destination.name}-recolour-backup-{uuid.uuid4().hex}"
    record = {"destination": destination.name, "stage": stage.name, "backup": backup.name, "token": complete["token"]}
    journal = _journal_path(destination)
    temp = journal.with_suffix(".tmp")
    temp.write_text(json.dumps(record, indent=1) + "\n")
    temp.replace(journal)
    try:
        if destination.exists():
            destination.rename(backup)
        stage.rename(destination)
    except BaseException:
        if backup.exists() and not destination.exists():
            backup.rename(destination)
        journal.unlink(missing_ok=True)
        raise
    recover_publish(destination)             # new token proves success before old backup cleanup


def main():
    destination = CACHE / "map"
    recover_publish(destination)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}-recolour-stage-", dir=CACHE))
    log("recolour staging", stage)
    c = con()
    try:
        c.execute(f"""
            CREATE TABLE w AS
            SELECT w.* EXCLUDE (community, sub, is_dust), a.community, w.is_ring OR coalesce(a.is_dust, TRUE) AS is_dust
            FROM read_parquet('{D}/coords_web.parquet') w
            JOIN (SELECT CAST(regexp_extract(id, '(\\d+)$', 1) AS BIGINT) AS num, community, is_dust
                  FROM read_parquet('{CACHE}/node_attrs.parquet')) a
              ON a.num = CAST(regexp_extract(w.id, '(\\d+)$', 1) AS BIGINT)""")
        c.execute(f"COPY w TO '{stage}/coords_web.parquet' (FORMAT PARQUET)")
        c.execute(f"""COPY (SELECT community, count(*) AS on_map, median(xw) AS xw, median(yw) AS yw FROM w
                            WHERE NOT is_ring GROUP BY 1) TO '{stage}/positions.parquet' (FORMAT PARQUET)""")
        log(c.execute("SELECT count(*), sum(is_dust::INT) FROM w").fetchone())
    finally:
        c.close()                           # release the joined table before starting the renderer
    subprocess.run([sys.executable, "-m", "pipeline", "tiles", "--zooms", "0-5", "--out", str(stage / "tiles"),
                    "--force-aggregate"], cwd=ROOT, env={**os.environ, "MAPADEMIC_DATA": str(stage)}, check=True)
    counts = {str(z): sum(1 for _ in (stage / "tiles" / str(z)).glob("*/*.png")) for z in range(6)}
    if not all(counts.values()):
        raise RuntimeError(f"incomplete map tile pyramid in {stage}: {counts}")
    (stage / COMPLETE).write_text(json.dumps({"token": uuid.uuid4().hex, "network": NET, "tiles_by_zoom": counts}, indent=1) + "\n")
    publish_map(stage, destination)
    log("published recoloured map", destination, counts)


if __name__ == "__main__":
    main()
