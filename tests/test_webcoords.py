import json
import math

import duckdb
import pytest

from pipeline.webcoords import build_webcoords


def write_coords(path, rows):
    """rows: list of (id, x, y). Other columns filled with constants."""
    vals = ", ".join(
        f"('{i}', 'N {i}', CAST({x} AS FLOAT), CAST({y} AS FLOAT), 1, 20, 100,"
        f" 'Inst', 'Biology')" for i, x, y in rows
    )
    duckdb.sql(
        f"COPY (SELECT * FROM (VALUES {vals}) t(id, display_name, x, y,"
        f" community, works_count, cited_by_count, institution, field))"
        f" TO '{path}' (FORMAT PARQUET)"
    )


@pytest.fixture
def coords_file(tmp_path):
    # a center-heavy cross plus two far halo points
    rows = [("A0", 0, 0), ("A1", 10, 0), ("A2", -10, 0), ("A3", 0, 10),
            ("A4", 0, -10), ("H1", 1000, 0), ("H2", 0, -1000)]
    p = tmp_path / "coords.parquet"
    write_coords(p, rows)
    return str(p)


def test_all_points_inside_margin(coords_file, tmp_path):
    out = str(tmp_path / "coords_web.parquet")
    stats = build_webcoords(coords_file, out)
    assert stats["n"] == 7
    lo, hi = duckdb.sql(
        f"SELECT least(min(xw), min(yw)), greatest(max(xw), max(yw)) FROM '{out}'"
    ).fetchone()
    assert lo >= 0.02 - 1e-9 and hi <= 0.98 + 1e-9
    schema = {
        name: typ
        for name, typ, *_ in duckdb.sql(
            f"DESCRIBE SELECT xw, yw FROM '{out}'"
        ).fetchall()
    }
    assert schema == {"xw": "DOUBLE", "yw": "DOUBLE"}


def test_radius_order_preserved_and_angles_kept(coords_file, tmp_path):
    out = str(tmp_path / "coords_web.parquet")
    build_webcoords(coords_file, out)
    r = {
        i: math.hypot(xw - 0.5, yw - 0.5)
        for i, xw, yw in duckdb.sql(f"SELECT id, xw, yw FROM '{out}'").fetchall()
    }
    assert r["A0"] < r["A1"] < r["H1"]          # monotonic in original radius
    assert r["A1"] == pytest.approx(r["A2"])     # symmetric points equal radius
    # halo compressed: H1 is 100x A1's radius in data, far less on the map
    assert r["H1"] / r["A1"] < 10
    # angle preserved: A1 lies due +x of center, H2 due -y
    x1, y1 = duckdb.sql(f"SELECT xw, yw FROM '{out}' WHERE id='A1'").fetchone()
    assert y1 == pytest.approx(0.5, abs=1e-6) and x1 > 0.5
    xh, yh = duckdb.sql(f"SELECT xw, yw FROM '{out}' WHERE id='H2'").fetchone()
    assert xh == pytest.approx(0.5, abs=1e-6) and yh < 0.5


def test_meta_json_written(coords_file, tmp_path):
    out = str(tmp_path / "coords_web.parquet")
    stats = build_webcoords(coords_file, out)
    meta = json.loads((tmp_path / "coords_web.parquet.meta.json").read_text())
    assert meta == stats
    assert set(stats) == {"n", "cx", "cy", "s", "r2max"}
