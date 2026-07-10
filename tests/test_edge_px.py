import duckdb

from pipeline.edge_px import build_edge_px
from pipeline.tiles import PIX


def _write_inputs(tmp_path):
    web = tmp_path / "web.parquet"
    duckdb.sql(
        "COPY (SELECT * FROM (VALUES"
        " ('A', 0.5, 0.5, FALSE), ('B', 0.5005, 0.5, FALSE),"   # short edge
        " ('C', 0.9, 0.9, FALSE),"                              # far away
        " ('D', 0.1, 0.1, TRUE))"                               # ring dust
        " t(id, xw, yw, is_ring)) TO '" + str(web) + "' (FORMAT PARQUET)")
    edges = tmp_path / "edges.parquet"
    duckdb.sql(
        "COPY (SELECT * FROM (VALUES"
        " ('A', 'B', 3),"    # kept
        " ('A', 'C', 3),"    # too long
        " ('A', 'B', 1),"    # weight below min  (weights are per-row here)
        " ('A', 'D', 9))"    # ring endpoint
        " t(src, dst, weight)) TO '" + str(edges) + "' (FORMAT PARQUET)")
    return str(edges), str(web)


def test_filters_and_pixel_coords(tmp_path):
    edges, web = _write_inputs(tmp_path)
    out = str(tmp_path / "edges_px.parquet")
    n = build_edge_px(edges, web, out, min_weight=2, max_len_px=768)
    rows = duckdb.sql(f"SELECT * FROM '{out}'").fetchall()
    assert n == len(rows) == 1
    x0, y0, x1, y1 = rows[0]
    assert (x0, y0) == (PIX // 2, PIX // 2)
    assert abs(x1 - x0) <= 768 and y1 == y0
