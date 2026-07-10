from pipeline.palette import FIELD_HUES, field_community_rgb, build_community_palette, community_rgb


def test_all_fields_covered():
    assert len(FIELD_HUES) == 26
    assert "Computer Science" in FIELD_HUES and "Neuroscience" in FIELD_HUES


def test_deterministic_and_community_varies():
    a = field_community_rgb("Neuroscience", 42)
    assert a == field_community_rgb("Neuroscience", 42)  # stable across calls
    b = field_community_rgb("Neuroscience", 43)
    assert a != b                                        # neighbors distinguishable
    # same family: hues near the Neuroscience anchor -> green channel dominates both
    assert a[1] == max(a) and b[1] == max(b)


def test_distinct_fields_distinct_hues():
    # Big fields must be color-distinguishable, not aliased onto a shared hue.
    assert field_community_rgb("Social Sciences", 1) != field_community_rgb("Arts and Humanities", 1)
    assert field_community_rgb("Physics and Astronomy", 1) != field_community_rgb("Materials Science", 1)


def test_null_field_is_grey():
    r, g, b = field_community_rgb(None, 7)
    assert abs(r - g) < 0.05 and abs(g - b) < 0.05   # near-neutral
    assert field_community_rgb(None, 7) == field_community_rgb(None, 7)


def test_rgb_in_unit_range():
    for field in list(FIELD_HUES) + [None]:
        for comm in (0, 1, 999999):
            assert all(0.0 <= c <= 1.0 for c in field_community_rgb(field, comm))


def test_big_community_uses_majority_field_hue_family():
    import colorsys
    r, g, b = community_rgb(7, "Medicine", 500000)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    assert s >= 0.80
    hue_deg = (h * 360.0) % 360.0
    dist = min(abs(hue_deg - 0.0), 360.0 - abs(hue_deg - 0.0))
    assert dist <= 12.0 + 1e-6            # Medicine anchor is 0.0
    assert community_rgb(7, "Medicine", 500000) == (r, g, b)  # deterministic


def test_small_or_fieldless_community_is_dim_grey():
    import colorsys
    for rgb in (community_rgb(3, "Medicine", 50), community_rgb(3, None, 10**6)):
        h, l, s = colorsys.rgb_to_hls(*rgb)
        assert s <= 0.05 and l <= 0.40


def test_same_field_communities_differ():
    assert community_rgb(1, "Medicine", 10**6) != community_rgb(2, "Medicine", 10**6)


def test_build_community_palette_covers_all(tmp_path):
    import duckdb
    p = str(tmp_path / "web.parquet")
    duckdb.sql(
        "COPY (SELECT 'A' || range::VARCHAR AS id, 'n' AS display_name,"
        " 0.5 AS xw, 0.5 AS yw, (range % 3)::INT AS community, 20 AS works_count,"
        " 10 AS cited_by_count, 'i' AS institution,"
        " CASE WHEN range % 3 = 0 THEN 'Medicine' ELSE 'Chemistry' END AS field,"
        " FALSE AS is_ring FROM range(3000))"
        f" TO '{p}' (FORMAT PARQUET)")
    pal = build_community_palette(duckdb.connect(), p, min_members=1000)
    assert set(pal) == {0, 1, 2}
    assert all(len(v) == 3 for v in pal.values())


def test_build_community_palette_majority_tie_and_null(tmp_path):
    import duckdb
    p = str(tmp_path / "web.parquet")
    duckdb.sql(
        "COPY (SELECT 'A' || range::VARCHAR AS id, 'n' AS display_name,"
        " 0.5 AS xw, 0.5 AS yw, 1 AS community, 20 AS works_count,"
        " 10 AS cited_by_count, 'i' AS institution, 'Medicine' AS field, FALSE AS is_ring FROM range(600)"
        " UNION ALL"
        " SELECT 'B' || range::VARCHAR AS id, 'n' AS display_name,"
        " 0.5 AS xw, 0.5 AS yw, 1 AS community, 20 AS works_count,"
        " 10 AS cited_by_count, 'i' AS institution, 'Chemistry' AS field, FALSE AS is_ring FROM range(400)"
        " UNION ALL"
        " SELECT 'C' || range::VARCHAR AS id, 'n' AS display_name,"
        " 0.5 AS xw, 0.5 AS yw, 2 AS community, 20 AS works_count,"
        " 10 AS cited_by_count, 'i' AS institution, 'Chemistry' AS field, FALSE AS is_ring FROM range(500)"
        " UNION ALL"
        " SELECT 'D' || range::VARCHAR AS id, 'n' AS display_name,"
        " 0.5 AS xw, 0.5 AS yw, 2 AS community, 20 AS works_count,"
        " 10 AS cited_by_count, 'i' AS institution, 'Physics and Astronomy' AS field, FALSE AS is_ring FROM range(500)"
        " UNION ALL"
        " SELECT 'E' || range::VARCHAR AS id, 'n' AS display_name,"
        " 0.5 AS xw, 0.5 AS yw, 3 AS community, 20 AS works_count,"
        " 10 AS cited_by_count, 'i' AS institution, 'Medicine' AS field, FALSE AS is_ring FROM range(200)"
        " UNION ALL"
        " SELECT 'F' || range::VARCHAR AS id, 'n' AS display_name,"
        " 0.5 AS xw, 0.5 AS yw, 3 AS community, 20 AS works_count,"
        " 10 AS cited_by_count, 'i' AS institution, NULL AS field, FALSE AS is_ring FROM range(800))"
        f" TO '{p}' (FORMAT PARQUET)")
    pal = build_community_palette(duckdb.connect(), p, min_members=1000)
    assert set(pal) == {1, 2, 3}
    # Community 1: 600 Medicine + 400 Chemistry -> Medicine majority
    assert pal[1] == community_rgb(1, 'Medicine', 1000)
    # Community 2: 500 Chemistry + 500 Physics and Astronomy (tie) -> Chemistry wins (alphabetic tie-break)
    assert pal[2] == community_rgb(2, 'Chemistry', 1000)
    # Community 3: 200 Medicine + 800 NULL -> NULL majority (grey)
    assert pal[3] == community_rgb(3, None, 1000)
