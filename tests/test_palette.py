from pipeline.palette import FIELD_HUES, field_community_rgb


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
