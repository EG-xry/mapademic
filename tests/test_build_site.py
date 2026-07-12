from scripts.build_site import BARE_SCRIPT_TAG, build_config_block, inject_config


def test_inject_config_inserts_before_first_bare_script_tag():
    html = (
        '<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>\n'
        "<script>\n"
        "const HIT_ZOOM = 10;\n"
        "</script>\n"
    )
    out = inject_config(html, data_base="https://example.test")

    block = build_config_block("https://example.test")
    assert block in out
    # inserted before the bare <script>, after the src= one
    assert out.index(block) > out.index('<script src=')
    assert out.index(block) < out.index("const HIT_ZOOM")
    # the original bare <script>...HIT_ZOOM... content is untouched, just pushed later
    assert out.endswith(html[html.index(BARE_SCRIPT_TAG):])


def test_inject_config_uses_given_data_base():
    html = "<div></div>\n<script>\nconsole.log(1);\n</script>\n"
    out = inject_config(html, data_base="https://example.test/foo")
    assert 'dataBase: "https://example.test/foo"' in out
