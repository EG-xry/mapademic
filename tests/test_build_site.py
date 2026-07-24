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


def test_config_block_omits_api_base_when_none():
    block = build_config_block("https://example.test", None)
    assert 'dataBase: "https://example.test"' in block
    assert "apiBase" not in block.replace("// apiBase intentionally unset", "")


def test_config_block_emits_api_base_when_given():
    block = build_config_block("https://example.test", "https://api.example.test")
    assert 'dataBase: "https://example.test",' in block
    assert 'apiBase: "https://api.example.test"' in block
    assert "intentionally unset" not in block


def test_inject_config_passes_api_base_through():
    html = "<div></div>\n<script>\nconsole.log(1);\n</script>\n"
    out = inject_config(html, data_base="https://d.test", api_base="https://a.test")
    assert 'apiBase: "https://a.test"' in out
