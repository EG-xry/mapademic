# Public Connect + Network-Graph UX Batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Connect (degree-of-separation) feature work for public visitors once the backend is deployed, and land five small fixes to the researcher-network overlays found in the 2026-07-24 feature audit.

**Architecture:** Two independent halves. (1) `scripts/build_site.py` gains an `api_base` parameter so the generated `docs/index.html` can point `window.MAPADEMIC_CONFIG.apiBase` at a deployed degree-of-separation backend; `web/dev.html`'s error copy stops naming `:8000` when an `apiBase` was configured. (2) Five self-contained edits inside `web/dev.html`'s overlay code: click-to-fit ray framing, per-hop paper titles in the hop list, a caption on the startup demo path, a real `mailto` on the browser-native OpenAlex calls, and an institution fallback via `affiliations` when `last_known_institutions` is null.

**Tech Stack:** Vanilla JS + MapLibre GL 4.7.1 (single-file `web/dev.html`), Python 3 + pytest for the site builder, no JS test harness (browser verification instead).

## Global Constraints

- **Do not deploy anything to Render.** The deploy needs Eric's account; Task 8 documents it as a manual step. Code lands ready, with a single constant to flip.
- `web/dev.html` is the single source of truth; `docs/index.html` is generated. Never hand-edit `docs/index.html` — regenerate via `scripts/build_site.py` (Task 8).
- Every `dev.html` change makes `docs/index.html` stale. `.venv/bin/python scripts/build_site.py --check` exits 1 until Task 8 runs. That is expected mid-plan, not a failure.
- Run python from the repo venv: `.venv/bin/pytest`, `.venv/bin/python`. A bare `python3` misses `duckdb` and fails collection.
- The full suite must stay green: `.venv/bin/pytest` — 137 passing as of 2026-07-24.
- Preserve the existing comment style in `dev.html`: each non-obvious block carries a `// R<n>:` note explaining *why*. New code follows suit, tagged `R31`.
- Do not restyle or re-color anything outside the elements each task names. Edge colors, palette, and the community-gradient scheme are settled decisions.
- Two local servers are needed for verification: `bash scripts/preview.sh` (port 8123, this repo) and `.venv/bin/uvicorn backend.app:app --port 8000` (run from `/Users/eric/Downloads/academic-degree-of-separation-master`).

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `scripts/build_site.py` | Injects `MAPADEMIC_CONFIG`; gains `api_base` | 1 |
| `tests/test_build_site.py` | Unit tests for the injector | 1 |
| `web/dev.html` | All viewer code: config read, overlays, status copy, CSS | 2–7 |
| `docs/index.html` | Generated deploy snapshot | 8 |
| `README.md` | Deploy steps for the public backend | 8 |

---

### Task 1: `build_site.py` learns `apiBase`

**Files:**
- Modify: `scripts/build_site.py:25-58`
- Test: `tests/test_build_site.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `build_config_block(data_base: str, api_base: str | None) -> str`, `inject_config(html: str, data_base: str = DATA_BASE, api_base: str | None = API_BASE) -> str`, `build(dev_html_path: Path = DEV_HTML, data_base: str = DATA_BASE, api_base: str | None = API_BASE) -> str`, and the module constant `API_BASE: str | None = None`. Task 8 flips `API_BASE` to the deployed URL.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_build_site.py`:

```python
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
```

Leave the existing import line untouched — the new tests use only names it already imports. (Don't import `API_BASE` just to have it; an unused import is lint noise.) The two pre-existing tests call `build_config_block("https://example.test")` with one argument; give `api_base` a default of `None` in Step 3 so they keep passing unchanged.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/eric/Downloads/mapademic && .venv/bin/pytest tests/test_build_site.py -v`
Expected: 2 passed, 3 failed — the new tests fail with `TypeError: build_config_block() takes 1 positional argument but 2 were given`.

- [ ] **Step 3: Implement**

In `scripts/build_site.py`, replace the `DATA_BASE` constant block (lines 25-28) with:

```python
# Exact config currently deployed (read out of the last docs/index.html
# generation).
DATA_BASE = "https://pub-3627024db5924826912107ea4f16e739.r2.dev"
# R31: the public degree-of-separation backend for Connect/coauthor-ray
# lookups. None = omit apiBase entirely, which leaves dev.html falling back to
# its localhost:8000 default (the pre-R31 behavior: the public page probes a
# backend nobody is running, and the features stay gracefully disabled). Set
# this to the deployed https:// origin to turn Connect on for public visitors
# -- see README's "Public path backend" section. No trailing slash: dev.html
# builds URLs as `${API_BASE}/api/path`.
API_BASE = None
```

Replace `build_config_block` and the two functions below it with:

```python
def build_config_block(data_base: str, api_base: str | None = None) -> str:
    if api_base:
        return (
            "<script>\n"
            "window.MAPADEMIC_CONFIG = {\n"
            f'  dataBase: "{data_base}",\n'
            f'  apiBase: "{api_base}"\n'
            "};\n"
            "</script>\n"
        )
    return (
        "<script>\n"
        "window.MAPADEMIC_CONFIG = {\n"
        f'  dataBase: "{data_base}"\n'
        "  // apiBase intentionally unset: path/ray features probe localhost:8000 and\n"
        "  // stay gracefully disabled for public visitors without a running backend.\n"
        "};\n"
        "</script>\n"
    )


def inject_config(html: str, data_base: str = DATA_BASE, api_base: str | None = API_BASE) -> str:
    """Insert the MAPADEMIC_CONFIG script tag before the first bare <script>
    tag (a <script> tag with no attributes -- dev.html's own inline code,
    as opposed to the maplibre-gl CDN <script src="..."> above it).

    Pure string transform, no I/O -- kept separate from build() so it's easy
    to unit test.
    """
    idx = html.index(BARE_SCRIPT_TAG)
    return html[:idx] + build_config_block(data_base, api_base) + html[idx:]


def build(
    dev_html_path: Path = DEV_HTML,
    data_base: str = DATA_BASE,
    api_base: str | None = API_BASE,
) -> str:
    return inject_config(dev_html_path.read_text(), data_base, api_base)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/eric/Downloads/mapademic && .venv/bin/pytest tests/test_build_site.py -v`
Expected: PASS, 5 tests.

Then the full suite: `.venv/bin/pytest`
Expected: 140 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/eric/Downloads/mapademic
git add scripts/build_site.py tests/test_build_site.py
git commit -m "feat(site): optional apiBase in the generated deploy config"
```

---

### Task 2: Public-safe Connect / ray error copy

**Files:**
- Modify: `web/dev.html:485` (add the flag), `web/dev.html:1727`, `web/dev.html:1948`, `web/dev.html:1961`

**Interfaces:**
- Consumes: `CFG` / `API_BASE` from `dev.html`'s existing deploy-config block.
- Produces: `const API_BASE_CONFIGURED` (boolean) and `function backendOfflineMessage(what)`, used by Tasks 2 only.

**Why:** With no `apiBase` configured the page falls back to `http://localhost:8000`, so a public visitor clicking Go currently reads "is the degree-of-separation server running on :8000?" — a dev instruction aimed at the wrong audience. Once Task 8 sets a real `apiBase`, the same message should read as a plain outage.

- [ ] **Step 1: Add the flag and the message helper**

In `web/dev.html`, immediately after the `const API_BASE = CFG.apiBase || "http://localhost:8000";` line (~line 485), insert:

```javascript
// R31: whether a real backend origin was configured (build_site.py's apiBase)
// or we're on dev.html's localhost fallback. Only the message copy branches on
// this -- every code path still probes/uses API_BASE identically.
const API_BASE_CONFIGURED = Boolean(CFG.apiBase);
// R31: one wording for "the path/coauthor backend didn't answer", phrased for
// whoever is actually looking. On a deployed page that's a visitor who can do
// nothing about it (state the outage, point at the browser-native fallback
// that still works); locally it's whoever forgot to start uvicorn.
function backendOfflineMessage(what) {
  return API_BASE_CONFIGURED
    ? `${what} is temporarily unavailable - the connection service isn't responding.`
    : `${what} needs the degree-of-separation backend on :8000 (uvicorn backend.app:app --port 8000).`;
}
```

- [ ] **Step 2: Use it at the three call sites**

Replace line ~1727 (inside `connectGo`'s `es.addEventListener("error", ...)`):

```javascript
    setPathStatus("path backend unreachable - is the degree-of-separation server running on :8000?");
```

with:

```javascript
    setPathStatus(backendOfflineMessage("Connect"));
```

Replace line ~1948 (inside `drawCoauthorRays`'s `app_error` handler):

```javascript
    let msg = "coauthor backend offline";
```

with:

```javascript
    let msg = backendOfflineMessage("Coauthors");
```

Replace line ~1961 (inside `drawCoauthorRays`'s `error` handler):

```javascript
    setSearchStatus("coauthor backend offline");
```

with:

```javascript
    setSearchStatus(backendOfflineMessage("Coauthors"));
```

Leave `drawCoauthorRays`'s browser-native fallback path untouched — when the probe fails, rays already re-route through `fetchCoauthorsBrowser` and never reach these handlers. These messages only fire when the backend answered the `/health` probe and then failed mid-stream.

- [ ] **Step 3: Verify locally (backend down)**

```bash
# stop any local backend first, then:
cd /Users/eric/Downloads/mapademic && bash scripts/preview.sh
```

Open `http://localhost:8123/web/dev.html`, click **Connect**, pick two authors, click **Go**.
Expected: the pill reads "Connect needs the degree-of-separation backend on :8000 (uvicorn backend.app:app --port 8000)." — the local wording, since no `apiBase` is configured in `dev.html`.

- [ ] **Step 4: Verify the deployed wording**

In the browser devtools console on that same page, run:

```javascript
API_BASE_CONFIGURED === false && backendOfflineMessage("Connect")
```

Expected: the `:8000` string. Then confirm the other branch's text by reading `backendOfflineMessage`'s source — it is exercised for real in Task 8's post-deploy check.

- [ ] **Step 5: Commit**

```bash
cd /Users/eric/Downloads/mapademic
git add web/dev.html
git commit -m "fix(web): audience-appropriate copy when the path backend is unreachable"
```

---

### Task 3: Click-to-fit framing for coauthor rays

**Files:**
- Modify: `web/dev.html:60-62` (CSS), `web/dev.html:2041-2047` (`finishCoauthorRays` tail)

**Interfaces:**
- Consumes: `activeRays` (`{originXw, originYw, originColor, endpoints}`), set by `finishCoauthorRays`.
- Produces: `function fitActiveRays()`, called only by the status-pill button added here.

**Design note (deviation worth knowing):** the audit item was "rays fitBounds framing". A plain auto-`fitBounds` on every ray draw is the wrong shape: the caller has just flown the camera to the author at z10.5, and rays reach clear across the map, so auto-fitting would immediately yank the user out to ~z3 and lose the person they searched for. Instead the status pill becomes clickable — "25 coauthors shown (0 off-map) · Fit" — so the overview is one click away and the deep-zoom arrival is preserved. If Eric prefers the yank, the same `fitActiveRays()` call moves into `finishCoauthorRays`'s tail.

- [ ] **Step 1: Add the button styling**

In the `<style>` block, right after the `#search-status` rule (~line 62), add:

```css
    /* R31: the ray status pill carries a "Fit" action -- see fitActiveRays. */
    #search-status .fit-rays { background: none; border: none; color: #7fae8e;
                       font: 12px system-ui; cursor: pointer; padding: 0 0 0 6px; }
    #search-status .fit-rays:hover { text-decoration: underline; }
```

- [ ] **Step 2: Add `fitActiveRays` and render the button**

In `web/dev.html`, replace the last line of `finishCoauthorRays` (~line 2047):

```javascript
  setSearchStatus(`${endpoints.length} coauthors shown (${offMap} off-map)`);
```

with:

```javascript
  // R31: rays routinely run off-viewport -- the caller just flew to the author
  // at z10.5, and coauthors sit anywhere on the map. Auto-fitting here would
  // throw that arrival away on every search, so the overview is offered as an
  // action on the pill instead of forced.
  setSearchStatus(`${endpoints.length} coauthors shown (${offMap} off-map)`);
  if (endpoints.length) {
    const fitBtn = document.createElement("button");
    fitBtn.type = "button";
    fitBtn.className = "fit-rays";
    fitBtn.textContent = "Fit";
    fitBtn.title = "Zoom out to show every drawn coauthor";
    fitBtn.addEventListener("click", fitActiveRays);
    searchStatusEl.appendChild(fitBtn);
  }
```

`setSearchStatus` sets `textContent`, which wipes children — so the append must come after it, as written above.

Then add `fitActiveRays` immediately after `finishCoauthorRays`'s closing brace:

```javascript
// R31: frame the origin plus every drawn ray endpoint. Reads the same
// activeRays snapshot repositionActiveOverlays uses, so it re-projects through
// the live toLngLat and stays correct at any spacing/equalize/wide setting.
function fitActiveRays() {
  if (!activeRays) return;
  const { originXw, originYw, endpoints } = activeRays;
  const origin = toLngLat(originXw, originYw);
  const bounds = new maplibregl.LngLatBounds(origin, origin);
  endpoints.forEach(p => bounds.extend(toLngLat(p.xw, p.yw)));
  map.fitBounds(bounds, { padding: 60, maxZoom: 11 });
}
```

- [ ] **Step 3: Verify**

Start both servers (preview on 8123, backend on 8000). Open the page, search "Geoffrey Hinton", pick the top result, wait for the rays.
Expected: pill reads "25 coauthors shown (0 off-map) Fit"; clicking **Fit** zooms out so ray endpoints are on screen; the rays and their labels stay attached to their dots throughout.

Then press Escape.
Expected: rays clear and the pill (with its button) disappears — `clearRays` calls `setSearchStatus("")`, which empties the element.

- [ ] **Step 4: Commit**

```bash
cd /Users/eric/Downloads/mapademic
git add web/dev.html
git commit -m "feat(web): fit-to-rays action on the coauthor status pill"
```

---

### Task 4: Per-hop paper titles in the hop list

**Files:**
- Modify: `web/dev.html:227-236` (CSS), `web/dev.html:1454-1476` (`resolvePathCoords`), `web/dev.html:1494-1516` (`renderHopList`), `web/dev.html:1543-1552` (`renderPath`), `web/dev.html:1668-1678` (`DEMO_PATH`)

**Interfaces:**
- Consumes: `/api/path` result steps, each `{author_id, author_name, connection_to_next, label, direction}` — `label` is the title of the paper joining this hop to the next.
- Produces: resolved point objects gain a `label` field (`string | undefined`); `renderHopList(points, hopColors)` renders it. `activePath` keeps the same `{points, hopColors}` shape, so `repositionActiveOverlays` needs no change.

**Why:** the backend already returns the paper title for every hop and `renderPath` throws it away. "Hinton → Sejnowski" is a claim; "co-authored *A Learning Algorithm for Boltzmann Machines*" is the evidence.

- [ ] **Step 1: Carry the label through path resolution**

In `renderPath` (~line 1547), replace:

```javascript
  const nodes = pathSteps
    .filter(s => !String(s.author_id).startsWith("W"))
    .map(s => ({ id: s.author_id, name: s.author_name }));
```

with:

```javascript
  // R31: `label` is the paper joining this step to the next one (the backend
  // sends it alongside connection_to_next); carried through so the hop list can
  // show the evidence for each hop rather than just the two names.
  const nodes = pathSteps
    .filter(s => !String(s.author_id).startsWith("W"))
    .map(s => ({ id: s.author_id, name: s.author_name, label: s.label }));
```

In `resolvePathCoords` (~line 1461), replace the two `return` statements that build resolved entries:

```javascript
    if (hits[i]) return { ...hits[i], ghost: false, name: n.name, id: n.id };
```

with:

```javascript
    if (hits[i]) return { ...hits[i], ghost: false, name: n.name, id: n.id, label: n.label };
```

and:

```javascript
      return { xw: prev.xw + (next.xw - prev.xw) * t,
               yw: prev.yw + (next.yw - prev.yw) * t,
               ghost: true, name: n.name, id: n.id };
```

with:

```javascript
      return { xw: prev.xw + (next.xw - prev.xw) * t,
               yw: prev.yw + (next.yw - prev.yw) * t,
               ghost: true, name: n.name, id: n.id, label: n.label };
```

- [ ] **Step 2: Render it**

Replace `renderHopList`'s body (~lines 1494-1516) with:

```javascript
function renderHopList(points, hopColors) {
  clearHopList();
  points.forEach((p, i) => {
    const row = document.createElement("div");
    row.className = "hop";
    const idx = document.createElement("span");
    idx.className = "hop-i"; idx.textContent = String(i + 1);
    const sw = document.createElement("span");
    sw.className = "hop-sw"; sw.style.background = hopColors[i];
    const name = document.createElement("span");
    name.className = "hop-name";
    const who = document.createElement("span");
    who.textContent = p.name;
    name.appendChild(who);
    // R31: the paper that links this hop to the next one. The last hop has no
    // outgoing edge, and a ghost's label describes an edge whose other end may
    // have been skipped -- both simply render without a subtitle.
    if (p.label && i < points.length - 1) {
      const via = document.createElement("span");
      via.className = "hop-via";
      via.textContent = p.label;
      via.title = p.label;
      name.appendChild(via);
    }
    row.append(idx, sw, name);
    if (p.ghost) {
      const ghost = document.createElement("span");
      ghost.className = "hop-ghost"; ghost.textContent = "ghost";
      row.append(ghost);
    }
    row.addEventListener("click", () => {
      map.flyTo({ center: toLngLat(p.xw, p.yw), zoom: 10.5 });
    });
    pathHopsEl.appendChild(row);
  });
}
```

- [ ] **Step 3: Style the subtitle**

In the `<style>` block, replace the `#path-hops .hop .hop-name` rule (~line 235):

```css
    #path-hops .hop .hop-name { flex: 1; }
```

with:

```css
    #path-hops .hop .hop-name { flex: 1; display: flex; flex-direction: column; gap: 1px;
                        min-width: 0; }
    /* R31: paper title under the name -- clipped to one line, full text on hover. */
    #path-hops .hop .hop-via { color: #888; font-size: 10px; white-space: nowrap;
                        overflow: hidden; text-overflow: ellipsis; }
```

Also widen the hop list so two-line rows don't crowd — replace the `#path-hops` rule (~line 227):

```css
    #path-hops { display: flex; flex-direction: column; gap: 2px; max-height: 220px; overflow-y: auto; }
```

with:

```css
    #path-hops { display: flex; flex-direction: column; gap: 2px; max-height: 260px; overflow-y: auto; }
```

- [ ] **Step 4: Bake the demo path's labels**

The demo path is a snapshot of a real `/api/path?from=A5108093963&to=A5067184382&edges=coauthor` response; these are that response's actual `label` values (re-queried 2026-07-24). Replace `DEMO_PATH.points` (~lines 1670-1676) with:

```javascript
  points: [
    { id: "A5108093963", name: "Geoffrey E. Hinton",    xw: 0.568363, yw: 0.553862, ghost: false,
      label: "A Learning Algorithm for Boltzmann Machines*" },
    { id: "A5044141636", name: "Terrence J. Sejnowski", xw: 0.513473, yw: 0.550736, ghost: false,
      label: "Running enhances neurogenesis, learning, and long-term potentiation in mice" },
    { id: "A5072968010", name: "Fred H. Gage",          xw: 0.356876, yw: 0.663661, ghost: false,
      label: "A Model for Neural Development and Treatment of Rett Syndrome Using Human Induced Pluripotent Stem Cells" },
    { id: "A5082310752", name: "G Yeo",                 xw: 0.449272, yw: 0.756635, ghost: false,
      label: "Programmable RNA Tracking in Live Cells with CRISPR/Cas9" },
    { id: "A5067184382", name: "Jennifer A. Doudna",    xw: 0.503588, yw: 0.732482, ghost: false },
  ],
```

- [ ] **Step 5: Verify**

Reload `http://localhost:8123/web/dev.html` with a bare hash (no `#`), open **Connect**.
Expected: five hop rows; rows 1-4 each show a paper title under the name, clipped to one line, full title in the tooltip; row 5 (Doudna) shows a name only.

With the backend running, run a real Connect (e.g. Yann LeCun ↔ Emmanuelle Charpentier).
Expected: same treatment on the live result; clicking any row still flies to that hop.

- [ ] **Step 6: Commit**

```bash
cd /Users/eric/Downloads/mapademic
git add web/dev.html
git commit -m "feat(web): show each hop's shared paper in the path hop list"
```

---

### Task 5: Caption the startup demo path

**Files:**
- Modify: `web/dev.html:1684-1687` (`showDemoPath`), `web/dev.html:1518-1538` (`clearPath`), `web/dev.html:121-122` (CSS)

**Interfaces:**
- Consumes: `setPathStatus`, `connectPanel`, `connectToggle`, `DEMO_PATH`.
- Produces: nothing consumed by later tasks.

**Why:** a first-time visitor lands on a drawn 4-hop path with no indication of what it is, who those five people are, or that they can run their own. The path renders inside a collapsed Connect panel, so even the "4 hops" status is hidden until they click Connect.

- [ ] **Step 1: Style the caption**

In the `<style>` block, after the `#connect-status` rule (~line 122), add:

```css
    /* R31: startup demo caption -- same pill language, one step louder since
       it is the first thing a visitor reads. */
    #connect-status .demo-note { color: #ddd; }
    #connect-status .demo-note b { color: #7fae8e; font-weight: 600; }
```

- [ ] **Step 2: Open the panel and caption it**

Replace `showDemoPath` (~lines 1684-1687) with:

```javascript
function showDemoPath() {
  pathGen++;
  drawResolvedPath(DEMO_PATH.points, DEMO_PATH.hopColors, DEMO_PATH.hops, DEMO_PATH.idA, DEMO_PATH.idB, { writeUrl: false });
  // R31: the drawn path lives inside the Connect panel, so a visitor who never
  // clicks "Connect" sees five labelled dots with no explanation of what they
  // are. Open the panel and replace the bare "4 hops" with a caption naming the
  // endpoints and inviting a real search. drawResolvedPath's own setPathStatus
  // runs first, so this overwrites it -- and any later Go/Clear overwrites this
  // in turn through the ordinary setPathStatus path, no special-casing.
  connectPanel.classList.add("open");
  connectToggle.classList.add("active");
  connectStatus.innerHTML =
    '<span class="demo-note">Example: <b>Geoffrey Hinton</b> to <b>Jennifer Doudna</b> in ' +
    DEMO_PATH.hops + ' co-authorship hops. Search any two researchers above.</span>';
}
```

- [ ] **Step 3: Make sure the caption is cleared like any other status**

`clearPath` ends with `setPathStatus("")`, and `setPathStatus` assigns `textContent` — which clears the injected markup too. Confirm by reading `setPathStatus` (~line 1415); no change needed. Note this in the commit message rather than adding defensive code.

- [ ] **Step 4: Verify**

Open `http://localhost:8123/web/dev.html` with no hash (clear the URL fragment; a stored hash suppresses the demo by design).
Expected: the Connect panel is open on load, showing the caption sentence with both names bolded, above the five hop rows.

Click **Clear**.
Expected: the path, hop list, and caption all disappear together.

Reload with a hash like `#3/20/0`.
Expected: no demo path and no caption — the existing bare-load gate still holds.

- [ ] **Step 5: Commit**

```bash
cd /Users/eric/Downloads/mapademic
git add web/dev.html
git commit -m "feat(web): caption the startup demo connection and open its panel"
```

---

### Task 6: Real mailto on browser-native OpenAlex calls

**Files:**
- Modify: `web/dev.html:1839-1840`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing.

**Why:** `fetchCoauthorsBrowser` identifies itself to OpenAlex as `mapademic@example.com`. OpenAlex's polite pool keys off a reachable address; a fake one is worse than none, and this is the request path every public visitor's ray draw takes.

- [ ] **Step 1: Replace the placeholder**

In `fetchCoauthorsBrowser` (~line 1840), replace:

```javascript
      `&per-page=200&select=authorships,cited_by_count&mailto=mapademic@example.com`;
```

with:

```javascript
      `&per-page=200&select=authorships,cited_by_count&mailto=ericgao432@gmail.com`;
```

- [ ] **Step 2: Verify**

Reload the page, clear the ray cache in the console with `localStorage.clear()`, then search an author and select them.
In devtools Network, find the `api.openalex.org/works?...` request.
Expected: the query string carries `mailto=ericgao432@gmail.com`; the response is 200 and rays draw.

- [ ] **Step 3: Commit**

```bash
cd /Users/eric/Downloads/mapademic
git add web/dev.html
git commit -m "fix(web): use a real contact address for OpenAlex polite-pool requests"
```

---

### Task 7: Institution fallback via `affiliations`

**Files:**
- Modify: `web/dev.html:5321-5329` (`fetchAuthorDetails`), `web/dev.html:5352-5362` (`openInfoCard`'s detail rows)

**Interfaces:**
- Consumes: OpenAlex People API response.
- Produces: `function mostRecentAffiliation(data) -> string | null`, used by `openInfoCard`.

**Why:** the info card shows "Institution unknown" for Geoffrey Hinton — verified 2026-07-24, OpenAlex returns `last_known_institutions: null` for him while `affiliations` holds 29 entries with years. Prominent authors are exactly the ones people click, so the most visible card in the app reads as broken data.

- [ ] **Step 1: Request the field**

In `fetchAuthorDetails` (~line 5323), replace:

```javascript
    infoCardCache.set(bareId, fetch(`https://api.openalex.org/people/${bareId}?select=display_name,works_count,cited_by_count,summary_stats,last_known_institutions,topics`)
```

with:

```javascript
    infoCardCache.set(bareId, fetch(`https://api.openalex.org/people/${bareId}?select=display_name,works_count,cited_by_count,summary_stats,last_known_institutions,affiliations,topics`)
```

- [ ] **Step 2: Add the fallback helper**

Immediately before `openInfoCard` (~line 5334), add:

```javascript
// R31: OpenAlex leaves last_known_institutions null for plenty of authors --
// including very prominent ones (Hinton) -- while still carrying a full
// `affiliations` list with years. Fall back to the affiliation with the latest
// year so the card stops reading "unknown" for the people most likely clicked.
// Ties and missing years sort last; returns null when there's nothing usable.
function mostRecentAffiliation(data) {
  const affs = (data && data.affiliations) || [];
  let best = null, bestYear = -Infinity;
  for (const a of affs) {
    const name = a && a.institution && a.institution.display_name;
    if (!name) continue;
    const year = Math.max(-Infinity, ...((a.years || []).filter(y => typeof y === "number")));
    if (year > bestYear) { bestYear = year; best = name; }
    else if (best === null) { best = name; }
  }
  return best;
}
```

- [ ] **Step 3: Use it**

In `openInfoCard`'s `fetchAuthorDetails().then(...)` block (~lines 5355-5356), replace:

```javascript
    const inst = data.last_known_institutions && data.last_known_institutions[0];
    addInfoRow("Institution", inst ? inst.display_name : "unknown");
```

with:

```javascript
    const inst = data.last_known_institutions && data.last_known_institutions[0];
    addInfoRow("Institution", (inst && inst.display_name) || mostRecentAffiliation(data) || "unknown");
```

- [ ] **Step 4: Verify**

In the devtools console:

```javascript
fetchAuthorDetails("A5108093963").then(d => console.log(d.last_known_institutions, mostRecentAffiliation(d)))
```

Expected: `null` for `last_known_institutions`, and a real institution name (e.g. "University of New Brunswick") from the helper.

Then zoom past z10 onto Hinton and click his dot.
Expected: the card's Institution row shows that institution rather than "unknown". Click an author who does have `last_known_institutions` and confirm that value still wins.

- [ ] **Step 5: Commit**

```bash
cd /Users/eric/Downloads/mapademic
git add web/dev.html
git commit -m "fix(web): fall back to most recent affiliation when no last-known institution"
```

---

### Task 8: Deploy handoff — regenerate `docs/index.html` and document the backend deploy

**Files:**
- Modify: `README.md` (new "Public path backend" subsection)
- Regenerate: `docs/index.html`
- Modify (Eric's step, after the service exists): `scripts/build_site.py` `API_BASE`

**Interfaces:**
- Consumes: Task 1's `API_BASE` constant, Task 2's `API_BASE_CONFIGURED` copy branch, Tasks 3-7's `dev.html` edits.
- Produces: the deployed page.

**Important:** the Render deploy itself is Eric's to run — it needs his account. Land the code and docs with `API_BASE = None` so the public page keeps behaving exactly as it does today; the flip is one constant plus one regenerate.

- [ ] **Step 1: Document the deploy**

Add to `README.md`, after the existing hosting/deploy material:

```markdown
### Public path backend (Connect)

Connect and the coauthor rays call the degree-of-separation FastAPI service
(`/api/path`, `/api/graph/expand`, `/health`) from
[academic-degree-of-separation](https://github.com/riptideiv/academic-degree-of-separation).
With no backend configured the viewer probes `localhost:8000`, so the public
page keeps Connect gracefully disabled and rays fall back to querying OpenAlex
directly from the browser.

To turn Connect on for public visitors:

1. Deploy that repo on Render — it ships a `render.yaml` (one web service,
   `uvicorn backend.app:app`). Set `OPENALEX_KEY` in the service's environment.
   Its CORS policy already allows any origin, so no change is needed there.
2. Note the service URL, e.g. `https://<service>.onrender.com`, and confirm
   `curl https://<service>.onrender.com/health` returns `{"status":"ok",...}`.
3. Set `API_BASE` in `scripts/build_site.py` to that origin (no trailing slash).
4. Regenerate and deploy the page: `.venv/bin/python scripts/build_site.py`,
   then commit `docs/index.html`.

Note Render's free tier sleeps an idle service; the first Connect after a nap
waits out a cold start (~30s) before the SSE stream begins.
```

- [ ] **Step 2: Regenerate the deploy snapshot**

```bash
cd /Users/eric/Downloads/mapademic
.venv/bin/python scripts/build_site.py
```

Expected: `wrote .../docs/index.html`.

- [ ] **Step 3: Verify the snapshot is faithful**

```bash
cd /Users/eric/Downloads/mapademic
.venv/bin/python scripts/build_site.py --check
git diff --stat docs/index.html
grep -c "apiBase intentionally unset" docs/index.html
```

Expected: "is up to date"; a diff covering only the Task 2-7 changes; `1` for the grep (the config block still omits a real `apiBase`, so public behavior is unchanged).

- [ ] **Step 4: Verify the public snapshot renders**

Open `http://localhost:8123/docs/index.html`.
Expected: map loads against the R2 `dataBase`; the demo path draws with its caption and per-hop titles; clicking **Go** shows the local-flavored offline message (correct — this build has no `apiBase` yet).

- [ ] **Step 5: Full suite + commit**

```bash
cd /Users/eric/Downloads/mapademic
.venv/bin/pytest
git add README.md docs/index.html
git commit -m "docs: public path-backend deploy steps; regenerate deploy snapshot"
```

Expected: 140 passed.

---

## Post-plan follow-ups (not implemented here)

- **Connection-lines default visibility.** The Line visibility slider sits at its 0.005 floor by default, so a first-time toggle of "Connection lines" looks like nothing happened until the slider is dragged. Deliberately left alone — it's a taste call Eric deferred.
- **Rays auto-fit.** If the click-to-fit affordance from Task 3 reads as too subtle, move `fitActiveRays()` into `finishCoauthorRays`'s tail.
- **Rays are top-25 by shared-work count, not nearest.** Documented in the 2026-07-11 proximity study; the long rays are correct, not a layout bug.
