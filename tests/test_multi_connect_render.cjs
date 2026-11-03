const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const web = fs.readFileSync(path.resolve(__dirname, '../web/dev.html'), 'utf8');
const embedded = web.match(/\/\/ ---- Multi-connect network renderer ----[\s\S]*?\/\/ ---- End multi-connect network renderer ----/);
assert.ok(embedded, 'viewer contains the multi-connect renderer');
const fragment = embedded[0];
const step = id => ({ id, name: 'Researcher ' + id });
const found = (source, target, ids) => ({ source, target, status: 'found', steps: ids.map(x => x === null ? null : step(x)), hops: ids.length - 1 });
const plain = value => JSON.parse(JSON.stringify(value));

function harness() {
  class Element {
    constructor() { this.style = {}; this.children = []; this.className = ''; this.hidden = false; this.offsetHeight = 10; }
    append(...children) { this.children.push(...children); }
    setAttribute(key, value) { this[key] = value; }
    getBoundingClientRect() { return { left: 0, top: 0, right: 80, bottom: 18, width: 80, height: 18 }; }
  }
  const sources = new Map(), layers = new Map(), markers = [], calls = [], share = new Element();
  const context = {
    console, Map, Set, Promise, Number, String, Array,
    activeConnectGroup: { gen: 1 }, pathGen: 1,
    pathMarkers: [], pathOverlayIds: [], activePath: null, pathDimActive: false,
    VECTOR_POINT_FALLBACK_COLOR: '#888888',
    stripId: id => String(id).split('/').pop(),
    resolvePathCoords: async nodes => nodes.map((n, i) => n && ({ ...n, xw: .1 + Number(n.id.slice(1)) / 100, yw: .3, ghost: false })),
    resolveCommunityColor: async () => '#112233',
    toLngLat: (xw, yw) => ({ toArray: () => [xw, yw] }),
    gradientSegmentFeatures: (from, to, a, b) => [{ type: 'Feature', geometry: { type: 'LineString', coordinates: [from, to] }, properties: { color: a, end: b } }],
    document: { createElement: () => new Element(), getElementById: id => id === 'connect-share' ? share : null },
    markerOnDot: (el, dot, coord) => {
      const marker = { el, dot, coord, removed: false, remove() { this.removed = true; }, getElement() { return this.el; } };
      markers.push(marker); return marker;
    },
    overlayLabelAdd: (owner, ids) => calls.push(['add-owner', owner, [...ids]]),
    overlayLabelClear: (owner, ids) => calls.push(['clear-owner', owner, [...ids]]),
    drawLabels: () => calls.push(['labels']),
    updateRasterOpacity: () => calls.push(['raster']), updateVectorPoints: () => calls.push(['vectors']),
    fitPadding: () => 60,
    maplibregl: { LngLatBounds: class { constructor(a, b) { this.values = [a, b]; } extend(x) { this.values.push(x); return this; } } },
    map: {
      getSource: id => sources.get(id), getLayer: id => layers.get(id),
      addSource(id, data) { sources.set(id, data); calls.push(['source', id]); },
      addLayer(layer) { layers.set(layer.id, layer); calls.push(['layer', layer.id]); },
      removeLayer(id) { layers.delete(id); calls.push(['remove-layer', id]); },
      removeSource(id) { sources.delete(id); calls.push(['remove-source', id]); },
      fitBounds: () => calls.push(['fit']), project: () => ({ x: 100, y: 100 }),
    },
  };
  vm.createContext(context); vm.runInContext(fragment, context);
  return { context, calls, sources, layers, markers, share };
}

test('merges shared and reversed undirected edges while retaining endpoint identities', () => {
  const { context: c } = harness();
  const net = plain(c.buildConnectNetwork([found(0, 1, ['A1', 'A2', 'A3']), found(1, 2, ['A3', 'https://openalex.org/A2', 'A4'])]));
  assert.equal(net.nodes.length, 4); assert.equal(net.edges.length, 3);
  const namedEdges = net.edges.map(([a, b]) => [net.nodes[a].id, net.nodes[b].id].sort().join('-')).sort();
  assert.deepEqual(namedEdges, ['A1-A2', 'A2-A3', 'A2-A4']);
  assert.equal(net.nodes.find(n => n.id === 'A2').endpoint, false);
  assert.deepEqual(net.nodes.find(n => n.id === 'A3').endpointIndexes, [1]);
});

test('focus includes only the selected found route and excludes pending/error results', () => {
  const { context: c } = harness();
  const results = [found(0, 1, ['A1', 'A2']), found(1, 2, ['A2', 'A3']), { source: 2, target: 3, status: 'error', steps: [step('A99')] }];
  assert.deepEqual(plain(c.buildConnectNetwork(results, '1:2')).nodes.map(n => n.id), ['A2', 'A3']);
  assert.deepEqual(plain(c.buildConnectNetwork(results, '9:9')).nodes, []);
});

test('zero-hop routes retain a single endpoint and never create self edges', () => {
  const { context: c } = harness();
  const net = plain(c.buildConnectNetwork([found(0, 1, ['A1']), found(2, 3, ['A1', 'A1'])]));
  assert.equal(net.nodes.length, 1); assert.deepEqual(net.edges, []);
  assert.deepEqual(net.nodes[0].endpointIndexes, [0, 1, 2, 3]);
});

test('missing/invalid steps do not create edges across gaps', () => {
  const { context: c } = harness();
  const net = plain(c.buildConnectNetwork([found(0, 1, ['A1', null, 'A3'])]));
  assert.deepEqual(net.edges, []); assert.equal(net.nodes.length, 2);
});

test('draw merges routes, labels only researcher endpoints, and fits once per overview generation', async () => {
  const h = harness(), c = h.context;
  const routes = [found(0, 1, ['A1', 'A2', 'A3']), found(1, 2, ['A3', 'A2', 'A4'])];
  const endpoints = [{ id: 'I1', name: 'University only' }, { id: 'A3', name: 'Selected researcher' }, { id: 'A4', name: 'Last researcher' }];
  assert.notEqual(await c.renderConnectGroup(routes, endpoints, 1), false);
  assert.equal(c.activePath.group, true); assert.equal(c.activePath.edges.length, 3);
  assert.equal(h.sources.get('path').data.features.length, 3);
  const intermediate = h.markers.find(m => m.el.title.includes('Researcher A2'));
  assert.ok(intermediate); assert.equal(intermediate.el.children.length, 1);
  assert.equal(h.markers.filter(m => m.el.children.length > 1).length, 3);
  assert.ok(!h.markers.some(m => m.el.children.some(ch => ch.textContent === 'University only')));
  assert.equal(h.share.hidden, true); assert.equal(c.pathDimActive, true);
  await c.renderConnectGroup(routes, endpoints, 1);
  await c.renderConnectGroup(routes, endpoints, 1, '0:1');
  assert.equal(h.calls.filter(c => c[0] === 'fit').length, 1);
});

test('draw never joins nodes on either side of an unresolved coordinate', async () => {
  const h = harness(), c = h.context;
  c.resolvePathCoords = async nodes => nodes.map(n => n.id === 'A2' ? null : { ...n, xw: .2, yw: .4, ghost: false });
  await c.renderConnectGroup([found(0, 1, ['A1', 'A2', 'A3'])], [], 1);
  assert.equal(c.activePath.points.length, 2); assert.deepEqual(plain(c.activePath.edges), []);
  assert.equal(h.sources.get('path').data.features.length, 0);
});

test('stale coordinate resolution leaves the current overlay untouched', async () => {
  const h = harness(), c = h.context; let release;
  c.resolvePathCoords = nodes => new Promise(resolve => { release = () => resolve(nodes.map(n => ({ ...n, xw: .2, yw: .4 }))); });
  const pending = c.renderConnectGroup([found(0, 1, ['A1', 'A2'])], [], 1);
  c.pathGen = 2; release();
  assert.equal(await pending, false); assert.equal(h.calls.length, 0); assert.equal(h.markers.length, 0);
});

test('a changed group during color resolution cannot overwrite the new group', async () => {
  const h = harness(), c = h.context; let release;
  c.resolveCommunityColor = () => new Promise(resolve => { release = () => resolve('#123456'); });
  const pending = c.renderConnectGroup([found(0, 1, ['A1'])], [], 1);
  await new Promise(resolve => setImmediate(resolve));
  c.activeConnectGroup = { gen: 1 }; release();
  assert.equal(await pending, false); assert.equal(h.calls.length, 0);
});

test('cleanup releases label ownership, markers and layers without clearing group state', async () => {
  const h = harness(), c = h.context, group = c.activeConnectGroup;
  await c.renderConnectGroup([found(0, 1, ['A1', 'A2'])], [], 1);
  c.removeConnectGroupOverlay();
  assert.equal(c.activeConnectGroup, group); assert.equal(c.activePath, null);
  assert.equal(c.pathMarkers.length, 0); assert.equal(c.pathOverlayIds.length, 0);
  assert.equal(h.sources.size, 0); assert.equal(h.layers.size, 0);
  assert.ok(h.markers.every(m => m.removed));
  assert.ok(h.calls.some(call => call[0] === 'clear-owner' && call[2].length === 2));
});

test('a newer incremental render wins even when both use the same generation and group', async () => {
  const h = harness(), c = h.context; let releaseOld;
  c.resolvePathCoords = nodes => nodes[0].id === 'A1'
    ? new Promise(resolve => { releaseOld = () => resolve(nodes.map(n => ({ ...n, xw: .2, yw: .4 }))); })
    : Promise.resolve(nodes.map(n => ({ ...n, xw: .7, yw: .6 })));
  const old = c.renderConnectGroup([found(0, 1, ['A1', 'A2'])], [], 1);
  await c.renderConnectGroup([found(1, 2, ['A3', 'A4'])], [], 1);
  releaseOld(); assert.equal(await old, false);
  assert.deepEqual(plain(c.activePath.points).map(p => p.id), ['A3', 'A4']);
  assert.equal(h.calls.filter(call => call[0] === 'source').length, 1);
});

test('zero-hop draw keeps an endpoint marker with no fabricated segment', async () => {
  const h = harness(), c = h.context;
  await c.renderConnectGroup([found(0, 1, ['A1'])], [], 1);
  assert.equal(c.activePath.points.length, 1); assert.equal(c.activePath.edges.length, 0);
  assert.equal(h.sources.get('path').data.features.length, 0);
  assert.equal(h.markers.length, 1);
});

test('a real coordinate from another route replaces an interpolated shared-node position', async () => {
  const h = harness(), c = h.context;
  c.resolvePathCoords = async nodes => nodes.map(n => ({ ...n,
    xw: n.id === 'A2' ? (nodes[0].id === 'A2' ? .8 : .4) : .2,
    yw: .3, ghost: n.id === 'A2' && nodes[0].id !== 'A2' }));
  await c.renderConnectGroup([found(0, 1, ['A1', 'A2', 'A3']), found(2, 3, ['A2', 'A4'])], [], 1);
  const shared = c.activePath.points.find(p => p.id === 'A2');
  assert.equal(shared.xw, .8); assert.equal(shared.ghost, false);
  assert.equal(c.activePath.points.length, 4); assert.equal(c.activePath.edges.length, 3);
});
