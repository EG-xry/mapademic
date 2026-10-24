// node --test tests/test_institution_connect.cjs (Node 18+, no npm dependencies)
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../web/dev.html'), 'utf8');
function section(start, end) {
  const from = html.indexOf(start), to = html.indexOf(end, from);
  assert.ok(from >= 0 && to > from);
  return html.slice(from, to);
}
const searchCode = section('async function browserShortestPath(', '// Indices i whose link');
const institutionCode = section('const INSTITUTION_CANDIDATE_LIMIT', '// ---- End Institution Connect ----');
const pooledCode = section('async function pooled(', '// The whole list-length index');
const connectCode = section('async function connectInBrowser(', 'connectGo.addEventListener("click"');
const plain = value => JSON.parse(JSON.stringify(value));
const stripId = id => String(id).replace(/^https?:\/\/openalex\.org\//, '');
const idBucketNum = id => Number(stripId(id).replace(/\D/g, '')) % 1000;

function searchHarness(graph, overrides = {}) {
  const context = vm.createContext({
    DOMException, PATH_MAX_HOPS: 14, PATH_MAX_LINKS: 8_000_000,
    eachPathList: async (_meta, nodes, visit) => {
      for (const u of [...nodes].sort((a, b) => a - b)) if (visit(u, graph.get(u) || [])) break;
    }, ...overrides
  });
  vm.runInContext(searchCode, context);
  return context.browserShortestPath;
}
function oracle(graph, starts, target, avoidSuspect = false) {
  const queue = starts.map(u => [u, 0]), seen = new Set(starts);
  for (let k = 0; k < queue.length; k++) {
    const [u, depth] = queue[k];
    if (u === target) return depth;
    for (const raw of graph.get(u) || []) {
      if (raw < 0 && avoidSuspect) continue;
      const v = raw < 0 ? ~raw : raw;
      if (!seen.has(v)) { seen.add(v); queue.push([v, depth + 1]); }
    }
  }
  return null;
}
test('multi-source search matches an independent BFS across random graphs, including flagged edges', async () => {
  let seed = 79;
  const rand = () => ((seed = (1664525 * seed + 1013904223) >>> 0) / 2 ** 32);
  for (let trial = 0; trial < 120; trial++) {
    const graph = new Map(Array.from({ length: 20 }, (_, i) => [i, []]));
    for (let u = 0; u < 20; u++) for (let v = u + 1; v < 20; v++) {
      if (rand() < 0.10) {
        const suspect = rand() < 0.2;
        graph.get(u).push(suspect ? ~v : v);
        graph.get(v).push(suspect ? ~u : u);
      }
    }
    const starts = [Math.floor(rand() * 20), Math.floor(rand() * 20), Math.floor(rand() * 20)];
    const target = Math.floor(rand() * 20);
    const search = searchHarness(graph);
    for (const avoidSuspect of [false, true]) {
      const result = await search(null, starts, target, null, { avoidSuspect });
      const distance = oracle(graph, starts, target, avoidSuspect);
      if (distance === null) { assert.equal(result, null); continue; }
      assert.equal(result.length - 1, distance);
      assert.ok(starts.includes(result[0]));
      assert.equal(result.at(-1), target);
      for (let i = 0; i < result.length - 1; i++) {
        assert.ok(graph.get(result[i]).some(raw => (!avoidSuspect || raw >= 0) && (raw < 0 ? ~raw : raw) === result[i + 1]));
      }
    }
  }
});
test('single-author searches, empty pools, duplicate roots, shared membership, and hop limits', async () => {
  const search = searchHarness(new Map([[1, [2]], [2, [1, 3]], [3, [2]]]));
  assert.deepEqual(plain(await search(null, 1, 3)), [1, 2, 3]);
  assert.equal(await search(null, [], 3), null);
  assert.deepEqual(plain(await search(null, [3, 3, 1], 3)), [3]);
  assert.equal(await search(null, [1], 3, null, { maxHops: 1 }), null);
  assert.deepEqual(plain(await search(null, [1], 3, null, { maxHops: 2 })), [1, 2, 3]);
});
test('cancellation stops the current search, and link limits return incomplete instead of no path', async () => {
  const graph = new Map([[1, [2]], [2, [1, 3]], [3, [2]]]);
  const capped = searchHarness(graph, { PATH_MAX_LINKS: 0 });
  assert.equal((await capped(null, [1], 3)).tooFar, true);
  let stale = false, reads = 0;
  const search = searchHarness(graph, { eachPathList: async (_m, _nodes, visit) => {
    reads++; stale = true; visit(1, [2]);
  } });
  await assert.rejects(search(null, [1], 3, null, { isStale: () => stale }), { name: 'AbortError' });
  assert.equal(reads, 1);
});

function institutionHarness(overrides = {}) {
  const context = vm.createContext({
    URLSearchParams, stripId, idBucketNum, normalize: s => s.toLowerCase(),
    PATH_PARALLEL: 64, fetchDisplayName: async id => id,
    fetchJsonOr404: async () => ({ results: [] }), ...overrides
  });
  vm.runInContext(pooledCode + '\n' + institutionCode, context);
  return context;
}
const member = (id, institutions = ['I42']) => ({ id, display_name: id,
  last_known_institutions: institutions.map(id => ({ id: 'https://openalex.org/' + id })) });
test('institution lookup uses the free autocomplete endpoint, encodes queries, keeps stable IDs, and shows a place hint', async () => {
  let requested;
  const h = institutionHarness({ fetchJsonOr404: async url => {
    requested = new URL(url);
    return { results: [{ id: 'https://openalex.org/I42', display_name: 'A & B', hint: 'Chicago, USA' },
      { id: 'https://openalex.org/I43', display_name: 'No hint', hint: null }, { id: 'A123', display_name: 'wrong entity' }] };
  } });
  assert.deepEqual(plain(await h.fetchInstitutionMatches('A & B')), [
    ['a & b', 'A & B', 'https://openalex.org/I42', null, null, 'Chicago, USA'],
    ['no hint', 'No hint', 'https://openalex.org/I43', null, null, '']]);
  assert.equal(requested.origin, 'https://api.openalex.org');
  assert.equal(requested.pathname, '/autocomplete/institutions');   // free; ?search= costs 10x a lookup
  assert.equal(requested.searchParams.get('q'), 'A & B');
  assert.equal(requested.searchParams.get('search'), null);
  assert.equal(requested.searchParams.get('per_page'), null);         // autocomplete answers 400 to it
});
test('candidate discovery caps the pool, checks every last-known institution, and deduplicates authors', async () => {
  let url, calls = 0;
  const h = institutionHarness({ fetchJsonOr404: async u => {
    url = new URL(u); calls++;
    return { results: [member('A1', ['I9', 'I42']), member('A1'), member('A2', ['I9']),
      ...Array.from({ length: 60 }, (_, i) => member('A' + (i + 100)))] };
  } });
  const candidates = await h.fetchInstitutionCandidates('I42');
  assert.equal(candidates.length, 48);
  assert.equal(candidates[0].id, 'A1');
  assert.equal(url.searchParams.get('filter'), 'last_known_institutions.id:I42,works_count:>14');
  assert.equal(url.searchParams.get('per_page'), '50');
  assert.equal(url.searchParams.get('sort'), 'cited_by_count:desc');
  await h.fetchInstitutionCandidates('I42');
  assert.equal(calls, 1);
  await assert.rejects(h.fetchInstitutionCandidates('I42,works_count:0'));
});
test('failed API responses are evicted so retries and replacement BYOK keys can work', async () => {
  let calls = 0;
  const h = institutionHarness({ fetchJsonOr404: async () => ++calls === 1 ? null : { results: [member('A1')] } });
  await assert.rejects(h.fetchInstitutionCandidates('I42'), /Couldn't load/);
  assert.equal((await h.fetchInstitutionCandidates('I42')).length, 1);
  assert.equal(calls, 2);
});
test('map intersection groups bucket reads and excludes authors absent from the graph', async () => {
  const read = [];
  const h = institutionHarness({
    fetchJsonOr404: async () => ({ results: [member('A1'), member('A1001'), member('A2'), member('A3')] }),
    fetchIdBucket: async bucket => { read.push(bucket); return new Map([
      ['A1', [0.1, 0.1, 0]], ['A1001', [0.2, 0.2, 9]], ['A2', [0.3, 0.3]], ['A3', [0, 0, 100]]
    ]); }
  });
  const source = await h.resolveInstitutionStart({ authors: 10 }, { id: 'I42' }, () => false);
  assert.deepEqual(plain(source.nodes), [0, 9]);
  assert.equal(source.names.get('A1001'), 'A1001');
  assert.equal(source.considered, 4);
  assert.equal(read.filter(x => x === 1).length, 1);
});
test('missing map indexes fail instead of quietly shrinking the candidate pool', async () => {
  const h = institutionHarness({
    fetchJsonOr404: async () => ({ results: [member('A1')] }), fetchIdBucket: async () => null
  });
  await assert.rejects(h.resolveInstitutionStart({ authors: 10 }, { id: 'I42' }, () => false), /researcher index/);
});
test('cancelling candidate resolution avoids subsequent bucket downloads', async () => {
  let calls = 0;
  const h = institutionHarness({ fetchJsonOr404: async () => ({ results: [member('A1')] }),
    fetchIdBucket: async () => { calls++; return new Map(); } });
  assert.equal(await h.resolveInstitutionStart({ authors: 10 }, { id: 'I42' }, () => true), null);
  assert.equal(calls, 0);
});

function connectHarness(overrides = {}) {
  const rendered = [], statuses = [];
  const context = vm.createContext({
    console, performance, pathGen: 1, PATH_MAX_HOPS: 14,
    connectA: { id: 'I42', name: 'Example University' }, connectB: { id: 'A9', name: 'Destination' },
    connectAffiliation: { textContent: '' },
    setPathStatus: text => statuses.push(text),
    resolveInstitutionStart: async () => ({ nodes: [1, 3], names: new Map([['A1', 'Institution Researcher']]), considered: 50 }),
    authorPathNumber: async () => 9, loadPathIndex() {}, suspectSteps: async () => [],
    browserShortestPath: async () => [1, 9], pathAuthorIds: async () => ['A1', 'A9'],
    fetchDisplayName: async id => id,
    sharedPaper: async () => ({ title: 'Shared paper', year: 2020, venue: 'Journal', url: 'https://doi.org/10.1/x' }),
    renderPath: async (...args) => rendered.push(plain(args)), ...overrides
  });
  vm.runInContext(connectCode, context);
  return { context, rendered, statuses };
}
test('institution results render actual researchers and paper labels, retaining institution share endpoints', async () => {
  const h = connectHarness();
  await h.context.connectInBrowser({}, 'I42', 'A9', 1);
  assert.equal(h.rendered[0][0][0].author_name, 'Institution Researcher');
  assert.equal(h.rendered[0][0][0].label, 'Shared paper');
  assert.equal(h.rendered[0][0][0].paper.url, 'https://doi.org/10.1/x');   // the paper reaches the hop list
  assert.deepEqual(h.rendered[0].slice(1), [1, 'I42', 'A9']);
  assert.match(h.context.connectAffiliation.textContent, /last-known affiliation: Example University/);
  assert.match(h.statuses.at(-1), /2 mapped institution candidates/);
});
test('zero-hop membership renders the destination researcher, not an institution masquerading as an author', async () => {
  const h = connectHarness({ browserShortestPath: async () => [9], pathAuthorIds: async () => ['A9'] });
  await h.context.connectInBrowser({}, 'I42', 'A9', 1);
  assert.equal(h.rendered[0][0][0].author_name, 'Destination');
  assert.equal(h.rendered[0][1], 0);
});
test('empty pools and search limits are reported as bounded failures', async () => {
  const empty = connectHarness({ resolveInstitutionStart: async () => ({ nodes: [], considered: 50 }) });
  await empty.context.connectInBrowser({}, 'I42', 'A9', 1);
  assert.match(empty.statuses.at(-1), /does not rule out a connection/);
  for (const result of [null, { tooFar: true }]) {
    const h = connectHarness({ browserShortestPath: async () => result });
    await h.context.connectInBrowser({}, 'I42', 'A9', 1);
    assert.equal(h.rendered.length, 0);
    assert.match(h.statuses.at(-1), /other institution researchers may connect|connection may still exist/);
  }
});
test('clearing or changing a selection during resolution prevents late results', async () => {
  let finish;
  const h = connectHarness({ resolveInstitutionStart: () => new Promise(resolve => { finish = resolve; }) });
  const searching = h.context.connectInBrowser({}, 'I42', 'A9', 1);
  h.context.pathGen++;
  finish({ nodes: [1], names: new Map(), considered: 1 });
  await searching;
  assert.equal(h.rendered.length, 0);
});
test('share links permit institution-to-author or author pairs and reject other endpoint types', () => {
  const context = vm.createContext({ stripId, location: { hash: '#1/0/0&p=I42,A9' } });
  vm.runInContext(section('function parseHash()', 'function writeHash()'), context);
  assert.deepEqual(plain(context.parseHash().p), ['I42', 'A9']);
  context.location.hash = '#1/0/0&p=A1,A9';
  assert.deepEqual(plain(context.parseHash().p), ['A1', 'A9']);
  for (const invalid of ['I42,I9', 'W1,A9', 'I42,A9&api_key=secret', 'I42foo,A9']) {
    context.location.hash = '#1/0/0&p=' + invalid;
    if (invalid.includes('&api_key')) assert.deepEqual(plain(context.parseHash().p), ['I42', 'A9']);
    else assert.equal(context.parseHash().p, undefined);
  }
});
test('institution share restoration selects the right mode and uses browser paths without a backend probe', async () => {
  let clicked = 0, synced = 0, probed = 0;
  const pair = ['I42', 'A9'];
  const context = vm.createContext({
    stripId, pendingP: pair, activeA: null, activePath: null, pathEventSource: null,
    connectPanel: { classList: { add() {} } }, connectToggle: { classList: { add() {} } },
    connectFromKind: { value: 'researcher' }, connectAInput: {}, connectBInput: {},
    syncConnectStartUI() { synced++; }, setPathStatus() {}, writeHash() {},
    fetchConnectStartName: async () => 'Institution', fetchDisplayName: async () => 'Researcher',
    pathsMeta: async () => ({ authors: 10 }), wakeBackend: async () => { probed++; return false; },
    connectGo: { click() { clicked++; } }
  });
  vm.runInContext(section('async function restoreFromHash(', '// Greedy screen-rect collision'), context);
  await context.restoreFromHash({ p: pair });
  assert.equal(context.connectFromKind.value, 'institution');
  assert.equal(context.connectAInput.value, 'Institution');
  assert.equal(context.connectBInput.value, 'Researcher');
  assert.equal(synced, 1);
  assert.equal(clicked, 1);
  assert.equal(probed, 0);
});
test('changing selection while a share link restores prevents its late Go', async () => {
  let finish, clicked = 0;
  const pair = ['I42', 'A9'];
  const context = vm.createContext({
    stripId, pendingP: pair, activeA: null, activePath: null, pathEventSource: null,
    connectPanel: { classList: { add() {} } }, connectToggle: { classList: { add() {} } },
    fetchConnectStartName: () => new Promise(resolve => { finish = resolve; }),
    fetchDisplayName: async () => 'Researcher', writeHash() {},
    connectGo: { click() { clicked++; } }
  });
  vm.runInContext(section('async function restoreFromHash(', '// Greedy screen-rect collision'), context);
  const restoring = context.restoreFromHash({ p: pair });
  context.pendingP = null;
  finish('Institution');
  await restoring;
  assert.equal(clicked, 0);
});
test('late lookup results cannot reopen an institution dropdown after a mode change', async () => {
  const timers = new Map(); let timerId = 0, finish;
  function element() {
    const classes = new Set();
    return { value: '', innerHTML: '', listeners: {},
      classList: { add: c => classes.add(c), remove: c => classes.delete(c), contains: c => classes.has(c) },
      addEventListener(type, callback) { this.listeners[type] = callback; },
      appendChild() {}, append() {}, blur() {} };
  }
  const input = element(), dropdown = element(), kind = { value: 'institution' };
  const context = vm.createContext({
    stripId, normalize: s => s.toLowerCase(), document: { createElement: element },
    setTimeout: fn => { const id = ++timerId; timers.set(id, fn); return id; },
    clearTimeout: id => timers.delete(id),
    fetchInstitutionMatches: () => new Promise(resolve => { finish = resolve; })
  });
  vm.runInContext(section('function createAuthorLookup(', 'const connectToggle ='), context);
  const lookup = context.createAuthorLookup(input, dropdown, () => {}, kind);
  input.value = 'Chalmers'; input.listeners.input();
  const running = [...timers.values()][0]();
  kind.value = 'researcher'; lookup.escape();
  finish([['chalmers', 'Chalmers', 'I42', null, null, 'SE']]);
  await running;
  assert.equal(lookup.isOpen(), false);
  assert.equal(input.value, '');
});
test('typing keeps the current matches visible until new ones arrive', async () => {
  const timers = new Map(); let timerId = 0;
  function element() {
    const classes = new Set();
    return { value: '', innerHTML: '', listeners: {},
      classList: { add: c => classes.add(c), remove: c => classes.delete(c), contains: c => classes.has(c) },
      addEventListener(type, callback) { this.listeners[type] = callback; },
      appendChild() {}, append() {}, blur() {} };
  }
  const input = element(), dropdown = element(), kind = { value: 'institution' };
  const context = vm.createContext({
    stripId, normalize: s => s.toLowerCase(), document: { createElement: element },
    setTimeout: fn => { const id = ++timerId; timers.set(id, fn); return id; },
    clearTimeout: id => timers.delete(id),
    fetchInstitutionMatches: async q => [[q.toLowerCase(), q, 'I42', null, null, 'Gothenburg, Sweden']]
  });
  vm.runInContext(section('function createAuthorLookup(', 'const connectToggle ='), context);
  const lookup = context.createAuthorLookup(input, dropdown, () => {}, kind);
  input.value = 'Chal'; input.listeners.input();
  await [...timers.values()].at(-1)();
  assert.equal(lookup.isOpen(), true);
  input.value = 'Chalm'; input.listeners.input();
  assert.equal(lookup.isOpen(), true);          // no blink while the next lookup is pending
});
test('the complete viewer script parses', () => {
  new vm.Script(html.match(/<script>\n([\s\S]*?)<\/script>/)[1]);
});
