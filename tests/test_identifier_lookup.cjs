// node --test tests/test_identifier_lookup.cjs (no npm dependencies)
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');
const html = readFileSync(join(__dirname, '../web/dev.html'), 'utf8');
const code = html.split('// ---- Identifier lookup helpers ----')[1].split('// ---- End identifier lookup helpers ----')[0];
const plain = x => JSON.parse(JSON.stringify(x));
function harness(overrides = {}) {
  const context = vm.createContext({ URL, URLSearchParams, normalize: x => x.toLowerCase(),
    idBucketNum: id => Number(id.slice(1)) % 1000, expandRepairedCandidates: async result => result,
    pooled: async tasks => Promise.all(tasks.map(task => task())), ...overrides });
  vm.runInContext(code, context);
  return context;
}

test('DOI forms preserve suffix punctuation and ORCID forms validate check digits, including X', () => {
  const { parseResearchIdentifier: parse } = harness();
  for (const value of ['10.1016/j.orbis.2022.02.007', 'doi:10.1016/j.orbis.2022.02.007',
    ' https://doi.org/10.1016/j.orbis.2022.02.007?via=copy ']) {
    assert.equal(parse(value, 'doi'), '10.1016/j.orbis.2022.02.007');
  }
  assert.equal(parse('https://dx.doi.org/10.1234/part%28a%29%3Fb', 'doi'), '10.1234/part(a)?b');
  for (const value of ['0000-0002-1825-0097', '0000000218250097', 'orcid:0000-0002-1825-0097',
    'https://orcid.org/0000-0002-1825-0097']) assert.equal(parse(value, 'orcid'), '0000-0002-1825-0097');
  assert.equal(parse('0000-0002-1694-233x', 'orcid'), '0000-0002-1694-233X');
});

test('invalid IDs and unrelated, credentialed, or misleading URLs are rejected before network access', async () => {
  const { fetchIdentifierCandidates } = harness({ fetchJsonOr404: () => assert.fail('unexpected network request') });
  for (const [query, kind] of [['0000-0002-1825-0098', 'orcid'], ['10.12/no', 'doi'],
    ['10.1234/a\nb', 'doi'], ['https://example.com/10.1234/a', 'doi'],
    ['https://doi.org.evil.test/10.1234/a', 'doi'], ['https://user:pass@doi.org/10.1234/a', 'doi'],
    ['https://orcid.org/0000-0002-1825-0097', 'doi'], ['https://doi.org/%zz', 'doi']]) {
    await assert.rejects(fetchIdentifierCandidates(query, kind));
  }
});

test('DOI lookup returns distinct same-name authors and the paper affiliations, never a name search', async () => {
  const calls = [];
  const { fetchIdentifierCandidates } = harness({ fetchJsonOr404: async url => {
    calls.push(url);
    return { title: 'Known paper', publication_year: 2022, authorships: [
      { raw_author_name: 'Alex Lee', author: { id: 'https://openalex.org/A1', display_name: 'A. Lee' }, institutions: [{ display_name: 'University One' }] },
      { raw_author_name: 'Alex Lee', author: { id: 'https://openalex.org/A2', display_name: 'Alex Lee' }, raw_affiliation_strings: ['University Two'] },
      { raw_author_name: 'Unresolved Author', author: { id: null } },
    ] };
  } });
  const result = await fetchIdentifierCandidates('10.1234/known?part#fragment', 'doi');
  assert.deepEqual(plain(result.candidates.map(c => [c.name, c.id, c.affiliation])), [
    ['Alex Lee', 'A1', 'University One'], ['Alex Lee', 'A2', 'University Two'], ['Unresolved Author', null, 'Affiliation unavailable'],
  ]);
  assert.match(result.message, /Known paper \(2022\)/);
  assert.equal(calls.length, 1);
  const url = new URL(calls[0]);
  assert.equal(url.origin, 'https://api.openalex.org');
  assert.equal(decodeURIComponent(url.pathname), '/works/doi:10.1234/known?part#fragment');
  assert.equal(url.hash, '');
  assert.equal(url.searchParams.has('search'), false);
});

test('ORCID uses the exact filter and preserves multiple matching profiles', async () => {
  let request;
  const { fetchIdentifierCandidates } = harness({ fetchJsonOr404: async url => {
    request = new URL(url);
    return { results: [
      { id: 'https://openalex.org/A5', display_name: 'Chris Lee', last_known_institutions: [{ display_name: 'Georgetown' }] },
      { id: 'https://openalex.org/A6', display_name: 'Chris Lee', orcid: 'https://orcid.org/0000-0002-1694-233X' },
    ], meta: { count: 2 } };
  }, fetchIdBucket: async () => new Map([['A5', [.2, .3]], ['A6', [.4, .5]]]) });
  const result = await fetchIdentifierCandidates('0000-0002-1825-0097', 'orcid');
  assert.equal(request.searchParams.get('filter'), 'orcid:0000-0002-1825-0097');
  assert.equal(request.searchParams.has('search'), false);
  assert.deepEqual(plain(result.candidates.map(c => c.id)), ['A5', 'A6']);
  assert.match(result.message, /2 mapped profiles/);
});

test('ORCID immediately excludes a live high-count profile absent from the published map', async () => {
  const c = harness({ fetchJsonOr404: async () => ({ results: [
    { id: 'A99', display_name: 'Off-map researcher', works_count: 10000 },
  ] }), fetchIdBucket: async () => new Map() });
  const result = await c.fetchIdentifierCandidates('0000-0002-1825-0097', 'orcid');
  assert.deepEqual(plain(result.candidates), []);
  assert.match(result.message, /not in the published map/i);
  assert.match(result.message, /15\+ qualifying works/);
});

test('ORCID release membership keeps a mapped researcher despite a low current count', async () => {
  const c = harness({ fetchJsonOr404: async () => ({ results: [
    { id: 'A5', display_name: 'Mapped researcher', works_count: 1 },
  ] }), fetchIdBucket: async () => new Map([['A5', [.2, .3]]]) });
  const result = await c.fetchIdentifierCandidates('0000-0002-1825-0097', 'orcid');
  assert.deepEqual(plain(result.candidates.map(p => p.id)), ['A5']);
});

test('ORCID checks expanded identities in shared buckets once and retains only mapped matches', async () => {
  const requests = [];
  const c = harness({ fetchJsonOr404: async () => ({ results: [
    { id: 'A5', display_name: 'Original duplicate' }, { id: 'A6', display_name: 'Other match' },
  ] }), expandRepairedCandidates: async result => ({ ...result, candidates: [
    { id: 'M1005', name: 'Repaired slice', works_count: 2 },
    { id: 'A5', name: 'Mapped original', works_count: 1 }, { id: 'A6', name: 'Absent match' },
  ] }), fetchIdBucket: async bucket => {
    requests.push(bucket);
    return new Map([['M1005', [.2, .3]], ['A5', [.4, .5]]]);
  } });
  const result = await c.fetchIdentifierCandidates('0000-0002-1825-0097', 'orcid');
  assert.deepEqual(plain(result.candidates.map(p => p.id)), ['M1005', 'A5']);
  assert.deepEqual(requests.sort(), [5, 6]);
  assert.match(result.message, /2 mapped profiles/);
});

test('an unavailable ORCID map bucket stays a retryable error instead of an off-map result', async () => {
  let attempt = 0;
  const c = harness({ fetchJsonOr404: async () => ({ results: [{ id: 'A5', display_name: 'Retry researcher' }] }),
    fetchIdBucket: async () => attempt++ === 0 ? null : new Map([['A5', [.2, .3]]]) });
  await assert.rejects(c.fetchIdentifierCandidates('0000-0002-1825-0097', 'orcid'), /Couldn't load.*retry/i);
  assert.equal((await c.fetchIdentifierCandidates('0000-0002-1825-0097', 'orcid')).candidates[0].id, 'A5');
  c.fetchIdBucket = async () => { throw new Error('Map request rate limited. Retry.'); };
  await assert.rejects(c.fetchIdentifierCandidates('0000-0002-1825-0097', 'orcid'), /rate limited/);
});

test('two ORCID source profiles merged into one map identity produce one result', async () => {
  const c = harness({ fetchJsonOr404: async () => ({ results: [
    { id: 'A5', display_name: 'Merged researcher' }, { id: 'A6', display_name: 'Merged researcher' },
  ] }), expandRepairedCandidates: async result => ({ ...result, candidates: [
    { id: 'M1005', name: 'Merged researcher', sourceId: 'A5' },
    { id: 'M1005', name: 'Merged researcher', sourceId: 'A6' },
  ] }), fetchIdBucket: async () => new Map([['M1005', [.2, .3]]]) });
  const result = await c.fetchIdentifierCandidates('0000-0002-1825-0097', 'orcid');
  assert.deepEqual(plain(result.candidates.map(p => p.id)), ['M1005']);
  assert.match(result.message, /1 mapped profile found/);
});

test('ORCID row deduplication preserves different people with the same name or numeric ID suffix', async () => {
  const c = harness({ fetchJsonOr404: async () => ({ results: [{ id: 'A5', display_name: 'Alex Lee' }] }),
    expandRepairedCandidates: async result => ({ ...result, candidates: [
      { id: 'M1005', name: 'Alex Lee' }, { id: 'M1005', name: 'Alex Lee' },
      { id: 'A1005', name: 'Alex Lee' }, { id: 'A7', name: 'Alex Lee' },
    ] }), fetchIdBucket: async () => new Map([
      ['M1005', [.2, .3]], ['A1005', [.4, .5]], ['A7', [.6, .7]],
    ]) });
  const result = await c.fetchIdentifierCandidates('0000-0002-1825-0097', 'orcid');
  assert.deepEqual(plain(result.candidates.map(p => p.id)), ['M1005', 'A1005', 'A7']);
  assert.match(result.message, /3 mapped profiles/);
});

test('missing identifiers, empty authors, and incomplete bylines have distinct messages', async () => {
  const states = [null, { title: 'No byline', authorships: [] }, { title: 'Large paper', authorships: [
    { author: { id: 'A1', display_name: 'First author' } },
  ], is_authors_truncated: true }, { results: [], meta: { count: 0 } }];
  const { fetchIdentifierCandidates } = harness({ fetchJsonOr404: async () => states.shift() });
  assert.match((await fetchIdentifierCandidates('10.1234/missing', 'doi')).message, /No paper/);
  assert.match((await fetchIdentifierCandidates('10.1234/empty', 'doi')).message, /No authors/);
  assert.match((await fetchIdentifierCandidates('10.1234/large', 'doi')).message, /incomplete author list/);
  assert.match((await fetchIdentifierCandidates('0000-0002-1825-0097', 'orcid')).message, /No OpenAlex profile/);
});

test('live author ID is kept through map selection, even for identical names', async () => {
  const buckets = new Map([['A1', [0.2, 0.3, 11]], ['A2', [0.8, 0.9, 22]]]);
  const { identifierMapEntry } = harness({ fetchIdBucket: async () => buckets });
  const candidate = { id: 'A2', name: 'Alex Lee', affiliation: 'Second institution', citations: 5 };
  assert.deepEqual(plain(await identifierMapEntry(candidate)), ['alex lee', 'Alex Lee', 'A2', 0.8, 0.9, 5, 'Second institution']);
});

test('off-map, unlinked, and failed map loads never substitute a same-name researcher', async () => {
  let calls = 0;
  const { identifierMapEntry } = harness({ fetchIdBucket: async () => { calls++; return new Map(); } });
  await assert.rejects(identifierMapEntry({ id: null, name: 'Alex Lee' }), /no OpenAlex author ID/);
  assert.equal(calls, 0);
  await assert.rejects(identifierMapEntry({ id: 'A3', name: 'Alex Lee' }), /not in the map/);
  const failed = harness({ fetchIdBucket: async () => null });
  await assert.rejects(failed.identifierMapEntry({ id: 'A3' }), /Couldn't load map positions/);
});

test('rate limits and transient failures remain errors and a retry can succeed', async () => {
  let attempt = 0;
  const { fetchIdentifierCandidates } = harness({ fetchJsonOr404: async () => {
    if (attempt++ === 0) throw new Error('OpenAlex rate limit reached.');
    return { results: [{ id: 'A3', display_name: 'Found after retry' }] };
  }, fetchIdBucket: async () => new Map([['A3', [.2, .3]]]) });
  await assert.rejects(fetchIdentifierCandidates('0000-0002-1825-0097', 'orcid'), /rate limit/);
  assert.equal((await fetchIdentifierCandidates('0000-0002-1825-0097', 'orcid')).candidates[0].id, 'A3');
});

test('the entire inline viewer script parses', () => {
  new vm.Script(html.slice(html.indexOf('<script>') + 8, html.lastIndexOf('</script>')));
});

function dialogHarness(overrides = {}) {
  class Element {
    constructor() { this.children = []; this.listeners = {}; this.value = ''; this.textContent = ''; this.open = false; }
    append(...nodes) { this.children.push(...nodes); }
    replaceChildren(...nodes) { this.children = nodes; }
    addEventListener(event, fn) { this.listeners[event] = fn; }
    focus() {}
    showModal() { this.open = true; }
    close() { this.open = false; this.listeners.close?.(); }
  }
  const elements = new Map();
  const element = id => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
  const selected = [];
  const context = vm.createContext({
    document: { getElementById: element, createElement: () => new Element(), querySelector: element },
    clearTimeout() {}, searchTimer: null, searchGen: 0, closeSearchDropdown() {},
    connectALookup: { pause() {}, select: entry => selected.push(['a', entry]) },
    connectBLookup: { pause() {}, select: entry => selected.push(['b', entry]) },
    searchInput: element('search-input'), selectSearchEntry: entry => selected.push(['search', entry]),
    parseResearchIdentifier: value => value, ...overrides,
  });
  const controller = html.split('// ---- Identifier lookup dialog ----')[1].split('// ---- End identifier lookup dialog ----')[0];
  vm.runInContext(controller, context);
  return { context, element, selected, submit: () => element('identifier-form').listeners.submit({ preventDefault() {} }) };
}

test('identifier selection reaches exactly the intended search or Connect field', async () => {
  const entry = ['alex lee', 'Alex Lee', 'A2', 0.8, 0.9, 0, 'University Two'];
  const h = dialogHarness({
    fetchIdentifierCandidates: async () => ({ message: 'Two namesakes', candidates: [
      { id: 'A1', name: 'Alex Lee' }, { id: 'A2', name: 'Alex Lee' },
    ] }), identifierMapEntry: async candidate => { assert.equal(candidate.id, 'A2'); return entry; },
  });
  for (const [button, target] of [['search-identifier', 'search'], ['connect-a-identifier', 'a'], ['connect-b-identifier', 'b']]) {
    h.element(button).listeners.click();
    await h.submit();
    const secondResult = h.element('identifier-results').children[1];
    await secondResult.children.at(-1).listeners.click();
    assert.equal(h.selected.at(-1)[0], target);
    assert.equal(h.selected.at(-1)[1], entry);
    assert.equal(h.element('identifier-dialog').open, false);
  }
});

test('editing or closing a lookup discards late API responses and late map selections', async () => {
  let resolveAPI;
  const h = dialogHarness({ fetchIdentifierCandidates: () => new Promise(resolve => { resolveAPI = resolve; }) });
  const response = { message: 'Old result', candidates: [{ id: 'A1', name: 'Old query' }] };
  h.element('search-identifier').listeners.click();
  let pending = h.submit();
  h.element('identifier-query').listeners.input();
  resolveAPI(response);
  await pending;
  assert.equal(h.element('identifier-results').children.length, 0);
  assert.equal(h.element('identifier-status').textContent, '');
  pending = h.submit();
  h.element('identifier-close').listeners.click();
  resolveAPI(response);
  await pending;
  assert.equal(h.element('identifier-results').children.length, 0);
  h.context.fetchIdentifierCandidates = async () => response;
  let resolveMap;
  h.context.identifierMapEntry = () => new Promise(resolve => { resolveMap = resolve; });
  h.element('connect-b-identifier').listeners.click();
  await h.submit();
  pending = h.element('identifier-results').children[0].children.at(-1).listeners.click();
  h.element('identifier-close').listeners.click();
  resolveMap(['old', 'Old query', 'A1', 0, 0, 0]);
  await pending;
  assert.equal(h.selected.length, 0);
});

test('a queued native close event from the previous lookup cannot cancel a reopened lookup', async () => {
  let resolveAPI;
  const h = dialogHarness({
    fetchIdentifierCandidates: () => new Promise(resolve => { resolveAPI = resolve; }),
    identifierMapEntry: async () => ['paul d miller', 'Paul D. Miller', 'A5088470719', 0.4, 0.5, 0],
  });
  h.element('connect-a-identifier').listeners.click();
  // HTMLDialogElement.close queues its event; it can arrive after a new open.
  h.element('identifier-dialog').close = function () { this.open = false; };
  h.element('identifier-close').listeners.click();
  h.element('connect-b-identifier').listeners.click();
  const pending = h.submit();
  h.element('identifier-dialog').listeners.close();
  resolveAPI({ message: 'Miller found', candidates: [{ id: 'A5088470719', name: 'Paul D. Miller' }] });
  await pending;
  assert.equal(h.element('identifier-results').children.length, 1);
  await h.element('identifier-results').children[0].children.at(-1).listeners.click();
  assert.equal(h.selected[0][0], 'b');
  assert.equal(h.selected[0][1][2], 'A5088470719');
});

test('changing the cross-check field cancels old evidence, and conflicting rows are disabled', async () => {
  let resolve;
  const h = dialogHarness({ fetchIdentifierCandidates: () => new Promise(r => { resolve = r; }) });
  h.element('search-identifier').listeners.click();
  let pending = h.submit();
  h.element('identifier-check').listeners.input();
  resolve({ message: 'Old pair', candidates: [{ id: 'A1', name: 'Old pair' }] });
  await pending;
  assert.equal(h.element('identifier-results').children.length, 0);
  pending = h.submit();
  resolve({ message: 'Conflict', candidates: [{ id: 'A1', name: 'Different author',
    evidence: { state: 'conflict', message: 'Publisher gives a different ORCID.', sources: [] } }] });
  await pending;
  const button = h.element('identifier-results').children[0].children.at(-1);
  assert.equal(button.disabled, true);
  assert.match(button.textContent, /Resolve conflicting evidence/);
});
