// node --test tests/test_data_flags.cjs (Node 18+, no npm dependencies)
// R46: data flags (merged, split and non-person profiles) and the snapshot-name fallback.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../web/dev.html'), 'utf8');
function section(start, end) {
  const from = html.indexOf(start), to = html.indexOf(end, from);
  assert.ok(from >= 0 && to > from, start);
  return html.slice(from, to);
}
const flagsCode = section('// ---- R46: data flags ----', '// ---- End data flags ----');
const namesCode = section('const displayNameCache = new Map();', 'async function probeBackendHealth(');
const snapshotCode = section('async function snapshotName(', '// Search-path community resolution');
const cacheCode = section('function cacheUnlessFailed(', 'const tileCache =');

const raw = { release: '2026-09-23', flags: {
  merged: { n: [10, 11], id: [101, 102], peak: [984, 40], year: [2026, 2019] },
  split: { n: [20], id: [201], then: [11318], now: [277] },
  notPerson: { n: [30], id: [301] },
} };
function flagsHarness(answer = async () => raw, href = 'https://mapademic.app/#3/0/0') {
  const asked = [];
  const context = vm.createContext({ URL, location: new URL(href),
    fetchJsonOr404: async url => { asked.push(url); return answer(url); } });
  vm.runInContext(flagsCode, context);
  return { context, asked };
}

test('flags load once, from next to the page, keyed by author number and by id', async () => {
  const h = flagsHarness();
  const flags = await h.context.dataFlags();
  await h.context.dataFlags();
  assert.deepEqual(h.asked, ['https://mapademic.app/flags-2026-09-23.json']);
  assert.equal(flags.byNumber.get(10).kind, 'merged');
  assert.equal(flags.byNumber.get(20).now, 277);
  assert.equal(h.context.flagForId(flags, 'A5000000102').peak, 40);
  assert.equal(h.context.flagForId(flags, 'A5000000301').kind, 'notPerson');
  assert.equal(h.context.flagForId(flags, 'A5000000999'), undefined);
  // top-level const is not a property of the context's global object: read it as an expression
  assert.equal(vm.runInContext('FLAGS_URL', flagsHarness(undefined, 'http://localhost:8124/web/dev.html').context),
    'http://localhost:8124/web/flags-2026-09-23.json');
});
test('a missing file means no flags, and a failed load is retried', async () => {
  let n = 0;
  const h = flagsHarness(async () => { n++; if (n === 1) throw new Error('503'); return null; });
  const first = await h.context.dataFlags();
  assert.equal(first.byNumber.size, 0);
  const second = await h.context.dataFlags();
  assert.equal(second.byId.size, 0);
  assert.equal(n, 2);
});
test('the busiest affiliation year comes from the live record', () => {
  const { context } = flagsHarness();
  const data = { affiliations: [
    { institution: { display_name: 'A' }, years: [2019, 2020] },
    { institution: { display_name: 'B' }, years: [2020] },
    { institution: { display_name: 'C' }, years: [2020, 2021] },
  ] };
  assert.equal(JSON.stringify(context.peakAffiliations(data)), JSON.stringify({ count: 3, year: 2020 }));
  assert.equal(context.peakAffiliations({ affiliations: [] }), null);
  assert.equal(context.peakAffiliations(null), null);
});
test('card notes: flags first, then a neutral note for a busy live record, else none', async () => {
  const { context } = flagsHarness();
  const flags = await context.dataFlags();
  const note = (id, peak) => context.profileNote(context.flagForId(flags, id), peak);
  assert.match(note('A5000000101'), /^OpenAlex lists 984 institutions for this profile in 2026\. It probably merges several people/);
  assert.match(note('A5000000201'), /split this profile since the 2026-09-23 snapshot \(11,318 works then, 277 now\)/);
  assert.match(note('A5000000301'), /organization, team or placeholder rather than a person/);
  assert.match(note('A5000000999', { count: 31, year: 2015 }), /31 institutions .* in 2015\. That can mean several people share it, or one researcher/);
  assert.equal(note('A5000000999', { count: 24, year: 2015 }), null);
  assert.equal(note('A5000000999', null), null);
  assert.match(note('A5000000101', { count: 3, year: 2020 }), /984 institutions/);   // the snapshot's evidence wins
});
test('Connect step labels name the flagged researcher', () => {
  const { context } = flagsHarness();
  assert.equal(context.flagStepLabel({ kind: 'merged' }, 'Wei Wang'),
    "possible name mix-up in OpenAlex: Wei Wang's profile probably merges several people with this name");
  assert.equal(context.flagStepLabel({ kind: 'notPerson' }, 'unknown unknown'),
    'not a person in OpenAlex: unknown unknown looks like an organization or placeholder');
});

function namesHarness({ live, bucket, entry }) {
  const asked = [];
  const context = vm.createContext({
    Map,
    fetchJsonOr404: async url => { asked.push(url); return live(url); },
    idBucketNum: id => Number(String(id).replace(/\D/g, '')) % 1000,
    fetchIdBucket: async () => bucket,
    fetchZ9Entry: async (id, xw, yw) => { asked.push(`z9:${id}:${xw}:${yw}`); return entry; },
  });
  vm.runInContext(cacheCode + namesCode + snapshotCode, context);
  return { context, asked };
}
test('live OpenAlex names come first', async () => {
  const h = namesHarness({ live: async () => ({ display_name: 'Live Name' }), bucket: null, entry: null });
  assert.equal(await h.context.fetchDisplayName('A5000000001'), 'Live Name');
  assert.equal(h.asked.length, 1);
});
test('an unreachable OpenAlex or an unknown id falls back to the snapshot name, then the id', async () => {
  const bucket = new Map([['A5000000001', [0.25, 0.75, 7]]]);
  const entry = ['Snapshot Name', 'https://openalex.org/A5000000001', 0.25, 0.75, 10, 3, 0];
  let h = namesHarness({ live: async () => { throw new Error('429'); }, bucket, entry });
  assert.equal(await h.context.fetchDisplayName('A5000000001'), 'Snapshot Name');
  assert.ok(h.asked.includes('z9:A5000000001:0.25:0.75'));
  assert.equal(vm.runInContext('displayNameCache.size', h.context), 0);   // a failure is not cached: the next call retries live
  h = namesHarness({ live: async () => null, bucket, entry });
  assert.equal(await h.context.fetchDisplayName('A5000000001'), 'Snapshot Name');
  h = namesHarness({ live: async () => null, bucket: new Map(), entry: null });
  assert.equal(await h.context.fetchDisplayName('A5000000002'), 'A5000000002');
  h = namesHarness({ live: async () => null, bucket, entry: undefined });
  assert.equal(await h.context.snapshotName('A5000000001'), 'A5000000001');
});
