// Run with node --test tests/test_openalex_byok.cjs (no npm dependencies).
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');
const html = readFileSync(join(__dirname, '../web/dev.html'), 'utf8');
const byok = html.split('// ---- OpenAlex BYOK:')[1].split('// ---- End OpenAlex BYOK ----')[0];
const jsonHelpers = html.slice(html.indexOf('function fetchJsonOr404('), html.indexOf('const tileCache ='));

function harness({ stored, blocked = false } = {}) {
  const elements = new Map();
  const storage = new Map(stored ? [['mapademic.openalex-key', stored]] : []);
  const calls = [];
  let status = 200;
  const context = vm.createContext({
    URL, Map,
    location: { href: 'https://mapademic.app/#6/0/0' },
    document: { getElementById(id) {
      if (!elements.has(id)) elements.set(id, {
        value: '', textContent: '', listeners: {},
        addEventListener(type, callback) { this.listeners[type] = callback; }
      });
      return elements.get(id);
    } },
    sessionStorage: {
      getItem(key) { if (blocked) throw Error('blocked'); return storage.get(key) || null; },
      setItem(key, value) { if (blocked) throw Error('blocked'); storage.set(key, value); },
      removeItem(key) { if (blocked) throw Error('blocked'); storage.delete(key); }
    },
    fetch: async (url, options) => {
      calls.push({ url, options });
      return { status, ok: status === 200, json: async () => ({ display_name: 'Researcher' }) };
    }
  });
  // The marker's trailing description is a comment in the production script.
  vm.runInContext('// ' + byok + '\n' + jsonHelpers, context);
  return {
    elements, storage, calls, context,
    run: code => vm.runInContext(code, context),
    setStatus(value) { status = value; },
    use(key) {
      elements.get('openalex-key').value = key;
      elements.get('openalex-key-form').listeners.submit({ preventDefault() {} });
    },
    clear() { elements.get('openalex-key-clear').listeners.click(); }
  };
}

test('anonymous requests work; saved keys are masked, tab-scoped, and cleared from input', async () => {
  const h = harness();
  await h.run('fetchOpenAlex("https://api.openalex.org/works")');
  assert.equal(h.calls[0].options.headers.Authorization, undefined);
  h.use('  test-personal-key  ');
  assert.equal(h.elements.get('openalex-key').value, '');
  assert.equal(h.storage.get('mapademic.openalex-key'), 'test-personal-key');
  await h.run('fetchOpenAlex("https://api.openalex.org/people/A123?select=display_name")');
  const { url, options } = h.calls[1];
  assert.equal(options.headers.Authorization, 'Bearer test-personal-key');
  assert.equal(options.redirect, 'error');
  assert.equal(options.credentials, 'omit');
  assert.equal(options.cache, 'no-store');
  assert.equal(options.referrerPolicy, 'no-referrer');
  assert.ok(!url.includes('test-personal-key'));
  assert.match(html, /id="openalex-key" type="password"/);
});

test('restore, replace, and clear credentials across reloads', async () => {
  const h = harness({ stored: 'old-key' });
  assert.match(h.elements.get('openalex-key-status').textContent, /Using your key/);
  h.use('new-key');
  await h.run('fetchOpenAlex("https://api.openalex.org/works")');
  assert.equal(h.calls[0].options.headers.Authorization, 'Bearer new-key');
  h.clear();
  assert.equal(h.storage.size, 0);
  await h.run('fetchOpenAlex("https://api.openalex.org/works")');
  assert.equal(h.calls[1].options.headers.Authorization, undefined);
  assert.equal(h.elements.get('openalex-key').value, '');
});

test('disabled storage still permits an in-memory key and clear', async () => {
  const h = harness({ blocked: true });
  h.use('memory-key');
  assert.match(h.elements.get('openalex-key-status').textContent, /until reload/);
  await h.run('fetchOpenAlex("https://api.openalex.org/works")');
  assert.equal(h.calls[0].options.headers.Authorization, 'Bearer memory-key');
  h.clear();
  await h.run('fetchOpenAlex("https://api.openalex.org/works")');
  assert.equal(h.calls[1].options.headers.Authorization, undefined);
});

test('invalid input never replaces the working key', async () => {
  const h = harness({ stored: 'working-key' });
  for (const invalid of ['', 'a b', 'a\nb', 'a'.repeat(513), 'non-ascii-é']) {
    h.use(invalid);
    assert.match(h.elements.get('openalex-key-status').textContent, /Enter a key/);
  }
  await h.run('fetchOpenAlex("https://api.openalex.org/works")');
  assert.equal(h.calls[0].options.headers.Authorization, 'Bearer working-key');
});

test('credentials cannot be sent to data hosts, backends, HTTP, or lookalike origins', async () => {
  const h = harness({ stored: 'private-key' });
  for (const url of ['https://data.mapademic.app/data/index/test.json',
    'https://example.test/api/path', 'http://api.openalex.org/works',
    'https://api.openalex.org.evil.test/works', 'https://user:pass@api.openalex.org/works']) {
    await assert.rejects(h.run(`fetchOpenAlex(${JSON.stringify(url)})`));
  }
  assert.equal(h.calls.length, 0);
  await h.run('fetchJsonOr404("https://data.mapademic.app/data/index/test.json")');
  await h.run('fetchJsonOr404("/data/index/test.json")');
  assert.equal(h.calls[0].options, undefined);
  assert.equal(h.calls[1].options, undefined);
});

test('auth and rate-limit failures evict cached lookups so replacement keys can retry', async () => {
  for (const status of [401, 403, 429]) {
    const h = harness({ stored: 'rejected-key' });
    h.setStatus(status);
    h.run('var cache = new Map()');
    const result = await h.run('cacheUnlessFailed(cache, "A123", fetchJsonOr404("https://api.openalex.org/people/A123"), null)');
    assert.equal(result, null);
    assert.equal(h.run('cache.size'), 0);
    assert.equal(h.calls.length, 1); // no anonymous retry
    assert.ok(!h.elements.get('openalex-key-status').textContent.includes('rejected-key'));
    h.use('replacement-key');
    h.setStatus(200);
    const retried = await h.run('fetchJsonOr404("https://api.openalex.org/people/A123")');
    assert.equal(retried.display_name, 'Researcher');
    assert.equal(h.calls[1].options.headers.Authorization, 'Bearer replacement-key');
  }
});

test('missing records still resolve null; transient server errors reject', async () => {
  const h = harness();
  h.setStatus(404);
  assert.equal(await h.run('fetchJsonOr404("https://api.openalex.org/people/A123")'), null);
  h.setStatus(503);
  await assert.rejects(h.run('fetchJsonOr404("https://api.openalex.org/people/A123")'));
});

test('all live OpenAlex URLs use the credential gateway', () => {
  // Protect the four integrations (details, names, coauthors, shared papers)
  // against a future direct fetch bypassing the origin and credential policy.
  assert.doesNotMatch(html, /\bfetch\([`'"]https:\/\/api\.openalex\.org/);
  new vm.Script(html.match(/<script>\n([\s\S]*?)<\/script>/)[1]);
});
