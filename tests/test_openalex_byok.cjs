// Run with node --test tests/test_openalex_byok.cjs (no npm dependencies).
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');
const html = readFileSync(join(__dirname, '../web/dev.html'), 'utf8');
const byok = html.split('// ---- OpenAlex BYOK:')[1].split('// ---- End OpenAlex BYOK ----')[0];
const jsonHelpers = html.slice(html.indexOf('function fetchJsonOr404('), html.indexOf('const tileCache ='));

function harness({
  stored, storedRaw, blocked = false, rejectWrites = false,
  jar = new Map(), href = 'https://mapademic.app/#6/0/0'
} = {}) {
  const elements = new Map();
  if (stored !== undefined) jar.set('mapademic.openalex-key', encodeURIComponent(stored));
  if (storedRaw !== undefined) jar.set('mapademic.openalex-key', storedRaw);
  const cookieWrites = [];
  const calls = [];
  let status = 200;
  const location = new URL(href);
  const document = {
    get cookie() {
      if (blocked) throw Error('cookies blocked');
      return [...jar].map(([name, value]) => `${name}=${value}`).join('; ');
    },
    set cookie(value) {
      if (blocked) throw Error('cookies blocked');
      cookieWrites.push(value);
      if (rejectWrites) return;
      const [pair, ...attributes] = value.split(';').map(part => part.trim());
      const equals = pair.indexOf('=');
      const name = pair.slice(0, equals);
      const encoded = pair.slice(equals + 1);
      const attrs = new Map(attributes.map(attribute => {
        const index = attribute.indexOf('=');
        return index < 0 ? [attribute.toLowerCase(), true]
          : [attribute.slice(0, index).toLowerCase(), attribute.slice(index + 1)];
      }));
      if (attrs.has('secure') && location.protocol !== 'https:') return;
      if (attrs.has('max-age') && Number(attrs.get('max-age')) <= 0) jar.delete(name);
      else jar.set(name, encoded);
    },
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, {
        value: '', textContent: '', listeners: {},
        addEventListener(type, callback) { this.listeners[type] = callback; }
      });
      return elements.get(id);
    }
  };
  const context = vm.createContext({
    URL, Map,
    location, document,
    fetch: async (url, options) => {
      calls.push({ url, options });
      return { status, ok: status === 200, json: async () => ({ display_name: 'Researcher' }) };
    }
  });
  // The marker's trailing description is a comment in the production script.
  vm.runInContext('// ' + byok + '\n' + jsonHelpers, context);
  return {
    elements, jar, cookieWrites, calls, context,
    run: code => vm.runInContext(code, context),
    setStatus(value) { status = value; },
    use(key) {
      elements.get('openalex-key').value = key;
      elements.get('openalex-key-form').listeners.submit({ preventDefault() {} });
    },
    clear() { elements.get('openalex-key-clear').listeners.click(); }
  };
}

function cookieAttributes(write) {
  const [pair, ...attributes] = write.split(';').map(part => part.trim());
  assert.match(pair, /^mapademic\.openalex-key=/);
  return new Map(attributes.map(attribute => {
    const index = attribute.indexOf('=');
    return index < 0 ? [attribute.toLowerCase(), true]
      : [attribute.slice(0, index).toLowerCase(), attribute.slice(index + 1)];
  }));
}

test('anonymous requests work; saved keys are masked and cleared from input', async () => {
  const h = harness();
  await h.run('fetchOpenAlex("https://api.openalex.org/works")');
  assert.equal(h.calls[0].options.headers.Authorization, undefined);
  h.use('  test-personal-key  ');
  assert.equal(h.elements.get('openalex-key').value, '');
  assert.equal(h.jar.get('mapademic.openalex-key'), 'test-personal-key');
  assert.match(h.elements.get('openalex-key-status').textContent, /30 days/);
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

test('restore, replace, and clear credentials across pages and reloads', async () => {
  const h = harness({ stored: 'old-key' });
  assert.match(h.elements.get('openalex-key-status').textContent, /Using your saved key/);
  await h.run('fetchOpenAlex("https://api.openalex.org/works")');
  assert.equal(h.calls[0].options.headers.Authorization, 'Bearer old-key');
  h.use('new-key');
  const nextPage = harness({ jar: h.jar, href: 'https://mapademic.app/other-page' });
  await nextPage.run('fetchOpenAlex("https://api.openalex.org/works")');
  assert.equal(nextPage.calls[0].options.headers.Authorization, 'Bearer new-key');
  nextPage.clear();
  assert.equal(nextPage.jar.size, 0);
  assert.equal(nextPage.elements.get('openalex-key').value, '');
  await nextPage.run('fetchOpenAlex("https://api.openalex.org/works")');
  assert.equal(nextPage.calls[1].options.headers.Authorization, undefined);
  const reloaded = harness({ jar: nextPage.jar });
  await reloaded.run('fetchOpenAlex("https://api.openalex.org/works")');
  assert.equal(reloaded.calls[0].options.headers.Authorization, undefined);
});

test('keys use an encoded, 30-day, host-only cookie with matching clear attributes', async () => {
  const key = 'key;=,%/?:@&+#!$[]';
  const h = harness();
  h.use(key);
  assert.equal(h.jar.get('mapademic.openalex-key'), encodeURIComponent(key));
  const saved = cookieAttributes(h.cookieWrites.at(-1));
  assert.equal(saved.get('max-age'), '2592000');
  assert.equal(saved.get('path'), '/');
  assert.equal(saved.get('samesite').toLowerCase(), 'strict');
  assert.equal(saved.get('secure'), true);
  assert.equal(saved.has('domain'), false);
  const reloaded = harness({ jar: h.jar });
  await reloaded.run('fetchOpenAlex("https://api.openalex.org/works")');
  assert.equal(reloaded.calls[0].options.headers.Authorization, `Bearer ${key}`);
  h.clear();
  const cleared = cookieAttributes(h.cookieWrites.at(-1));
  assert.equal(cleared.get('max-age'), '0');
  assert.equal(cleared.get('path'), saved.get('path'));
  assert.equal(cleared.get('samesite').toLowerCase(), saved.get('samesite').toLowerCase());
  assert.equal(cleared.get('secure'), true);
  assert.equal(cleared.has('domain'), false);
  assert.equal(h.jar.size, 0);
});

test('HTTP local development can save and clear the key without Secure', async () => {
  const h = harness({ href: 'http://localhost:8000/web/dev.html' });
  h.use('local-key');
  assert.equal(cookieAttributes(h.cookieWrites.at(-1)).has('secure'), false);
  const reloaded = harness({ jar: h.jar, href: 'http://localhost:8000/web/dev.html' });
  await reloaded.run('fetchOpenAlex("https://api.openalex.org/works")');
  assert.equal(reloaded.calls[0].options.headers.Authorization, 'Bearer local-key');
  h.clear();
  assert.equal(cookieAttributes(h.cookieWrites.at(-1)).has('secure'), false);
  assert.equal(h.jar.size, 0);
});

test('only the exact credential cookie is loaded among unrelated cookies', async () => {
  const jar = new Map([
    ['other-mapademic.openalex-key', 'wrong-prefix-key'],
    ['mapademic.openalex-key-extra', 'wrong-suffix-key'],
    ['display-settings', 'unrelated']
  ]);
  const withoutKey = harness({ jar });
  await withoutKey.run('fetchOpenAlex("https://api.openalex.org/works")');
  assert.equal(withoutKey.calls[0].options.headers.Authorization, undefined);
  jar.set('mapademic.openalex-key', encodeURIComponent('right;key=value'));
  const withKey = harness({ jar });
  await withKey.run('fetchOpenAlex("https://api.openalex.org/works")');
  assert.equal(withKey.calls[0].options.headers.Authorization, 'Bearer right;key=value');
  withKey.clear();
  assert.equal(jar.size, 3);
  assert.equal(jar.get('display-settings'), 'unrelated');
});

test('malformed or invalid saved cookies are ignored', async () => {
  for (const storedRaw of ['%E0%A4%A', '', encodeURIComponent('a b'),
    encodeURIComponent('a\nb'), encodeURIComponent('non-ascii-é'), 'a'.repeat(513)]) {
    const h = harness({ storedRaw });
    await h.run('fetchOpenAlex("https://api.openalex.org/works")');
    assert.equal(h.calls[0].options.headers.Authorization, undefined);
    assert.match(h.elements.get('openalex-key-status').textContent, /without a key/);
  }
});

test('blocked cookies and silently rejected writes permit an in-memory key and clear', async () => {
  for (const options of [{ blocked: true }, { rejectWrites: true }]) {
    const h = harness(options);
    h.use('memory-key');
    assert.match(h.elements.get('openalex-key-status').textContent, /until reload/);
    await h.run('fetchOpenAlex("https://api.openalex.org/works")');
    assert.equal(h.calls[0].options.headers.Authorization, 'Bearer memory-key');
    const reloaded = harness({ ...options, jar: h.jar });
    await reloaded.run('fetchOpenAlex("https://api.openalex.org/works")');
    assert.equal(reloaded.calls[0].options.headers.Authorization, undefined);
    h.clear();
    await h.run('fetchOpenAlex("https://api.openalex.org/works")');
    assert.equal(h.calls[1].options.headers.Authorization, undefined);
  }
});

test('rejected replacement and clear writes report failure when the old cookie remains', async () => {
  const h = harness({ stored: 'old-key', rejectWrites: true });
  h.use('memory-key');
  assert.match(h.elements.get('openalex-key-status').textContent, /until reload/);
  await h.run('fetchOpenAlex("https://api.openalex.org/works")');
  assert.equal(h.calls[0].options.headers.Authorization, 'Bearer memory-key');
  h.clear();
  await h.run('fetchOpenAlex("https://api.openalex.org/works")');
  assert.equal(h.calls[1].options.headers.Authorization, undefined);
  assert.equal(h.jar.get('mapademic.openalex-key'), 'old-key');
  assert.match(h.elements.get('openalex-key-status').textContent, /this page/);
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

test('the credential gateway rejects data hosts, backends, HTTP, and lookalike origins', async () => {
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
