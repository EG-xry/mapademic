// node --test tests/test_shared_paper.cjs (Node 18+, no npm dependencies)
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../web/dev.html'), 'utf8');
const from = html.indexOf('// The paper behind one step'), to = html.indexOf('async function authorPathNumber(', from);
assert.ok(from >= 0 && to > from);
const code = html.slice(from, to);
const plain = value => JSON.parse(JSON.stringify(value));

function harness(answer) {
  const asked = [];
  const context = vm.createContext({
    fetchJsonOr404: async url => { asked.push(url); return answer(url, asked.length); },
  });
  vm.runInContext(code, context);
  return { sharedPaper: context.sharedPaper, asked };
}
const work = (over = {}) => ({
  id: 'https://openalex.org/W1', title: 'Small paper', publication_year: 2017,
  doi: 'https://doi.org/10.1/small', primary_location: { landing_page_url: 'https://example.org/p',
    source: { display_name: 'A Journal', type: 'journal' } }, ...over });

test('asks only for works the coauthor network counts: 2-50 authors, no excluded types', async () => {
  const h = harness(() => ({ results: [work()] }));
  assert.deepEqual(plain(await h.sharedPaper('A1', 'A2')),
    { title: 'Small paper', year: 2017, venue: 'A Journal', url: 'https://doi.org/10.1/small' });
  assert.equal(h.asked.length, 1);
  const filter = new URL(h.asked[0]).searchParams.get('filter');
  assert.match(filter, /^authorships\.author\.id:A1,authorships\.author\.id:A2,authors_count:<51,/);
  for (const type of ['dataset', 'software', 'erratum', 'retraction', 'paratext']) assert.match(filter, new RegExp(`type:!${type}(,|$)`));
});
test('falls back to any shared work when none qualifies', async () => {
  const h = harness((url, n) => ({ results: n === 1 ? [] : [work({ title: 'Big collaboration' })] }));
  assert.equal((await h.sharedPaper('A1', 'A2')).title, 'Big collaboration');
  assert.equal(h.asked.length, 2);
  assert.doesNotMatch(new URL(h.asked[1]).searchParams.get('filter'), /authors_count/);
});
test('a repository is not a venue, and the link prefers DOI, then landing page, then OpenAlex', async () => {
  const repo = { landing_page_url: 'https://figshare.com/x', source: { display_name: 'Figshare', type: 'repository' } };
  let h = harness(() => ({ results: [work({ doi: null, primary_location: repo })] }));
  assert.deepEqual(plain(await h.sharedPaper('A1', 'A2')),
    { title: 'Small paper', year: 2017, venue: null, url: 'https://figshare.com/x' });
  h = harness(() => ({ results: [work({ doi: null, primary_location: null })] }));
  assert.equal((await h.sharedPaper('A1', 'A2')).url, 'https://openalex.org/W1');
});
test('never links anything but http(s)', async () => {
  const h = harness(() => ({ results: [work({ doi: 'javascript:alert(1)',
    primary_location: { landing_page_url: 'data:text/html,x' }, id: 'ftp://example.org/W1' })] }));
  assert.equal((await h.sharedPaper('A1', 'A2')).url, null);
});
test('no shared work, or OpenAlex failing, gives no paper rather than an error', async () => {
  assert.equal(await harness(() => ({ results: [] })).sharedPaper('A1', 'A2'), null);
  assert.equal(await harness(() => { throw new Error('429'); }).sharedPaper('A1', 'A2'), null);
});
