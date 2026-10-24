// node --test tests/test_share_card.cjs (Node 18+, no npm dependencies)
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
const context = vm.createContext({});
vm.runInContext(section('function normalize(name)', '\n}\n') + '\n}\n', context);
vm.runInContext(section('// What the card and the share sheet say', 'function fitText('), context);
const plain = value => JSON.parse(JSON.stringify(value));
const people = (...names) => names.map(name => ({ name }));

test('the card and share text name both ends and count the steps', () => {
  const s = context.shareChainSummary(people('S. R. Hinton', 'J. R. Mould', 'Allan Sandage', 'Edwin Hubble'), 3);
  assert.equal(s.steps, '3 steps');
  assert.equal(s.text, 'S. R. Hinton is 3 coauthor steps from Edwin Hubble.');
  assert.equal(s.fileName, 'mapademic-hinton-to-hubble.png');
  assert.equal(context.shareChainSummary(people('A B', 'C D'), 1).steps, '1 step');
});
test('file names survive accents and punctuation', () => {
  assert.equal(context.shareChainSummary(people('Svante Pääbo', 'Theodor W. Hänsch'), 2).fileName,
               'mapademic-paabo-to-hansch.png');
});
test('a long chain keeps both ends and marks the gap', () => {
  assert.deepEqual(plain(context.elideChain(4, 5)), [0, 1, 2, 3]);
  assert.deepEqual(plain(context.elideChain(10, 8)), [0, 1, 2, 3, null, 7, 8, 9]);
  assert.deepEqual(plain(context.elideChain(6, 5)), [0, 1, null, 4, 5]);
});
test('a step shows its paper and year, else its label', () => {
  assert.equal(context.shareStepPaper({ paper: { title: 'Taipan', year: 2017 } }), 'Taipan (2017)');
  assert.equal(context.shareStepPaper({ paper: { title: 'Untitled year' } }), 'Untitled year');
  assert.equal(context.shareStepPaper({ label: 'possible name mix-up in OpenAlex' }), 'possible name mix-up in OpenAlex');
  assert.equal(context.shareStepPaper({}), '');
});
