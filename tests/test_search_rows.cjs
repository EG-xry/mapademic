// node --test tests/test_search_rows.cjs (Node 18+, no npm dependencies)
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
const rowsCode = section('// ---- Search rows (issue #4) ----', '// ---- End search rows ----');

// Just enough DOM for the row builders.
class FakeElement {
  constructor(tag) { this.tag = tag; this.children = []; this.className = ''; this.textContent = ''; this.title = ''; this.listeners = {}; }
  append(...kids) { this.children.push(...kids); }
  addEventListener(type, fn) { this.listeners[type] = fn; }
}
function rows() {
  const context = vm.createContext({ document: { createElement: tag => new FakeElement(tag) } });
  vm.runInContext(rowsCode, context);
  return context;
}
function parts(row) {
  const [name, meta] = row.children, [inst, n] = meta.children;
  return { name: name.textContent, inst: inst.textContent, instClass: inst.className, title: inst.title, n: n.textContent };
}

test('a researcher row shows the last-known institution and labelled citations', () => {
  const { researcherRow } = rows();
  const row = researcherRow(['wei wang', 'Wei Wang', 'A1', 0.3, 0.3, 1234, 'Nanjing Tech University'], true, () => {});
  assert.equal(row.className, 'row researcher active');
  assert.deepEqual(parts(row), {
    name: 'Wei Wang', inst: 'Nanjing Tech University', instClass: 'inst', title: 'Nanjing Tech University',
    n: `${(1234).toLocaleString()} citations`
  });
});

test('six-field entries and missing institutions show the unknown state', () => {
  const { researcherRow } = rows();
  for (const e of [['wei wang', 'Wei Wang', 'A2', 0.4, 0.4, 7], ['wei wang', 'Wei Wang', 'A3', 0.4, 0.4, 7, '']]) {
    const p = parts(researcherRow(e, false, () => {}));
    assert.equal(p.inst, 'institution unknown');
    assert.equal(p.instClass, 'inst unknown');
    assert.equal(p.title, '');
  }
  assert.equal(parts(researcherRow(['x', 'X', 'A4', 0, 0, 1], false, () => {})).n, '1 citation');
});

test('picking a row hands back the entry, so selections stay tied to OpenAlex ids', () => {
  const { researcherRow } = rows();
  const e = ['wei wang', 'Wei Wang', 'A1', 0.3, 0.3, 50, 'Nanjing Tech University'];
  let picked = null, prevented = false;
  researcherRow(e, false, x => { picked = x; }).listeners.mousedown({ preventDefault() { prevented = true; } });
  assert.equal(picked, e);
  assert.ok(prevented);
});

test('the caption names the institution as last known in the snapshot', () => {
  const { researcherNote } = rows();
  assert.match(researcherNote().textContent, /last-known institutions in the snapshot/i);
});
