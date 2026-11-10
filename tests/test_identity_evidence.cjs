const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../web/dev.html'), 'utf8');
const code = html.split('// ---- Identifier lookup helpers ----')[1].split('// ---- End identifier lookup helpers ----')[0];
const doi = '10.1017/9781108876544', orcid = '0000-0003-0443-3957', other = '0000-0002-7287-1353';
const normalize = text => text.normalize('NFKD').replace(/\p{M}/gu, '').toLowerCase().replace(/[^\p{L}\p{N}]/gu, ' ').trim().replace(/\s+/g, ' ');
function harness(extra = {}) {
  const c = vm.createContext({ URL, URLSearchParams, AbortController, setTimeout, clearTimeout, normalize,
    expandRepairedCandidates: async result => result, ...extra });
  vm.runInContext(code, c);
  return c;
}
function example() {
  return {
    candidate: { id: 'A5088470719', name: 'Paul D. Miller', orcid, paperIndex: 0 },
    paper: { doi: 'https://doi.org/' + doi, authorships: [{ raw_author_name: 'Paul D. Miller', raw_orcid: null,
      author: { id: 'https://openalex.org/A5088470719', orcid } }] },
    crossref: { DOI: doi, author: [{ given: 'Paul D.', family: 'Miller' }] },
    registry: { 'orcid-identifier': { path: orcid }, person: { name: { 'given-names': { value: 'Paul' }, 'family-name': { value: 'Miller' } } },
      'activities-summary': { works: { group: [{ 'work-summary': [{ 'external-ids': { 'external-id': [
        { 'external-id-type': 'doi', 'external-id-value': doi, 'external-id-relationship': 'self' },
      ] } }] }] } } },
  };
}
function assess(e, overrides = {}) {
  const c = harness();
  return c.publicationIdentityEvidence(e.candidate, doi, orcid, e.paper, e.crossref, e.registry, overrides.unavailable || []);
}
test('a profile ORCID alone never becomes publication corroboration (the Boyle failure)', () => {
  const e = example(); e.registry['activities-summary'].works.group = [];
  const r = assess(e);
  assert.equal(r.state, 'unconfirmed');
  assert.match(r.message, /Missing or private data is not a contradiction/);
});
test('a DOI explicitly listed in ORCID plus a compatible unique publisher byline corroborates only that publication', () => {
  const r = assess(example());
  assert.equal(r.state, 'corroborated');
  assert.match(r.message, /supports this publication only/);
  assert.equal(r.sources.length, 2);
});
test('an empty or private ORCID works list is missing evidence, not an identity conflict', () => {
  const e = example(); e.registry = { 'orcid-identifier': { path: orcid } };
  assert.equal(assess(e).state, 'unconfirmed');
});
test('a deposited authorship ORCID is separate from the profile ORCID', () => {
  const e = example(); e.registry = null; e.crossref = null;
  e.paper.authorships[0].raw_orcid = orcid; e.paper.authorships[0].author.orcid = other;
  assert.equal(assess(e).state, 'corroborated');
  e.paper.authorships[0].raw_orcid = other; e.paper.authorships[0].author.orcid = orcid;
  assert.equal(assess(e).state, 'conflict');
});
test('conflicting publisher evidence takes precedence over a matching registry work', () => {
  const e = example(); e.crossref.author[0].ORCID = 'https://orcid.org/' + other;
  const r = assess(e); assert.equal(r.state, 'conflict'); assert.match(r.message, /different ORCID/);
});
test('publisher metadata assigning the ORCID to another author is a reviewable conflict', () => {
  const e = example(); e.crossref.author.push({ given: 'Michael', family: 'Boyle', ORCID: orcid });
  assert.equal(assess(e).state, 'conflict');
});
test('an authenticated publisher ORCID retains its provenance and works when ORCID API is unavailable', () => {
  const e = example(); Object.assign(e.crossref.author[0], { ORCID: orcid, 'authenticated-orcid': true }); e.registry = null;
  const r = assess(e, { unavailable: ['ORCID'] });
  assert.equal(r.state, 'corroborated'); assert.match(r.message, /authenticated/); assert.match(r.message, /ORCID could not be checked/);
});
test('a DOI for a parent volume does not corroborate the chapter DOI', () => {
  const e = example(); e.registry['activities-summary'].works.group[0]['work-summary'][0]['external-ids']['external-id'][0]['external-id-relationship'] = 'part-of';
  assert.equal(assess(e).state, 'unconfirmed');
});
test('identical names and a shared ORCID on a multi-author paper do not silently identify a byline', () => {
  const e = example(); e.paper.authorships.push({ raw_author_name: 'Paul D. Miller', author: { id: 'A999' } });
  assert.equal(assess(e).state, 'unconfirmed');
  e.crossref.author = [{ given: 'Paul D.', family: 'Miller', ORCID: orcid }, { given: 'Michael', family: 'Boyle', ORCID: orcid }];
  assert.equal(assess(e).state, 'conflict');
});
test('wrong DOI responses, wrong ORCID responses, and incompatible registry names cannot confirm identity', () => {
  let e = example(); e.paper.doi = '10.1234/other'; assert.equal(assess(e).state, 'unavailable');
  e = example(); e.registry['orcid-identifier'].path = other; assert.equal(assess(e).state, 'unconfirmed');
  e = example(); e.registry.person.name['given-names'].value = 'Michael'; assert.equal(assess(e).state, 'unconfirmed');
  e = example(); e.crossref.DOI = '10.1234/other'; assert.equal(assess(e).state, 'unconfirmed');
  e = example(); e.candidate.id = 'A999'; assert.equal(assess(e).state, 'unconfirmed');
});
test('ORCID profile without a unique authorship on the submitted paper is unconfirmed', () => {
  const e = example(); delete e.candidate.paperIndex;
  e.paper.authorships[0].author.id = 'A999'; assert.equal(assess(e).state, 'unconfirmed');
  e.paper.authorships.push({ raw_author_name: 'Another Author', author: { id: 'A5088470719' } });
  e.paper.authorships.push({ raw_author_name: 'Paul D. Miller', author: { id: 'A5088470719' } });
  assert.equal(assess(e).state, 'unconfirmed');
});
test('network failures remain incomplete checks, not conflicts or positive identities', () => {
  const e = example(); e.crossref = null; e.registry = null;
  assert.equal(assess(e, { unavailable: ['Crossref', 'ORCID'] }).state, 'unavailable');
});
test('the pair checker uses fixed public sources and validates both fields before requests', async () => {
  const e = example(), requests = [];
  const c = harness({ fetchJsonOr404: async url => { requests.push(url); return e.paper; } });
  c.fetchIdentitySource = async url => { requests.push(url); return url.includes('crossref') ? { message: e.crossref } : e.registry; };
  const r = await c.fetchIdentifierCandidates(doi, 'doi', orcid);
  assert.equal(r.candidates[0].evidence.state, 'corroborated');
  assert.equal(requests.length, 3);
  requests.length = 0;
  await assert.rejects(c.fetchIdentifierCandidates(doi, 'doi', 'not-an-orcid'));
  assert.equal(requests.length, 0);
});
test('a conflict cannot be selected even if the row button is bypassed', async () => {
  const c = harness({ fetchIdBucket: () => assert.fail('must not load positions for a conflict') });
  await assert.rejects(c.identifierMapEntry({ id: 'A1', evidence: { state: 'conflict' } }), /Conflicting publication evidence/);
});
test('evidence APIs receive no credentials or OpenAlex key, and other origins are rejected', async () => {
  let args;
  const c = harness({ fetch: async (...a) => { args = a; return { ok: true, json: async () => ({}) }; } });
  await c.fetchIdentitySource('https://pub.orcid.org/v3.0/' + orcid + '/record');
  assert.equal(args[1].credentials, 'omit'); assert.deepEqual(Object.keys(args[1].headers), ['Accept']);
  await assert.rejects(c.fetchIdentitySource('https://attacker.test/data'));
});
